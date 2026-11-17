'''
Main GRS processor: from a L1C image to the L2A product (lazy, written chunk by chunk).

Memory / dask design
====================

``prod.raster`` is normally dask-backed. Nothing of full size is computed in ``Process.execute``:

* the gaseous absorption correction and the interpolation of the wind map are applied block by block
  (``xr.map_blocks``): the coarse maps are interpolated to the grid of each block;
* the aerosol retrieval works on a coarse raster (``aot_resolution`` pixel size) which is computed once;
* ``Kernel.final_process`` returns the lazy L2A variables; the flag for negative Rrs, the mask and the
  wind map are lazy too;
* everything is evaluated in a single pass by ``write_output`` (one ``dask.compute`` in ``L2aProduct``), so
  ``Rrs`` and the intermediate arrays only exist chunk by chunk.

Number of threads: ``execute`` and ``write_output`` run the dask threaded scheduler with ``processor.ncpu``
workers (configuration), instead of one per core. Every worker holds its blocks in memory, so this sets the
peak memory (about 150 MB per worker for 512 px blocks). The speed stops increasing at about 8 workers
(GIL, memory bandwidth; measured on 40 cores), so more workers only cost memory.

Format of the output: ``write_output`` takes it from the name of the output path (``.zarr`` or ``.nc``).
'''
import os
from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr

# keep attributes through operatyion on xarray objects
xr.set_options(keep_attrs=True)
import rioxarray as rio
import logging
import gc
import functools

import dask

import GRSdriver

from . import utils, remote, __version__
from .config import load_config, lut_path, check_luts, ensure_luts
from .product import Product
from .cams import CamsProduct
from .output import L2aProduct, quicklook_coarsen, quicklook_factor, quicklook_rgb
from .mask import Masking
from .grs_kernel import _zero_crossing, Kernel
from .grs_exceptions import GRS_IO_Exception

opj = os.path.join

NETCDF_ENGINE = load_config()['processor']['netcdf_engine']

# extensions that tell the output format (see write_output)
_OUTPUT_EXTENSIONS = ('.zarr', '.nc', '.nc4', '.cdf')
# arrays smaller than this (bytes) are loaded in memory once
_SMALL_ARRAY_BYTES = 512 * 2 ** 20


def _compute_small(obj, max_bytes=_SMALL_ARRAY_BYTES):
    '''Load a small dask-backed xarray object in memory (once); anything else is returned unchanged.'''
    if isinstance(obj, (xr.DataArray, xr.Dataset)) and obj.chunks and obj.nbytes <= max_bytes:
        return obj.compute()
    return obj


def _correct_gas_absorption(bands, Tg_raster):
    '''
    Divide the bands by the gaseous transmittance (coarse grid, interpolated to the pixels).

    * dask-backed `bands`: lazy, done block by block, the transmittance is only interpolated on the block
      (no full-size interpolation, no sequence of assignments into a big array);
    * numpy-backed `bands`: in place band by band, as before.

    In both cases the result keeps the dtype of `bands`.

    :param bands: DataArray (wl, y, x) of the TOA reflectance
    :param Tg_raster: DataArray of the gaseous transmittance on a coarse grid, with the wavelengths of `bands`
    :return: corrected bands (same object as `bands` for numpy-backed data)
    '''
    if bands.chunks is None:
        for wl in bands.wl.values:
            bands.loc[wl] = bands.sel(wl=wl) / Tg_raster.sel(wl=wl).interp(x=bands.x, y=bands.y)
        return bands

    # without its scalar coordinates: a spatial_ref different from the one of the bands would be dropped from
    # the result by the division, and map_blocks would reject the block
    Tg = _compute_small(Tg_raster.sel(wl=bands.wl)).reset_coords(drop=True)

    def block(b):
        corrected = b / Tg.interp(x=b.x, y=b.y)
        corrected = corrected.drop_vars([c for c in corrected.coords if c not in b.coords])
        return corrected.transpose(*b.dims).astype(b.dtype)

    return xr.map_blocks(block, bands, template=bands)


def _interp_like(coarse, like, name=None):
    '''
    Interpolate a coarse DataArray (dimensions x, y) onto the grid of `like`, block by block.

    Lazy if `like` is dask-backed: only the part of the coarse map needed by a block is interpolated when
    the block is computed (no full-size array in memory). Same values as ``coarse.interp(x=like.x, y=like.y)``;
    the scalar coordinates of `coarse` are kept.

    :param coarse: DataArray on a coarse grid (loaded in memory if small)
    :param like: DataArray (y, x) giving the target grid and chunks
    :param name: name of the result
    :return: DataArray (y, x) float64
    '''
    coarse = _compute_small(coarse)
    like = like.reset_coords(drop=True)
    template = xr.full_like(like, np.nan, dtype=np.float64)

    def block(b):
        out = coarse.interp(x=b.x, y=b.y).reset_coords(drop=True)
        return out.transpose(*b.dims).astype(np.float64)

    out = xr.map_blocks(block, like, template=template).rename(name or coarse.name)
    scalar_coords = {k: v for k, v in coarse.coords.items() if k not in coarse.dims}
    return out.assign_coords(scalar_coords)


def _with_workers(method):
    '''Run a `Process` method with ``self.Nproc`` dask workers (threaded scheduler; None: one per core).'''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with dask.config.set(num_workers=self.Nproc):
            return method(self, *args, **kwargs)
    return wrapper


def _check_remote_driver(l1c_prod):
    '''
    Raise a clear error if the installed GRSdriver cannot read a Level-1 product from a URL.

    :raises GRS_IO_Exception: without ``GRSdriver.remote``
    '''
    import importlib.util
    if importlib.util.find_spec('GRSdriver.remote') is None:
        raise GRS_IO_Exception(l1c_prod, RuntimeError(
            f'reading a Level-1 product from a URL needs a GRSdriver version with remote reading '
            f'(GRSdriver.remote), installed: {GRSdriver.__version__}'))


def no_water_marker(odir):
    '''
    Path of the marker file written instead of the product for an image without water (see
    ``Process.execute(min_water_area=...)``), so that batch runs can skip it like a processed image:
    ``<directory>/<basename>_no_water.txt``, with the directory and base name of the product
    (``/data/scene.zarr`` -> ``/data/scene_no_water.txt``; a directory without extension, as given by the
    ``grs`` command, ``/data/scene`` -> ``/data/scene/scene_no_water.txt``).

    :param odir: output path given to `Process.execute` (local path or URL)
    :return: Path, or URL string for a remote output
    '''
    odir = str(odir).rstrip('/')
    if remote.is_url(odir):
        name = remote.basename(odir)
        stem, ext = os.path.splitext(name)
        if ext.lower() in _OUTPUT_EXTENSIONS:
            return odir[:-len(name)] + stem + '_no_water.txt'
        return remote.join(odir, name + '_no_water.txt')
    path = Path(odir)
    if path.suffix.lower() in _OUTPUT_EXTENSIONS:
        return path.with_name(path.stem + '_no_water.txt')
    return path / (path.name + '_no_water.txt')


def _water_area(prod):
    '''
    Area (km²) of the water pixels of the Level-1 image, with the water test of the processing:
    (blue - SWIR1) / (blue + SWIR1) > ``vis_swir_index_threshold`` and SWIR2 < ``sunglint_threshold``, on the
    TOA reflectance (before the gaseous correction and the cloud mask, so it is only used to skip the images
    without water).

    :param prod: :py:class:`~grs.product.Product` with the TOA reflectance ``prod.raster.bands``
    :return: area in km² (float)
    '''
    bands = prod.raster.bands
    vis = bands.sel(wl=prod.bvis, method='nearest')
    swir = bands.sel(wl=prod.bswir, method='nearest')
    swir2 = bands.sel(wl=prod.bswir2, method='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        water = ((vis - swir) / (vis + swir) > prod.vis_swir_index_threshold) & (swir2 < prod.sunglint_threshold)
    npix = int(water.sum())
    pixel_area = abs(float(prod.raster.x[1] - prod.raster.x[0]) * float(prod.raster.y[1] - prod.raster.y[0]))
    return npix * pixel_area / 1e6


def _cams_from_archive(raster, cams_dir):
    """
    CAMS data of an image from the ``YYYY/MM/DD`` archive of `cams_dir` (``path: cams`` of the
    configuration): global file of the day if it exists, else the file of the tile, downloaded if needed.
    A file of the former layout ``<cams_dir>/<tile>/<date>-forecast_<tile>.nc`` is still used.
    """
    tile = raster.attrs['tile']
    cams = CamsProduct(raster, dir=cams_dir, suffix='_' + tile)
    legacy = os.path.join(cams_dir, tile, f'{cams.date_str}-forecast_{tile}.nc')
    if not os.path.exists(cams.filepath) and os.path.exists(legacy):
        cams = CamsProduct(raster, cams_file=legacy)
    logging.info('CAMS file: %s', cams.filepath)
    return cams


# settings of the aerosol retrieval, ``aerosol`` section of the configuration
AEROSOL_SETTINGS = ('aot_resolution', 'min_pixel_prop', 'aot_chunk', 'outlier_factor', 'aot_smoothing_blocks',
                    'fine_step', 'fine_window', 'aot_step')


# optional settings (default when missing in the configuration): hybrid estimator under sunglint
AEROSOL_OPTIONAL_SETTINGS = {'glint_aot': False, 'glint_aot_min_glint': 0.02}


def _aerosol_settings(**overrides):
    """
    Settings of the aerosol retrieval: ``aerosol`` section of the configuration, overridden by the arguments
    that are not None.
    """
    config = load_config().get('aerosol') or {}
    missing = [key for key in AEROSOL_SETTINGS if key not in config and overrides.get(key) is None]
    if missing:
        raise KeyError(f'aerosol settings missing in the configuration: {missing}')
    settings = {key: config.get(key) for key in AEROSOL_SETTINGS}
    settings.update({key: config.get(key, default) for key, default in AEROSOL_OPTIONAL_SETTINGS.items()})
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def _glint_hybrid_aot(aot_ref_img, min_glint):
    """
    Hybrid AOT of the blocks under strong sunglint.

    Under sunglint the retrieval of `Kernel.aerosol_visible_chunk` keeps the top of the admissible AOT window
    and over-estimates the AOT. The scene glint-based AOT is the AOT at which the slope of the corrected
    visible reflectance against the sunglint, pooled over the blocks under strong sunglint (block means
    removed: sum of ``glint_sxy`` / sum of ``glint_sxx``), crosses zero. In these blocks it replaces
    ``aot_ref``, clipped to the admissible window ``[aot_low, aot_ref]``; the other blocks keep ``aot_ref``.

    :param aot_ref_img: merged block results of `Kernel.aerosol_visible_chunk`
    :param min_glint: strong sunglint: block median ``glint_ref`` above this value
    :return: (aot_ref_img with the hybrid ``aot_ref``, scene glint-based AOT or None)
    """
    if 'glint_sxy' not in aot_ref_img:
        return aot_ref_img, None
    strong = (aot_ref_img.glint_ref > min_glint) & (aot_ref_img.number_of_pixel > 0) & (aot_ref_img.glint_sxx > 0)
    if not bool(strong.any()):
        logging.info('glint-based AOT: no block under strong sunglint, retrieval unchanged')
        return aot_ref_img, None
    # NaN (outside the tested values of a block) where any block is NaN: the pooled slope uses all the blocks
    block_sxy = aot_ref_img.glint_sxy.where(strong)
    sxy = block_sxy.sum(['y', 'x']).where(block_sxy.notnull().sum(['y', 'x']) == int(strong.sum()))
    slope = sxy / float(aot_ref_img.glint_sxx.where(strong).sum())
    aot_glint = _zero_crossing(slope.aot_grid.values, slope.values)
    if not np.isfinite(aot_glint):
        logging.info('glint-based AOT: no crossing of the pooled slope (%d blocks), retrieval unchanged',
                     int(strong.sum()))
        return aot_ref_img, None
    low = aot_ref_img.aot_low.fillna(0.)
    hybrid = np.minimum(np.maximum(aot_glint, low), aot_ref_img.aot_ref)
    logging.info('glint-based AOT %.3f, applied to %d blocks under strong sunglint', aot_glint, int(strong.sum()))
    return aot_ref_img.assign(aot_ref=aot_ref_img.aot_ref.where(~strong, hybrid)), aot_glint


class Process:
    '''
    Main GRS class.

    '''

    def __init__(self):
        # configuration read at run time (see grs.config), so that it can be changed without reinstalling
        config = load_config()
        self.lut_file = str(lut_path('toa'))
        self.trans_lut_file = str(lut_path('transmittance'))
        self.cams_dir = config['path']['cams']
        self.Nproc = config['processor']['ncpu']
        # device of OmniCloudMask (grs.mask.Masking), 'auto': GPU when available
        self.ocm_device = config['processor'].get('ocm_device', 'cpu')
        self.pressure_ref = 101500.
        # mask recipe (bits of grs.mask): keep the water index pixels (3); mask nodata (0), SURFWATER cloud
        # and shadow (10), OmniCloudMask thick cloud, thin cloud and shadow (11-13), Landsat dilated cloud (14),
        # snow and ice (21, 22) and negative Rrs (23)
        self.flags_tokeep = [3]
        self.flags_tomask = [0, 10, 11, 12, 13, 14, 21, 22, 23]
        self.min_water_area = config['settings'].get('min_water_area', 0.)
        # wave slope probability in the sunglint normalisation (Kernel.p_slope), off by default
        self.p_slope = bool(config['settings'].get('p_slope', False))
        self.successful = False
        # True when execute stopped because the image has (almost) no water (see min_water_area)
        self.no_water = False
        self.l2a = None

    @_with_workers
    def execute(self, l1c_prod,
                odir='',
                cams_file=None,
                surfwater_file=None,
                dem_file=None,
                resolution=20,
                megapix_size=4,
                chunk=512,
                aot_resolution=None,
                aot_chunk=None,
                aot_smoothing_blocks=None,
                min_pixel_prop=None,
                outlier_factor=None,
                fine_step=None,
                fine_window=None,
                aot_step=None,
                glint_aot=None,
                glint_aot_min_glint=None,
                scale_aot=1,
                opac_model=None,
                allpixels=False,
                snap_compliant=False,
                provide_kernel=False,
                min_water_area=None,
                p_slope=None,
                storage_options=None
                ):

        '''
        Main program calling all GRS steps

        The processing is lazy: when the function returns, ``self.l2a.l2_prod`` holds dask-backed variables
        (``Rrs``, ``BRDFg``, ``aot550``, ``flags``, ``mask``, ``wind``...) that are computed, chunk by chunk,
        by ``write_output`` (or by an explicit ``.compute()``).

        The settings of the aerosol retrieval (``aot_resolution`` to ``aot_step``) default to the ``aerosol``
        section of the configuration (see :py:func:`grs.config.load_config`): None takes the value of the
        configuration, another value overrides it.

        :param l1c_prod: xarray L1C object or L1C input file (path) to be processed. The files (Level-1, CAMS,
            SURFWATER, DEM) can be local paths or URLs (``s3://``, ``https://``..., see :mod:`grs.remote`): the
            Level-1 bands are read in place by GRSdriver (needs GRSdriver with remote reading,
            ``GRSdriver.remote``), the other files are copied once into a local cache
        :param odir: Output path, see `write_output` (``.zarr`` / ``.nc`` extension tells the format)
        :param cams_file: CAMS file of the image (global daily or monthly file, or file of the area); None:
            taken from (or downloaded into) the ``YYYY/MM/DD`` archive of ``path: cams`` of the
            configuration, see :py:class:`~grs.cams.CamsProduct`
        :param surfwater_file: Absolute path the surfwater file (.tif)
        :param dem_file: Absolute path of the DEM geotiff file
        :param resolution: pixel resolution in meter
        :param megapix_size: not used for the moment (the wind comes from CAMS, see the TODO below)
        :param chunk: size (pixels) of the blocks of the final processing; 512 divides the 1024 shards of
            the Zarr output. A block needs ~150 MB (512) or ~600 MB (1024) per thread.
        :param aot_resolution: pixel size (meter) of the coarse raster of the aerosol retrieval, e.g. 60, 120,
            240 (default), 480; must be a multiple of ``resolution``
        :param storage_options: fsspec options of the remote input files (e.g. ``key``, ``secret``,
            ``endpoint_url`` for S3); default: the environment (``AWS_PROFILE``, ``AWS_ENDPOINT_URL``...)
        :param aot_chunk: size (pixels, on the coarse raster) of the blocks of the aerosol retrieval; one AOT
            value is retrieved per block of ``aot_chunk * aot_resolution`` meters (default 64: 15 km at 240 m)
        :param aot_smoothing_blocks: size, in blocks, of the window of the weighted mean of the block AOT values
            (default 3: each block is averaged with its neighbours)
        :param min_pixel_prop: coarse pixels with fewer water pixels than this proportion of the fullest one are
            discarded (default 0.5)
        :param outlier_factor: blocks with an AOT above ``outlier_factor`` times the median get the median AOT
            and a null weight (default 1.5)
        :param fine_step: the transitions between blocks are smoothed on a grid of ``fine_step`` coarse pixels
            (default 5)
        :param fine_window: size (cells of this grid) of the NaN-mean window of this smoothing; None in the
            configuration: about one block (default)
        :param aot_step: AOT step of the look-up tables of the final processing (default 0.01)
        :param glint_aot: hybrid AOT under strong sunglint (see `_glint_hybrid_aot`; default False)
        :param glint_aot_min_glint: strong sunglint: median sunglint reflectance of the reference band of a block
            above this value (default 0.02)
        :param scale_aot: scaling factor applied to CAMS aod550 raster
        :param opac_model: If set force OPAC aerosol model for LUT interpolation (taken from CAMS data otherwise)
        :param allpixels: if True process all pixels (no water and cloud masking); otherwise only the clear-sky
            (OmniCloudMask class 0) water pixels are processed
        :param snap_compliant: Output format compliant with SNAP software for practical analysis
        :param provide_kernel: dev option, return the kernel before the retrieval
        :param min_water_area: images with less water (km²) are not processed: the function stops right
            after reading the image, sets ``self.no_water`` and returns None, and ``write_output`` writes
            the marker file `no_water_marker` instead of the product. None: ``settings: min_water_area`` of
            the configuration (1 km²); 0: stop only without any water pixel. Not used with `allpixels`.
            The processing stops in the same way (``self.no_water``, None returned, marker file) when no
            clear-sky water pixel is left after the cloud masking (water entirely under clouds).
        :param p_slope: use the Cox-Munk wave slope probability in the sunglint normalisation (see
            :py:meth:`grs.grs_kernel.Kernel.p_slope`, not validated); None: ``settings: p_slope`` of the
            configuration (False by default: P_slope = 1)
        :return: the kernel (None if the image has no water or no clear-sky water pixel)

        Examples
        --------

        >>> import grs
        >>> file ='$YOUR_PATH_TO_IMG/S2A_MSIL1C_20181019T102031_N0500_R065_T30PYT_20230815T043850.SAFE'
        >>> tile = file.split('_')[-2][1:]
        >>> dem_file = '$YOUR_PATH_TO_DEM/COP-DEM_GLO-30-DGED_'+tile+'.tif'
        >>> cams_file = '$YOUR_PATH_TO_CAMS/cams_forecast_2018-10.nc'
        >>> process_ = grs.Process()

        In this example, you force the aerosol model to be 'DESE_rh70' for desert dust

        >>> process_.execute(file_nc,
        ...                 cams_file=cams_file,
        ...                 surfwater_file=None,
        ...                 dem_file=dem_file,
        ...                 scale_aot=1,
        ...                 opac_model='DESE_rh70')
        INFO:root:pass netcdf image as grs product object
        INFO:root:get CAMS auxilliary data
        INFO:root:flagging from l1c data
        INFO:root:OmniCloudMask processing on 606 x 841 pixels (60 m)
        INFO:root:loading look-up tables
        INFO:root:compute gaseous transmittance from cams data
        INFO:root:correct for gaseous absorption
        INFO:root:compute spectral index (e.g., NDWI)
        INFO:root:apply water and cloud masking
        INFO:root:lut interpolation
        INFO:root:selected aerosol model: DESE_rh70
        INFO:root:scaling aot by: 1
        INFO:root:set final parameters
        INFO:root:compute surface pressure from dem
        INFO:root:run grs process
        INFO:root:success
        INFO:root:construct final product
        INFO:root:construct l2a

        >>> process_.l2a.l2_prod
        <xarray.Dataset>
        Dimensions:      (wl: 11, y: 1818, x: 2523)
        Coordinates:
          * wl           (wl) int64 443 490 560 665 705 740 783 842 865 1610 2190
            time         datetime64[ns] 2018-10-19T10:20:31.024000
          * x            (x) float64 7.299e+05 7.299e+05 7.3e+05 ... 7.803e+05 7.804e+05
          * y            (y) float64 1.3e+06 1.3e+06 1.3e+06 ... 1.264e+06 1.264e+06
            band         int64 1
            spatial_ref  int64 0
        Data variables:
            Rrs          (wl, y, x) float32 nan nan nan nan nan ... nan nan nan nan nan
            BRDFg        (y, x) float32 nan nan nan nan nan nan ... nan nan nan nan nan
            aot550       (y, x) float32 0.187 0.187 0.187 0.187 ... 0.1929 0.1929 0.1929
            vza          (y, x) float32 6.489 6.489 6.483 6.483 ... 2.13 2.125 2.125
            sza          (y, x) float32 nan nan nan nan nan nan ... nan nan nan nan nan
            raa          (y, x) float32 332.5 332.5 332.5 332.5 ... 290.0 289.9 289.9
            flags    (y, x) int64 184 56 184 184 184 184 ... 160 160 160 160 160 160
            dem          (y, x) float32 282.9 282.7 282.5 282.3 ... 237.3 237.3 237.4
            surfwater    (y, x) int8 1 1 1 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1 1 1 1 1
        Attributes: (12/71)
            long_name:                           CA BLUE GREEN RED VRE_1 VRE_2 VRE_3 ...
            constellation:                       Sentinel-2
            constellation_id:                    S2
            product_path:                        /data/satellite/Sentinel-2/L1C/30PYT...
            product_name:                        S2A_MSIL1C_20181019T102031_N0500_R06...
            product_filename:                    S2A_MSIL1C_20181019T102031_N0500_R06...
            ...                                  ...
            ndwi_threshold:                      0.0
            vis_swir_index_threshold:            0.0
            hcld_threshold:                      0.003
            dirdata:                             /data/grs/grsdata
            abs_gas_file:                        /home/harmel/Dropbox/Dropbox/work/gi...
            water_vapor_transmittance_file:      /home/harmel/Dropbox/Dropbox/work/gi...

        You can either further play with the l2a xarray or save it (zarr or netcdf, from the name):


        >>> process_.odir='./name_of_your_output_l2a.zarr'
        >>> process_.write_output()
        INFO:root:export final product

        '''

        self.odir = odir
        self.snap_compliant = snap_compliant
        self.successful = False
        self.no_water = False
        self.l2a = None

        # stop before reading the image if the LUTs are missing; the LUTs of the configuration are downloaded
        # from Zenodo the first time (see grs.config.ensure_luts), LUT files set by hand are only checked
        if [self.lut_file, self.trans_lut_file] == [str(lut_path('toa')), str(lut_path('transmittance'))]:
            ensure_luts()
        else:
            check_luts([self.lut_file, self.trans_lut_file])

        ##################################
        # Get image data
        ##################################
        if isinstance(l1c_prod, str | PathLike):
            # get extension (of a local path or of a URL)
            name = remote.basename(l1c_prod)
            extension = Path(name).suffix
            basename = Path(name).stem
            if remote.is_url(l1c_prod) and 'nc' not in extension:
                _check_remote_driver(l1c_prod)
            driver_kwargs = {'storage_options': storage_options} if remote.is_url(l1c_prod) else {}
            if 'nc' in extension:
                logging.info('pass netcdf image as grs product object')
                prod = Product(xr.open_dataset(remote.local_file(l1c_prod, storage_options), engine=NETCDF_ENGINE))
            elif 'SAFE' in extension:
                logging.info('Open L1C Sentinel 2 image and compute angle parameters')
                try:
                    l1c = GRSdriver.Sentinel2Driver(str(l1c_prod), resolution=resolution, lazy_geom=True,
                                                    **driver_kwargs)
                    l1c.load_product()
                except Exception as exc:
                    raise GRS_IO_Exception(l1c_prod, exc)
                logging.info('pass raw image as grs product object')
                prod = Product(l1c.prod)
                # clear memory (TODO make it work!!)
                del l1c
                gc.collect()
            elif ('LC09_L1' in basename) or ('LC08_L1' in basename):
                logging.info('Open L1TP Landsat image')
                try:
                    l1c = GRSdriver.LandsatDriver(str(l1c_prod), resolution=resolution, **driver_kwargs)
                    l1c.load_mask()
                    l1c.load_product()
                except Exception as exc:
                    raise GRS_IO_Exception(l1c_prod, exc)
                logging.info('pass raw image as grs product object')
                prod = Product(l1c.prod)
                # clear memory (TODO make it work!!)
                del l1c
                gc.collect()
            else:
                logging.info('input file format not recognized, stop')
                return
        elif isinstance(l1c_prod, xr.Dataset):
            try:
                prod = Product(l1c_prod)
            except Exception as exc:
                logging.info('input file format not recognized, stop')
                raise GRS_IO_Exception(l1c_prod, exc)
        else:
            raise TypeError('l1c_prod must be a file path or an xarray Dataset, got ' + type(l1c_prod).__name__)

        self.prod = prod

        ##################################
        # Stop early if the image has (almost) no water
        ##################################
        if min_water_area is None:
            min_water_area = self.min_water_area
        if not allpixels:
            area = _water_area(prod)
            if area <= 0 or area < min_water_area:
                logging.warning('no water to process: %.3f km² of water pixels (minimum %.3f km²), '
                                'image not processed', area, min_water_area)
                self.no_water = True
                self.no_water_info = {'image': str(prod.raster.attrs.get('product_name', l1c_prod)
                                                   if isinstance(l1c_prod, xr.Dataset) else l1c_prod),
                                      'water_area_km2': round(area, 4),
                                      'min_water_area_km2': min_water_area}
                return None

        ##################################
        # GET ANCILLARY DATA (Pressure, O3, water vapor, NO2...
        ##################################
        logging.info('get CAMS auxilliary data')
        try:
            if cams_file:
                cams = CamsProduct(prod.raster, cams_file=remote.local_file(cams_file, storage_options))
            else:
                cams = _cams_from_archive(prod.raster, self.cams_dir)
            cams.load(daily_stats=False)
        except Exception as exc:
            raise GRS_IO_Exception(cams_file, exc)

        ##################################
        # Pixel classification
        # Generate the flags raster
        ##################################
        logging.info('flagging from l1c data')

        if surfwater_file:
            try:
                logging.info('loading surfwater data file')
                prod.raster['surfwater'] = rio.open_rasterio(remote.local_file(surfwater_file, storage_options)
                                                             ).astype(np.uint8
                                                                      ).squeeze().interp(x=prod.x,
                                                                                         y=prod.y,
                                                                                         method='nearest')
                prod.raster.surfwater.name = 'surfwater'
                prod.raster.surfwater.attrs = {
                    'description': 'surfwater file not provided as input, all pixels flagged as water (e.g., surfwater=1)'}
            except Exception as exc:
                raise GRS_IO_Exception(surfwater_file, exc)

        masking_ = Masking(prod.raster)
        masking_.OCM_inference_device = masking_.OCM_mosaic_device = self.ocm_device
        prod.raster = masking_.process(output="prod")
        ocm_device_used = getattr(masking_, 'OCM_device_used', None)

        # -- clean up
        prod.raster = prod.raster.drop_vars(["surfwater"])

        #####################################
        # SUBSET RASTER TO KEEP REQUESTED BANDS
        #####################################
        # TODO check if we can remove cirrus and water vapor band from output object
        # copies: a selection is a view that would keep all the L1C bands in memory
        if prod.bcirrus:
            prod.cirrus = prod.raster.bands.sel(wl=prod.bcirrus, method='nearest').copy()
        if prod.bwv:
            prod.wv = prod.raster.bands.sel(wl=prod.bwv, method='nearest').copy()

        prod.raster = prod.raster.sel(wl=prod.wl_process, method='nearest')

        # coarse L1C RGB of the quicklook, taken before the gas correction and the water masking of the bands
        # (lazy for dask-backed bands: evaluated with the export, in the same pass)
        prod.quicklook_l1c = quicklook_coarsen(quicklook_rgb(prod.raster.bands),
                                               quicklook_factor(prod.raster.sizes['y'], prod.raster.sizes['x']))

        #####################################
        # LOAD LUT FOR ATMOSPHERIC CORRECTION
        #####################################
        logging.info('loading look-up tables')
        try:
            trans_lut = xr.open_dataset(self.trans_lut_file, engine=NETCDF_ENGINE)
        except Exception as exc:
            raise GRS_IO_Exception(self.trans_lut_file, exc)
        # convert wavelength into nm
        trans_lut['wl'] = trans_lut['wl'] * 1000

        try:
            aero_lut = xr.open_dataset(self.lut_file, engine=NETCDF_ENGINE)
        except Exception as exc:
            raise GRS_IO_Exception(self.lut_file, exc)
        aero_lut['wl'] = aero_lut['wl'] * 1000
        aero_lut['aot'] = aero_lut.aot.isel(wind=0).squeeze()

        ####################################
        # absorbing gases correction
        ####################################
        logging.info('compute gaseous transmittance from cams data')
        try:
            gas_trans = utils.GaseousTransmittance(prod, cams)
            gases = ['co2', 'o2', 'o4', 'ch4', 'no2', 'o3', 'h2o']
            for gas in gases:
                gas_trans.coef_abs_scat[gas] = 1
            Tg_raster = gas_trans.get_gaseous_transmittance(gases=['o3', 'no2'])
        except Exception as exc:
            raise GRS_IO_Exception(cams_file, exc)

        logging.info('correct for gaseous absorption')
        bands_attrs = dict(prod.raster.bands.attrs)
        prod.raster['bands'] = _correct_gas_absorption(prod.raster.bands, Tg_raster)
        prod.raster.bands.attrs.update(bands_attrs)
        prod.raster.bands.attrs['gas_absorption_correction'] = True

        ######################################
        # Water mask
        ######################################
        # TODO remove ndwi export / replace this part with flags masking instead
        logging.info('compute spectral index (e.g., NDWI)')

        vis = prod.raster.bands.sel(wl=prod.bvis, method='nearest')
        nir = prod.raster.bands.sel(wl=prod.bnir, method='nearest')
        swir = prod.raster.bands.sel(wl=prod.bswir, method='nearest')
        swir2 = prod.raster.bands.sel(wl=prod.bswir2, method='nearest')

        ndwi = (vis - nir) / (vis + nir)
        ndwi_swir = (vis - swir) / (vis + swir)

        prod.raster['ndwi'] = ndwi
        prod.raster.ndwi.attrs = {
            'description': 'Normalized difference spectral index between bands at ' + str(prod.bvis) + ' and ' + str(
                prod.bnir) + ' nm', 'units': '-'}
        prod.raster['ndwi_swir'] = ndwi_swir
        prod.raster.ndwi_swir.attrs = {
            'description': 'Normalized difference spectral index between bands at ' + str(prod.bvis) + ' and ' + str(
                prod.bswir) + ' nm', 'units': '-'}

        if allpixels:
            pass  # masked_raster = prod.raster.bands
        else:
            # processed pixels: water (SWIR index, no bright SWIR), clear sky (OmniCloudMask class 0: no
            # thick cloud, thin cloud or cloud shadow), neither snow nor ice (snow and ice pass the SWIR index)
            logging.info('apply water, cloud, snow and ice masking')
            mask = ((ndwi_swir > prod.vis_swir_index_threshold) & (swir2 < prod.sunglint_threshold)
                    & Masking.clear_sky(prod.raster.flags) & ~Masking.snow_ice(prod.raster.flags))
            bands = prod.raster.bands
            if bands.chunks is None and mask.chunks is None and bands.dims == ('wl', 'y', 'x'):
                # in memory: set the pixels not processed to NaN in place (``where`` would copy the whole cube)
                bands.values[:, ~mask.transpose('y', 'x').values] = np.nan
            else:
                prod.raster['bands'] = bands.where(mask)
            prod.raster['sza'] = prod.raster['sza'].where(mask)
            del bands

            # stop if no clear-sky water pixel is left (e.g. water entirely under clouds): nothing to process,
            # and the angle tables of the LUT preparation would be empty. Same outcome as an image without
            # water (min_water_area): self.no_water, None returned, marker file written by write_output
            n_clear = int(mask.sum())
            if n_clear == 0:
                logging.warning('no clear-sky water pixel (water under clouds or cloud shadows, or frozen), '
                                'image not processed')
                self.no_water = True
                self.no_water_info = {'image': str(prod.raster.attrs.get('product_name', l1c_prod)
                                                   if isinstance(l1c_prod, xr.Dataset) else l1c_prod),
                                      'reason': 'no clear-sky water pixel',
                                      'clear_water_pixels': 0}
                return None

        # the band selections are views of the bands: release them now rather than at the end of execute
        del vis, nir, swir, swir2

        ######################################
        # get DEM information
        ######################################
        if dem_file:
            logging.info('load dem')
            dem = xr.open_dataset(remote.local_file(dem_file, storage_options)).squeeze().interp(y=prod.raster.y,
                                                             x=prod.raster.x,
                                                             method='nearest')
            dem = dem.rename_vars({'band_data': 'dem'})
            dem.dem.attrs['long_name'] = 'digital elevation model'
            dem.dem.attrs['units'] = 'm'
            dem.dem.attrs['source'] = dem_file
            # TODO use the dem to adjust the pressure raster. The kernel uses the CAMS surface pressure for the
            #  moment; the pressure computed here was never used, and computing it evaluated two full-size
            #  arrays, so it is not done any more:
            # presure_msl = cams.raster.msl.interp(y=prod.raster.y, x=prod.raster.x)
            # _pressure = presure_msl * (1. - 0.0065 * dem.dem / 288.15) ** 5.255
        else:
            dem = None

        ######################################
        # load algorithm kernel
        ######################################
        if p_slope is None:
            p_slope = self.p_slope
        self.kernel = Kernel(prod,
                             aero_lut,
                             trans_lut,
                             cams,
                             use_p_slope=p_slope)
        kernel = self.kernel

        # option for dev purposes
        if provide_kernel:
            return kernel

        ######################################
        # Aerosol type selection
        ######################################
        # select appropriate opac aerosol model from CAMS aod
        # remove URBAN for the moment
        opac_models = aero_lut.model.values
        models = aero_lut.drop_sel(model=['URBA_rh70']).model.values
        # get mean aot and aot550 from CAMS
        cams_aot_mean = cams.cams_aod.mean(['x', 'y'])
        cams_aot_ref = cams.cams_aod.interp(wl=550, method='quadratic')
        cams_aot_ref_mean = cams_aot_ref.mean(['x', 'y'])

        # get the model that has the closest aot spectral shape
        if opac_model is None:
            lut_aod = aero_lut.aot.sel(model=models, aot_ref=1).interp(wl=cams.cams_aod.wl)
            idx = np.abs((cams_aot_mean / cams_aot_ref_mean) - lut_aod).sum('wl').argmin()
            opac_model = aero_lut.sel(model=models).model.values[idx]

        logging.info('selected aerosol model: ' + opac_model)

        aerosol_combination = (opac_models == opac_model).astype(int)

        ######################################
        # LUT preparation
        ######################################
        logging.info('lut interpolation')
        kernel.lut_preparation(aerosol_combination=aerosol_combination)
        kernel.set_gas_transmittance()

        ######################################
        # surface rugosity (wind) estimation
        ######################################
        # # TODO speed up process (e.g. proper loop)
        # logging.info('surface rugosity (wind) estimation')
        # raster = kernel.get_coarse_masked_raster(resolution=megapix_size * resolution).compute()
        # _Nwl, _height, _width = raster.bands.shape
        # res = []
        # for iy in range(0, _height, chunk):
        #     yc = min(_height, iy + chunk)
        #
        #     for ix in range(0, _width, chunk):
        #         xc = min(_width, ix + chunk)
        #         res.append(kernel.rugosity_est_chunk(raster[dict(x=slice(ix, xc),
        #                                                          y=slice(iy, yc))]))
        #
        # wind_img = xr.merge(res)
        # # TODO add uncertainty weight for convolution
        # kernel.wind_img = kernel.smoothing(wind_img,
        #                                    varname='wind',
        #                                    mask=np.ones((3, 3)),
        #                                    windows=np.array([5, 5]))
        # coarse CAMS map: loaded once, the blocks of the final process only interpolate it
        kernel.wind_img = cams.raster.wind.to_dataset().compute()

        ######################################
        # aerosol load estimation
        ######################################
        logging.info('aerosol optical thickness estimation')
        aerosol = _aerosol_settings(aot_resolution=aot_resolution, aot_chunk=aot_chunk,
                                    aot_smoothing_blocks=aot_smoothing_blocks, min_pixel_prop=min_pixel_prop,
                                    outlier_factor=outlier_factor, fine_step=fine_step, fine_window=fine_window,
                                    aot_step=aot_step, glint_aot=glint_aot,
                                    glint_aot_min_glint=glint_aot_min_glint)
        logging.info('aerosol retrieval settings: %s', aerosol)
        aot_resolution, aot_chunk = aerosol['aot_resolution'], aerosol['aot_chunk']

        # the coarse raster is small: compute it once (one pass over the image) instead of re-evaluating the
        # lazy graph for every block and every aot_ref of the retrieval
        raster = kernel.get_coarse_masked_raster(resolution=aot_resolution,
                                                 min_pixel_prop=aerosol['min_pixel_prop']).compute()
        _Nwl, _height, _width = raster.bands.shape
        res = []
        for iy in range(0, _height, aot_chunk):
            yc = min(_height, iy + aot_chunk)

            for ix in range(0, _width, aot_chunk):
                xc = min(_width, ix + aot_chunk)
                raster_ = raster[dict(x=slice(ix, xc), y=slice(iy, yc))]
                number_of_pixel = raster_.water_pixel_number.sum()
                if number_of_pixel == 0:
                    continue

                # first to get rough aot values
                kernel.aerosol_swir_chunk(raster_)

                # second refine retrieval of aot_max
                if kernel.aot_ref_max < 0.12:
                    kernel.aot_ref_max = 0.12
                kernel.aerosol_swir_chunk(raster_,
                                          aot_refs=np.linspace(0, kernel.aot_ref_max, 21))

                # thrird apply non-negativity of visible bands
                # (one pixel per block: aot_ref and number_of_pixel, the number of water pixels of the block)
                res_ = kernel.aerosol_visible_chunk(raster_)
                res.append(res_)

        kernel.aot_ref_img = xr.merge(res)
        if aerosol['glint_aot']:
            kernel.aot_ref_img, kernel.aot_glint = _glint_hybrid_aot(kernel.aot_ref_img,
                                                                     aerosol['glint_aot_min_glint'])
        aot_ref_img = kernel.aot_ref_img

        logging.info('smoothing aerosol optical thickness raster')
        # the windows scale with the blocks: weighted mean over aot_smoothing_blocks blocks on the block grid,
        # then smoothing of the transitions (by default over one block) on a grid of fine_step coarse pixels
        fine_step = aerosol['fine_step']
        block_windows = np.array([aerosol['aot_smoothing_blocks']] * 2)
        cells = aerosol['fine_window']
        if cells is None:
            cells = max(1, int(round(aot_chunk / fine_step)))
            cells += 1 - cells % 2  # odd, centred window
        fine_windows = np.array([cells, cells])

        # filter out outliers from median: outliers and blocks without water pixels (NaN) get the median
        # aot_ref and a null weight, so that they do not contribute to the weighted mean of filter2d
        aot_median = kernel.aot_ref_img.aot_ref.median()
        valid = aot_ref_img.aot_ref < aerosol['outlier_factor'] * aot_median
        kernel.aot_ref_img = aot_ref_img.assign(aot_ref=aot_ref_img.aot_ref.where(valid, aot_median),
                                                number_of_pixel=aot_ref_img.number_of_pixel.where(valid, 0))

        # weighted mean, weights = number of water pixels of the blocks
        aot_ref_raster = kernel.aot_ref_img
        param = aot_ref_raster.aot_ref.values
        _weights = aot_ref_raster.number_of_pixel.values

        raster_smoothed = kernel.filter2d(param, _weights, block_windows)
        aot_ref_raster = xr.Dataset({'aot_ref': (["y", "x"], raster_smoothed)},
                                    coords=dict(y=kernel.aot_ref_img.y, x=kernel.aot_ref_img.x))
        # get both limit of x and y values for extrapolation using nearest method:
        new_x = [*raster.x.values[0:-1:fine_step], raster.x.values[-1]]
        new_y = [*raster.y.values[0:-1:fine_step], raster.y.values[-1]]
        aot_ref_raster = aot_ref_raster.interp(x=new_x,
                                               y=new_y,
                                               method='nearest',
                                               kwargs={"fill_value": "extrapolate"})

        aot_ref_raster = kernel.smoothing(
            aot_ref_raster,
            varname='aot_ref',
            mask=np.ones(fine_windows),
            windows=fine_windows)
        # the box filter can turn a null AOT into a tiny negative value (rounding, e.g. -4e-17): the LUTs of
        # the final processing would then start below 0 and give NaN Rrs for the whole image
        aot_ref_raster['aot_ref'] = aot_ref_raster.aot_ref.clip(min=0)

        kernel.aot_ref_raster = aot_ref_raster.interp(x=raster.x,
                                                      y=raster.y,
                                                      method='linear',
                                                      kwargs={"fill_value": "extrapolate"})

        ######################################
        # refined LUT preparation
        ######################################
        logging.info('refined LUT preparation')
        aot_step = aerosol['aot_step']
        aot_min = float(aot_ref_raster.aot_ref.min())
        aot_max = float(aot_ref_raster.aot_ref.max())
        logging.info('aot range of the refined LUT: %.4f - %.4f', aot_min, aot_max + aot_step)
        kernel.lut_preparation(aerosol_combination=aerosol_combination,
                               aot_refs=[0, *np.arange(aot_min, aot_max + aot_step, aot_step)])

        ######################################
        # final grs process
        ######################################
        logging.info('final grs process')
        # lazy: nothing is computed here, the blocks are evaluated when the product is written
        kernel.final_process(kernel.prod.raster,
                             aot_ref_raster.aot_ref,
                             chunk=chunk)

        # filter wind retrievals: the kernel works on a frozen copy of the wind map taken in final_process,
        # so this does not change the Rrs
        kernel.wind_img['wind'] = kernel.wind_img.wind.where(kernel.wind_img.wind < 12)
        # interpolated block by block (lazy), instead of a full-size interpolation in memory
        wind = _interp_like(kernel.wind_img.wind, kernel.xres.BRDFg, name='wind')
        # own attributes, not the ones of the BRDFg template (product attributes)
        wind.attrs = {'long_name': 'wind speed at 10 m (CAMS), NaN above 12 m s-1', 'units': 'm s-1'}
        l2_prod = xr.merge([kernel.xres, wind], compat='override')

        ##############################################
        # Update flags and create mask from recipe
        ##############################################
        logging.info('Update flags and create mask from recipe')
        # flags for negative blue/green Rrs (lazy, uint32 like the other flags); bit 23, free for both sensors
        # (bit 18, used up to 3.1.1, is also the Landsat l1_snow bit), set with OR so no other bit is changed
        bitmask = 23
        negative_rrs = ((l2_prod.Rrs.sel(wl=490, method='nearest') < -0.0005) |
                        (l2_prod.Rrs.sel(wl=565, method='nearest') < -0.0005))
        prod.raster['flags'] = prod.raster.flags | (negative_rrs.astype(np.uint32) << bitmask)
        # add name and description
        prod.raster.flags.attrs['flag_descriptions'][bitmask] = 'negative Rrs for blue or green bands'
        prod.raster.flags.attrs['flag_names'][bitmask] = 'neg_rrs'

        # mask from recipe
        mask = masking_.create_mask(prod.raster.flags,
                                    tomask=self.flags_tomask,
                                    tokeep=self.flags_tokeep,
                                    mask_name="mask")
        l2_prod = xr.merge([l2_prod, mask], compat='override')

        l2_prod['central_wavelength'] = ('wl', prod.raster.wl_true.values)
        l2_prod = l2_prod.set_coords('central_wavelength')
        prod.raster.attrs['opac_model'] = opac_model
        # wave slope probability of the sunglint normalisation: 'cox_munk_iso' or 'off' (P_slope = 1)
        prod.raster.attrs['p_slope'] = 'cox_munk_iso' if kernel.use_p_slope else 'off'
        # device on which OmniCloudMask ran ('cpu' or 'cuda'): same mask in fp32
        if ocm_device_used:
            prod.raster.attrs['ocm_device'] = ocm_device_used
        # pixel size (meter) of the coarse raster of the aerosol retrieval
        prod.raster.attrs['aot_resolution'] = kernel.coarse_resolution

        ######################################
        # Write final product
        ######################################
        logging.info('construct final product')
        # self.l2_prod = l2_prod
        self.l2a = L2aProduct(prod, l2_prod, cams, gas_trans, dem)

        del prod, l2_prod, cams, gas_trans, dem
        gc.collect()
        self.successful = True
        return kernel

    @_with_workers
    def write_output(self, fmt=None, **export_kwargs):
        '''
        Write the L2A product; this is where the lazy processing is evaluated, chunk by chunk.

        The format comes from the name of ``self.odir``: ``.zarr`` gives a Zarr store (with pyramids),
        ``.nc`` a NetCDF file (plus ``_anc.nc`` for the ancillary data); the ``.zarr`` path can be a remote
        URL (``s3://bucket/name.zarr``). A path with no such extension is a directory and gets a Zarr store
        (as before), unless `fmt` says otherwise.

        :param fmt: ``'zarr'`` or ``'netcdf'`` to force the format (None: from the name of the output)
        :param export_kwargs: options of ``L2aProduct.export`` (e.g. ``chunks``, ``shards``, ``compression``,
            ``storage_options``, ``max_concurrency``, ``pyramids``)
        :return: for an image without water (see ``execute(min_water_area=...)``), the path of the marker file
            written instead of the product (`no_water_marker`)
        '''
        if self.l2a is None:
            if self.no_water:
                # image without water (see min_water_area): a marker file instead of the product
                marker = no_water_marker(self.odir)
                remote.write_text(marker, ''.join(f'{key}: {value}\n' for key, value in
                                                  {**self.no_water_info, 'grs_version': __version__}.items()),
                                  storage_options=export_kwargs.get('storage_options'))
                logging.warning('image without water: no L2A product, marker %s', marker)
                return marker
            logging.warning('no L2A product to write (execute did not complete)')
            return None
        logging.info('export final product')
        if fmt is None and Path(str(self.odir).rstrip('/')).suffix.lower() not in _OUTPUT_EXTENSIONS:
            fmt = 'zarr'
        return self.l2a.export(self.odir,
                               fmt=fmt,
                               snap_compliant=self.snap_compliant,
                               **export_kwargs)