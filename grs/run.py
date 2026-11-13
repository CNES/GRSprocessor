''' Executable to process Sentinel-2 L1C images for aquatic environment

Usage:
  grs <input_file> [--cams_file file] [--odir <odir>] [--resolution res] [--aot_resolution res]\
   [--max_cloud_cover max_cc] [--scale_aot factor] [--opac_model name] [--levname <lev>] [--no_clobber]\
   [--allpixels] [--surfwater file] [--dem_file file] [--suffix suffix] [--snap_compliant]\
   [--min_water_area area] [--log_dir dir]
  grs -h | --help
  grs -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <input_file>     Input file to be processed: local path or URL (s3://, https://..., see grs.remote);
                   the files of the options below and --odir can be URLs too

  --cams_file file     Absolute path of the CAMS file to be used (mandatory)

  -o <odir>, --odir <odir>  Full (absolute or relative) path to output L2 image.
  --levname lev    Level naming used for output product [default: L2AGRS]
  --no_clobber     Do not process <input_file> if <output_file> already exists (or its marker file
                   <name>_no_water.txt for an image without water, see --min_water_area).
  --resolution=res  spatial resolution of the scene pixels [default: 60]
  --aot_resolution res  pixel size (m) of the coarse grid of the aerosol retrieval, multiple of the
                        resolution (default: aot_resolution of the configuration, 240)
  --max_cloud_cover max_cc  Skip process if image level 1 cloud cover is greater than max_cc
                            in decimal number [default: 1]
  --allpixels      force to process all pixels whatever they are masked (cloud, vegetation...) or not
  --surfwater file  Absolute path of the surfwater geotiff file to be used
  --dem_file file  Absolute path of the DEM geotiff file (already subset for the S2 tile)
  --scale_aot factor  scaling factor applied to CAMS aod550 raster
                      [default: 1]
  --opac_model name  Force the aerosol model (OPAC) to be 'name'
                     (choice: ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70',
                     'MACL_rh70', 'URBA_rh70'])
  --suffix suffix  A suffix to append to the output dir name to personalize a run default value is "V<GRS version>"
  --snap_compliant  Write Rrs as one variable per band (Rrs_<wl>), "beam" layout for ESA SNAP software
  --min_water_area area  Do not process the image if it has less water than this area (km2): no product
                         is written, only the marker file <name>_no_water.txt (default: min_water_area of
                         the configuration, 1 km2; 0: only images without any water pixel are skipped)
  --log_dir dir    Local folder of the log files (log_file.log, error.log); by default the output folder of
                   the image, or ./<name> for a URL --odir (the log files are then copied next to the product)


  Example:
      grs s3://bucket/l1c/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE --cams_file s3://bucket/cams/cams_forecast_2022-07.nc --odir s3://bucket/l2a --resolution 20
      (credentials and endpoint of the bucket: AWS_PROFILE, AWS_ENDPOINT_URL... environment variables)
      grs /data/satellite/S2/L1C/S2B_MSIL1C_20220731T103629_N0400_R008_T31TFJ_20220731T124834.SAFE --cams_file /data/cams/world/cams_forecast_2022-07.nc --resolution 60
  For CNES datalake:
      grs /work/datalake/S2-L1C/31TFJ/2023/06/16/S2B_MSIL1C_20230616T103629_N0509_R008_T31TFJ_20230616T111826.SAFE --cams_file /work/datalake/watcal/ECMWF/CAMS/2023/06/16/2023-06-16-cams-global-atmospheric-composition-forecasts.nc --odir /work/datalake/watcal/test --resolution 20 --dem_file /work/datalake/static_aux/MNT/COP-DEM_GLO-30-DGED_S2_tiles/COP-DEM_GLO-30-DGED_31TFJ.tif

Exit status:
  0  the L2A product is written (or, with --no_clobber, already exists)
  1  processing error (traceback in the log files)
  2  invalid arguments, missing or unreadable input file (Level-1, CAMS, SURFWATER, LUT)
  3  image skipped on purpose, no product: cloud cover above --max_cloud_cover, or not enough water
     (marker file <name>_no_water.txt written or, with --no_clobber, already there)
  Only exit status 0 gives a product for the next steps of a processing chain (e.g. Level-2B).

'''

import ctypes
import ctypes.util
import logging
import os

import sys
from pathlib import Path
from xml.etree import ElementTree
from docopt import docopt, DocoptExit

from . import __package__, __version__

# exit status of the grs command (and of exe/launcher.py), see the Exit status section above
EXIT_OK = 0         # product written (or already there with --no_clobber)
EXIT_ERROR = 1      # processing error
EXIT_INPUT = 2      # invalid arguments, missing or unreadable input
EXIT_SKIPPED = 3    # skipped on purpose (cloud cover, no water): no product


def set_ofile(filename: str, odir='', level_name='L2GRS', suffix='') -> Path:
    """Output path: level name of `filename` replaced by `level_name`, extension removed, `suffix` appended."""
    outfile = filename
    for level in ('L1C', 'L1GT', 'L1TP'):
        outfile = outfile.replace(level, level_name)
    return Path(odir, Path(outfile).stem + suffix)


def s2_cloud_cover(safe, storage_options=None):
    '''
    Cloud cover (0-1) of a Sentinel-2 L1C image, from ``Cloud_Coverage_Assessment`` of its
    ``MTD_MSIL1C.xml`` (read directly: no GDAL needed).

    :param safe: path or URL of the ``.SAFE`` folder
    :param storage_options: fsspec options of a URL (see :mod:`grs.remote`)
    '''
    from . import remote
    root = ElementTree.fromstring(remote.read_bytes(remote.join(safe, 'MTD_MSIL1C.xml'), storage_options))
    node = next(e for e in root.iter() if e.tag.split('}')[-1] == 'Cloud_Coverage_Assessment')
    return float(node.text) / 100


def limit_malloc_arenas(n=2):
    """
    Limit the number of glibc malloc arenas to `n` (as ``MALLOC_ARENA_MAX=n``), unless ``MALLOC_ARENA_MAX`` is
    set in the environment.

    With one arena per thread, the memory freed by the threads that read and process the image is kept by
    the process: on a Sentinel-2 tile at 20 m the peak memory goes from about 12-13 GB to 9.5-10 GB with 2
    arenas, with the same output. Call it at start-up, before the threads are created; no effect on systems
    without glibc.
    """
    if os.environ.get('MALLOC_ARENA_MAX'):
        return
    try:
        libc = ctypes.CDLL(ctypes.util.find_library('c'))
        libc.mallopt(-8, int(n))  # M_ARENA_MAX
    except (OSError, AttributeError, TypeError):
        pass


def _upload_logs(log_files, outdir):
    '''Copy the local log files next to a remote product (a failed copy is only reported).'''
    from . import remote
    for log_file in log_files:
        if Path(log_file).is_file():
            try:
                remote.upload(log_file, remote.join(outdir, Path(log_file).name))
            except Exception as exc:
                print(f'log file {log_file} not copied to {outdir}: {exc}', file=sys.stderr)


def main(argv=None):
    '''
    ``grs`` command: process one Level-1 image.

    :param argv: command line arguments (default: ``sys.argv[1:]``)
    :return: exit status, see the module constants ``EXIT_*`` (the console script passes it to ``sys.exit``)
    '''
    # before any thread is started
    limit_malloc_arenas()
    # -h / -v are handled (and exit with status 0) here, before the heavy imports below
    try:
        args = docopt(__doc__, argv=argv, version=__package__ + '_' + __version__)
    except DocoptExit as exc:
        print(exc, file=sys.stderr)
        return EXIT_INPUT
    print(args)

    from . import class_logger
    from .grs_process import Process, no_water_marker
    from .grs_exceptions import GRS_IO_Exception

    from . import remote

    # a URL is kept as a string (Path would turn s3://bucket into s3:/bucket)
    file = args['<input_file>'] if remote.is_url(args['<input_file>']) else Path(args['<input_file>'])
    for name, path in (('input', file), ('CAMS', args['--cams_file']), ('SURFWATER', args['--surfwater']),
                       ('DEM', args['--dem_file'])):
        try:
            found = not path or remote.exists(path)
        except Exception as exc:  # e.g. no credentials or no fsspec driver for the URL
            print(f'{name} file not readable: {path} ({exc})', file=sys.stderr)
            return EXIT_INPUT
        if not found:
            print(f'{name} file not found: {path}', file=sys.stderr)
            return EXIT_INPUT

    lev = args['--levname']
    cams_file = args['--cams_file']
    surfwater_file = args['--surfwater']
    dem_file = args['--dem_file']
    noclobber = args['--no_clobber']
    allpixels = args['--allpixels']
    resolution = int(args['--resolution'])
    aot_resolution = int(args['--aot_resolution']) if args['--aot_resolution'] else None   # None: configuration
    max_cc = float(args['--max_cloud_cover'])
    scale_aot = float(args['--scale_aot'])
    opac_model = args['--opac_model']
    snap_compliant = args['--snap_compliant']
    suffix = args['--suffix']
    # None: configuration
    min_water_area = float(args['--min_water_area']) if args['--min_water_area'] else None

    ##################################
    # File naming convention
    ##################################
    basename = remote.basename(file)
    # first check cloud cover (for S2, not implemented for Landsat)
    if ('MSIL1C' in basename) and ('SAFE' in basename):
        cc = s2_cloud_cover(file)
        if cc >= max_cc:
            logging.info('input file not processed since cloud cover {:.3f} is greater than {:.3f}'.format(cc, max_cc))
            print(f'cloud cover {cc:.3f} greater than {max_cc:.3f}: image not processed', file=sys.stderr)
            return EXIT_SKIPPED

    odir = args['--odir']
    if odir == './' or not odir:
        odir = Path.cwd()

    if not suffix:
        suffix = f'_V{__version__}'

    # output folder <odir>/<name> and its Zarr store <odir>/<name>/<name>.zarr (Process.write_output)
    name = set_ofile(basename, level_name=lev, suffix=suffix).name
    outdir = remote.join(odir, name)
    outfile = remote.join(outdir, name + '.zarr')
    remote_output = remote.is_url(outdir)

    # skip if already processed, or already found without water (marker file of Process.write_output);
    # checked before the logger is set up, which would overwrite the log file of the previous run
    if noclobber:
        if remote.exists(outfile):
            print(f'File {outfile} already processed; skip!')
            return EXIT_OK
        if remote.exists(no_water_marker(outdir)):
            print(f'File {no_water_marker(outdir)} found (image without water); skip!')
            return EXIT_SKIPPED

    # log files: in the output folder, or in a local folder for a remote output (copied there at the end)
    log_dir = Path(args['--log_dir']) if args['--log_dir'] else (Path.cwd() / name if remote_output else outdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = [log_dir / 'log_file.log', log_dir / 'error.log']
    class_logger.ServiceLogger(log_file=str(log_files[0]), error_log=str(log_files[1]), log_level='INFO',
                               log_console=False)

    logging.info(f'call grs_process for the following paramater. '
                 f'File: {file}, '
                 f'output directory: {odir}, '
                 f'cams_file:{cams_file}, '
                 f'resolution: {resolution}, '
                 f'aot_resolution: {aot_resolution}')

    try:
        process_ = Process()
        process_.execute(file,
                         # the Zarr store itself for a URL (an object storage has no folders to create)
                         odir=outfile if remote_output else outdir,
                         cams_file=cams_file,
                         resolution=resolution,
                         aot_resolution=aot_resolution,
                         scale_aot=scale_aot,
                         opac_model=opac_model,
                         dem_file=dem_file,
                         allpixels=allpixels,
                         surfwater_file=surfwater_file,
                         snap_compliant=snap_compliant,
                         min_water_area=min_water_area)
        # for an image without water: marker file <outdir>/<name>_no_water.txt instead of the product
        process_.write_output()
    except GRS_IO_Exception:
        logging.error("Input error in grs_process", exc_info=True)
        return EXIT_INPUT
    except Exception:
        logging.error("Fatal error in grs_process", exc_info=True)
        return EXIT_ERROR
    finally:
        # log the run statistics and close the log files
        class_logger.get_instance().close()
        if remote_output:
            _upload_logs(log_files, outdir)
    if getattr(process_, 'no_water', False):
        return EXIT_SKIPPED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
