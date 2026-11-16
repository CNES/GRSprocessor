"""
Level-1C subsets of Sentinel-2 images around a point (AERONET-OC site), in the layout of the existing archive
``/data/satellite/Sentinel-2/subset/aeronet/L1C/<tile>/<year>/<image>.nc`` (made by
grstbx/notebook/validation/box_subset_and_save_aeronet_all_sites.ipynb): box of 521 x 521 pixels at 20 m
(10.4 km) centred on the site, angles computed at 60 m, bands packed in int16. GRSprocessor reads these files
directly (``Process.execute(<file.nc>)``, ~15 s per image instead of minutes on the full tile).

    python make_l1c_subsets.py <lon> <lat> <odir> <SAFE> [<SAFE> ...] [--half 5200] [--workers 4]

One subprocess per image (the memory is released between the images); an existing subset is skipped; a failure
writes ``<file>.nc.err``.
"""
import argparse
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def subset_path(odir, safe):
    name = Path(safe).name.replace('.SAFE', '')
    tile = re.search(r'_T(\d{2}[A-Z]{3})_', name).group(1)
    year = re.search(r'_MSIL1C_(\d{4})', name).group(1)
    return Path(odir) / tile / year / f'{name}.nc'


def _packing(values, nbit=16):
    """int16 scale factor and offset covering the range of the values"""
    vmax, vmin = np.nanmax(values), np.nanmin(values)
    scale = (vmax - vmin) / (2 ** nbit - 2)
    return float(scale), float(vmin + 2 ** (nbit - 1) * scale)


def one(safe, lon, lat, odir, half=5200., resolution=20, resolution_angle=60, complevel=5):
    import GRSdriver
    from pyproj import Transformer

    out = subset_path(odir, safe)
    if out.exists():
        return f'{out.name}: exists'
    try:
        zone = int(re.search(r'_T(\d{2})[A-Z]{3}_', Path(safe).name).group(1))
        epsg = (32600 if lat >= 0 else 32700) + zone
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        subset = [(x0 - half, y0 - half, x0 + half, y0 + half)]
        l1c = GRSdriver.Sentinel2Driver(safe, subset=subset, resolution=resolution, resolution_angle=resolution_angle)
        l1c.load_product()
        prod = l1c.prod
        if prod.sizes['x'] == 0 or prod.sizes['y'] == 0 or not np.isfinite(prod.bands.values).any():
            raise ValueError('no data in the subset (outside the swath)')
        for obj in [prod] + [prod[v] for v in prod.variables]:
            obj.attrs = {k: (int(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in obj.attrs.items()}
        scale, offset = _packing(prod.bands.values)
        enc = dict(zlib=True, complevel=complevel)
        encoding = {'bands': {'dtype': 'int16', 'scale_factor': scale, 'add_offset': offset, '_FillValue': -32768, **enc},
                    'vza': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, **enc},
                    'raa': {'dtype': 'int16', 'scale_factor': 0.1, '_FillValue': -9999, **enc},
                    'sza': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, **enc}}
        encoding = {k: v for k, v in encoding.items() if k in prod}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.nc.part')
        prod.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, out)
        return f'{out.name}: written ({prod.sizes["y"]} x {prod.sizes["x"]})'
    except Exception as e:
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + '.err').write_text(f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}')
        return f'{out.name}: FAILED {type(e).__name__}: {e}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('lon', type=float); ap.add_argument('lat', type=float); ap.add_argument('odir')
    ap.add_argument('safes', nargs='+')
    ap.add_argument('--half', type=float, default=5200., help='half size of the box (m)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    todo = [s for s in a.safes if not subset_path(a.odir, s).exists()
            and not Path(str(subset_path(a.odir, s)) + '.err').exists()]
    print(f'{len(a.safes)} images, {len(todo)} subsets to make', flush=True)
    # max_tasks_per_child=1: a new process per image, its memory is released at the end
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as pool:
        for msg in pool.map(one, todo, [a.lon] * len(todo), [a.lat] * len(todo), [a.odir] * len(todo),
                            [a.half] * len(todo)):
            print(msg, flush=True)


if __name__ == '__main__':
    sys.exit(main())
