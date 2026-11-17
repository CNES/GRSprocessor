"""
GRSprocessor (code of this repository) around a point of a Sentinel-2 L1C image; keeps a small box of the L2A
variables around the point in a netCDF file. The input is either an L1C subset (``.nc`` of make_l1c_subsets.py,
processed whole, fast) or a full image (``.SAFE``, a subset of +-``--half`` m is read). One image per call, so that the memory is released
between the images (run by aaot_s2_matchups.ipynb, one subprocess per image).

    python process_subset.py <SAFE|subset.nc> <cams_file> <lon> <lat> <out.nc> [--half 10000] [--extract 15]
                             [--resolution 20] [--options '{"glint_aot": true}']
    python process_subset.py --batch <jobs.json>

``--batch``: several images in the same process, one after the other (a JSON list of objects with the keys
``safe``, ``cams``, ``lon``, ``lat``, ``out`` and optionally ``half``, ``extract``, ``resolution``, ``options``):
the imports (torch, GRSdriver, numba), about 10 s, are made once instead of once per image. The images already
done (``out``, ``.err`` or ``.nowater`` there) are skipped, so that a batch stopped on the way (killed for its
memory) can be run again; the caller keeps the batches short so that the memory is still released regularly.

On failure, writes <out>.err with the exception and exits with status 1. An image without clear-sky water pixel
(GRSprocessor stops, ``Process.no_water``) writes <out>.nowater with the reason and exits with status 0.
"""
import argparse
import gc
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))   # before site-packages (an installed grsprocessor would hide the code under test)

import GRSdriver  # noqa: E402
import grs  # noqa: E402
from pyproj import Transformer  # noqa: E402

VARS = ['Rrs', 'BRDFg', 'aot550', 'mask', 'flags', 'wind', 'sza', 'vza', 'raa']


def version():
    """GRSprocessor version + last commit of grs/ (the code under test; '.dirty' if grs/ has uncommitted changes)"""
    git = ['git', '-C', str(REPO)]
    commit = subprocess.run(git + ['log', '-1', '--format=%h', '--', 'grs'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ['diff', '--quiet', 'HEAD', '--', 'grs']).returncode != 0
    return f'{grs.__version__}+{commit}' + ('.dirty' if dirty else '')


def done(out):
    """The image has its extraction, its failure or its no-water marker"""
    return any(Path(str(out) + s).exists() for s in ('', '.err', '.nowater'))


def run(safe, cams, lon, lat, out, half=10_000, extract=15, resolution=20, options=None):
    """GRSprocessor on one image and extraction of the box around the point; returns the exit status (0 or 1)"""
    out = Path(out)
    options = dict(options or {})
    t0 = time.time()
    try:
        zone = int(re.search(r'_T(\d{2})[A-Z]{3}_', Path(safe).name).group(1))   # UTM zone of the tile
        epsg = (32600 if lat >= 0 else 32700) + zone
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        if safe.endswith('.nc'):
            l1c_prod = safe
        else:
            subset = [(x0 - half, y0 - half, x0 + half, y0 + half)]
            l1c = GRSdriver.Sentinel2Driver(safe, subset=subset, resolution=resolution, resolution_angle=resolution)
            l1c.load_product()
            l1c_prod = l1c.prod
        p = grs.Process()
        p.execute(l1c_prod, cams_file=cams, resolution=resolution, **options)
        if p.no_water:   # no (clear-sky) water pixel in the subset: no product, not an error
            info = {**p.no_water_info, 'grs_version': version()}
            Path(str(out) + '.nowater').write_text(''.join(f'{k}: {v}\n' for k, v in info.items()))
            print(f'{out.name}: no water ({info.get("reason", "water area below the minimum")}), {time.time() - t0:.0f} s', flush=True)
            return 0
        l2 = p.l2a.l2_prod
        iy, ix = int(np.abs(l2.y - y0).argmin()), int(np.abs(l2.x - x0).argmin())
        h = extract // 2
        box = l2[[v for v in VARS if v in l2]].isel(y=slice(iy - h, iy + h + 1), x=slice(ix - h, ix + h + 1)).compute()
        box = box.drop_vars([c for c in box.coords if c not in ('wl', 'x', 'y', 'central_wavelength')])
        for obj in [box] + [box[v] for v in box.variables]:
            obj.attrs = {k: (int(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in obj.attrs.items()
                         if isinstance(v, (str, int, float, bool, np.number, np.bool_))}
        box.attrs.update(image=Path(safe).name.replace('.SAFE', '').replace('.nc', ''), l1c=safe,
                         time=str(np.datetime_as_string(l2.time.values, unit='s')) if 'time' in l2.coords else '',
                         lon=lon, lat=lat, x0=x0, y0=y0, grs_version=version(),
                         grsdriver_version=GRSdriver.__version__, cams_file=cams, options=json.dumps(options),
                         resolution=resolution)
        out.parent.mkdir(parents=True, exist_ok=True)
        box.to_netcdf(out)
        print(f'{out.name}: {time.time() - t0:.0f} s', flush=True)
    except Exception as e:
        Path(str(out) + '.err').write_text(f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}')
        print(f'{out.name}: FAILED {type(e).__name__}: {e}', flush=True)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('safe', nargs='?'); ap.add_argument('cams', nargs='?')
    ap.add_argument('lon', type=float, nargs='?'); ap.add_argument('lat', type=float, nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--half', type=float, default=10_000, help='half size of the processed subset (m)')
    ap.add_argument('--extract', type=int, default=15, help='pixels kept around the point')
    ap.add_argument('--resolution', type=int, default=20)
    ap.add_argument('--options', default='{}', help='options of Process.execute (JSON)')
    ap.add_argument('--batch', help='JSON file of the images to process in this process')
    a = ap.parse_args(argv)
    if a.batch:
        status = 0
        for job in json.loads(Path(a.batch).read_text()):
            if not done(job['out']):
                status |= run(**job)
                gc.collect()   # the products of the previous image are released before the next one
        return status
    if None in (a.safe, a.cams, a.lon, a.lat, a.out):
        ap.error('safe, cams, lon, lat and out are required without --batch')
    return run(a.safe, a.cams, a.lon, a.lat, a.out, a.half, a.extract, a.resolution, json.loads(a.options))


if __name__ == '__main__':
    sys.exit(main())
