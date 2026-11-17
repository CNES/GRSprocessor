"""
CAMS files of a site: the CAMS cells within ``--margin`` degrees of a point, all the times of the file, cut out of
global CAMS files (monthly ``cams_forecast_YYYY-MM.nc`` or daily files) into ``<out_dir>/<name of the file>``.

GRSprocessor reads them as any CAMS file (``Process.execute(cams_file=...)``): ``CamsProduct.load`` takes the CAMS
cells within 1 degree of the image, so a margin of 2 degrees covers the L1C subsets and the images cut around
the point. A global monthly file is several GB of compressed chunks on a hard disk, read again for every image
(one to several minutes per image when the file is not in the page cache); the file of the site is read once,
here, and is a few MB. The files already made are kept (delete them to make them again).

    python make_cams_site.py <lon> <lat> <out_dir> <cams file> [<cams file> ...] [--margin 2]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))   # before site-packages (an installed grsprocessor would hide the code under test)

import numpy as np  # noqa: E402

try:   # the tool also runs with the code of older versions under test (git worktrees of other commits)
    from grs.cams import _netcdf_engine  # noqa: E402
except ImportError:
    def _netcdf_engine(filepath):
        with open(filepath, 'rb') as f:
            return 'scipy' if f.read(4) in (b'CDF\x01', b'CDF\x02') else None
try:
    from grs.cams import _select_longitudes  # noqa: E402
except ImportError:
    def _select_longitudes(cams, lonmin, lonmax, margin=1):
        """Copy of grs.cams._select_longitudes (GRSprocessor > 3.1.1+11dd607)"""
        lon = cams.longitude.values
        if lon.min() >= 0:
            if lonmin <= 0 <= lonmax:
                lon = ((lon + 180) % 360) - 180
            else:
                lonmin, lonmax = lonmin % 360, lonmax % 360
        idx = np.flatnonzero((lon >= lonmin - margin) & (lon <= lonmax + margin))
        idx = idx[np.argsort(lon[idx], kind='stable')]
        if idx.size and np.all(np.diff(idx) == 1):
            cams = cams.isel(longitude=slice(int(idx[0]), int(idx[-1]) + 1))
        else:
            cams = cams.isel(longitude=idx)
        return cams.assign_coords(longitude=lon[idx]), lonmin, lonmax


def site_file(cams_file, out_dir):
    """Path of the CAMS file of the site made from `cams_file`"""
    return Path(out_dir) / Path(cams_file).name


def make(cams_file, lon, lat, out_dir, margin=2.):
    """Cuts the cells of the site out of `cams_file` into `out_dir` (kept if it exists); returns its path"""
    out = site_file(cams_file, out_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(cams_file, engine=_netcdf_engine(cams_file)) as ds:
        lat_desc = bool(ds.latitude[0] > ds.latitude[-1])
        ds = ds.sel(latitude=slice(lat + margin, lat - margin) if lat_desc else slice(lat - margin, lat + margin))
        ds, _, _ = _select_longitudes(ds, lon, lon, margin=margin)
        ds = ds.load()
    for v in ds.variables.values():   # chunks and compression of the global file do not fit the small arrays
        v.encoding = {k: x for k, x in v.encoding.items() if k in ('units', 'calendar', 'dtype', '_FillValue')}
    ds.attrs.update(grs_site_source=str(cams_file), grs_site_lon=lon, grs_site_lat=lat, grs_site_margin=margin)
    tmp = out.with_suffix('.nc.tmp')
    ds.to_netcdf(tmp)
    os.replace(tmp, out)   # complete files only
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('lon', type=float); ap.add_argument('lat', type=float); ap.add_argument('out_dir')
    ap.add_argument('cams', nargs='+')
    ap.add_argument('--margin', type=float, default=2., help='half size (deg) of the area kept around the point')
    a = ap.parse_args(argv)
    status = 0
    for f in a.cams:
        t0 = time.time()
        if site_file(f, a.out_dir).exists():
            continue
        try:
            out = make(f, a.lon, a.lat, a.out_dir, a.margin)
            print(f'{out.name}: {time.time() - t0:.0f} s, {out.stat().st_size / 1e6:.1f} MB', flush=True)
        except Exception as e:   # truncated or unreadable file: the image keeps the global file
            print(f'{Path(f).name}: FAILED {type(e).__name__}: {e}', flush=True)
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
