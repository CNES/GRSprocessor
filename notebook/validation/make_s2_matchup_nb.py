"""
Writes the Sentinel-2 / AERONET-OC matchup notebook of a site, ``<site>_s2_matchups.ipynb`` next to this script
(edit the notebook here, not the .ipynb):

    python make_s2_matchup_nb.py [SITE]      # name of aeronet_oc_db.site_info(), default AAOT

SITE can also be a platform with several AERONET-OC series (``INSITU``), e.g. ``Venise+AAOT``: the in situ records of
all its series are merged (same coordinates).
"""
import sys
from pathlib import Path

import nbformat as nbf

SITE = sys.argv[1] if len(sys.argv) > 1 else 'AAOT'
# platforms with several AERONET-OC series (same coordinates), merged
INSITU = {'Venise+AAOT': ['Venise', 'AAOT']}.get(SITE, [SITE])
# period of the images: the whole Sentinel-2 record for the merged platforms, a couple of years up to now otherwise
PERIOD = {'Venise+AAOT': ('2015-07-01', '2026-11-14')}.get(SITE, ('2024-11-14', '2026-11-14'))
DESCRIPTION = {
    'AAOT': 'the Acqua Alta Oceanographic Tower (**AAOT**, northern Adriatic, 45.314 N, 12.508 E, tile 32TQR). '
            'Venise is the former name of the same platform, with data until 2023-03: it can be used for the '
            "older years (`SITE = 'Venise'`)",
    'Section-7_Platform': 'the Section-7 platform (**Section-7_Platform**, north-western Black Sea off the Danube '
                          'delta, 44.546 N, 29.447 E, tile 35TPK). Gloria is the former platform nearby, with '
                          "data until 2019-08 (`SITE = 'Gloria'`)",
    'Venise+AAOT': 'the Acqua Alta Oceanographic Tower (northern Adriatic, 45.314 N, 12.508 E, tile 32TQR), with the '
                   'two AERONET-OC series of the platform merged: **Venise** (2002 to 2023-03) and **AAOT** (since '
                   '2017-10; both instruments ran together from 2017-10 to 2023-03). Whole Sentinel-2 record',
}
NB_NAME = {'AAOT': 'aaot', 'Section-7_Platform': 'section7', 'Venise+AAOT': 'venise_aaot'}.get(SITE, SITE.lower())

nb = nbf.v4.new_notebook()
C = []


def _site(s):
    return (s.replace('__SITE__', SITE).replace('__NB__', f'{NB_NAME}_s2_matchups.ipynb')
            .replace('__INSITU__', repr(INSITU)).replace('__START__', PERIOD[0]).replace('__END__', PERIOD[1]))


def md(s):
    C.append(nbf.v4.new_markdown_cell(_site(s).strip('\n')))


def code(s):
    C.append(nbf.v4.new_code_cell(_site(s).strip('\n')))


md(f'''
# GRSprocessor validation at {SITE} (AERONET-OC), Sentinel-2

Matchups of the Sentinel-2 L2A products of the **current version of GRSprocessor** (the code of this repository)
with the AERONET-OC measurements of {DESCRIPTION[SITE]}. Other sites: `SITE` in the settings (a name of
`aeronet_oc_db.site_info()`, which gives the coordinates and the tile).
''' + r'''
Steps:

1. **In situ**: AERONET-OC v3 (`aeronet_oc_db`), level 2.0 when available, else level 1.5 (flagged);
2. **Images**: the Sentinel-2 L1C acquisitions of the tile of the site in the period (CDSE catalogue), with an AERONET-OC
   record within `MAX_LAG` of the overpass; local archive check and optional download (`grstbx.download`);
3. **L1C subsets**: 10.4 x 10.4 km around the platform at 20 m (`make_l1c_subsets.py`), in the archive of the
   AERONET-OC subsets (`SUBSET_DIR/<tile>/<year>/<image>.nc`, 2017-2023 already there), made once per image;
4. **Processing**: GRSprocessor on each L1C subset (`process_subset.py`, one subprocess per image, ~15-20 s and
   ~1.4 GB), one cached extraction per image and version (`CACHE/<image>.nc`, box of `EXTRACT` pixels): nothing is
   recomputed on a rerun;
5. **Matchups**: median over a ring of pixels around the platform (`RING`, 80-140 m from the AERONET-OC
   coordinates at 20 m: the platform and its shadow are left out), water pixels only, homogeneity (CV) filter,
   nearest in situ record in time;
6. **Evaluation**: Rrs per band (slope, r, RMSE, bias, MdSA, SSPB, Pahlevan et al. 2020) and AOT(550), with all
   the matchups and with the Level 2.0 ones only.

Long steps (3-4) the first time: run the notebook with
`research/run_nb.sh notebook/validation/__NB__ 20000` (memory cap, see `CLAUDE.local.md`), or set
`PROCESS = False` to only evaluate what is cached.
''')

code(r'''
import glob
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# the repository first: the grsprocessor of site-packages (3.1.1) would hide the code under test
REPO = Path.cwd().resolve().parents[1] if Path.cwd().name == 'validation' else Path.cwd().resolve()
sys.path.insert(0, str(REPO))
import grs
import GRSdriver
import aeronet_oc_db
from grstbx import download
from grstbx.matchup import matchup_metrics
from shapely.geometry import Point, shape

assert Path(grs.__file__).resolve().is_relative_to(REPO), grs.__file__
# version of the code under test: last commit of grs/ (commits of the notebooks, docs... keep the cache);
# '.dirty' if grs/ has uncommitted changes (results of such a run are not attached to a commit)
GIT = ['git', '-C', str(REPO)]
COMMIT = subprocess.run(GIT + ['log', '-1', '--format=%h', '--', 'grs'], capture_output=True, text=True).stdout.strip()
DIRTY = subprocess.run(GIT + ['diff', '--quiet', 'HEAD', '--', 'grs']).returncode != 0
VERSION = f'{grs.__version__}+{COMMIT}' + ('.dirty' if DIRTY else '')
print('GRSprocessor', VERSION, '| GRSdriver', GRSdriver.__version__, '| aeronet_oc_db root', aeronet_oc_db.default_root())
warnings.filterwarnings('ignore', category=RuntimeWarning)
''')

md('## Settings')
code(r'''
SITE = '__SITE__'
INSITU_SITES = __INSITU__    # AERONET-OC series of the platform (merged when several)
START, END = '__START__', '__END__'      # period of the images
MAX_LAG = pd.Timedelta(hours=1)              # in situ record to overpass (Zibordi et al. 2009: +-1 h)
SEARCH_LAG = pd.Timedelta(hours=2)           # wider lag to select the images worth processing
MAX_CLOUD = 50                               # cloud cover of the tile (%), for the selection only

L1C_DIR = '/data/satellite/s3/grstbx-s3/sentinel-2/l1c'   # <tile>/YYYY/MM/DD/<image>.SAFE
CAMS_DIR = '/data/cams/world'   # daily global files YYYY/MM/DD/YYYY-MM-DD-cams-...nc, else monthly cams_forecast_YYYY-MM.nc
CAMS_SITE_DIR = Path('/data/cams/sites') / SITE   # CAMS cells around the site cut out of these files (make_cams_site.py)
CAMS_SITE_MARGIN = 2.0   # deg
CACHE = Path('/data/grs/validation/aeronet_oc') / SITE / VERSION
CACHE.mkdir(parents=True, exist_ok=True)

RESOLUTION = 20
SUBSET_DIR = '/data/satellite/Sentinel-2/subset/aeronet/L1C'   # L1C subsets <tile>/<year>/<image>.nc
HALF_SUBSET = 5200        # m: L1C subsets of 10.4 x 10.4 km (521 x 521 pixels at 20 m), as the existing archive
SUBSET_WORKERS = 4        # L1C subsets made at the same time (~3 GB each)
EXTRACT = 15              # pixels kept in the cache around the platform (15 x 15 at 20 m)
RING = (4, 7)             # matchup window: pixels at 4 to 7 pixels (Chebyshev distance) from the site, i.e. a ring
                          # 80-140 m away at 20 m, without the platform and its shadow (large at Section-7)
MIN_VALID = 0.5           # minimum fraction of valid water pixels in the window
MAX_CV = 0.2              # maximum coefficient of variation of Rrs(560) in the window
PLATFORM_NIR = 8e-4       # sr-1: pixels brighter than the median of the extraction at 865 nm (platform) left out

GRS_KW = {}               # other options of Process.execute, e.g. {'glint_aot': True}

PROCESS = True            # False: evaluation of the cached extractions only (no subset, no processing)
DOWNLOAD = False          # True: download the missing L1C images (CDSE S3 keys in ~/.s3cfg)

# Sentinel-2 bands (nm, GRSprocessor wl coordinate) -> AERONET-OC band (nm)
BANDS = {443: 443, 490: 490, 560: 560, 665: 667, 783: 779, 865: 865}
''')

md('## 1. In situ data')
code(r'''
info = aeronet_oc_db.site_info().loc[INSITU_SITES]
assert len(info[['latitude', 'longitude', 's2_tile']].drop_duplicates()) == 1, 'series of different locations'
LON, LAT, TILE = float(info.longitude.iloc[0]), float(info.latitude.iloc[0]), info.s2_tile.iloc[0]
print(SITE, INSITU_SITES, LAT, LON, TILE)


def insitu(sites):
    """Level 2.0 records, completed by the level 1.5 ones not (yet) in level 2.0 (quality 'lev15'), of all the
    AERONET-OC series of the platform (`site` column)"""
    out = []
    for site in sites:
        lev = []
        for level in ['lev20', 'lev15']:
            ds = aeronet_oc_db.open_site(site, level)
            df = ds[['Rrs_Lwn', 'Rrs_Lwn_fQ', 'aot']].to_dataframe().unstack('wl')
            df.columns = [f'{v}_{int(w)}' for v, w in df.columns]
            for v in ['sza', 'wind', 'chl']:
                df[v] = ds[v].to_series()
            df['level'] = level
            df['site'] = site
            lev.append(df)
        lev20, lev15 = lev
        out.append(pd.concat([lev20, lev15[~lev15.index.isin(lev20.index)]]))
    df = pd.concat(out).sort_values('level', ascending=False, kind='stable')
    return df[~df.index.duplicated(keep='first')].sort_index()     # same time in two series: level 2.0 first


ins = insitu(INSITU_SITES)
ins = ins[(ins.index >= pd.Timestamp(START) - SEARCH_LAG) & (ins.index <= pd.Timestamp(END) + SEARCH_LAG)]
print(len(ins), 'records in the period;', ins.groupby(['site', 'level']).size().to_dict())
''')

code(r'''
def aot550(df):
    """AOT at 550 nm from 490 and 667 nm (Angstrom law)"""
    a = -np.log(df['aot_667'] / df['aot_490']) / np.log(667 / 490)
    return df['aot_490'] * (550 / 490) ** -a


ins['aot_550'] = aot550(ins)
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for lev, c in [('lev20', 'k'), ('lev15', 'tab:orange')]:
    s = ins[ins.level == lev]
    ax[0].plot(s.index, s.Rrs_Lwn_560, '.', ms=2, c=c, label=lev)
    ax[1].plot(s.index, s.aot_550, '.', ms=2, c=c)
ax[0].set_ylabel('Rrs(560) [sr$^{-1}$]'); ax[1].set_ylabel('AOT(550)'); ax[0].legend()
ax[0].set_title(f'{SITE} AERONET-OC')
''')

md('## 2. Sentinel-2 images: catalogue, matchup candidates, local archive')
code(r'''
images = download.search(TILE, START, END, level='L1C')
images['time'] = pd.to_datetime(images.time).dt.tz_localize(None)


def _key(name):
    m = re.search(r'(S2[A-D])_MSIL1C_(\d{8}T\d{6})', name)
    return m.group(1) + m.group(2)


local = {_key(os.path.basename(p)): p for p in glob.glob(f'{L1C_DIR}/{TILE}/*/*/*/*.SAFE')}
images['path'] = images.name.map(lambda n: local.get(_key(n)))     # any processing baseline
t = ins.index.values
images['n_insitu'] = [int(((t > x - SEARCH_LAG) & (t < x + SEARCH_LAG)).sum()) for x in images.time]


def cams_file(t):
    """Daily global CAMS file of the date (preferred), else the monthly global file"""
    for f in [t.strftime(f'{CAMS_DIR}/%Y/%m/%d/%Y-%m-%d-cams-global-atmospheric-composition-forecasts.nc'),
              t.strftime('/data/cams/daily/cams_forecast_%Y-%m-%d.nc'),
              t.strftime(f'{CAMS_DIR}/cams_forecast_%Y-%m.nc')]:
        if os.path.exists(f):
            return f
    return None


images['cams'] = images.time.map(cams_file)


_cams_cache = {}


def cams_covers(path, t):
    """The CAMS file exists and its forecasts cover the date (a truncated download holds only a few days)"""
    if path is None:
        return False
    if path not in _cams_cache:
        try:
            with xr.open_dataset(path) as ds:
                # new ADS layout (forecast_reference_time x forecast_period), or the older one (time, up to 2024-06)
                ft = pd.to_datetime((ds.forecast_reference_time if 'forecast_reference_time' in ds.coords
                                     else ds.time).values)
                _cams_cache[path] = set(ft.normalize()) if ds.sizes['latitude'] > 100 else set()
        except (FileNotFoundError, OSError, AttributeError, KeyError):
            _cams_cache[path] = set()
    return t.normalize() in _cams_cache[path]


images['cams_ok'] = [cams_covers(c, t) for c, t in zip(images.cams, images.time)]

# the image covers the site (the tile is also delivered for the edges of the swath, which can miss the site)
site_point = Point(LON, LAT)
images['covers_site'] = images.footprint.map(lambda fp: shape(fp).contains(site_point))
cand = images[(images.n_insitu > 0) & (images.cloud_cover <= MAX_CLOUD) & images.covers_site].reset_index(drop=True)
print(f'{len(images)} images of {TILE} in the catalogue ({images.covers_site.sum()} covering the site), '
      f'{len(cand)} matchup candidates (in situ within {SEARCH_LAG}, cloud cover <= {MAX_CLOUD} %)')
print(f'  on disk: {cand.path.notna().sum()}, missing: {cand.path.isna().sum()} '
      f'({cand[cand.path.isna()]["size"].sum() / 1e9:.0f} GB); without CAMS: {(~cand.cams_ok).sum()} '
      f'(months {sorted(set(cand.loc[~cand.cams_ok, "time"].dt.strftime("%Y-%m")))})')
cand.groupby([cand.time.dt.to_period('Q'), cand.path.notna().rename('on_disk')]).size().unstack(fill_value=0)
''')

code(r'''
missing = cand[cand.path.isna()]
if DOWNLOAD and len(missing):
    summary = download.download(missing, L1C_DIR, storage_options=download.cdse_storage_options(), workers=4)
    print(summary)
    local = {_key(os.path.basename(p)): p for p in glob.glob(f'{L1C_DIR}/{TILE}/*/*/*/*.SAFE')}
    cand['path'] = cand.name.map(lambda n: local.get(_key(n)))
else:
    print(f'{len(missing)} images to download (DOWNLOAD = False)')
''')

md(r'''
## 3. L1C subsets around the platform

`make_l1c_subsets.py` (current GRSdriver) cuts 521 x 521 pixels at 20 m around the platform out of each image
(angles at 60 m, bands packed in int16, ~5 MB), in `SUBSET_DIR/<tile>/<year>/<image>.nc`, the layout of the
existing AERONET-OC subsets. Made once per image (a few minutes each, `SUBSET_WORKERS` at a time); the subsets
already there are reused, also without their `.SAFE`. A failure (e.g. platform outside the swath) leaves
`<image>.nc.err`.
''')
code(r'''
def subset_file(name):
    m = re.search(r'_MSIL1C_(\d{4})', name)
    return Path(SUBSET_DIR) / TILE / m.group(1) / name.replace('.SAFE', '.nc')


subsets = {_key(p.name): p for p in Path(SUBSET_DIR, TILE).glob('*/*.nc')}
cand['subset'] = cand.name.map(lambda n: subsets.get(_key(n)))
to_cut = cand[cand.subset.isna() & cand.path.notna()
              & ~cand.name.map(lambda n: Path(str(subset_file(n)) + '.err').exists())]
print(f'{cand.subset.notna().sum()} subsets on disk, {len(to_cut)} to make from the .SAFE')
if PROCESS and len(to_cut):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(Path('make_l1c_subsets.py').resolve()), str(LON), str(LAT),
                        SUBSET_DIR, *to_cut.path, '--half', str(HALF_SUBSET), '--workers', str(SUBSET_WORKERS)],
                       capture_output=True, text=True)
    print(*[l for l in r.stdout.splitlines() if 'FAILED' in l], sep='\n')
    print(f'{(time.time() - t0) / 60:.0f} min, status {r.returncode}')
    subsets = {_key(p.name): p for p in Path(SUBSET_DIR, TILE).glob('*/*.nc')}
    cand['subset'] = cand.name.map(lambda n: subsets.get(_key(n)))
print(f'{cand.subset.notna().sum()} / {len(cand)} candidates with a subset')
''')

md(r'''
## 4. GRSprocessor on the subsets

The L1C subsets are processed by `process_subset.py --batch`, `BATCH` images per worker process (its imports made
once per batch, the memory released between the batches; the images left by a worker killed on the way are run
one per process); `EXTRACT` x `EXTRACT` pixels around the
platform are kept in `CACHE/<image>.nc` with the attributes of the run (versions, CAMS file, options). Failures
are logged in `CACHE/<image>.nc.err` and not retried (delete the file to retry); the subsets without any clear-sky
water pixel (fully cloudy) get `CACHE/<image>.nc.nowater` instead of an extraction. The images without a CAMS file
covering their date are left out (`cams_ok`). The runs read the CAMS cells around the site, cut once out of each
global CAMS file by `make_cams_site.py` into `CAMS_SITE_DIR` (a few MB instead of a global monthly file of several
GB read again for every image, which took 20 s to several minutes per image from the hard disk).
''')
code(r'''
from make_cams_site import make as make_cams_site, site_file as site_cams_file

WORKER = Path('process_subset.py').resolve()       # worker processes: memory released between the batches
PY = sys.executable
BATCH = 20      # images per worker process (its imports, ~10 s, made once per batch)


def _out(row):
    return CACHE / f"{row['name'].replace('.SAFE', '')}.nc"


def _done(f):
    return f.exists() or Path(str(f) + '.err').exists() or Path(str(f) + '.nowater').exists()


def process(row):
    name = row['name'].replace('.SAFE', '')
    f = CACHE / f'{name}.nc'
    if f.exists() or Path(str(f) + '.err').exists() or Path(str(f) + '.nowater').exists():
        return f
    cmd = [PY, str(WORKER), str(row['subset']), row['cams'], str(LON), str(LAT), str(f),
           '--extract', str(EXTRACT), '--resolution', str(RESOLUTION), '--options', json.dumps(GRS_KW)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f'{name}: no output (status {r.returncode})')
    if r.returncode and not Path(str(f) + '.err').exists():   # killed (memory): keep the trace
        Path(str(f) + '.err').write_text(f'status {r.returncode}\n{r.stderr[-3000:]}')
    return f


todo = cand[cand.subset.notna() & cand.cams_ok].copy()
print(len(todo), 'images to process or already cached in', CACHE)
if PROCESS:   # CAMS files of the site, made once per global CAMS file
    t0 = time.time()
    for c in sorted(set(todo.cams)):
        if not site_cams_file(c, CAMS_SITE_DIR).exists():
            try:
                make_cams_site(c, LON, LAT, CAMS_SITE_DIR, CAMS_SITE_MARGIN)
            except Exception as e:   # the images of this file keep the global file
                print(Path(c).name, 'FAILED', type(e).__name__, e)
    print(f'CAMS files of the site in {CAMS_SITE_DIR}: {(time.time() - t0) / 60:.0f} min')
todo['cams'] = [str(site_cams_file(c, CAMS_SITE_DIR)) if site_cams_file(c, CAMS_SITE_DIR).exists() else c
                for c in todo.cams]


def process_batch(rows):
    """Images of `rows` in one worker process; those left undone by a failed worker (killed) are run alone"""
    rows = [r for r in rows if not _done(_out(r))]
    if not rows:
        return
    jobs = [dict(safe=str(r['subset']), cams=r['cams'], lon=LON, lat=LAT, out=str(_out(r)), extract=EXTRACT,
                 resolution=RESOLUTION, options=GRS_KW) for r in rows]
    jf = CACHE / f'.jobs_{os.getpid()}.json'
    jf.write_text(json.dumps(jobs))
    r = subprocess.run([PY, str(WORKER), '--batch', str(jf)], capture_output=True, text=True)
    jf.unlink()
    for line in r.stdout.splitlines():
        if '.nc: ' in line:
            print(line)
    left = [x for x in rows if not _done(_out(x))]
    if left:   # worker stopped on the way (killed for its memory, crash): the rest is run one per process
        print(f'worker stopped (status {r.returncode}, {r.stderr.strip().splitlines()[-1:]}), '
              f'{len(left)} images run alone')
        for x in left:
            process(x)


if PROCESS:
    t0 = time.time()
    rows = [row for _, row in todo.iterrows()]
    for i in range(0, len(rows), BATCH):
        process_batch(rows[i:i + BATCH])
    print(f'{(time.time() - t0) / 60:.0f} min')
for e in sorted(CACHE.glob('*.err')):
    print(e.name, '->', e.read_text().splitlines()[0])
nowater = sorted(CACHE.glob('*.nowater'))
print(len(nowater), 'images without clear-sky water pixel (cloudy):', ', '.join(e.name[:-11] for e in nowater))
''')

md(r'''
## 5. Matchups

Ring of pixels around the AERONET-OC coordinates, from `RING[0]` to `RING[1]` pixels away (Chebyshev distance,
80-140 m at 20 m): the platform and its shadow, not masked by GRSprocessor, stay out of it (at Section-7, a ~100 m
structure, a centred window is contaminated and fails the CV test). Water pixels (`mask == 0`) with a valid Rrs,
without the platform (Rrs(865) above the median of the extraction by more than `PLATFORM_NIR`). A matchup
is kept if at least `MIN_VALID` of the window is valid and the CV of Rrs(560) is below `MAX_CV` (Bailey and
Werdell 2006). The in situ record is the nearest Level 2.0 one within `MAX_LAG`, else the nearest one (at
a platform with two series, the Level 2.0 record of one instrument is preferred to a closer Level 1.5 one of the
other). GRSprocessor gives the directional
Rrs (no BRDF correction): it is compared with `Rrs_Lwn` (Lwn / F0, no f/Q correction); `Rrs_Lwn_fQ` is kept
for reference.
''')
code(r'''
def window_stats(ds):
    iy, ix = ds.sizes['y'] // 2, ds.sizes['x'] // 2
    dy = xr.DataArray(np.abs(np.arange(ds.sizes['y']) - iy), dims='y')
    dx = xr.DataArray(np.abs(np.arange(ds.sizes['x']) - ix), dims='x')
    dist = np.maximum(dy, dx)                                     # Chebyshev distance (pixels) to the site
    ring = (dist >= RING[0]) & (dist <= RING[1])
    # pixels of the platform left in the ring (much brighter than the water in the NIR) are also left out
    nir = ds.Rrs.sel(wl=865, method='nearest')
    platform = (nir - nir.median()) > PLATFORM_NIR
    water = ((ds['mask'] == 0) if 'mask' in ds else xr.ones_like(ds.aot550, dtype=bool))
    water = ring & water & ~platform & np.isfinite(ds.Rrs).all('wl')
    w = ds
    rrs = ds.Rrs.where(water)
    n = int(water.sum())
    r560 = rrs.sel(wl=560, method='nearest')
    out = {'n_valid': n, 'frac_valid': n / int(ring.sum()),
           'cv560': float(r560.std() / r560.mean()) if n > 1 else np.nan,
           'aot550_sat': float(w.aot550.where(water).median()),
           'BRDFg': float(w.BRDFg.where(water).median()) if 'BRDFg' in w else np.nan,
           'wind_sat': float(w.wind.median()) if 'wind' in w else np.nan,
           'sza_sat': float(w.sza.median()) if 'sza' in w else np.nan}
    for wl_s in BANDS:
        out[f'Rrs_sat_{wl_s}'] = float(rrs.sel(wl=wl_s, method='nearest').median())
        out[f'Rrs_sat_std_{wl_s}'] = float(rrs.sel(wl=wl_s, method='nearest').std())
    return out


rows = []
for f in sorted(CACHE.glob('*.nc')):
    with xr.open_dataset(f) as ds:
        r = window_stats(ds)
        r.update(image=ds.attrs['image'],
                 time=pd.to_datetime(re.search(r'_MSIL1C_(\d{8}T\d{6})', ds.attrs['image']).group(1)))
    rows.append(r)
sat = pd.DataFrame(rows, columns=list(rows[0]) if rows else ['image', 'time'])
sat = sat[(sat.time >= pd.Timestamp(START)) & (sat.time <= pd.Timestamp(END) + pd.Timedelta(days=1))].reset_index(drop=True)
print(len(sat), 'cached extractions in the period;', len(list(CACHE.glob('*.err'))), 'failures')
if sat.empty:
    raise SystemExit('no cached extraction yet: download the images / CAMS months and run section 3')

# nearest in situ record within MAX_LAG: the nearest Level 2.0 one if within MAX_LAG, else the nearest one
def nearest(df):
    return df.iloc[df.index.get_indexer(sat.time, method='nearest')].reset_index().rename(columns={'time': 'time_insitu'})


near = nearest(ins)
if (ins.level == 'lev20').any():
    near20 = nearest(ins[ins.level == 'lev20'])
    use20 = ((near20.time_insitu - sat.time).abs() <= MAX_LAG).values
    near.loc[use20] = near20.loc[use20]
mu = pd.concat([sat, near], axis=1)
mu['lag_min'] = (mu.time_insitu - mu.time).dt.total_seconds() / 60
mu['ok'] = (mu.lag_min.abs() <= MAX_LAG.total_seconds() / 60) & (mu.frac_valid >= MIN_VALID) & (mu.cv560 <= MAX_CV)
print(f"{mu.ok.sum()} valid matchups ({(mu.ok & (mu.level == 'lev20')).sum()} with level 2.0)")
print(mu[mu.ok].groupby(['site', 'level']).size().to_dict())
mu.to_csv(CACHE / f'matchups_{SITE}.csv', index=False)
mu.loc[mu.ok, ['image', 'lag_min', 'site', 'level', 'n_valid', 'cv560', 'aot550_sat', 'aot_550', 'BRDFg', 'wind_sat']].round(4)
''')

md('## 6. Evaluation')
code(r'''
ok = mu[mu.ok]
if ok.empty:
    raise SystemExit('no valid matchup')
ok20 = ok[ok.level == 'lev20']          # Level 2.0 only (quality-assured in situ data)
stats, stats20 = {}, {}
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, (wl_s, wl_a) in zip(axs.flat, BANDS.items()):
    x, y = ok[f'Rrs_Lwn_{wl_a}'], ok[f'Rrs_sat_{wl_s}']
    stats[wl_s] = matchup_metrics(x, y)
    stats20[wl_s] = matchup_metrics(ok20[f'Rrs_Lwn_{wl_a}'], ok20[f'Rrs_sat_{wl_s}'])
    vmax = np.nanmax([x.max(), y.max()]) * 1.1
    for lev, c in [('lev20', 'k'), ('lev15', 'tab:orange')]:
        s = ok.level == lev
        ax.errorbar(x[s], y[s], yerr=ok.loc[s, f'Rrs_sat_std_{wl_s}'], fmt='o', ms=4, c=c, alpha=.7, label=lev)
    ax.plot([0, vmax], [0, vmax], 'k--', lw=.8)
    ax.axhline(0, c='grey', lw=.5)
    ax.set(xlim=(0, vmax), ylim=(min(0, np.nanmin(y) * 1.1), vmax), title=f'{wl_s} nm (AERONET {wl_a})',
           xlabel='Rrs AERONET-OC [sr$^{-1}$]', ylabel='Rrs GRSprocessor [sr$^{-1}$]')
    for k, (lab, m) in enumerate([('all', stats[wl_s]), ('L2.0', stats20[wl_s])]):
        ax.text(.03, .97 - .3 * k, f"{lab}: N={m['N']}  r={m['r']:.2f}\nslope={m['slope']:.2f}  RMSE={m['rmse']:.1e}\n"
                f"bias={m['bias']:.1e}\nMdSA={m['eps']:.0f}%  SSPB={m['beta']:.0f}%",
                transform=ax.transAxes, va='top', fontsize=7, color='k' if k else 'tab:orange')
axs.flat[0].legend(loc='lower right')

ax = axs.flat[len(BANDS)]
x, y = ok.aot_550, ok.aot550_sat
stats['aot550'] = matchup_metrics(x, y)
stats20['aot550'] = matchup_metrics(ok20.aot_550, ok20.aot550_sat)
for lev, c in [('lev20', 'k'), ('lev15', 'tab:orange')]:
    ax.plot(x[ok.level == lev], y[ok.level == lev], 'o', c=c); vmax = np.nanmax([x.max(), y.max()]) * 1.1
ax.plot([0, vmax], [0, vmax], 'k--', lw=.8)
ax.set(xlim=(0, vmax), ylim=(0, vmax), title='AOT(550)', xlabel='AERONET-OC', ylabel='GRSprocessor')
for k, (lab, m) in enumerate([('all', stats['aot550']), ('L2.0', stats20['aot550'])]):
    ax.text(.03, .97 - .15 * k, f"{lab}: N={m['N']} r={m['r']:.2f}\nbias={m['bias']:.3f}  MdSA={m['eps']:.0f}%",
            transform=ax.transAxes, va='top', fontsize=7, color='k' if k else 'tab:orange')

ax = axs.flat[-1]
wls = list(BANDS)
for _, r in ok.iterrows():
    ax.plot(wls, [r[f'Rrs_sat_{w}'] - r[f'Rrs_Lwn_{BANDS[w]}'] for w in wls], c='grey', alpha=.3)
ax.plot(wls, [stats[w]['bias'] for w in wls], 'o-', c='tab:orange', label='bias, all')
ax.plot(wls, [stats20[w]['bias'] for w in wls], 'ko-', label='bias, L2.0')
ax.legend(fontsize=8)
ax.axhline(0, c='k', lw=.5); ax.set(xlabel='wavelength [nm]', ylabel='Rrs sat - in situ', title='Spectral residuals')
fig.suptitle(f'GRSprocessor {VERSION} vs AERONET-OC {SITE} ({START} to {END}, |lag| <= {MAX_LAG})')
fig.tight_layout()
fig.savefig(CACHE / f'matchups_{SITE}.png', dpi=150)
''')

code(r'''
# statistics with all the matchups (levels 2.0 and 1.5) and with the level 2.0 ones only
table = pd.concat({'all': pd.DataFrame(stats).T, 'lev20': pd.DataFrame(stats20).T}, names=['insitu', 'band'])
table.to_csv(CACHE / f'metrics_{SITE}.csv')
table.round({'slope': 3, 'intercept': 5, 'r': 3, 'rmse': 5, 'bias': 5, 'mape': 1, 'eps': 1, 'beta': 1})
''')

md(r'''
### Residuals against the conditions of the scene

Error of Rrs(865) and of the AOT against the sunglint (`BRDFg`), the wind and the sun zenith angle: the
over-estimated AOT under glint (see `research/03_aot_overestimation.ipynb`) shows here as a negative
Rrs(865) residual growing with `BRDFg`.
''')
code(r'''
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex='col')
res865 = ok.Rrs_sat_865 - ok.Rrs_Lwn_865
resaot = ok.aot550_sat - ok.aot_550
for j, v in enumerate(['BRDFg', 'wind_sat', 'sza_sat']):
    axs[0, j].plot(ok[v], res865, 'o'); axs[0, j].axhline(0, c='k', lw=.5)
    axs[1, j].plot(ok[v], resaot, 'o', c='tab:red'); axs[1, j].axhline(0, c='k', lw=.5)
    axs[1, j].set_xlabel(v)
axs[0, 0].set_ylabel('Rrs(865) sat - in situ'); axs[1, 0].set_ylabel('AOT(550) sat - in situ')
fig.tight_layout()
''')

nb['cells'] = C
nb.metadata['kernelspec'] = {'name': 'py12', 'display_name': 'py12', 'language': 'python'}
nbf.write(nb, Path(__file__).resolve().parent / f'{NB_NAME}_s2_matchups.ipynb')
print('written')
