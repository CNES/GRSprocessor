# Changelog

All notable changes to GRS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates are given for tagged releases only.

## [Unreleased] - 3.1.1

### Added
- Inputs and outputs on object storage (S3 or S3-compatible) or web servers (issue #11): the Level-1 product, the
  CAMS, DEM and SURFWATER files, the output (`--odir`, Zarr) and the configuration file (`GRS_CONFIG`) can be
  URLs (`s3://`, `https://`...), in `Process.execute` (`storage_options` argument), the `grs` command and
  `exe/launcher.py`. New module `grs.remote`: URL helpers on fsspec, credentials from the AWS environment; the
  Level-1 bands are read in place by GRSdriver (needs its remote reading, `GRSdriver.remote`), the other input
  files are copied once into a local cache (`GRS_CACHE_DIR`). The no-water marker is also written for a remote
  output. `grs --log_dir`: local folder of the log files, copied next to a remote product at the end.
- Images without water are not processed: `Process.execute` stops right after reading the image (no CAMS,
  cloud mask, LUTs nor aerosol retrieval) when the water test of the processing finds less water than
  `min_water_area` (km², `settings: min_water_area` of the configuration, default 1 km²; argument of
  `execute` and `--min_water_area` option of the CLI; 0: only images without any water pixel). Instead of
  the product, `write_output` writes the marker file `<name>_no_water.txt` (image, water area, minimum,
  version), skipped like a product by `grs --no_clobber` and by the batch tutorial (status `no_water`).
  Not used with `allpixels`.
- Hybrid aerosol retrieval under sunglint (opt-in, `aerosol: glint_aot` of the configuration or `glint_aot`
  argument of `execute`, default off): `Kernel.aerosol_visible_chunk` also returns the slopes of the visible
  Rrs (560, 665 nm) against the glint for a grid of AOT; pooled over the blocks under strong glint
  (`glint_aot_min_glint`, default 0.02), their zero crossing gives an AOT clipped to the admissible window
  of the retrieval. Removes most of the glint stripes due to the over-estimated AOT on glinted scenes.
- Unit tests of the `Kernel` retrievals (`grs/tests/unit/kernel/`: LUT preparation, coarse rasters, wind,
  SWIR and visible aerosol, final processing), of the masking, output product, configuration, command line,
  exceptions and utilities (from `develop`, PR #24 and #26).
- CI from `develop`: GitLab synchronisation in its own workflow (`gitlab-sync.yml`, Gitflow branches,
  merge requests and releases), unit tests before the PyPI publication, test and coverage summary,
  release/hotfix/bugfix branches, documentation published on tags; `notebooks` extra in `pyproject.toml`.

- Snow and ice flags (`grs.mask`, bits 21 `snow` and 22 `ice`): Fmask-like test on the TOA reflectance (NDSI(560,
  1610) > 0.4, R(842) > 0.11, R(560) > 0.10, NDVI < 0.1, no opaque cirrus) with a whiteness test R(490) > 1.03
  R(560) that removes the very turbid waters (brown, NDSI ~0.6); ice when R(560) < 0.5, snow above. Snow and lake
  ice passed the SWIR water index and were processed as water: they are now excluded from the processed pixels
  (`Masking.snow_ice`) and set in `mask`. Thresholds in `grs.mask.Settings`. Checked on the AERONET-OC L1C subsets
  of Lake Erie (ice kept) and Rio de la Plata (turbid plume no longer flagged).

### Changed
- Requires GRSdriver >= 2.1.1 (was 2.0.4): 2.1 is the first version reading the Level-1 products in place from
  a bucket (`GRSdriver.remote`), used by the URL inputs above; 2.1.1 gives the Landsat relative azimuth in the
  convention of Sentinel-2 (`(vaa - saa) % 360`), needed by the Landsat sunglint fix below.
- `grs`: `error.log` is written next to `log_file.log` in the output folder (was the current folder, where
  the runs started from the same folder overwrote each other's error log).
- Faster processing, same products (bit-identical L2A on a Sentinel-2 tile; `execute` about 295 s ->
  150 s on the Loire tile at 20 m):
  - CAMS NetCDF3 files (monthly global files `cams_forecast_YYYY-MM.nc`) read with the xarray `scipy`
    engine (memory map): the area of an image is read in ~5 s instead of ~50 s from a hard disk when the
    file is not in the cache. The NetCDF4 files (daily archive, ADS downloads) keep the default engine.
  - LUT preparation: only the window of the radiance LUT around the angles of the image is read, once
    per run, and the aerosol models are mixed before the angular interpolation (45 + 26 s -> 21 + 7 s for
    the first and refined LUTs).
  - Aerosol retrieval: the loops over the tested AOT values work on numpy arrays (AOT LUT interpolated
    once per block, nearest diffuse light LUT by index, direct transmittance without xarray): 33 -> 19 s.
- Lower memory of the final processing: blocks computed in float32 with in-place products, peak memory
  per block halved (~150 MB for blocks of 512 pixels, ~600 MB for 1024), same speed and Rrs.
- One numba thread per block of the final processing (the blocks already run in `processor: ncpu` dask
  threads): `ncpu` is the number of busy cores, as fast as all the cores per block before.
- Cloud screening with OmniCloudMask only, in the core of the processing: the pixels processed (aerosol
  retrieval and correction) are the clear-sky water pixels, OmniCloudMask class 0 (no thick cloud, thin
  cloud or cloud shadow, `Masking.clear_sky`); cloudy pixels get NaN `Rrs`. The `mask` recipe masks the
  OmniCloudMask bits 11, 12 and 13 (was: s2cloudless bit 1 and OmniCloudMask shadow bit 13).
- s2cloudless removed (masking step, settings, `Masking.s2cloudless_mask` and the `s2cloudless`
  requirement, with LightGBM): flag bits 1 and 2 (`cloud_p06`, `cloud_p08`) are no longer set.
- Merge of `develop` (CNES): Docker images on Ubuntu 24.04 / Python 3.12 without GDAL, SLURM test
  environment on Python 3.12, `exe/launcher.py` without `osgeo` (cloud cover from `s2_cloud_cover`),
  light `environment.yml`, CWL descriptor reading its inputs from S3; requirements with minimum versions
  and GRSdriver >= 2.0.4 (no EOReader, sertit, fiona). The processing code of 3.1.1 is kept.

### Fixed
- Negative Rrs flag (`neg_rrs`) moved from bit 18 to bit 23 and set with a bitwise OR: bit 18 is also the Landsat
  `l1_snow` bit, and the addition carried into bit 19 (`l1_clear`) for a snow pixel with a negative Rrs. Readers of
  the `neg_rrs` bit of earlier products: bit 18 (the names are in the `flag_names` attribute of `flags`).
- Images whose water is entirely under clouds (no clear-sky water pixel after the cloud masking) crashed in
  `Kernel.lut_preparation` (`ValueError: zero-size array to reduction operation fmin`, empty angle tables):
  `Process.execute` now logs a warning and stops like an image without water (`no_water`, None returned,
  marker file written by `write_output`, with the reason `no clear-sky water pixel`).
- Sunglint removal of the multi-view sensors (Sentinel-2), `Kernel.final_process_chunk`: the glint estimated
  in the SWIR was normalised with `sunglint_eps` and `p_slope` of the last band (2190 nm) but propagated to
  the other bands with those of the 1610 nm band, so that only ε(2190)/ε(1610) ≈ 0.89 of the glint was
  removed in every band (regression of 3.0, 2.2.1 was consistent). On a glinted Sentinel-2 scene the
  residual glint was ~3e-3 sr-1 at 1610 nm and ~2e-3 sr-1 in the green. Unit test on a glint-only scene
  (Rrs = 0).
- Sunglint removal of the monoview sensors (Landsat), `Kernel.aerosol_visible_chunk` and
  `Kernel.final_process_chunk`: the glint was divided twice by `sunglint_eps`; same normalisation as
  Sentinel-2. On a glinted Landsat 8 scene, the residual Rrs(1610) drops from 2.6e-3 to 0 sr-1 and the
  retrieved AOT from 0.17 to 0.05 (CAMS 0.07).
- Thread-safe numba threading layer by default (`NUMBA_THREADING_LAYER=threadsafe`: TBB, else OpenMP,
  unless set by the user): the fallback `workqueue` layer aborted the process on the concurrent calls of
  the numba kernels from the dask threads.
- `grs --no_clobber`: the existing product (or no-water marker) is checked before the logger is set up,
  so a skipped run no longer overwrites the `log_file.log` of the previous run (the skip message is
  printed).
- Wind retrieval: `_find_min_cost` picked the first candidate with a NaN cost (`np.argmin` returns the
  first NaN) instead of the smallest valid cost, giving a NaN cost and a wrong wind on the pixels where
  only some wind candidates have a valid cost.
- Configuration tests no longer read the user file of the computer (`~/.config/grs/config.yml`).
- Exit status of `grs` and of `exe/launcher.py` (issue #13): a processing error ended with status 0, so that the
  processing chains went on to the Level-2B with no Level-2A product. The status is now 0 when the product is
  written (or already there with `--no_clobber`), 1 for a processing error, 2 for invalid arguments or a missing /
  unreadable input (was -1, i.e. 255), 3 for an image skipped on purpose without product (cloud cover, no water,
  was 0). `grs.run.main` returns it (`argv` argument for tests) and closes its log files; the `--no_clobber` check of
  the launcher also finds the Zarr products and the no-water marker.

## [3.1.0]

### Added
- New masking procedure including OmniCloudMask (building on the former s2cloudless development).
- Export to Zarr (Zarr v3) with multiscale pyramids and geo-information.
- PNG quicklooks next to the product (Zarr and NetCDF): L1C RGB (TOA reflectance), L2A RGB (Rrs of the
  water pixels) and sunglint (`BRDFg` of the water pixels), at most 1000 pixels, fixed stretches,
  computed in the same pass as the product; `quicklooks=False` export option to skip them.
- AOT coarse resolution stored in output attributes (default 240 m, a multiple of the 10, 20, 30 and
  60 m pixel sizes).
- `--aot_resolution` option of the CLI (pixel size of the coarse grid of the aerosol retrieval, default
  from the configuration, 240 m).
- Documentation of the kernel physics with equations; reworked Sphinx documentation for Read the Docs
  (light/dark logos, partner logos, reorganised example notebooks: processing, auxiliary data,
  visualization, Landsat).
- `CONTRIBUTING.md` and a separate `README_CNES.md`.

- `grs_config` command: prints the configuration in use and checks the LUTs (exit code 1 if one is
  missing); `--init [DATA_ROOT]` writes the user configuration file; `--download` downloads the
  LUTs from Zenodo (MD5 checked) into the LUT folder.
- Configuration page of the documentation.
- CAMS archive in a `YYYY/MM/DD` tree: without CAMS file, `Process.execute` (and `CamsProduct` with `dir`)
  reads `<path: cams>/YYYY/MM/DD/YYYY-MM-DD-cams-global-atmospheric-composition-forecasts.nc`, the global
  file of the day, else downloads the file of the tile next to it (`..._<tile>.nc`); the files of the former
  layout `<path: cams>/<tile>/<date>-forecast_<tile>.nc` are still read. `grs.cams.cams_daily_file` gives
  the path of a date. CAMS tutorial on a 2026 image with the daily archive.
- `aerosol` section of the configuration: settings of the AOT retrieval (`aot_resolution`,
  `min_pixel_prop`, `aot_chunk`, `outlier_factor`, `aot_smoothing_blocks`, `fine_step`, `fine_window`,
  `aot_step`), overridden by the arguments of the same name of `Process.execute` (new arguments
  `min_pixel_prop`, `outlier_factor`, `fine_step`, `fine_window`, `aot_step`, previously fixed). Default
  `aot_resolution` 240 m (was 480 m: one AOT value per Sentinel-2 tile), also for `--aot_resolution`.
- Default `aot_chunk` 64 (was 256): blocks of 15 km at 240 m. Validated against AERONET-OC (Venise,
  Section-7 Platform; 73 matchups): AOT RMSE 0.066 instead of 0.094, bias -0.013 instead of -0.054, Rrs
  RMSE 23 to 33 % lower (study in `notebook/settings_aerosols/AERONET-OC`).
- Unit tests (`grs/tests/unit`, about 20 s): configuration and aerosol settings, CAMS archive, smoothing
  filters, gaseous correction with dask, command line helpers, and end-to-end processing of a small L1C
  subset (`grs/tests/data`, 128 x 128 pixels at 60 m over the Camargue, made with `make_test_data.py`);
  run in the CI with the integration tests.
- Batch processing tutorial (documentation): Level-1 images of a date range processed from Python, or
  one `grs` command per image for the command line.
- Automatic download of the missing LUTs from Zenodo before the first processing
  (`grs.config.ensure_luts`); switched off with `zenodo: auto_download: false` or `GRS_NO_DOWNLOAD=1`.
- PyPI packaging as `GRSprocessor` (package and commands still `grs`): version read from
  `grs.__version__`, dependencies matching the imports, wheel without tests nor LUTs; release recipe in
  `CONTRIBUTING.md` and GitHub workflow `pypi.yml` (trusted publishing on `v*` tags).

### Changed
- Configuration read at run time from the packaged defaults (`grs/config.yml`, now documented), the
  user file `~/.config/grs/config.yml` (or `GRS_CONFIG`) and `GRS_DATA_ROOT`, instead of editing
  `grs/config.yml` before installing (`grs.config` module). When `grsdata` is not set, the LUTs are
  read from the user data folder (`~/.local/share/grs/lut` on Linux). New dependency: `platformdirs`.
- `Process.execute` checks the LUT files before reading the image.
- Coarse raster now defined by spatial resolution instead of pixel factors.
- Optimised LUT interpolation; `acutils` module renamed to `utils`.
- Lazy package loading for faster imports; Sentinel-2 viewing geometry loaded lazily.
- AOT smoothing windows scale with the retrieval blocks (new `aot_smoothing_blocks` parameter of
  `Process.execute`, default 3).
- Python >= 3.11. The CLI reads the cloud cover of Sentinel-2 images from `MTD_MSIL1C.xml` directly:
  the GDAL Python bindings are no longer needed.
- `execute` and `write_output` use `processor: ncpu` dask threads (one per core before), the L1C bands
  are no longer held by the CAMS product, and the water masking is done in place: about 35 % less
  memory and 20 % less time for a Sentinel-2 tile at 20 m.
- Two glibc malloc arenas (`MALLOC_ARENA_MAX=2`) in the Docker images, and set at start-up by the `grs`
  command and `exe/launcher.py` (unless `MALLOC_ARENA_MAX` is set): the memory freed by the reading and
  processing threads is reused, about 2-3 GB less peak memory with the parallel band reading of GRSdriver.
- Blocks without water pixels are skipped in the final processing; the disabled sunglint slope
  probability is no longer computed.
- Integration tests moved to the unit tests (Level-1 product, CAMS data, gaseous transmittance, log
  files) on the small Camargue subset of `grs/tests/data`: the Git LFS test data (`grs/tests/inputs`)
  and `grs/tests/integration_tests` are removed, and CI no longer fetches LFS files.

### Removed
- Legacy eager methods of `Kernel`: `rugosity_est`, `aerosol_swir`, `aerosol_visible` and
  `aerosol_visible_chunk_old` (use the `*_chunk` methods), with the `recalib_aot_ref` and
  `retrieved_aot_ref` attributes they set.
- Unused arguments and attributes of `Kernel`: `chunk` of the `*_chunk` block methods, `update_state` of
  `aerosol_visible_chunk`, and `neg_pix_max`.
- `grs.auxdata` module: `AuxData` moved to `grs.utils` (`from grs import AuxData` still works); the
  obsolete `SensorData` (it could no longer be instantiated) and the masking thresholds are removed, with
  `grs/TEST/src/test_auxdata.py`, which only tested them.

### Fixed
- Installation page of the documentation brought up to date (Python >= 3.11 instead of 3.9, no conda
  GDAL/eoreader stack, install from PyPI, CPU build of PyTorch, venv or conda, editable install, automatic
  LUT download with `GRS_DATA_ROOT`/`GRS_NO_DOWNLOAD`, ADS key for CAMS).
- `wind` was written as uncompressed float64 (Zarr: without Blosc, inner chunks or shards), about 60 % of
  a Zarr product: it is packed as int16 (scale 0.001), 1.2 MB instead of 209 MB at level 0 for a 20 m
  tile, and has its own attributes (`units`, `long_name`) instead of the product ones. The Zarr export
  now compresses and chunks every data variable, also those without packing.
- NetCDF export failed (`TypeError: illegal data type for attribute b'gas_absorption_correction'`):
  the boolean attributes are written as integers (0/1) in the NetCDF files; Zarr keeps the booleans.
- `Process.cams_dir` was set from the `trans_lut` setting instead of `path: cams`.
- NaN `Rrs` on the edges of the image (about half a pixel of the AOT grid), where the AOT map was not
  extrapolated.
- Vertical bands of NaN in Landsat products, and biased sunglint normalisation in the Landsat AOT
  retrieval (slope probability of the reference band taken from a row of the processing block).
- Weights of the AOT smoothing: the number of water pixels of each retrieval block is used as weight
  (it was overwritten, so that only the empty blocks were weighted); outliers and empty blocks get the
  median AOT with a null weight.
- `Kernel.filter2d`: window wrapping around the image edges and in-place computation from already
  smoothed values.
- `--no_clobber` behaviour for Zarr output and `--snap_compliant` help text in the CLI.
- Unresolved imports with lazy package loading.
- Autosummary class template path in the API reference.
- Compatibility with recent xarray versions (air-mass broadcasting in the kernel, `coarsen().mean()`
  in masking).
- Gas absorption correction of dask-backed bands (e.g. GRSdriver with `EOREADER_USE_DASK=1`), which
  failed on the `spatial_ref` coordinate of the CAMS grid.

## [3.0.0]

### Changed
- Merge of the scientific developments with the CI/CD pipeline from 2.1.9: major update of the
  rugosity (surface roughness) and aerosol retrievals, and refinement of the aerosol retrieval
  algorithm (developed as 2.2.0 and 2.2.1, not released: the released 2.2.0 is the CNES CI version
  below).

## [2.2.0] - 2026-10-16

Version with the CI for the CNES operation.

### Added
- CI/CD: GitHub Actions pipeline for the pull requests and the gitflow branches, synchronisation with
  the CNES GitLab, PyPI and Docker Hub publication, documentation build.
- Integration tests (`grs/tests/integration_tests`) and their README.
- `README_CNES.md` (CNES installation and operation) and `CONTRIBUTING.md`.

### Changed
- Masking updated with the s2cloudless developments (`s2cloudless` added to the requirements).
- CWL descriptor updated.

### Fixed
- CLI: accept `--odir` as long alias for `-o`; pass a string path to `gdal.Open`.
- Docker: upgrade datashader to match the dask API.

## [2.1.9] - 2026-05-27

### Added
- L1C NetCDF accepted as input in the CLI.
- CI/CD: GitLab synchronisation, PyPI publication, Sphinx documentation build, coverage badge,
  CWL descriptor.

### Fixed
- Azimuth handling in the sunglint correction.

## [2.1.8] - 2025-03-19

### Changed
- Improved logging and CI.

## [2.1.7] - 2024-11-07

### Changed
- Account for absorption/scattering interaction in the transmittance of absorbing gases.

### Fixed
- Bugs related to true (band-averaged) wavelengths.
- Support for the new ADS CAMS format.

## [2.1.6] - 2024-09-24

### Changed
- Use `GRSdriver` instead of `s2driver`.

## [2.1.5]

### Added
- Landsat 8 and 9 imagery support.

### Changed
- Repackaging of some classes.

## [2.1.4]

### Changed
- `grs` process can be called as a simple function within a script (new input/output handling).
- Output variables now include bitmask flags.

## [2.1.3]

### Changed
- New automated CAMS loading and revised parameters.

## [2.1.2]

### Fixed
- Tiles straddling the Greenwich meridian.

## [2.1.1]

### Changed
- Package data.

## [2.1.0] - 2024-02-22

### Changed
- Major update of the radiative transfer look-up tables and aerosol models.

## [2.0.5]

### Changed
- New output format compliant with the SNAP "BEAM" format.

## [2.0.4]

### Changed
- Output format adapted for QGIS.

## [2.0.3]

### Added
- SurfWater integration.

## [2.0.2]

### Changed
- Memory optimisation.

## [2.0.1]

### Fixed
- I/O issues.

## [2.0.0]

### Removed
- snappy skeleton (SNAP dependency); simplification of previous options; Landsat handling dropped.

## [1.5.0]

### Changed
- Output parameters extended with `ndwi_nir` and `ndwi_swir`.

## [1.4.0]

### Changed
- Image processed by rectangular chunks.

## [1.3.4]

### Added
- Slope and shade from DEM.

## [1.3.3]

### Fixed
- Sunglint BRDF output.

## [1.3.2]

### Changed
- Major improvement of LUT interpolation and access.

## [1.3.1]

### Fixed
- CAMS data extraction with xarray and longitude conventions.

## [1.3.0]

### Changed
- CAMS data update (CDS version): new AOD wavelengths and spectral SSA from 2018 onwards.
- Adjustment for absorbing aerosols through SSA.

## [1.2.4]

### Added
- Option to load MAJA and WaterDetect masks and export them in the output file.
- Option to process WaterDetect water pixels only.

## [1.2.3]

### Changed
- Compliance with SNAP 8.
- Output written directly in compressed NetCDF4 at the end of the process.

### Added
- New utility `get_subset`.

## [1.2.2]

### Changed
- Nearest-neighbour interpolation to keep tile-edge pixels.
- `product.dispose` to minimise memory usage in the SNAP JVM.

## [1.2.1]

### Added
- High-latitude processing (DEM change).
- Option to process all pixels before masking non-water pixels.

## [1.2.0]

### Changed
- Full data matrix loaded from the image instead of line-by-line pixel extraction
  (preparation for multi-pixel retrieval).

## [1.1.4]

### Added
- Cloud mask; compressed NetCDF4 output format.

## [1.1.3]

### Changed
- Minor changes.

## [1.1.2]

### Changed
- Output in Rrs units.

## [1.1.1]

### Changed
- Minor changes.

## [1.1.0]

### Added
- Landsat 4, 5, 7 and 8 support.

## [1.0.0]

### Added
- GRS for Sentinel-2.
