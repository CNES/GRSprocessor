"""
Ficture for end-to-end integration tests.

The test case is supplied through the GRS_TEST_CASE environment variable:

    GRS_TEST_CASE=/path/to/case.yml pytest -m integration

The YAML file contains the SAFE, CAMS, DEM and the expected product contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from grs import Process

DEFAULT_CASE = Path(__file__).parent / "reference" / "case_S2A_20230129.yml"


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_case() -> tuple[dict, Path]:
    filename = Path(os.environ.get("GRS_TEST_CASE", DEFAULT_CASE)).expanduser().resolve()
    if not filename.exists():
        pytest.fail(
            f"Files of test cas not found: {filename}\n"
            "Please define GRS_TEST_CASE to your YAML reference file."
        )

    with filename.open("r", encoding="utf-8") as stream:
        case = yaml.safe_load(stream) or {}

    if not isinstance(case, dict):
        pytest.fail(f"The file {filename} must contain a YAML object")

    return case, filename.parent


def _check_required_input(case: dict, key: str, base_dir: Path) -> Path:
    value = case.get("inputs", {}).get(key)
    if not value:
        pytest.fail(f"The test case must define inputs. {key}")

    path = _resolve_path(value, base_dir)
    assert path is not None
    if not path.exists():
        pytest.fail(f"Files/Repository not found for {key}: {path}")

    return path


@pytest.fixture(scope="session")
def test_case() -> dict:
    case, base_dir = _load_case()

    inputs = case.setdefault("inputs", {})
    inputs["safe"] = str(_check_required_input(case, "safe", base_dir))
    inputs["cams"] = str(_check_required_input(case, "cams", base_dir))
    inputs["dem"] = str(_check_required_input(case, "dem", base_dir))
    inputs["hypernets"] = str(_check_required_input(case, "hypernets", base_dir))   
    inputs["grsdata"] = str(_check_required_input(case, "grsdata", base_dir)) 

    return case


@pytest.fixture(scope="session")
def hypernets(test_case) -> xr.Dataset:
    path = test_case["inputs"]["hypernets"]

    with xr.open_dataset(path) as dataset:
        yield dataset.load()


@pytest.fixture(scope="session")
def process_result(test_case, tmp_path_factory):
    """Run the complete GRS processing pipeline"""

    output_dir = tmp_path_factory.mktemp("grs_l2a")
    options = test_case.get("processing", {})

    process = Process()

    grsdata = test_case["inputs"].get("grsdata")
    if grsdata:
        process.lut_file = str(Path(grsdata) / Path(process.lut_file).name)
        process.trans_lut_file = str(Path(grsdata) / Path(process.trans_lut_file).name)

    process.execute(
        test_case["inputs"]["safe"],
        odir = str(output_dir),
        cams_file=test_case['inputs']['cams'],
        dem_file=test_case["inputs"]["dem"],
        surfwater_file=options.get("surfwater_file"),
        resolution=options.get('resolution', 20),
        scale_aot=options.get("scale_aot", 1),
        opac_model=options.get("opac_model"),
        allpixels=options.get("allpixels", False),
        snap_compliant=options.get("snap_compliant", False),
    )

    assert process.l2a is not None, "The process didn't construct the L2A product"
    assert process.l2a.construct_l2a is not None, 'The L2A product is empty'

    # Keep the output directory on the process object so output tests can reuse it.
    process._test_output_dir = output_dir
    return process


@pytest.fixture(scope="session")
def l2_product(process_result) -> xr.Dataset:
    return process_result.l2a.l2_prod


@pytest.fixture(scope="session")
def output_product(process_result) -> xr.Dataset:
    """Write the netCDF output once, then reopen it with xarray"""

    process_result.write_output()

    output_dir = process_result._test_output_dir
    nc_files = sorted(
        path for path in output_dir.glob("*.nc")
        if not path.name.endswith("_anc.nc")
    )

    assert len(nc_files) == 1, (
        f"Only one L2A netcdf product was expected in {output_dir}, "
        f"Found : {nc_files}"
    )

    dataset = xr.open_dataset(nc_files[0], decode_coords='all')
    yield dataset
    dataset.close()


@pytest.fixture(scope="session")
def ancillary_output(process_result) -> xr.Dataset:
    output_dir = process_result._test_output_dir
    ancillary_files = sorted(output_dir.glob("*_anc.nc"))

    assert len(ancillary_files) == 1, (
        f"Only one L2A netcdf product was expected in {output_dir}, "
        f"Found : {ancillary_files}"
    )

    dataset = xr.open_dataset(ancillary_files[0], decode_coords='all')
    yield dataset
    dataset.close()