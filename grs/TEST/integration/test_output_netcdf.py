"""
Test of the actual NetCDF files written by L2AProduct.to_netcdf()
"""

from __future__ import annotations

import numpy as np
import pytest

from .helpers import sample_product

pytestmark = pytest.mark.integration

def test_l2a_netcdf_exists(process_result, output_product):
    assert output_product is not None
    assert len(output_product.data_vars) > 0

def test_l2a_netcdf_has_expected_variables(output_product, test_case):
    expected = set(test_case.get("expected", {}).get("variables", []))
    missing = expected - set(output_product.data_vars)
    assert not missing, f"Variables absentes du NetCDF final: {sorted(missing)}"


def test_l2a_netcdf_has_expected_dimensions(output_product, test_case):
    expected = test_case.get("expected", {}).get("dimensions", {})
    for dimension, expected_size in expected.items():
        assert dimension in output_product.dims
        assert output_product.sizes[dimension] == expected_size


def test_l2a_netcdf_preserves_rrs_shape(output_product, l2_product):
    assert output_product["Rrs"].shape == l2_product["Rrs"].shape
    assert output_product["Rrs"].dims == ("wl", "y", "x")


def test_l2a_netcdf_preserves_wavelengths(output_product, l2_product):
    np.testing.assert_array_equal(output_product.wl.values, l2_product.wl.values)

    if "central_wavelength" in l2_product.coords:
        np.testing.assert_allclose(
            output_product.central_wavelength.values,
            l2_product.central_wavelength.values,
            rtol=0,
            atol=1e-6,
        )


def test_l2a_netcdf_has_crs(output_product, test_case):
    assert output_product.rio.crs is not None

    expected_epsg = test_case.get("expected", {}).get("epsg")
    if expected_epsg is not None:
        assert output_product.rio.crs.to_epsg() == expected_epsg


# def test_l2a_netcdf_in_situ_values(output_product, test_case):
#     """Also validate the values after NetCDF encoding/decoding."""

#     measurements = test_case.get("insitu", [])
#     if not measurements:
#         pytest.skip("No in situ measurement in the reference case.")

#     for measurement in measurements:
#         point = measurement["point"]
#         default_window = measurement.get("window_pixels", 1)
#         name = measurement.get("name", "in situ")

#         for variable, entries in measurement["variables"].items():
#             for entry in entries:
#                 wavelength = entry.get("wavelength")
#                 observed = float(entry["value"])
#                 actual = sample_product(
#                     output_product,
#                     variable=variable,
#                     point=point,
#                     wavelength=wavelength,
#                     window=entry.get("window_pixels", default_window),
#                 )

#                 np.testing.assert_allclose(
#                     actual,
#                     observed,
#                     atol=float(entry.get("atol", 0.0)),
#                     rtol=float(entry.get("rtol", 0.0)),
#                     err_msg=(
#                         f"Failure on the comparison with in situ in the NetCDF: "
#                         f"{name}, {variable}, wl={wavelength}"
#                     ),
#                 )


def test_ancillary_netcdf_exists(ancillary_output):
    assert ancillary_output is not None
    assert "xc" in ancillary_output.dims
    assert "yc" in ancillary_output.dims