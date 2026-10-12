"""End-to-end tests"""

from __future__ import annotations

import numpy as np
import pytest

from .helpers import sample_product, get_hypernets_location, interpolate_hypernets_reflectance

pytestmark = pytest.mark.integration

DEFAULT_VARIABLES = {
    "Rrs",
    "BRDFg",
    "aot550",
    "vza",
    "sza",
    "raa",
    "flags",
    "dem",
    "mask",
}

def test_product_is_created(l2_product):
    assert l2_product is not None
    assert len(l2_product.data_vars) > 0
    assert "x" in l2_product.dims
    assert "y" in l2_product.dims


def test_expected_variables(l2_product, test_case):
    expected = set(test_case.get("expected", {}).get("variables", DEFAULT_VARIABLES))
    actual = set(l2_product.data_vars)

    missing = expected - actual
    unexpected = set(test_case.get("expected", {}).get("forbidden_variables", [])) & actual

    assert not missing, f"Expected variables missing : {sorted(missing)}"
    assert not unexpected, f"Unexpected variables present: {sorted(unexpected)}"


def test_expected_dimensions(l2_product, test_case):
    expected = test_case.get("expected", {}).get("dimensions", {})
    assert expected, "Expected dimensions must be indicate in the reference case."

    for dimension, expected_size in expected.items():
        assert dimension in l2_product.dims, f"Dimension missing: {dimension}"
        assert l2_product.sizes[dimension] == expected_size, (
            f"Dimension {dimension}: expected {expected_size}, "
            f"obtained {l2_product.sizes[dimension]}"
        )


def test_expected_shapes(l2_product, test_case):
    expected_shapes = test_case.get("expected", {}).get("shapes", {})

    for variable, shape in expected_shapes.items():
        assert variable in l2_product, f"Variable missing: {variable}"
        assert tuple(l2_product[variable].shape) == tuple(shape), (
            f"Shape of {variable}: expected {tuple(shape)}, "
            f"obtained {l2_product[variable].shape}"
        )


def test_rrs_structure(l2_product):
    assert "Rrs" in l2_product
    assert l2_product["Rrs"].dims == ("wl", "y", "x")


def test_2d_variables_structure(l2_product):
    variables = [
        "BRDFg",
        "aot550",
        "vza",
        "sza",
        "raa",
        "flags",
        "dem",
        "mask",
    ]

    for variable in variables:
        if variable in l2_product:
            assert l2_product[variable].dims == ("y", "x"), (
                f"{variable}: unexpected dimensions {l2_product[variable].dims}"
            )


def test_expected_wavelengths(l2_product, test_case):
    expected = test_case.get("expected", {}).get("wavelengths")
    if expected is None:
        pytest.skip("No expected.wavelengths list provided.")

    assert "wl" in l2_product.coords
    np.testing.assert_allclose(
        l2_product.wl.values,
        np.asarray(expected),
        rtol=0,
        atol=test_case.get("expected", {}).get("wavelength_tolerance_nm", 1),
    )


def test_crs(l2_product, test_case):
    expected_epsg = test_case.get("expected", {}).get("epsg")
    assert l2_product.rio.crs is not None, "The L2A product hasn't CRS"

    if expected_epsg is not None:
        assert l2_product.rio.crs.to_epsg() == expected_epsg


def test_spatial_dimensions(l2_product):
    assert l2_product.rio.x_dim == "x"
    assert l2_product.rio.y_dim == "y"


def test_xy_monotonic(l2_product):
    assert np.all(np.diff(l2_product.x.values) > 0), "x must be strictly increasing."
    assert np.all(np.diff(l2_product.y.values) < 0), "y must be strictly decreasing."


def test_data_are_not_empty(l2_product):
    for variable in ("Rrs", "aot550", "dem"):
        assert variable in l2_product
        values = l2_product[variable].values
        assert np.isfinite(values).any(), f"No value provided in {variable}."


def test_dem_range(l2_product, test_case):
    limits = test_case.get("expected", {}).get("dem_range_m")
    if limits is None:
        pytest.skip("No expected.dem_range_m range provided.")

    values = l2_product.dem.values
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert finite.min() >= limits[0]
    assert finite.max() <= limits[1]


def test_rrs_against_hypernets(
    l2_product,
    hypernets,
    test_case,
):
    validation = test_case["validation"]["hypernets"]

    lon, lat = get_hypernets_location(hypernets)
    point = {"lon": lon, "lat": lat}

    window = validation.get("spatial_window_pixels", 3)
    target_wavelengths = l2_product.wl.values

    # Rrs HYPERNETS (rho_w / pi), NaN hors de la plage mesurée (400-899 nm)
    reference = interpolate_hypernets_reflectance(hypernets, target_wavelengths)

    # on ne compare que les bandes couvertes par HYPERNETS
    covered = np.isfinite(reference)
    assert covered.sum() >= 2, "HYPERNETS ne couvre pas assez de bandes S2."

    wavelengths = target_wavelengths[covered]
    reference = reference[covered]

    actual = np.asarray([
        sample_product(
            l2_product,
            variable="Rrs",
            point=point,
            wavelength=float(wl),
            window=window,
        )
        for wl in wavelengths
    ])

    details = "\n".join(
        f"  {wl:6.0f} nm : GRS={a:.5f}  HYPERNETS={r:.5f}  écart relatif={(a - r) / r:+.1%}"
        for wl, a, r in zip(wavelengths, actual, reference)
    )

    np.testing.assert_allclose(
        actual,
        reference,
        rtol=validation["tolerance"]["rtol"],
        atol=validation["tolerance"]["atol"],
        err_msg=f"Rrs GRS vs HYPERNETS :\n{details}",
    )