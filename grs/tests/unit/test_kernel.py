from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage
import dask
import dask.array as dsa
import pandas as pd
import logging
import builtins

from types import SimpleNamespace

from grs.grs_kernel import Kernel
from grs.grs_kernel import _find_min_cost, _aot_ref_candidates, _unique_values, _load_small
import grs.grs_kernel as kernel_module

# ------------------------------------------------------------------
# _load_small
# ------------------------------------------------------------------
# region _load_small
def test_load_small_loads_small_dask_dataarray():
    obj = xr.DataArray(
        dsa.from_array(np.arange(10), chunks=5)
    )

    res = _load_small(obj, max_bytes=1000)

    assert not res.chunks
    np.testing.assert_array_equal(res.values, np.arange(10))

def test_load_small_loads_small_dask_dataset():
    obj = xr.Dataset({
        "a": ("x", dsa.from_array(np.arange(10), chunks=5)),
        "b": ("x", dsa.from_array(np.arange(10) * 2, chunks=5)),
    })

    res = _load_small(obj, max_bytes=1000)

    assert not res.chunks
    np.testing.assert_array_equal(res["a"].values, np.arange(10))
    np.testing.assert_array_equal(res["b"].values, np.arange(10) * 2)   

def test_load_small_keeps_large_dask_dataarray_lazy():
    obj = xr.DataArray(
        dsa.from_array(np.arange(100), chunks=10)
    )

    res = _load_small(obj, max_bytes=10)

    assert res is obj
    assert res.chunks is not None

def test_load_small_returns_numpy_dataarray_unchanged():
    obj = xr.DataArray(np.arange(10))

    res = _load_small(obj, max_bytes=1000)

    assert res is obj
#endregion


# ------------------------------------------------------------------
# _unique_values
# ------------------------------------------------------------------
#region _unique_values
def test_unique_values_numpy_arrays():
    a = np.array([3, 1, 2, 1, 3])
    b = np.array([1,20,20])
    
    res_a, res_b = _unique_values(a,b)
    np.testing.assert_array_equal(res_a, np.array([1,2,3])) 
    np.testing.assert_array_equal(res_b, np.array([1,20]))   

def test_unique_values_empty_array():
    res, = _unique_values(np.array([]))

    np.testing.assert_array_equal(res, np.array([]))

def test_unique_values_dask_array():
    a = xr.DataArray(
        dsa.from_array(
            np.array([3, 1, 2, 1, 3]),
            chunks=2,
        )
    )

    res, = _unique_values(a)

    np.testing.assert_array_equal(res, [1, 2, 3])

def test_unique_value_import_error(monkeypatch):
    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "dask" or name.startswith("dask."):
            raise ImportError("No module named 'dask'")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr(builtins, "__import__", mock_import)
    
    a = np.array([3, 1, 2, 1])
    b = np.array([5, 4, 5])

    result = _unique_values(a, b)
    
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result[1], np.array([4, 5]))
#endregion

    
# ------------------------------------------------------------------
# _find_min_cost
# ------------------------------------------------------------------
# region _find_min_cost
def test_find_min_cost_selects_minimum():
    values=np.array([10,20,30])
    cost = np.array([
        [[5.0,1.0]],
        [[2.0,4.0]],
        [[3.0,2.0]],
    ])
    
    best_value, best_cost = _find_min_cost.py_func(cost, values)
    
    np.testing.assert_array_equal(best_value, np.array([[20.0,10.0]]))
    np.testing.assert_array_equal(best_cost, np.array([[2.0,1.0]]))

def test_find_min_cost_all_NaN():
    values = np.array([10,20,30])
    cost = np.array([
            [[np.nan,np.nan]],
            [[np.nan,np.nan]],
            [[np.nan,np.nan]],
    ])
        
    best_value, best_cost = _find_min_cost.py_func(cost, values)
    
    assert np.isnan(best_value[0,0])
    assert np.isnan(best_cost[0,0])

def test_find_min_cost_ignores_nan_candidates():
    """This test reveals a weird behaviour because we use np.argmin instead of np.nanargmin"""
    values = np.array([10, 20, 30])

    cost = np.array([
        [[np.nan]],
        [[2.0]],
        [[3.0]],
    ])

    best_value, best_cost = _find_min_cost.py_func(cost, values)

    assert best_value[0] == 10
    assert np.isnan(best_cost[0])
#endregion


# ------------------------------------------------------------------
# _aot_ref_candidates
# ------------------------------------------------------------------
# region _aot_ref_candidates
def test_aot_ref_candidates_relative_threshold():
    aot_refs = np.array([1.0, 3.0, 5.0, 10.0])
    counts = np.array([0, 1, 2, 4])
    previous_max = 4.0
    rel_thresh = 0.5

    res = _aot_ref_candidates(aot_refs, counts, previous_max, rel_thresh)

    np.testing.assert_array_equal(res, np.array([1.0, 3.0]))

def test_aot_ref_candidates_values_with_NaN():
    aot_refs = np.array([1.0,np.nan,-4.0,10.0])
    counts = np.array([0,0,2,0])
    previous_max = 2.0
    rel_thresh = 1
    
    res = _aot_ref_candidates(aot_refs, counts, previous_max, rel_thresh)
    np.testing.assert_array_equal(res, np.array([1.0, np.nan, 10.0]))

def test_aot_ref_candidates_uses_previous_max():
    aot_refs = np.array([1.0, 3.0, 5.0])
    counts = np.array([0, 1, 2])
    previous_max = 10.0
    rel_thresh = 0.5

    res = _aot_ref_candidates(aot_refs, counts, previous_max, rel_thresh)

    np.testing.assert_array_equal(res, np.array([1.0, 3.0, 5.0]))
#endregion


# ------------------------------------------------------------------
# _cached
# ------------------------------------------------------------------
# region _cached
def test_cached_computes_and_stores_value():
    kernel = Kernel.__new__(Kernel) #init without constructing complex objects prod, aero_lut, trans_lut and cams
    kernel._cache = {}

    factory = Mock(return_value=42)

    result = kernel._cached("my_key", factory)

    assert result == 42
    assert kernel._cache["my_key"] == 42
    factory.assert_called_once_with()


def test_cached_reuses_existing_value():
    kernel = Kernel.__new__(Kernel)
    kernel._cache = {"my_key": 42}

    factory = Mock(return_value=99)

    result = kernel._cached("my_key", factory)

    assert result == 42
    factory.assert_not_called()
#endregion
    

# ------------------------------------------------------------------
# _Negative_counter
# ------------------------------------------------------------------
# region negative_counter
def test_negative_counter_counts_valid_pixels_with_negative_reflectance():
    """Compte les pixels valides dont la réflectance minimale devient négative."""

    from grs.grs_kernel import _NegativeCounter
    counter = _NegativeCounter(base_threshold=-0.0003)

    # Candidat AOT = 0.0 : les deux pixels sont positifs.
    reflectance_aot_0 = np.array([
        [[0.02, 0.01]],
        [[0.03, 0.02]],
    ])

    counter.add(reflectance_aot_0)

    assert counter.counts.tolist() == [0]

    # Candidat AOT = 0.1 : le pixel 0 devient négatif.
    reflectance_aot_01 = np.array([
        [[-0.01, 0.01]],
        [[ 0.02, 0.02]],
    ])

    counter.add(reflectance_aot_01)

    assert counter.counts.tolist() == [0, 1]

    # Candidat AOT = 0.2 : les deux pixels deviennent négatifs.
    reflectance_aot_02 = np.array([
        [[-0.02, -0.01]],
        [[ 0.01,  0.02]],
    ])

    counter.add(reflectance_aot_02)

    assert counter.counts.tolist() == [0, 1, 2]
    
def test_negative_counter_uses_only_first_seven_bands():
    """Ignores bands after seven first"""
    from grs.grs_kernel import _NegativeCounter
    counter = _NegativeCounter(base_threshold=-0.0003)
    
     # 9 bandes, 2 pixels.
    reflectance = np.full((9,1,2),0.01)
    
    # Pixel 1 : negative reflectance on band 8
    reflectance[7,0,1] = -0.5
    
    counter.add(reflectance, nwl=7)

    # Deuxième appel : on vérifie les mêmes pixels avec les 9 bandes.
    counter.add(reflectance)
    assert counter.counts[0] == 0
    assert counter.counts[1] == 1
# endregion

    
# ------------------------------------------------------------------
# load_block
# ------------------------------------------------------------------
# region load_block
def test_load_block_selects_requested_variables():
    raster = xr.Dataset({
        "a" : ("x", np.array([1,2,3])),
        "b" : ("x", np.array([4,5,6])),
        "c" : ("x", np.array([7,8,9])),
    })
    
    res = Kernel._load_block(raster, ["a", "c"])
    
    assert set(res.data_vars) == { "a", "c" }
    np.testing.assert_array_equal(res["a"].values, [1,2,3])
    np.testing.assert_array_equal(res["c"].values, [7,8,9])

def test_load_block_ignores_missing_variable():
    raster = xr.Dataset({
        "a" : ("x", np.array([1,2,3]))
    })
    
    res = Kernel._load_block(raster, ["a", "missing"])
    
    assert set(res.data_vars) == { "a" }
    np.testing.assert_array_equal(res["a"].values, [1,2,3])

def test_load_block_computes_dask_data():
    raster = xr.Dataset({
        "a": ("x", dsa.from_array(np.array([1, 2, 3]), chunks=2)),
    })

    res = Kernel._load_block(raster, ["a"])

    assert res["a"].chunks is None
    np.testing.assert_array_equal(res["a"].values, [1, 2, 3])
# endregion


# ------------------------------------------------------------------
# coarsening_factor
# ------------------------------------------------------------------
# region coarsening_factor

@pytest.fixture
def kernel():
    """Create a Kernel instance without calling its constructor."""
    instance = Kernel.__new__(Kernel)

    instance.raster = xr.Dataset(
        coords={
            "x": [10, 30, 50, 70, 90],
            "y": [100, 120, 140, 160, 180],
        }
    )

    return instance

def test_coarsening_factor_standard_resolution(kernel):
    """Test coarsening with a target resolution that is a multiple
    of the pixel size.
    """
    result = kernel.coarsening_factor(resolution=60)

    assert result == {"x": 3, "y": 3}


def test_coarsening_factor_different_pixel_sizes():
    """Test coarsening when x and y have different pixel sizes."""
    instance = Kernel.__new__(Kernel)

    instance.raster = xr.Dataset(
        coords={
            "x": [0, 20, 40, 60, 80],
            "y": [0, 30, 60, 90, 120],
        }
    )

    result = instance.coarsening_factor(resolution=120)

    assert result == {"x": 6, "y": 4}


@pytest.mark.parametrize(
    "resolution, expected",
    [
        (20, {"x": 1, "y": 1}),
        (40, {"x": 2, "y": 2}),
        (60, {"x": 3, "y": 3}),
        (120, {"x": 6, "y": 6}),
    ],
)
def test_coarsening_factor_multiple_resolutions(kernel, resolution, expected):
    """Test several valid target resolutions."""
    result = kernel.coarsening_factor(resolution)

    assert result == expected


@pytest.mark.parametrize("resolution", [10, 30, 50, 70])
def test_coarsening_factor_invalid_resolution(kernel, resolution):
    """Test that a resolution that is not a multiple of the pixel size
    raises a ValueError.
    """
    with pytest.raises(ValueError, match="not a multiple"):
        kernel.coarsening_factor(resolution)


@pytest.mark.parametrize("resolution", [0, -20])
def test_coarsening_factor_non_positive_resolution(kernel, resolution):
    """Test that a non-positive resolution raises a ValueError."""
    with pytest.raises(ValueError, match="not a multiple"):
        kernel.coarsening_factor(resolution)


def test_coarsening_factor_descending_coordinates():
    """Test that descending coordinates are handled correctly."""
    instance = Kernel.__new__(Kernel)

    instance.raster = xr.Dataset(
        coords={
            "x": [90, 70, 50, 30, 10],
            "y": [180, 160, 140, 120, 100],
        }
    )

    result = instance.coarsening_factor(resolution=60)

    assert result == {"x": 3, "y": 3}
# endregion


# ------------------------------------------------------------------
# get_coarse_masked_raster
# ------------------------------------------------------------------
# region get_coarse_marked_rast
@pytest.fixture
def kernel_coarse():
    """Create a minimal Kernel instance for testing coarse raster generation."""
    instance = Kernel.__new__(Kernel)

    instance.raster = xr.Dataset(
        data_vars={
            "sza": (
                ("y", "x"),
                [
                    [10., 20., 30., 40.],
                    [30., 40., 50., 60.],
                    [50., 60., 70., 80.],
                    [70., 80., 90., 100.],
                ],
            ),
            "vza": (
                ("y", "x"),
                [
                    [1., 2., 3., 4.],
                    [3., 4., 5., 6.],
                    [5., 6., 7., 8.],
                    [7., 8., 9., 10.],
                ],
            ),
            "raa": (
                ("y", "x"),
                [
                    [100., 110., 120., 130.],
                    [110., 120., 130., 140.],
                    [120., 130., 140., 150.],
                    [130., 140., 150., 160.],
                ],
            ),
            "bands": (
                ("y", "x"),
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.5, 0.6],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.7, 0.8, 0.9, 1.0],
                ],
            ),
        },
        coords={
            "x": [10, 30, 50, 70],
            "y": [10, 30, 50, 70],
        },
    )

    return instance

def test_get_coarse_masked_raster_mean(kernel_coarse):
    """Test that raster variables are averaged over coarse pixel blocks."""

    result = kernel_coarse.get_coarse_masked_raster(
        resolution=40,
        min_pixel_prop=0
    )

    expected_sza = np.array([
        [25., 45.],
        [65., 85.],
    ])

    np.testing.assert_allclose(
        result["sza"].values,
        expected_sza
    )

    assert result.sizes["x"] == 2
    assert result.sizes["y"] == 2

def test_get_coarse_masked_raster_with_mask(kernel_coarse):
    """Test that masked pixels are excluded from coarse averages."""

    mask = xr.DataArray(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dims=("y", "x"),
        coords=kernel_coarse.raster.coords,
    )

    result = kernel_coarse.get_coarse_masked_raster(
        mask=mask,
        resolution=40,
        min_pixel_prop=0
    )

    # The pixel with sza=20 is masked.
    # The remaining values are 10, 30 and 40.
    expected_mean = (10. + 30. + 40.) / 3.

    np.testing.assert_allclose(
        result["sza"].values[0, 0],
        expected_mean
    )

    assert result["water_pixel_number"].values[0, 0] == 3

def test_get_coarse_masked_raster_water_pixel_number(kernel_coarse):
    """Test the number of valid pixels in each coarse block."""

    result = kernel_coarse.get_coarse_masked_raster(
        resolution=40,
        min_pixel_prop=0
    )

    expected = np.full((2, 2), 4)

    np.testing.assert_array_equal(
        result["water_pixel_number"].values,
        expected
    )
    
def test_get_coarse_masked_raster_min_pixel_prop(kernel_coarse):
    """Test filtering of coarse blocks according to their valid pixel count."""

    mask = xr.DataArray(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dims=("y", "x"),
        coords=kernel_coarse.raster.coords,
    )

    result = kernel_coarse.get_coarse_masked_raster(
        mask=mask,
        resolution=40,
        min_pixel_prop=0.5
    )

    # Pixel counts:
    # [[4, 1],
    #  [1, 2]]
    #
    # Maximum = 4
    # Threshold = 0.5 * 4 = 2
    # Only blocks with more than 2 valid pixels are retained.

    expected_counts = np.array([
        [4, 1],
        [1, 2],
    ])

    np.testing.assert_array_equal(
        kernel_coarse.coarse_masked_raster["water_pixel_number"].values,
        expected_counts
    )

    assert result["sza"].values[0, 0] == pytest.approx(25.)

    assert np.isnan(result["sza"].values[0, 1])
    assert np.isnan(result["sza"].values[1, 0])
    assert np.isnan(result["sza"].values[1, 1])
    
def test_get_coarse_masked_raster_attributes(kernel_coarse):
    """Test that the coarse raster attributes are correctly updated."""

    result = kernel_coarse.get_coarse_masked_raster(
        resolution=40,
        min_pixel_prop=0
    )

    assert kernel_coarse.coarse_resolution == 40
    assert kernel_coarse.coarse_masked_raster.attrs["resolution"] == 40
    assert result.attrs["resolution"] == 40

@pytest.fixture
def kernel_coarse_odd():
    """Create a 3x3 raster to test incomplete coarse blocks."""
    instance = Kernel.__new__(Kernel)

    instance.raster = xr.Dataset(
        data_vars={
            "sza": (
                ("y", "x"),
                [
                    [10., 20., 30.],
                    [30., 40., 50.],
                    [50., 60., 70.],
                ],
            ),
            "vza": (
                ("y", "x"),
                np.ones((3, 3)),
            ),
            "raa": (
                ("y", "x"),
                np.ones((3, 3)),
            ),
            "bands": (
                ("y", "x"),
                np.ones((3, 3)),
            ),
        },
        coords={
            "x": [10, 30, 50],
            "y": [10, 30, 50],
        },
    )

    return instance
def test_get_coarse_masked_raster_incomplete_blocks(kernel_coarse_odd):
    """Test that incomplete blocks at the raster edges are padded."""

    result = kernel_coarse_odd.get_coarse_masked_raster(
        resolution=40,
        min_pixel_prop=0
    )

    expected_sza = np.array([
        [25., 40.],
        [55., 70.],
    ])

    np.testing.assert_allclose(
        result["sza"].values,
        expected_sza
    )

    expected_counts = np.array([
        [4, 2],
        [2, 1],
    ])

    np.testing.assert_array_equal(
        result["water_pixel_number"].values,
        expected_counts
    )

# endregion


# ------------------------------------------------------------------
# lut_preparation
# ------------------------------------------------------------------
# region lut_preparation
@pytest.fixture
def mock_auxdata(monkeypatch):
    """Replace AuxData with synthetic auxiliary data."""

    def fake_auxdata(wl=None):
        if wl is None:
            wl = np.array([500., 600.])

        wl = np.asarray(wl)

        return SimpleNamespace(
            sunglint_eps=xr.DataArray(
                np.ones(len(wl)),
                dims=("wl",),
                coords={"wl": wl},
            ),
            rot=xr.DataArray(
                np.full(len(wl), 0.1),
                dims=("wl",),
                coords={"wl": wl},
            ),
        )

    monkeypatch.setattr(kernel_module, "AuxData", fake_auxdata)
    
#  Cette fixture est une base de travail, pas encore une fixture garantie compatible avec toutes les versions du code.

# En particulier, il faudra vérifier :

#     si wl_true doit être une coordonnée du raster ou un attribut du raster ;
#     si les transmittances utilisent exactement les mêmes dimensions que I ;
#     si les six modèles synthétiques correspondent bien à AEROSOL_MODELS

@pytest.fixture
def kernel_lut():
    """Create a Kernel with synthetic aerosol LUTs."""

    # --------------------------------------------------
    # 1. Synthetic raster
    # --------------------------------------------------

    wl_true = np.array([500., 600., 700., 800.])

    raster = xr.Dataset(
        data_vars={
            "sza": (
                ("y", "x"),
                [
                    [30., 30.],
                    [40., 40.],
                ],
            ),
            "vza": (
                ("y", "x"),
                [
                    [10., 10.],
                    [20., 20.],
                ],
            ),
            "raa": (
                ("y", "x"),
                [
                    [0., 0.],
                    [90., 90.],
                ],
            ),
            "bands": (
                ("y", "x", "wl"),
                np.ones((2, 2, 4)),
            ),
        },
        coords={
            "x": [0, 20],
            "y": [0, 20],
            "wl": wl_true,
        },
    )

    raster = raster.assign_coords(wl_true=("wl", wl_true))

    prod = SimpleNamespace(
        raster=raster,
        sensor="S2",
        _type=np.float32,
    )

    # --------------------------------------------------
    # 2. LUT coordinates
    # --------------------------------------------------

    winds = [1., 2., 3.]
    models = kernel_module.AEROSOL_MODELS
    aot_refs = np.linspace(0., 0.8, 5)
    szas = [0., 10., 20., 30., 40., 50.]
    vzas = [0., 10., 20., 30.]
    azis = [0., 90., 180., 270.]
    wls = [400., 500., 600., 700., 800., 900.]

    dims = (
        "wind",
        "model",
        "aot_ref",
        "sza",
        "vza",
        "azi",
        "wl",
    )

    coords = {
        "wind": winds,
        "model": models,
        "aot_ref": aot_refs,
        "sza": szas,
        "vza": vzas,
        "azi": azis,
        "wl": wls,
    }

    shape = tuple(len(coords[dim]) for dim in dims)

    # --------------------------------------------------
    # 3. Synthetic aerosol LUT
    # --------------------------------------------------

    I = np.full(shape, 0.2, dtype=float)

    aot = np.ones(shape, dtype=float)

    aero_lut = xr.Dataset(
        data_vars={
            "I": (dims, I),
            "aot": (dims, aot),
        },
        coords=coords,
    )

    # --------------------------------------------------
    # 4. Placeholder transmittance LUT
    # --------------------------------------------------

    trans_lut = xr.Dataset(
        data_vars={
            "Ttot_Ed": (dims, np.ones(shape, dtype=float)),
        },
        coords=coords,
    )

    # --------------------------------------------------
    # 5. Build Kernel without calling __init__
    # --------------------------------------------------

    kernel = Kernel.__new__(Kernel)

    kernel.prod = prod
    kernel.raster = raster
    kernel.aero_lut = aero_lut
    kernel.trans_lut = trans_lut
    kernel.wl_true = wl_true
    kernel._cache = {}

    return kernel

def test_lut_preparation_truncates_angles(kernel_lut, mock_auxdata):
    """Check that the angle rasters are rounded as expected."""

    kernel_lut.lut_preparation()
    np.testing.assert_array_equal(
        kernel_lut.prod.raster["sza_trunc"].values, 
        [[30., 30.],[40., 40.]]
    )
    np.testing.assert_array_equal(
        kernel_lut.prod.raster["vza_trunc"].values,
        [[10., 10.], [20., 20.]],
    )

    np.testing.assert_array_equal(
        kernel_lut.prod.raster["raa_trunc"].values,
        [[0., 0.], [90., 90.]],
    )

def test_lut_preparation_ignores_nan_angles(
    kernel_lut,
    mock_auxdata,
):
    """Check that NaN angles are excluded from the LUT coordinates."""

    kernel_lut.prod.raster["sza"].values[0, 0] = np.nan
    kernel_lut.prod.raster["vza"].values[1, 1] = np.nan
    kernel_lut.prod.raster["raa"].values[0, 1] = np.nan

    kernel_lut.lut_preparation()

    assert not np.isnan(kernel_lut.szas).any()
    assert not np.isnan(kernel_lut.vzas).any()
    assert not np.isnan(kernel_lut.azis).any()


@pytest.fixture
def kernel_lut_spectral(kernel_lut):
    """Create a LUT with a wavelength-dependent intensity."""

    kernel = kernel_lut

    lut = kernel.aero_lut["I"]
    wls = lut["wl"].values

    # Intensity varies linearly with wavelength.
    intensities = np.broadcast_to(
        (wls / 1000)[None, None, None, None, None, None, :],
        lut.shape,
    ).copy()

    kernel.aero_lut["I"] = (lut.dims, intensities)

    return kernel

def test_lut_preparation_rayleigh_spectral_interpolation(
    kernel_lut_spectral,
    mock_auxdata,
):
    """Check spectral interpolation of Rayleigh reflectance."""

    kernel_lut_spectral.lut_preparation()

    result = kernel_lut_spectral.Rray

    expected_intensity = result["wl"] / 1000

    expected = (
        expected_intensity / np.cos(np.radians(result["sza"]))
    ).broadcast_like(result)

    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
        atol=1e-8,
    )

def test_lut_preparation_rayleigh_reflectance(kernel_lut, mock_auxdata):
    """Check Rayleigh reflectance against explicitly expected angles."""

    kernel_lut.lut_preparation()

    result = kernel_lut.Rray

    # Rayleigh intensity is constant: I0 = 0.2.
    # The raster contains SZA values of 30 and 40 degrees.
    expected_sza = np.array([30., 40.])

    expected = 0.2 / np.cos(np.radians(expected_sza))

    np.testing.assert_allclose(
        result["sza"].values,
        expected_sza,
    )

    # The other dimensions must not affect the Rayleigh intensity.
    expected = xr.DataArray(
        expected,
        dims=("sza",),
        coords={"sza": expected_sza},
    ).broadcast_like(result)

    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
    )


@pytest.fixture
def kernel_lut_diffuse(kernel_lut):
    """Create a Kernel with distinct intensities for each aerosol model."""

    kernel = kernel_lut

    dims = kernel.aero_lut["I"].dims
    shape = kernel.aero_lut["I"].shape
    intensities = np.broadcast_to(
        np.arange(1, 7, dtype=float)[None, :, None, None, None, None, None],
        shape,
    ).copy()

    kernel.aero_lut["I"] = (dims, intensities)

    return kernel

def test_lut_preparation_diffuse_reflectance_combination(
    kernel_lut_diffuse,
    mock_auxdata,
):
    """Check the weighted combination of aerosol models in Rdiff_lut."""

    weights = np.array([0., 0.5, 0., 0.5, 0., 0.])

    kernel_lut_diffuse.lut_preparation(
        aerosol_combination=weights,
    )

    result = kernel_lut_diffuse.Rdiff_lut

    # Models have constant intensities: 1, 2, 3, 4, 5, 6.
    # The selected weights give: 0.5 * 2 + 0.5 * 4 = 3.
    expected = 3. / np.cos(np.radians(result["sza"]))
    expected = expected.broadcast_like(result)

    assert len(kernel_module.AEROSOL_MODELS) == 6
    
    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
    )

@pytest.fixture
def kernel_lut_aot(kernel_lut):
    """Create a Kernel with synthetic AOT values varying by model,
    reference AOT and wavelength.
    """

    kernel = kernel_lut

    lut = kernel.aero_lut["aot"]

    # Model factors: 1, 2, 3, 4, 5, 6.
    model_factors = np.arange(1, 7, dtype=float).reshape(
        1, 6, 1, 1, 1, 1, 1
    )

    # Reference AOT and wavelength coordinates.
    aot_refs = lut.aot_ref.values
    wls = lut.wl.values

    # Build a function that is linear in aot_ref and wl.
    base = (
        aot_refs[None, None, :, None, None, None, None]
        + wls[None, None, None, None, None, None, :] / 1000
    )

    values = np.broadcast_to(
        model_factors * base,
        lut.shape,
    ).copy()

    kernel.aero_lut["aot"] = (lut.dims, values)

    return kernel

def test_lut_preparation_aot_combination_and_interpolation(
    kernel_lut_aot,
    mock_auxdata,
):
    """Check weighted AOT combination and interpolation."""

    weights = np.array([0., 0.5, 0., 0.5, 0., 0.])

    kernel_lut_aot.lut_preparation(
        aerosol_combination=weights,
    )

    result = kernel_lut_aot.aot_lut

    # The model factors are 1, 2, 3, 4, 5, 6.
    # The selected weights give:
    # 0.5 * 2 + 0.5 * 4 = 3.
    #
    # Expected AOT:
    # 3 * (aot_ref + wl / 1000)

    expected = 3.0 * (
        result["aot_ref"] + result["wl"] / 1000
    )

    expected = expected.broadcast_like(result)

    assert len(kernel_module.AEROSOL_MODELS) == 6
    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
        atol=1e-8,
    )

@pytest.fixture
def kernel_lut_transmittance(kernel_lut):
    """Create a transmittance LUT with distinct values per aerosol model."""

    kernel = kernel_lut

    dims = kernel.trans_lut["Ttot_Ed"].dims
    shape = kernel.trans_lut["Ttot_Ed"].shape

    transmittances = np.broadcast_to(
        np.arange(1, 7, dtype=float)[None, :, None, None, None, None, None],
        shape,
    ).copy()

    kernel.trans_lut["Ttot_Ed"] = (dims, transmittances)
    return kernel

def test_lut_preparation_transmittance_combination(
    kernel_lut_transmittance,
    mock_auxdata,
):
    """Check the weighted combination of transmittance models."""

    weights = np.array([0., 0.5, 0., 0.5, 0., 0.])

    kernel_lut_transmittance.lut_preparation(
        aerosol_combination=weights,
    )

    result = kernel_lut_transmittance.trans_aero_lut["Ttot_Ed"]

    # The selected models have transmittances 2 and 4.
    # Their weighted mean is 3.
    expected = xr.full_like(result, 3.)

    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
        atol=1e-8,
    )

def test_lut_preparation_aot_reference_coordinates(
    kernel_lut,
    mock_auxdata,
):
    """Check the reference AOT coordinates after interpolation."""

    requested_aot_refs = np.linspace(0.0, 0.8, 25)

    kernel_lut.lut_preparation(
        aot_refs=requested_aot_refs,
    )

    np.testing.assert_allclose(
        kernel_lut.aot_refs,
        requested_aot_refs,
    )

    np.testing.assert_allclose(
        kernel_lut.Rdiff_lut["aot_ref"].values,
        requested_aot_refs,
    )

    np.testing.assert_allclose(
        kernel_lut.aot_lut["aot_ref"].values,
        requested_aot_refs,
    )

    np.testing.assert_allclose(
        kernel_lut.trans_aero_lut["aot_ref"].values,
        requested_aot_refs,
    )

def test_lut_preparation_removes_model_dimension(
    kernel_lut,
    mock_auxdata,
):
    """Check that no prepared LUT retains the aerosol model dimension."""

    kernel_lut.lut_preparation()

    for name in (
        "Rray",
        "Rdiff_lut",
        "aot_lut",
        "trans_aero_lut",
    ):
        result = getattr(kernel_lut, name)

        assert "model" not in result.dims, (
            f"{name} unexpectedly contains the model dimension"
        )

def test_lut_preparation_azimuth_values(kernel_lut, mock_auxdata):
    """Check the conversion from relative azimuth to LUT azimuth."""

    kernel_lut.lut_preparation()

    expected_azimuths = np.array([90., 180.])

    np.testing.assert_allclose(
        kernel_lut.azis,
        expected_azimuths,
    )

def test_lut_preparation_loads_prepared_luts(
    kernel_lut,
    mock_auxdata,
    monkeypatch,
):
    """Check that _load_small is called for all prepared LUTs."""

    original = kernel_module._load_small
    calls = []

    def spy_load_small(obj, *args, **kwargs):
        calls.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(
        kernel_module,
        "_load_small",
        spy_load_small,
    )

    kernel_lut.lut_preparation()

    # Four LUTs and two auxiliary datasets.
    assert len(calls) >= 4

    assert kernel_lut.trans_aero_lut is not None
    assert kernel_lut.Rray is not None
    assert kernel_lut.Rdiff_lut is not None
    assert kernel_lut.aot_lut is not None

def test_lut_preparation_uses_separate_cache_for_angle_resolution(
    kernel_lut,
    mock_auxdata,
    monkeypatch,
):
    """Check that different angle resolutions use different cache entries."""

    original = kernel_module._unique_values
    calls = []

    def spy_unique_values(*arrays):
        calls.append(arrays)
        return original(*arrays)

    monkeypatch.setattr(
        kernel_module,
        "_unique_values",
        spy_unique_values,
    )

    kernel_lut.lut_preparation()

    kernel_lut.lut_preparation(
        ang_resol={
            "sza": 0,
            "vza": 0,
            "raa": 0,
        },
    )

    assert len(calls) == 2

def test_lut_preparation_caches_unique_angles(
    kernel_lut,
    mock_auxdata,
    monkeypatch,
):
    """Check that unique angle values are cached."""

    original = kernel_module._unique_values
    calls = []

    def spy_unique_values(*arrays):
        calls.append(arrays)
        return original(*arrays)

    monkeypatch.setattr(
        kernel_module,
        "_unique_values",
        spy_unique_values,
    )

    kernel_lut.lut_preparation()

    assert len(calls) == 1

    # Repeating the preparation with the same angular resolution
    # must reuse the cached angle values.
    kernel_lut.lut_preparation()

    assert len(calls) == 1

@pytest.fixture
def kernel_lut_wind_dependent(kernel_lut):
    """Create an aerosol LUT whose intensity depends on wind speed."""

    kernel = kernel_lut

    dims = kernel.aero_lut["I"].dims
    shape = kernel.aero_lut["I"].shape

    wind_values = kernel.aero_lut["wind"].values.reshape(
        -1, 1, 1, 1, 1, 1, 1
    )

    intensities = np.broadcast_to(
        wind_values,
        shape,
    ).copy()

    kernel.aero_lut["I"] = (dims, intensities)

    return kernel

def test_lut_preparation_selects_nearest_wind(
    kernel_lut_wind_dependent,
    mock_auxdata,
):
    """Check that the nearest wind-dependent LUT is selected."""

    kernel_lut_wind_dependent.lut_preparation(
        wind=2.4,
    )

    result = kernel_lut_wind_dependent.Rdiff_lut

    expected = 2. / np.cos(np.radians(result["sza"]))
    expected = expected.broadcast_like(result)

    np.testing.assert_allclose(
        result.values,
        expected.values,
        rtol=1e-6,
    )

def test_lut_preparation_auxiliary_data(kernel_lut, mock_auxdata):
    """Check that auxiliary spectral data are initialized."""

    kernel_lut.lut_preparation()

    np.testing.assert_allclose(
        kernel_lut.sunglint_eps.values,
        np.ones(len(kernel_lut.wl_true)),
    )

    np.testing.assert_allclose(
        kernel_lut.rot.values,
        np.full(len(kernel_lut.wl_true), 0.1),
    )

    np.testing.assert_allclose(
        kernel_lut.sunglint_eps["wl"].values,
        kernel_lut.wl_true,
    )

    np.testing.assert_allclose(
        kernel_lut.rot["wl"].values,
        kernel_lut.wl_true,
    )

def test_lut_preparation_invalid_aerosol_combination(
    kernel_lut,
    mock_auxdata,
    caplog,
):
    """Check that invalid aerosol weights stop LUT preparation."""

    with caplog.at_level(logging.ERROR):
        result = kernel_lut.lut_preparation(
            aerosol_combination="invalid",
        )

    assert result is None

    assert any(
        "aerosol_combination" in record.message
        for record in caplog.records
    )

    assert not hasattr(kernel_lut, "Rdiff_lut")

def test_lut_preparation_accepts_dataarray_combination(
    kernel_lut,
    mock_auxdata,
):
    """Check that aerosol weights can be supplied as a DataArray."""

    weights = xr.DataArray(
        [0., 0.5, 0., 0.5, 0., 0.],
        dims=("model",),
        coords={"model": kernel_module.AEROSOL_MODELS},
    )

    kernel_lut.lut_preparation(
        aerosol_combination=weights,
    )

    xr.testing.assert_equal(
        kernel_lut.aerosol_combination,
        weights,
    )

def test_lut_preparation_default_aerosol_combination(
    kernel_lut,
    mock_auxdata,
):
    """Check that the default aerosol combination is used."""

    kernel_lut.lut_preparation()

    xr.testing.assert_equal(
        kernel_lut.aerosol_combination,
        xr.DataArray(
            kernel_module.AEROSOL_COMBINATION,
            coords={"model": kernel_module.AEROSOL_MODELS},
        ),
    )

@pytest.fixture
def kernel_lut_decimal_angles(kernel_lut):
    """Create a raster with non-integer viewing angles."""

    kernel = kernel_lut

    kernel.prod.raster["sza"][:] = [
        [30.14, 30.24],
        [40.16, 40.24],
    ]

    kernel.prod.raster["vza"][:] = [
        [10.14, 10.24],
        [20.16, 20.24],
    ]

    kernel.prod.raster["raa"][:] = [
        [0.14, 0.24],
        [90.16, 90.24],
    ]

    return kernel

def test_lut_preparation_rounds_angles(kernel_lut_decimal_angles, mock_auxdata):
    """Check that angle resolution controls the rounding."""

    kernel_lut_decimal_angles.lut_preparation(
        ang_resol={
            "sza": 1,
            "vza": 1,
            "raa": 0,
        },
    )

    raster = kernel_lut_decimal_angles.prod.raster

    np.testing.assert_allclose(
        raster["sza_trunc"].values,
        [
            [30.1, 30.2],
            [40.2, 40.2],
        ],
    )

    np.testing.assert_allclose(
        raster["vza_trunc"].values,
        [
            [10.1, 10.2],
            [20.2, 20.2],
        ],
    )

    np.testing.assert_allclose(
        raster["raa_trunc"].values,
        [
            [0., 0.],
            [90., 90.],
        ],
    )

def test_lut_preparation_output_dimensions(kernel_lut, mock_auxdata):
    """Check dimensions and wavelength coordinates of prepared LUTs."""

    kernel_lut.lut_preparation()

    expected_wavelengths = kernel_lut.wl_true

    for result in (
        kernel_lut.Rray,
        kernel_lut.Rdiff_lut,
        kernel_lut.aot_lut,
        kernel_lut.trans_aero_lut,
    ):
        np.testing.assert_allclose(
            result["wl"].values,
            expected_wavelengths,
        )

    assert "model" not in kernel_lut.Rdiff_lut.dims
    assert "model" not in kernel_lut.aot_lut.dims
    assert "model" not in kernel_lut.trans_aero_lut.dims

    assert "aot_ref" in kernel_lut.Rdiff_lut.dims
    assert "aot_ref" in kernel_lut.aot_lut.dims
    assert "aot_ref" in kernel_lut.trans_aero_lut.dims

    np.testing.assert_allclose(
        kernel_lut.aot_refs,
        kernel_lut.Rdiff_lut["aot_ref"].values,
    )

    np.testing.assert_allclose(
        kernel_lut.szas,
        kernel_lut.Rdiff_lut["sza"].values,
    )

    np.testing.assert_allclose(
        kernel_lut.vzas,
        kernel_lut.Rdiff_lut["vza"].values,
    )

    np.testing.assert_allclose(
        kernel_lut.azis,
        kernel_lut.Rdiff_lut["azi"].values,
    )

@pytest.fixture
def kernel_lut_spectral(kernel_lut):
    """Create a LUT with a wavelength-dependent intensity."""

    kernel = kernel_lut

    lut = kernel.aero_lut["I"]
    wls = lut["wl"].values

    # Intensity varies linearly with wavelength.
    intensities = np.broadcast_to(
        (wls / 1000)[None, None, None, None, None, None, :],
        lut.shape,
    ).copy()

    kernel.aero_lut["I"] = (lut.dims, intensities)

    return kernel

# endregion


# ------------------------------------------------------------------
# mu_N
# ------------------------------------------------------------------
# region mu_N
def test_mu_N_monoview():
    """Check mu_N for zero angles in monoview mode."""
    kernel = Kernel.__new__(Kernel)
    sza = xr.DataArray(
            np.full((2, 2), 60.),
            dims=("y", "x"),
    )

    vza = xr.DataArray(
        np.full((2, 2), 60.),
        dims=("y", "x"),
    )

    azi = xr.DataArray(
        np.full((2, 2), 180.),
        dims=("y", "x"),
    )

    result = kernel.mu_N(
        sza=sza,
        vza=vza,
        azi=azi,
        monoview=True,
    )

    np.testing.assert_allclose(
        result.values,
        np.ones((2, 2)),
        rtol=1e-12,
        atol=1e-12,
    )
    
def test_mu_N_multiview(kernel):
    """Check mu_N calculation and dimensions in multiview mode."""

    wl = [500., 600., 700., 800.]

    # Angles vary with wavelength.
    sza_values = np.array([0., 30., 45., 60.])
    vza_values = np.array([0., 10., 20., 30.])

    shape = (4, 2, 2)

    sza = xr.DataArray(
        np.broadcast_to(sza_values[:, None, None], shape).copy(),
        dims=("wl", "y", "x"),
        coords={"wl": wl},
    )
    
    vza = xr.DataArray(
        np.broadcast_to(vza_values[:, None, None], shape).copy(),
        dims=("wl", "y", "x"),
        coords={"wl": wl},
    )

    azi = xr.DataArray(
        np.zeros(shape),
        dims=("wl", "y", "x"),
        coords={"wl": wl},
    )

    result = kernel.mu_N(
        sza=sza,
        vza=vza,
        azi=azi,
        monoview=False,
    )

    # Check output dimensions.
    assert result.dims == ("wl", "y", "x")

    # With azi = 0°, cos(alpha) = cos(sza - vza).
    # Therefore, mu_N = cos((sza + vza) / 2).
    expected_values = np.cos(
        (np.radians(sza_values) + np.radians(vza_values)) / 2
    )

    expected = np.broadcast_to(
        expected_values[:, None, None],
        shape,
    )

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

# endregion


# ------------------------------------------------------------------
# rugosity_est_chunk
# ------------------------------------------------------------------
# region rugosity_est_chunk
class FakeRasterization:
    """Replace the interpolation of the atmospheric LUT"""

    def __init__(self, diffuse_value=0.0):
        self.diffuse_value = diffuse_value
        self.calls = []

    def interp_Rlut_rayleigh(
        self,
        szas,
        sza,
        vzas,
        vza,
        azis,
        azi,
        nwl,
        ny,
        nx,
        lut_values,
    ):
        self.calls.append({
            "sza": sza.copy(),
            "vza": vza.copy(),
            "azi": azi.copy(),
            "nwl": nwl,
            "ny": ny,
            "nx": nx,
        })

        return np.full(
            (nwl, ny, nx),
            self.diffuse_value,
            dtype=np.float64,
        )

def make_test_raster():
    """Create an artificial raster with 4 bands"""

    wl = np.array([700.0, 800.0, 900.0, 1000.0])
    y = np.array([10.0, 20.0])
    x = np.array([100.0, 110.0, 120.0])

    # TOA reflectance : dimensions (wl, y, x).
    bands = np.array([
        np.full((2, 3), 0.10),
        np.full((2, 3), 0.12),
        np.full((2, 3), 0.14),
        np.full((2, 3), 0.16),
    ])

    # Identical angles for each band
    sza = np.full((4, 2, 3), 30.0)
    vza = np.full((4, 2, 3), 10.0)
    raa = np.full((4, 2, 3), 180.0)

    return xr.Dataset(
        data_vars={
            "bands": (("wl", "y", "x"), bands),
            "sza": (("wl", "y", "x"), sza),
            "vza": (("wl", "y", "x"), vza),
            "raa": (("wl", "y", "x"), raa),
        },
        coords={
            "wl": wl,
            "y": y,
            "x": x,
        },
    )

@pytest.fixture
def kernel_and_raster():
    """Create a minimal test Kernel and raster"""

    kernel = Kernel.__new__(Kernel)

    raster = make_test_raster()

    kernel.monoview = False
    kernel.pressure_ref = 101320.0

    kernel.wl_true = raster.wl

    kernel.szas = np.array([30.0])
    kernel.vzas = np.array([10.0])
    kernel.azis = np.array([0.0])

    # Artificial LUT diffuse.
    kernel.Rdiff_lut = xr.DataArray(
        np.zeros((4, 2)),
        dims=("wl", "aot_ref"),
        coords={
            "wl": raster.wl,
            "aot_ref": [0.0, 0.8],
        },
    )

    # Pressure is equal to reference pressure
    pressure = xr.DataArray(
        np.full((2, 3), 101320.0),
        dims=("y", "x"),
        coords={
            "y": raster.y,
            "x": raster.x,
        },
    )

    kernel._cache = {"sp": pressure}

    # Gaseous transmittances are equal to 1.
    transmittance = xr.DataArray(
        np.ones((4, 2, 3)),
        dims=("wl", "y", "x"),
        coords={
            "wl": raster.wl,
            "y": raster.y,
            "x": raster.x,
        },
    )

    kernel.Tg_raster = transmittance
    kernel.Tg_diff_raster = transmittance

    kernel._R_ = FakeRasterization(diffuse_value=0.0)

    return kernel, raster

def test_rugosity_est_chunk_output_structure(kernel_and_raster):
    kernel, raster = kernel_and_raster

    result = kernel.rugosity_est_chunk(
        raster,
        winds=np.array([1.0, 5.0, 10.0]),
        idx_nir=[0, 1, 2, 3],
    )

    assert isinstance(result, xr.Dataset)

    assert set(result.data_vars) == {
        "wind",
        "cost",
        "brdf_g",
    }

    for variable in ("wind", "cost", "brdf_g"):
        assert result[variable].dims == ("y", "x")
        assert result[variable].shape == (2, 3)

    np.testing.assert_array_equal(
        result.x.values,
        raster.x.values,
    )

    np.testing.assert_array_equal(
        result.y.values,
        raster.y.values,
    )

    assert result.attrs["units"] == "m.s-1"

def test_rugosity_est_chunk_selects_first_wind_when_costs_are_equal(
    kernel_and_raster,
):
    # This test will need to be readapted if p_slope is reactivated
    
    kernel, raster = kernel_and_raster

    winds = np.array([2.0, 5.0, 8.0])

    result = kernel.rugosity_est_chunk(
        raster,
        winds=winds,
        idx_nir=[0, 1, 2, 3],
    )

    np.testing.assert_allclose(
        result["wind"].values,
        np.full((2, 3), winds[0]),
    )

    np.testing.assert_allclose(
        result["cost"].values,
        0.0,
        atol=1e-12,
    )

def test_rugosity_est_chunk_subtracts_diffuse_reflectance(
    kernel_and_raster,
):
    kernel, raster = kernel_and_raster

    kernel._R_ = FakeRasterization(diffuse_value=0.01)

    result = kernel.rugosity_est_chunk(
        raster,
        winds=np.array([2.0, 5.0]),
        idx_nir=[0, 1, 2, 3],
    )

    # The last band is 0.16.
    #The diffuse contribution is 0.01.
    # The relative pressure and transmittances are equal to 1.
    expected_brdf_g = 0.16 - 0.01

    np.testing.assert_allclose(
        result["brdf_g"].values,
        expected_brdf_g,
    )

def test_rugosity_est_chunk_azimuth_conversion(kernel_and_raster):
    kernel, raster = kernel_and_raster
    
    kernel.rugosity_est_chunk(
        raster,
        winds=np.array([2.0]),
        idx_nir=[0,1,2,3],
    )
    
    assert len(kernel._R_.calls) == 1
    call = kernel._R_.calls[0]
    
    np.testing.assert_allclose(
        call["azi"],
        0,0
    )
    
    assert call["nwl"] == 4
    assert call["ny"] == 2
    assert call["nx"] == 3

@pytest.mark.parametrize(
    "empty_dimension",
    ["x", "y"],
)
def test_rugosity_est_chunk_empty_block(kernel_and_raster, empty_dimension):
    kernel, raster = kernel_and_raster
    if empty_dimension == "x":
        empty_raster = raster.isel(x=slice(0,0))
    else:
        empty_raster = raster.isel(y=slice(0,0))
    
    result = kernel.rugosity_est_chunk(
        empty_raster,
        winds=np.array([2.0,5.0]),
        idx_nir=[0,1,2,3]
    )
    
    assert result is None

def test_rugosity_est_chunk_nan_angles(kernel_and_raster):
    kernel, raster = kernel_and_raster
    
    #Invalid angles at pixel (0,0)
    raster["sza"].values[:,0,0] = np.nan
    result = kernel.rugosity_est_chunk(
        raster,
        winds=np.array([2.0,5.0]),
        idx_nir=[0,1,2,3]
    )
    
    assert np.isnan(result["wind"].values[0,0])
    assert np.isnan(result["cost"].values[0,0])
    
    #Other pixels have a valid result
    assert np.isfinite(result["wind"].values[1,1])
    assert np.isfinite(result["cost"].values[1,1])
    
# endregion


# ------------------------------------------------------------------
# aerosol_swir_chunk
# ------------------------------------------------------------------
# region aerosol_swir_chunk
@pytest.fixture
def kernel_swir():
    """Construit un Kernel minimal pour tester aerosol_swir_chunk."""

    kernel = Kernel.__new__(Kernel)

    kernel.pressure_ref = 101320.0
    kernel.aot_ref_lim = 0.8
    kernel.aot_ref_max = 0.5
    kernel._cache = {}

    x = np.array([100.0, 120.0])
    y = np.array([200.0, 220.0])
    wl = np.array([560.0, 865.0, 1610.0])

    bands = np.array([
        [[0.50, 0.50], [0.50, 0.50]],
        [[0.10, 0.10], [0.10, 0.10]],
        [[0.08, 0.08], [0.08, 0.08]],
    ])

    angles = np.full((2, 2), 30.0)

    raster = xr.Dataset(
        data_vars={
            "bands": (("wl", "y", "x"), bands),
            "sza": (("y", "x"), angles),
            "vza": (("y", "x"), angles),
            "raa": (("y", "x"), angles),
        },
        coords={"wl": wl, "x": x, "y": y},
    )

    # Données de pression CAMS
    kernel.cams = Mock()
    kernel.cams.raster.sp = xr.DataArray(
        np.full((2, 2), 101320.0),
        dims=("y", "x"),
        coords={"x": x, "y": y},
    )

    # Objet chargé de simuler l'interpolation atmosphérique
    kernel._R_ = Mock()

    # Coordonnées angulaires de la LUT
    kernel.szas = np.array([30.0])
    kernel.vzas = np.array([30.0])
    kernel.azis = np.array([150.0])
    
    # LUT artificielle : la valeur de chaque bande identifie l'AOT.
    aot_refs = np.array([0.0, 0.1, 0.2])
    lut_values = np.zeros((3, 3, 1, 1, 1))

    for i, aot in enumerate(aot_refs):
        lut_values[:, i, 0, 0, 0] = aot

    kernel.Rdiff_lut = xr.DataArray(
        lut_values,
        dims=("wl", "aot_ref", "sza", "vza", "azi"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
            "sza": [30.0],
            "vza": [30.0],
            "azi": [150.0],
        },
    )

    # Transmittance gazeuse diffuse égale à 1
    kernel.Tg_diff_raster = xr.DataArray(
        np.ones((3, 2, 2)),
        dims=("wl", "y", "x"),
        coords={"wl": wl, "x": x, "y": y},
    )

    kernel.test_raster = raster

    return kernel

def test_aerosol_swir_chunk_empty_raster(kernel_swir):
    raster = kernel_swir.test_raster.isel(x=slice(0, 0))

    result = kernel_swir.aerosol_swir_chunk(raster)

    assert result is None
    kernel_swir._R_.interp_Rlut_rayleigh.assert_not_called()

def configure_diffuse_interpolation(kernel, diffuse_by_aot):
    """Configure le mock pour renvoyer une contribution diffuse connue."""

    def fake_interp(*args):
        lut_values = np.asarray(args[-1])
        aot = float(lut_values.flat[0])

        diffuse = diffuse_by_aot[aot]

        return np.full((2, 2, 2), diffuse, dtype=float)

    kernel._R_.interp_Rlut_rayleigh.side_effect = fake_interp

def test_aerosol_swir_chunk_selects_largest_valid_aot(kernel_swir):
    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.02,
            0.2: 0.09,
        },
    )

    aot_refs = [0.0, 0.1, 0.2]

    result = kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=aot_refs,
        idx_swir=[-2, -1],
        neg_pix_rel_thresh=0.05,
    )

    assert result == pytest.approx(0.1)
    assert kernel_swir.aot_ref_max == pytest.approx(0.1)


def test_aerosol_swir_chunk_uses_fallback_when_all_counts_are_zero(
    kernel_swir,
):
    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.0,
            0.2: 0.0,
        },
    )

    result = kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=[0.0, 0.1, 0.2],
    )

    assert result == pytest.approx(0.16)
    assert kernel_swir.aot_ref_max == pytest.approx(0.16)


def test_aerosol_swir_chunk_selects_largest_valid_aot(kernel_swir):
    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.02,
            0.2: 0.09,
        },
    )

    aot_refs = [0.0, 0.1, 0.2]

    result = kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=aot_refs,
        idx_swir=[-2, -1],
        neg_pix_rel_thresh=0.05,
    )

    assert result == pytest.approx(0.1)
    assert kernel_swir.aot_ref_max == pytest.approx(0.1)

def test_aerosol_swir_chunk_enforces_lut_limit(kernel_swir):
    kernel_swir.aot_ref_lim = 0.2
    kernel_swir.aot_ref_cams_max = 0.15

    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.02,
            0.2: 0.09,
        },
    )

    result = kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=[0.0, 0.1, 0.2],
        neg_pix_rel_thresh=2.0,
    )

    assert result == pytest.approx(0.15)
    assert kernel_swir.aot_ref_max == pytest.approx(0.15)

def test_aerosol_swir_chunk_keep_xres(kernel_swir):
    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.02,
            0.2: 0.09,
        },
    )

    aot_refs = [0.0, 0.1, 0.2]

    kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=aot_refs,
        keep_xres=True,
    )

    assert hasattr(kernel_swir, "xres")

    assert "Rrs" in kernel_swir.xres
    assert "aot_ref" in kernel_swir.xres.dims

    assert kernel_swir.xres.sizes["aot_ref"] == 3
    assert kernel_swir.xres.sizes["wl"] == 2
    assert kernel_swir.xres.sizes["y"] == 2
    assert kernel_swir.xres.sizes["x"] == 2

    np.testing.assert_allclose(
        kernel_swir.xres.aot_ref.values,
        aot_refs,
    )

def test_aerosol_swir_chunk_keep_xres(kernel_swir):
    configure_diffuse_interpolation(
        kernel_swir,
        {
            0.0: 0.0,
            0.1: 0.02,
            0.2: 0.09,
        },
    )

    aot_refs = [0.0, 0.1, 0.2]

    kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=aot_refs,
        keep_xres=True,
    )

    assert hasattr(kernel_swir, "xres")

    assert "Rrs" in kernel_swir.xres
    assert "aot_ref" in kernel_swir.xres.dims

    assert kernel_swir.xres.sizes["aot_ref"] == 3
    assert kernel_swir.xres.sizes["wl"] == 2
    assert kernel_swir.xres.sizes["y"] == 2
    assert kernel_swir.xres.sizes["x"] == 2

    np.testing.assert_allclose(
        kernel_swir.xres.aot_ref.values,
        aot_refs,
    )

def test_aerosol_swir_chunk_ignores_pixels_negative_at_zero_aot(
    kernel_swir,
):
    diffuse_by_aot = {
        0.0: np.array([
            [[0.00, 0.00], [0.00, 0.00]],
            [[0.10, 0.00], [0.00, 0.00]],
        ]),
        0.1: np.array([
            [[0.00, 0.02], [0.00, 0.00]],
            [[0.10, 0.02], [0.00, 0.00]],
        ]),
        0.2: np.array([
            [[0.09, 0.09], [0.09, 0.09]],
            [[0.09, 0.09], [0.09, 0.09]],
        ]),
    }

    def fake_interp(*args):
        aot = float(np.asarray(args[-1]).flat[0])
        return diffuse_by_aot[aot]

    kernel_swir._R_.interp_Rlut_rayleigh.side_effect = fake_interp

    result = kernel_swir.aerosol_swir_chunk(
        kernel_swir.test_raster,
        aot_refs=[0.0, 0.1, 0.2],
        neg_pix_rel_thresh=0.5,
    )

    assert result == pytest.approx(0.1)


# endregion


# ------------------------------------------------------------------
# aerosol_visible_chunk
# ------------------------------------------------------------------
# region aerosol_visible_chunk

@pytest.fixture
def kernel_visible():
    """Construit un Kernel minimal pour tester aerosol_visible_chunk."""

    kernel = Kernel.__new__(Kernel)

    # --------------------------------------------------
    # Paramètres généraux
    # --------------------------------------------------
    kernel.pressure_ref = 101320.0
    kernel.aot_ref_lim = 0.8
    kernel.aot_ref_max = 0.2
    kernel.neg_pix_max = 0
    kernel.monoview = True
    kernel._cache = {}

    # --------------------------------------------------
    # Géométrie et grille
    # --------------------------------------------------
    x = np.array([100.0, 120.0])
    y = np.array([200.0, 220.0])

    # Sept bandes visibles/NIR et deux bandes SWIR.
    wl = np.array([
        443.0, 490.0, 560.0, 665.0,
        705.0, 740.0, 783.0, 865.0, 1610.0
    ])

    # Réflectance TOA constante pour simplifier le test.
    bands = np.full((len(wl), 2, 2), 0.1, dtype=np.float32)

    angles = np.full((2, 2), 30.0)

    raster = xr.Dataset(
        data_vars={
            "bands": (("wl", "y", "x"), bands),
            "sza": (("y", "x"), angles),
            "vza": (("y", "x"), angles),
            "raa": (("y", "x"), angles),
            "water_pixel_number": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.int32),
            ),
        },
        coords={
            "wl": wl,
            "x": x,
            "y": y,
        },
    )

    # --------------------------------------------------
    # Paramètres et LUT atmosphériques
    # --------------------------------------------------
    kernel.szas = np.array([30.0])
    kernel.vzas = np.array([30.0])
    kernel.azis = np.array([150.0])

    aot_refs = np.array([0.0, 0.1, 0.2, 0.4, 0.8])

    # LUT de réflectance diffuse : (wl, aot_ref, sza, vza, azi)
    lut_values = np.zeros(
        (len(wl), len(aot_refs), 1, 1, 1),
        dtype=np.float32,
    )

    for i, aot in enumerate(aot_refs):
        lut_values[:, i, 0, 0, 0] = aot * 0.01

    kernel.Rdiff_lut = xr.DataArray(
        lut_values,
        dims=("wl", "aot_ref", "sza", "vza", "azi"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
            "sza": [30.0],
            "vza": [30.0],
            "azi": [150.0],
        },
    )

    # LUT d'épaisseur optique des aérosols.
    aot_lut_values = np.zeros(
        (len(wl), len(aot_refs)),
        dtype=np.float32,
    )

    for i, aot in enumerate(aot_refs):
        aot_lut_values[:, i] = aot * 0.1

    kernel.aot_lut = xr.DataArray(
        aot_lut_values,
        dims=("wl", "aot_ref"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
        },
    )

    # --------------------------------------------------
    # Transmittances gazeuses
    # --------------------------------------------------
    kernel.Tg_diff_raster = xr.DataArray(
        np.ones((len(wl), 2, 2), dtype=np.float32),
        dims=("wl", "y", "x"),
        coords={"wl": wl, "x": x, "y": y},
    )

    kernel.Tg_raster = xr.DataArray(
        np.ones((len(wl), 2, 2), dtype=np.float32),
        dims=("wl", "y", "x"),
        coords={"wl": wl, "x": x, "y": y},
    )

    # --------------------------------------------------
    # Rayleigh et sunglint
    # --------------------------------------------------
    kernel.rot = xr.DataArray(
        np.full(len(wl), 0.01, dtype=np.float32),
        dims=("wl",),
        coords={"wl": wl},
    )

    kernel.sunglint_eps = xr.DataArray(
        np.ones(len(wl), dtype=np.float32),
        dims=("wl",),
        coords={"wl": wl},
    )

    kernel.wind_img = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.full((2, 2), 2.0, dtype=np.float32),
            )
        },
        coords={"x": x, "y": y},
    )

    # --------------------------------------------------
    # Données CAMS
    # --------------------------------------------------
    kernel.cams = Mock()

    kernel.cams.raster.sp = xr.DataArray(
        np.full((2, 2), 101320.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": x, "y": y},
    )

    kernel.cams.cams_aod = xr.DataArray(
        np.full((3,2,2), 0.2, dtype=np.float32),
        dims=("wl", "y", "x"),
        coords={
            "wl": [440.0, 550.0, 670.0],
            "x": x,
            "y": y,
        },
    )

    # --------------------------------------------------
    # Interpolation atmosphérique simulée
    # --------------------------------------------------
    kernel._R_ = Mock()

    # Le résultat sera remplacé dans les tests individuels
    # pour contrôler les réflectances calculées.
    kernel._R_.interp_Rlut_rayleigh.return_value = np.zeros(
        (len(wl), 2, 2),
        dtype=np.float32,
    )

    # Multiplication spectrale utilisée pour construire le sunglint.
    def fake_multiplicate(spectrum, field, output):
        output[:] = spectrum[:, None, None] * field[None, :, :]
        return output

    kernel._R_._multiplicate.side_effect = fake_multiplicate

    kernel.test_raster = raster

    return kernel

def test_aerosol_visible_chunk_empty_block_returns_None(kernel_visible):
    raster = kernel_visible.test_raster.isel(x=slice(0, 0))
    
    result = kernel_visible.aerosol_visible_chunk(raster)

    assert result is None

def test_aerosol_visible_chunk_nominal(kernel_visible):
    """Checks that aerosol_visible_chunk returns a valid output"""

    kernel = kernel_visible
    raster = kernel.test_raster

    result = kernel.aerosol_visible_chunk(raster)

    assert result is not None
    assert isinstance(result, xr.Dataset)

    assert "aot_ref" in result
    assert "number_of_pixel" in result

    assert result["aot_ref"].shape == (1, 1)
    assert result["number_of_pixel"].shape == (1, 1)

    assert np.isfinite(result["aot_ref"].values).all()
    assert result["number_of_pixel"].values[0, 0] == 4

def test_aerosol_visible_chunk_fallback_cams(kernel_visible, monkeypatch):
    """Use CAMS when no AOT candidate has been accepted"""
    kernel = kernel_visible
    raster = kernel_visible.test_raster
    
    monkeypatch.setattr("grs.grs_kernel._aot_ref_candidates", lambda *args, **kwargs: np.array([]))
    kernel.cams.cams_aod.loc[{"wl": 550.0}] = 0.35
    result = kernel_visible.aerosol_visible_chunk(raster)
    
    assert result is not None
    assert isinstance(result, xr.Dataset)

    #AOD Cams at 550nm is equal to 0.2 on each pixel
    #its spatial mean must be equal to 0.2 as well
    assert np.isclose(
        result["aot_ref"].values[0,0],
        0.35
    )
    
    assert result["number_of_pixel"].values[0,0] == 4
    
def test_aerosol_visible_chunk_selects_larges_valid_aot(kernel_visible, monkeypatch):
    """Selects the largest AOT candidate amongst admissible candidates"""
    kernel = kernel_visible
    raster = kernel_visible.test_raster
    
    expected_candidates = np.array([0.0,0.1,0.3])
    monkeypatch.setattr("grs.grs_kernel._aot_ref_candidates", lambda *args, **kwargs: expected_candidates)
    
    result = kernel_visible.aerosol_visible_chunk(raster)
    
    assert np.isclose(
        result["aot_ref"].values[0,0],
        0.3
    )
    
    assert result["number_of_pixel"].values[0,0] == 4

def test_aerosol_visible_chunk_keeps_xres(kernel_visible, monkeypatch):
    """Stores reflectance arrays for every tested AOT when keep_xres=True."""
    kernel = kernel_visible

    monkeypatch.setattr(
        "grs.grs_kernel._aot_ref_candidates",
        lambda *args, **kwargs: np.array([0.1]),
    )

    kernel.aerosol_visible_chunk(
        kernel.test_raster,
        keep_xres=True,
    )

    assert hasattr(kernel, "xres")
    assert "Rrs" in kernel.xres
    assert kernel.xres.sizes["aot_ref"] == 21
    assert kernel.xres.sizes["wl"] == 9
    assert kernel.xres.sizes["y"] == 2
    assert kernel.xres.sizes["x"] == 2
    
# endregion


# ------------------------------------------------------------------
# final_process_chunk
# ------------------------------------------------------------------
# region final_process_chunk
@pytest.fixture
def kernel_final_process():
    """Build a minimal kernel to test final_process_chunk."""

    kernel = Kernel.__new__(Kernel)

    # --------------------------------------------------
    # Paramètres généraux
    # --------------------------------------------------
    kernel.pressure_ref = 101320.0
    kernel.monoview = True
    kernel._cache = {}

    # Valeurs d'AOT disponibles dans les LUT
    aot_refs = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    kernel.aot_refs = aot_refs

    # Type numérique des sorties
    kernel.prod = Mock()
    kernel.prod._type = np.float32

    # --------------------------------------------------
    # Géométrie et grille
    # --------------------------------------------------
    x = np.array([100.0, 120.0])
    y = np.array([200.0, 220.0])

    wl = np.array([
        443.0, 490.0, 560.0, 665.0,
        705.0, 740.0, 783.0, 865.0, 1610.0
    ])

    # Réflectances TOA constantes
    bands = np.full((len(wl), 2, 2), 0.1, dtype=np.float32)

    angles = np.full((2, 2), 30.0, dtype=np.float32)

    raster = xr.Dataset(
        data_vars={
            "bands": (("wl", "y", "x"), bands),
            "sza": (("y", "x"), angles),
            "vza": (("y", "x"), angles),
            "raa": (("y", "x"), angles),
        },
        coords={
            "wl": wl,
            "x": x,
            "y": y,
        },
    )

    kernel.test_raster = raster

    # --------------------------------------------------
    # Carte d'AOT fournie en entrée
    # --------------------------------------------------
    kernel.test_aot_ref_raster = xr.DataArray(
        np.full((2, 2), 0.2, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="aot_ref",
    )

    # --------------------------------------------------
    # Pression atmosphérique
    # --------------------------------------------------
    kernel.cams = Mock()

    kernel.cams.raster.sp = xr.DataArray(
        np.full((2, 2), 101320.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": x, "y": y},
    )

    # --------------------------------------------------
    # LUT et paramètres atmosphériques
    # --------------------------------------------------
    kernel.szas = np.array([30.0])
    kernel.vzas = np.array([30.0])
    kernel.azis = np.array([150.0])

    # LUT de réflectance diffuse : (wl, aot_ref, sza, vza, azi)
    rdiff_values = np.zeros(
        (len(wl), len(aot_refs), 1, 1, 1),
        dtype=np.float32,
    )

    kernel.Rdiff_lut = xr.DataArray(
        rdiff_values,
        dims=("wl", "aot_ref", "sza", "vza", "azi"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
            "sza": [30.0],
            "vza": [30.0],
            "azi": [150.0],
        },
    )

    # LUT d'AOT spectrale
    aot_values = np.zeros(
        (len(wl), len(aot_refs)),
        dtype=np.float32,
    )

    kernel.aot_lut = xr.DataArray(
        aot_values,
        dims=("wl", "aot_ref"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
        },
    )

    # Transmittance totale descendante
    # Les valeurs seront simulées par _interp_Tlut dans ce fixture.
    ttot_values = np.ones(
        (len(wl), len(aot_refs)),
        dtype=np.float32,
    )

    Ttot_Ed = xr.DataArray(
        ttot_values,
        dims=("wl", "aot_ref"),
        coords={
            "wl": wl,
            "aot_ref": aot_refs,
        },
    )

    kernel.trans_aero_lut = Mock()
    kernel.trans_aero_lut.Ttot_Ed = Ttot_Ed

    # --------------------------------------------------
    # Transmittances gazeuses
    # --------------------------------------------------
    kernel.Tg_diff_raster = xr.DataArray(
        np.ones((len(wl), 2, 2), dtype=np.float32),
        dims=("wl", "y", "x"),
        coords={"wl": wl, "x": x, "y": y},
    )

    kernel.Tg_raster = xr.DataArray(
        np.ones((len(wl), 2, 2), dtype=np.float32),
        dims=("wl", "y", "x"),
        coords={"wl": wl, "x": x, "y": y},
    )

    # --------------------------------------------------
    # Rayleigh et paramètres du sunglint
    # --------------------------------------------------
    kernel.rot = xr.DataArray(
        np.full(len(wl), 0.01, dtype=np.float32),
        dims=("wl",),
        coords={"wl": wl},
    )

    kernel.sunglint_eps = xr.DataArray(
        np.ones(len(wl), dtype=np.float32),
        dims=("wl",),
        coords={"wl": wl},
    )

    kernel.wind_img = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.full((2, 2), 2.0, dtype=np.float32),
            )
        },
        coords={"x": x, "y": y},
    )

    # --------------------------------------------------
    # Interpolation atmosphérique simulée
    # --------------------------------------------------
    kernel._R_ = Mock()

    def fake_multiplicate(spectrum, field, output):
        output[:] = spectrum[:, None, None] * field[None, :, :]
        return output

    def fake_interp_Rlut(
        szas, sza, vzas, vza, azis, azi,
        aot_refs, aot_ref_raster, Nwl, Ny, Nx, lut_values,
    ):
        return np.zeros((Nwl, Ny, Nx), dtype=np.float32)

    def fake_interp_aotlut(
        aot_ref_values, aot_ref_raster, Nwl, Ny, Nx, lut_values,
    ):
        return np.zeros((Nwl, Ny, Nx), dtype=np.float32)

    def fake_interp_Tlut(
        angles, angle_values, aot_ref_values,
        aot_ref_raster, Nwl, Ny, Nx, lut_values,
    ):
        return np.ones((Nwl, Ny, Nx), dtype=np.float32)

    kernel._R_._multiplicate.side_effect = fake_multiplicate
    kernel._R_.interp_Rlut.side_effect = fake_interp_Rlut
    kernel._R_._interp_aotlut.side_effect = fake_interp_aotlut
    kernel._R_._interp_Tlut.side_effect = fake_interp_Tlut

    # --------------------------------------------------
    # Calcul de la pente de probabilité du sunglint
    # --------------------------------------------------
    def fake_p_slope(
        sza, vza, raa, sigma2=None, monoview=True,
    ):
        return xr.DataArray(
            np.ones(sza.shape, dtype=np.float32),
            dims=sza.dims,
            coords=sza.coords,
        )

    kernel.p_slope = Mock(side_effect=fake_p_slope)

    return kernel


def test_final_process_chunk_empty_block(kernel_final_process):
    """Returns a dataset with empty arrays when the block is empty"""
    raster = kernel_final_process.test_raster.isel(x=slice(0, 0))
    result = kernel_final_process.final_process_chunk(raster, kernel_final_process.test_aot_ref_raster)
    assert isinstance(result, xr.Dataset)
    
    assert "Rrs" in result
    assert "BRDFg" in result
    assert "aot550" in result
    
    assert result["Rrs"].shape == (9,2,0)
    assert result["BRDFg"].shape == (2,0)
    assert result["aot550"].shape == (2,0)

def test_final_process_chunk_no_valid_pixel(kernel_final_process):
    """Returns empty sets when no valid pixels are available"""
    kernel = kernel_final_process
    raster = kernel_final_process.test_raster.copy(deep=True)
    raster["bands"][:] = np.nan
    
    result = kernel.final_process_chunk(raster, kernel.test_aot_ref_raster)
    
    assert isinstance(result, xr.Dataset)
        
    assert set(result.data_vars) == {"Rrs", "BRDFg", "aot550"}
    #no reflectance or sunglint has been calculated
    assert np.isnan(result["Rrs"].values).all()
    assert np.isnan(result["BRDFg"].values).all()
    #aot is interpolated independantly
    assert np.isfinite(result["aot550"].values).all()

def test_final_process_chunk_partially_valid_pixels(kernel_final_process):
    """Enables compact mode when more then half of the pixels are invalid."""
    kernel = kernel_final_process
    raster = kernel.test_raster.copy(deep=True)
    
    raster["bands"][:] = np.nan
    
    #we keep only one valid pixel (y=0, x=0)
    raster["bands"].values[:,0,0] = 0.1
    
    result = kernel.final_process_chunk(raster, kernel.test_aot_ref_raster)
    
    assert isinstance(result, xr.Dataset)
    
    #the invalid pixels stay Nan for Rrs
    assert np.isnan(result["Rrs"].values[:,0,1:]).all()
    assert np.isnan(result["Rrs"].values[:,1,:]).all()
    
    #the valid pixel has been treated
    assert np.isfinite(result["Rrs"].values[:,0,0]).all()
    
    #the sunglint is calculated only for the valid pixel
    assert np.isfinite(result["BRDFg"].values[0,0])
    assert np.isnan(result["BRDFg"].values[0,1:]).all()
    assert np.isnan(result["BRDFg"].values[1, :]).all()
    
    #the aot card is finite for all pixels
    assert np.isfinite(result["aot550"].values).all()

def test_final_process_chunk_not_compacted(kernel_final_process):
    """Disables compact mode when half of the pixels are valid."""
    kernel = kernel_final_process
    raster = kernel.test_raster.copy(deep=True)
    
    raster["bands"][:] = np.nan
    
    #We keep two valid pixels out of four
    raster["bands"].values[:,0,0] = 0.1    
    raster["bands"].values[:,1,0] = 0.1
        
    result = kernel.final_process_chunk(raster, kernel.test_aot_ref_raster)
    
    assert isinstance(result, xr.Dataset)
    
    #the two valid pixels were treated
    assert np.isfinite(result["Rrs"].values[:,0,0]).all()
    assert np.isfinite(result["Rrs"].values[:,1,0]).all()
    
    #the invalid pixel stay nan
    assert np.isnan(result["Rrs"].values[:,0,1]).all()
    assert np.isnan(result["Rrs"].values[:,1,1]).all()
    
    #the sunglint is calculated only for the valid pixel
    assert np.isfinite(result["BRDFg"].values[0,0]).all()
    assert np.isfinite(result["BRDFg"].values[1,0]).all()
    assert np.isnan(result["BRDFg"].values[0,1]).all()
    assert np.isnan(result["BRDFg"].values[1,1]).all()
    
    #the aot card is finite for all pixels
    assert np.isfinite(result["aot550"].values).all()
    
def test_final_process_chunk_clips_aot_to_lut_range(kernel_final_process):
    """Clips the interpolated AOT to the LUT range"""
    kernel = kernel_final_process
    raster = kernel.test_raster
    # Values voluntarily outside the authorized range [0.0,0.8]    
    aot_ref_raster = xr.DataArray(
        np.array([
            [-0.5,0.2],
            [0.4,1.5]
        ]),
        dims=("y","x"),
        coords={
            "x": raster.x.values,
            "y": raster.y.values
        },
        name="aot_ref"
    )
    
    result = kernel.final_process_chunk(raster, aot_ref_raster)
    expected_aot = np.array([
        [0.0, 0.2],
        [0.4, 0.8],
    ])

    np.testing.assert_allclose(
        result["aot550"].values,
        expected_aot,
        atol=1e-6,
    )

def test_final_process_chunk_negative_reflectance_skips_transmittance(kernel_final_process, monkeypatch):
    """Does not divide reflectances by the total transmittance."""
    kernel = kernel_final_process
    raster = kernel.test_raster.copy(deep=True)
    #Force negative reflectance
    raster["bands"][:] = -0.1
    
    #The total transmittance is 0.5 before condition
    monkeypatch.setattr(
        kernel._R_,
        "_interp_Tlut",
        lambda *args, **kwargs: np.full(
            (args[4], args[5], args[6]),
            0.5,
            dtype=np.float32,
        ),
    )

    result = kernel.final_process_chunk(raster, kernel.test_aot_ref_raster)
    
    #The reflectance stays negative 
    assert np.all(result["Rrs"].values < 0)

def test_final_process_chunk_positive_reflectance_applies_transmittance( kernel_final_process, monkeypatch ): 
    """Divides positive reflectances by the total transmittance.""" 
    kernel = kernel_final_process 
    raster = kernel.test_raster.copy(deep=True)

    # Réflectance TOA positive et constante.
    raster["bands"][:] = 0.1

    # Simulates a null contribution from _multiplicate
    def fake_multiplicate(spectrum, field, output):
        output.fill(0)
        return output
    
    monkeypatch.setattr(
        kernel._R_,
        "_multiplicate",
        fake_multiplicate,
    )
    # On impose une transmittance totale égale à 0.5.
    monkeypatch.setattr(
        kernel._R_,
        "_interp_Tlut",
        lambda *args, **kwargs: np.full(
            (args[4], args[5], args[6]),
            0.5,
            dtype=np.float32,
        ),
    )

    result = kernel.final_process_chunk(
        raster,
        kernel.test_aot_ref_raster,
    )
    # Les réflectances finales doivent être positives.
    assert np.all(result["Rrs"].values > 0)

# endregion


# ------------------------------------------------------------------
# conv_mapping
# ------------------------------------------------------------------
# region conv_mapping
def test_conv_mapping_central_value_is_Nan():
    kernel = Kernel.__new__(Kernel)
    test_array = np.array([2.0,np.nan,4.0])
    result = kernel.conv_mapping(test_array)
    
    assert result == 3.0

def test_conv_mapping_all_Nan_but_central():
    kernel = Kernel.__new__(Kernel)
    test_array = np.array([np.nan, 3.0, np.nan])
    result = kernel.conv_mapping(test_array)
    
    assert result == 3.0

def test_conv_mapping_finite_values():
    kernel = Kernel.__new__(Kernel)
    test_array = np.array([2.0, 3.0, 4.0])
    result = kernel.conv_mapping(test_array)
    
    assert result == 3.0
# endregion


# ------------------------------------------------------------------
# smoothing
# ------------------------------------------------------------------
# region smoothing
def test_smoothing_uses_nanmean_box(kernel, monkeypatch):
    raster = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.array([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]),
            )
        },
        coords={
            "y": [10, 20],
            "x": [100, 200],
        },
    )

    filtered = np.array([
        [1.5, 2.5],
        [3.5, 4.5],
    ])

    expected = np.array([
        [2.0, 2.5],
        [3.5, 4.0],
    ])

    calls = {}

    def fake_filter2d(image, weights, windows):
        calls["filter2d"] = True
        np.testing.assert_array_equal(image, raster["wind"].values)
        np.testing.assert_array_equal(weights, np.ones((2, 2)))
        np.testing.assert_array_equal(windows, [3, 3])
        return filtered

    def fake_nanmean_box(image, size):
        calls["nanmean_box"] = True
        np.testing.assert_array_equal(image, filtered)
        assert size == (3, 3)
        return expected

    monkeypatch.setattr(kernel, "filter2d", fake_filter2d)
    monkeypatch.setattr(kernel, "_nanmean_box", fake_nanmean_box)

    result = kernel.smoothing(
        raster,
        windows=np.array([3, 3]),
        mask=np.ones((3, 3)),
    )

    assert calls["filter2d"]
    assert calls["nanmean_box"]

    np.testing.assert_array_equal(result["wind"].values, expected)
    np.testing.assert_array_equal(result["y"].values, [10, 20])
    np.testing.assert_array_equal(result["x"].values, [100, 200])

def test_smoothing_uses_generic_filter(monkeypatch):
    kernel = Kernel.__new__(Kernel)
    raster = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.array([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]),
            )
        },
        coords={
            "y": [10, 20],
            "x": [100, 200],
        },
    )

    filtered = np.array([
        [1.5, 2.5],
        [3.5, 4.5],
    ])

    expected = np.array([
        [2.0, 2.5],
        [3.5, 4.0],
    ])

    calls = {}

    monkeypatch.setattr(
        kernel,
        "filter2d",
        lambda image, weights, windows: filtered,
    )

    def fail_nanmean_box(*args, **kwargs):
        pytest.fail("_nanmean_box ne devrait pas être appelé")

    monkeypatch.setattr(kernel, "_nanmean_box", fail_nanmean_box)

    def fake_generic_filter(image, function, footprint, mode):
        calls["function"] = function
        calls["footprint"] = footprint
        calls["mode"] = mode
        np.testing.assert_array_equal(image, filtered)
        return expected

    monkeypatch.setattr(
        "scipy.ndimage.generic_filter",
        fake_generic_filter,
    )

    mask = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    result = kernel.smoothing(
        raster,
        windows=np.array([3, 3]),
        mask=mask,
    )

    assert calls["function"] == Kernel.conv_mapping
    assert calls["mode"] == "nearest"
    np.testing.assert_array_equal(calls["footprint"], mask)
    np.testing.assert_array_equal(result["wind"].values, expected)


# endregion


# ------------------------------------------------------------------
# _nanmean_box
# ------------------------------------------------------------------
# region _nanmean_box
@pytest.mark.parametrize(
    "image, size, expected",
    [
        # local mean withou NaN
        (
            np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]]),
            (3, 3),
            np.array([
                [7/3, 3., 11/3],
                [13/3, 5., 17/3],
                [19/3, 7., 23/3]
            ])
        ),
        # Horizontal window
        (
            np.array([[1., 2., 3., 4.]]),
            (1, 3),
            np.array([[4/3, 2., 3., 11/3]])
        ),
        # Size 1 window: values remain unchanged 
        (
            np.array([[1., np.nan],
                      [3., 4.]]),
            (1, 1),
            np.array([[1., np.nan],
                      [3., 4.]])
        ),
        # Array full of NaN
        (
            np.full((3, 3), np.nan),
            (3, 3),
            np.full((3, 3), np.nan)
        ),
    ],
)
def test_nanmean_box(image, size, expected):
    result = Kernel._nanmean_box(image, size)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_nanmean_box_single_valid_pixel():
    """Un seul pixel valide doit donner sa valeur dans tout le voisinage."""
    image = np.full((3, 3), np.nan)
    image[1, 1] = 42.

    result = Kernel._nanmean_box(image, (3, 3))

    np.testing.assert_allclose(result, 42.)


def test_nanmean_box_nearest_edge_extension():
    """Vérifie le traitement des bords avec mode='nearest'."""
    image = np.array([[1., 2.],
                      [3., 4.]])

    result = Kernel._nanmean_box(image, (3, 3))

    expected = np.array([[2., 7/3],
                         [8/3, 3.]])

    np.testing.assert_allclose(result, expected)


def test_nanmean_box_output_properties():
    """La sortie conserve la forme et utilise float64."""
    image = np.array([[1, 2], [3, 4]], dtype=np.int32)

    result = Kernel._nanmean_box(image, (3, 3))

    assert result.shape == image.shape
    assert result.dtype == np.float64
# endregion


# ------------------------------------------------------------------
# set_gas_transmittance
# ------------------------------------------------------------------
# region set_gas_transmittance

@pytest.fixture
def kernel_for_gas_transmittance():
    """Create a Kernel instance without running its constructor."""
    kernel = Kernel.__new__(Kernel)
    kernel.prod = object()
    kernel.cams = object()
    return kernel


@pytest.fixture
def gas_transmittance_data():
    """Create synthetic gaseous transmittance rasters."""
    coords = {
        "wl": [443, 560, 665],
        "y": [10, 20],
        "x": [100, 200],
    }

    diffuse = xr.DataArray(
        np.full((3, 2, 2), 0.8, dtype=np.float32),
        dims=("wl", "y", "x"),
        coords=coords,
        name="transmittance",
    )

    total = xr.DataArray(
        np.full((3, 2, 2), 0.9, dtype=np.float32),
        dims=("wl", "y", "x"),
        coords=coords,
        name="transmittance",
    )

    return diffuse, total


def test_set_gas_transmittance_initializes_both_rasters(
    monkeypatch,
    kernel_for_gas_transmittance,
    gas_transmittance_data,
):
    """Test that diffuse and total gaseous transmittance rasters are initialized."""
    diffuse, total = gas_transmittance_data
    calls = []

    class FakeGaseousTransmittance:
        def __init__(self, prod, cams):
            assert prod is kernel_for_gas_transmittance.prod
            assert cams is kernel_for_gas_transmittance.cams

            self.coef_abs_scat = {
                "h2o": 1,
                "ch4": 1,
            }

        def get_gaseous_transmittance(self, gases, background):
            calls.append({
                "gases": list(gases),
                "background": background,
                "coefficients": dict(self.coef_abs_scat),
            })

            if self.coef_abs_scat["h2o"] == 0.5:
                return diffuse.transpose("x", "wl", "y")

            return total.transpose("y", "x", "wl")

    loaded = []

    def fake_load_small(data):
        loaded.append(data)
        return data

    monkeypatch.setattr(
        kernel_module,
        "GaseousTransmittance",
        FakeGaseousTransmittance,
    )
    monkeypatch.setattr(kernel_module, "_load_small", fake_load_small)

    kernel = kernel_for_gas_transmittance
    kernel.set_gas_transmittance()

    assert hasattr(kernel, "Tg_diff_raster")
    assert hasattr(kernel, "Tg_raster")

    xr.testing.assert_equal(kernel.Tg_diff_raster, diffuse)
    xr.testing.assert_equal(kernel.Tg_raster, total)

    assert len(loaded) == 2
    assert len(calls) == 2


def test_set_gas_transmittance_uses_expected_gases_and_coefficients(
    monkeypatch,
    kernel_for_gas_transmittance,
    gas_transmittance_data,
):
    """Test the gases and absorption-scattering coefficients used in each calculation."""
    diffuse, total = gas_transmittance_data
    calls = []

    class FakeGaseousTransmittance:
        def __init__(self, prod, cams):
            self.coef_abs_scat = {
                "h2o": 1,
                "ch4": 1,
            }

        def get_gaseous_transmittance(self, gases, background):
            calls.append({
                "gases": list(gases),
                "background": background,
                "coefficients": dict(self.coef_abs_scat),
            })

            if len(calls) == 1:
                return diffuse

            return total

    monkeypatch.setattr(
        kernel_module,
        "GaseousTransmittance",
        FakeGaseousTransmittance,
    )
    monkeypatch.setattr(kernel_module, "_load_small", lambda data: data)

    kernel_for_gas_transmittance.set_gas_transmittance()

    assert calls[0]["gases"] == ["h2o", "ch4"]
    assert calls[0]["background"] is False
    assert calls[0]["coefficients"]["h2o"] == 0.5
    assert calls[0]["coefficients"]["ch4"] == 0.65

    assert calls[1]["gases"] == ["h2o", "ch4"]
    assert calls[1]["background"] is False
    assert calls[1]["coefficients"]["h2o"] == 1
    assert calls[1]["coefficients"]["ch4"] == 1


def test_set_gas_transmittance_transposes_dimensions(
    monkeypatch,
    kernel_for_gas_transmittance,
    gas_transmittance_data,
):
    """Test that both output rasters use the (wl, y, x) dimension order."""
    diffuse, total = gas_transmittance_data

    class FakeGaseousTransmittance:
        def __init__(self, prod, cams):
            self.coef_abs_scat = {
                "h2o": 1,
                "ch4": 1,
            }

        def get_gaseous_transmittance(self, gases, background):
            if self.coef_abs_scat["h2o"] == 0.5:
                return diffuse.transpose("x", "y", "wl")

            return total.transpose("y", "x", "wl")

    monkeypatch.setattr(
        kernel_module,
        "GaseousTransmittance",
        FakeGaseousTransmittance,
    )
    monkeypatch.setattr(kernel_module, "_load_small", lambda data: data)

    kernel_for_gas_transmittance.set_gas_transmittance()

    assert kernel_for_gas_transmittance.Tg_diff_raster.dims == ("wl", "y", "x")
    assert kernel_for_gas_transmittance.Tg_raster.dims == ("wl", "y", "x")


def test_set_gas_transmittance_calls_load_small_twice(
    monkeypatch,
    kernel_for_gas_transmittance,
    gas_transmittance_data,
):
    """Test that both transmittance rasters are passed through the loading helper."""
    diffuse, total = gas_transmittance_data
    loaded = []

    class FakeGaseousTransmittance:
        def __init__(self, prod, cams):
            self.coef_abs_scat = {
                "h2o": 1,
                "ch4": 1,
            }

        def get_gaseous_transmittance(self, gases, background):
            if self.coef_abs_scat["h2o"] == 0.5:
                return diffuse

            return total

    def fake_load_small(data):
        loaded.append(data.copy())
        return data

    monkeypatch.setattr(
        kernel_module,
        "GaseousTransmittance",
        FakeGaseousTransmittance,
    )
    monkeypatch.setattr(kernel_module, "_load_small", fake_load_small)

    kernel_for_gas_transmittance.set_gas_transmittance()

    assert len(loaded) == 2
    assert loaded[0].dims == ("wl", "y", "x")
    assert loaded[1].dims == ("wl", "y", "x")


def test_set_gas_transmittance_preserves_transmittance_values(
    monkeypatch,
    kernel_for_gas_transmittance,
    gas_transmittance_data,
):
    """Test that the loading helper does not alter transmittance values."""
    diffuse, total = gas_transmittance_data

    class FakeGaseousTransmittance:
        def __init__(self, prod, cams):
            self.coef_abs_scat = {
                "h2o": 1,
                "ch4": 1,
            }

        def get_gaseous_transmittance(self, gases, background):
            if self.coef_abs_scat["h2o"] == 0.5:
                return diffuse

            return total

    monkeypatch.setattr(
        kernel_module,
        "GaseousTransmittance",
        FakeGaseousTransmittance,
    )
    monkeypatch.setattr(kernel_module, "_load_small", lambda data: data)

    kernel_for_gas_transmittance.set_gas_transmittance()

    np.testing.assert_allclose(
        kernel_for_gas_transmittance.Tg_diff_raster.values,
        0.8,
    )
    np.testing.assert_allclose(
        kernel_for_gas_transmittance.Tg_raster.values,
        0.9,
    )

# endregion

# ------------------------------------------------------------------
# load_process
# ------------------------------------------------------------------
# region load_process


@pytest.fixture
def kernel_for_final_process():
    """Kernel minimal permettant de tester final_process sans initialiser les LUT."""
    kernel = Kernel.__new__(Kernel)

    kernel.prod = MagicMock()
    kernel.prod._type = np.float32
    kernel.prod.sensor = "S2A"

    kernel.wind_img = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            )
        },
        coords={
            "x": np.arange(4),
            "y": np.arange(4),
        },
    )

    # Le snapshot est remplacé dans les tests : aucune LUT n'est nécessaire.
    kernel._cache = {}
    return kernel


@pytest.fixture
def raster():
    """Raster minimal compatible avec la préparation de final_process."""
    wl = np.array([443, 560, 665, 865, 1610, 2200])
    y = np.arange(8)
    x = np.arange(8)

    shape = (len(wl), len(y), len(x))

    return xr.Dataset(
        {
            "bands": (
                ("wl", "y", "x"),
                np.ones(shape, dtype=np.float32),
            ),
            "sza": (
                ("y", "x"),
                np.full((8, 8), 30, dtype=np.float32),
            ),
            "raa": (
                ("y", "x"),
                np.full((8, 8), 10, dtype=np.float32),
            ),
            "vza": (
                ("y", "x"),
                np.full((8, 8), 5, dtype=np.float32),
            ),
            # Variable qui ne doit pas être transmise à map_blocks.
            "unused": (
                ("y", "x"),
                np.zeros((8, 8), dtype=np.float32),
            ),
        },
        coords={
            "wl": wl,
            "y": y,
            "x": x,
        },
    )


@pytest.fixture
def aot_ref_raster():
    return xr.DataArray(
        np.full((4, 4), 0.2, dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.arange(4) * 2 + 0.5,
            "x": np.arange(4) * 2 + 0.5,
        },
        name="aot_ref",
    )


def test_final_process_creates_expected_template(
    kernel_for_final_process,
    raster,
    aot_ref_raster,
    monkeypatch,
):
    """Vérifie les dimensions, variables et type du template passé à map_blocks."""
    kernel = kernel_for_final_process

    # Le snapshot est un objet minimal dont final_process_chunk est une méthode factice.
    snapshot = MagicMock()
    snapshot.final_process_chunk = MagicMock(return_value=None)
    monkeypatch.setattr(kernel, "_snapshot", lambda: snapshot)

    captured = {}

    def fake_map_blocks(func, obj, kwargs, template):
        captured["func"] = func
        captured["obj"] = obj
        captured["kwargs"] = kwargs
        captured["template"] = template
        return template

    monkeypatch.setattr(xr, "map_blocks", fake_map_blocks)

    result = kernel.final_process(
        raster,
        aot_ref_raster,
        chunk=4,
        lazy=True,
    )

    assert set(result.data_vars) == {"Rrs", "BRDFg", "aot550"}

    assert result["Rrs"].dims == ("wl", "y", "x")
    assert result["BRDFg"].dims == ("y", "x")
    assert result["aot550"].dims == ("y", "x")

    assert result.sizes["wl"] == raster.sizes["wl"]
    assert result.sizes["y"] == raster.sizes["y"]
    assert result.sizes["x"] == raster.sizes["x"]

    assert result["Rrs"].dtype == np.float32
    assert result["BRDFg"].dtype == np.float32
    assert result["aot550"].dtype == np.float32

    # Vérifie que seules les variables utiles sont transmises.
    assert set(captured["obj"].data_vars) == {
        "bands",
        "sza",
        "raa",
        "vza",
    }

    # Vérifie que le raster a été découpé en blocs de 4 x 4.
    assert captured["obj"]["bands"].chunksizes["y"] == (4, 4)
    assert captured["obj"]["bands"].chunksizes["x"] == (4, 4)
    assert captured["obj"]["bands"].chunksizes["wl"] == (raster.sizes["wl"],)

    assert captured["func"] == snapshot.final_process_chunk
    assert captured["kwargs"]["iwl_swir"] == [-2, -1]

    assert kernel.xres is result


def test_final_process_passes_custom_parameters(
    kernel_for_final_process,
    raster,
    aot_ref_raster,
    monkeypatch,
):
    """Vérifie la transmission de chunk et iwl_swir."""
    kernel = kernel_for_final_process

    snapshot = MagicMock()
    snapshot.final_process_chunk = MagicMock(return_value=None)
    monkeypatch.setattr(kernel, "_snapshot", lambda: snapshot)

    captured = {}

    def fake_map_blocks(func, obj, kwargs, template):
        captured["obj"] = obj
        captured["kwargs"] = kwargs
        return template

    monkeypatch.setattr(xr, "map_blocks", fake_map_blocks)

    result = kernel.final_process(
        raster,
        aot_ref_raster,
        iwl_swir=[-3, -1],
        chunk=2,
        lazy=True,
    )

    assert captured["kwargs"]["iwl_swir"] == [-3, -1]
    assert captured["obj"]["bands"].chunksizes["y"] == (2, 2, 2, 2)
    assert captured["obj"]["bands"].chunksizes["x"] == (2, 2, 2, 2)

    assert result is kernel.xres


def test_final_process_computes_result_when_lazy_false(
    kernel_for_final_process,
    raster,
    aot_ref_raster,
    monkeypatch,
):
    """Vérifie que lazy=False déclenche compute sur le résultat."""
    kernel = kernel_for_final_process

    template = xr.Dataset(
        {
            "Rrs": (
                ("wl", "y", "x"),
                np.ones((6, 8, 8), dtype=np.float32),
            ),
            "BRDFg": (
                ("y", "x"),
                np.ones((8, 8), dtype=np.float32),
            ),
            "aot550": (
                ("y", "x"),
                np.ones((8, 8), dtype=np.float32),
            ),
        },
        coords={
            "wl": raster.wl,
            "y": raster.y,
            "x": raster.x,
        },
    )

    snapshot = MagicMock()
    snapshot.final_process_chunk = MagicMock(return_value=None)
    monkeypatch.setattr(kernel, "_snapshot", lambda: snapshot)

    result_mock = MagicMock()
    result_mock.compute.return_value = template

    def fake_map_blocks(*args, **kwargs):
        return result_mock

    monkeypatch.setattr(xr, "map_blocks", fake_map_blocks)

    result = kernel.final_process(
        raster,
        aot_ref_raster,
        chunk=4,
        lazy=False,
    )

    result_mock.compute.assert_called_once_with()
    assert result is template
    assert kernel.xres is template


def test_final_process_uses_snapshot(
    kernel_for_final_process,
    raster,
    aot_ref_raster,
    monkeypatch,
):
    """Vérifie que le snapshot est créé et utilisé pour les calculs par blocs."""
    kernel = kernel_for_final_process

    snapshot = MagicMock()
    snapshot.final_process_chunk = MagicMock(return_value=None)

    snapshot_factory = MagicMock(return_value=snapshot)
    monkeypatch.setattr(kernel, "_snapshot", snapshot_factory)

    def fake_map_blocks(func, obj, kwargs, template):
        assert func == snapshot.final_process_chunk
        return template

    monkeypatch.setattr(xr, "map_blocks", fake_map_blocks)

    kernel.final_process(
        raster,
        aot_ref_raster,
        chunk=4,
    )

    snapshot_factory.assert_called_once_with()

# endregion


# ------------------------------------------------------------------
# load_process
# ------------------------------------------------------------------
# region snapshot

def test_snapshot_copies_state_and_drops_heavy_attributes():
    """Test that _snapshot creates a lightweight copy of the kernel."""
    kernel = Kernel.__new__(Kernel)

    # Set up the product and the attributes required by _snapshot.
    kernel.prod = SimpleNamespace(_type=np.float32, sensor='S2A')
    kernel._cache = {'cached_value': np.array([1, 2, 3])}

    kernel.wind_img = xr.Dataset({
        'wind': (('y', 'x'), np.array([[2.0, 3.0], [4.0, 5.0]]))
    })

    # Simulate the heavy attributes that must not be retained.
    kernel.raster = object()
    kernel.aero_lut = object()
    kernel.trans_lut = object()
    kernel.cams = object()
    kernel.coarse_masked_raster = object()
    kernel.xres = object()
    kernel.aot_ref_img = object()
    kernel.aot_ref_raster = object()

    # Mock the property to avoid loading actual surface pressure data.
    surface_pressure = xr.DataArray(
        [[101325.0, 101300.0], [101200.0, 101100.0]],
        dims=('y', 'x'),
    )

    with patch.object(
        Kernel,
        '_surface_pressure',
        new_callable=PropertyMock,
        create=True,
        return_value=surface_pressure,
    ) as mocked_pressure:
        state = kernel._snapshot()

    # Verify that the original kernel is not modified.
    for name in (
        'raster',
        'aero_lut',
        'trans_lut',
        'cams',
        'coarse_masked_raster',
        'xres',
        'aot_ref_img',
        'aot_ref_raster',
    ):
        assert hasattr(kernel, name)

    # Verify that heavy attributes are removed from the snapshot.
    for name in (
        'raster',
        'aero_lut',
        'trans_lut',
        'cams',
        'coarse_masked_raster',
        'xres',
        'aot_ref_img',
        'aot_ref_raster',
    ):
        assert not hasattr(state, name)

    # Verify that lightweight attributes are preserved.
    assert state is not kernel
    assert state.prod._type is np.float32
    assert state.prod.sensor == 'S2A'
    assert state._cache == kernel._cache
    assert state._cache is not kernel._cache

    # Verify that the wind dataset is copied without deep-copying its data.
    assert state.wind_img is not kernel.wind_img
    xr.testing.assert_identical(state.wind_img, kernel.wind_img)

    # Verify that surface pressure is accessed before the CAMS reference is removed.
    mocked_pressure.assert_called_once_with()

# endregion


