from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import dask.array as dsa
import pandas as pd
import re

from grs.mask import Settings, _Tile, Masking, _Step
from grs.mask import _set, _chunks, _make_block_fn

# ============================================================
# Fixtures and Helpers
# ============================================================

@pytest.fixture
def masking():
    prod = xr.Dataset(
        data_vars={
            "bands": (
                ("wl", "y", "x"),
                np.arange(48, dtype=np.float32).reshape(3, 4, 4),
            ),
            "ndwi": (
                ("y", "x"),
                np.zeros((4, 4), dtype=np.float32),
            ),
        },
        coords={
            "wl": [550, 665, 842],
            "y": [10, 20, 30, 40],
            "x": [10, 30, 50, 70],
        },
        attrs={
            "constellation_id": "S2",
        },
    )

    return Masking(prod=prod, tile_size=2)

@pytest.fixture
def masking_small_raster():
    prod = xr.Dataset(
                data_vars={
                    "bands": (
                        ("wl", "y", "x"),
                        np.arange(3, dtype=np.float32).reshape(3, 1, 1),
                    ),
                    "ndwi": (
                        ("y", "x"),
                        np.zeros((1, 1), dtype=np.float32),
                    ),
                },
                coords={
                    "wl": [550, 665, 842],
                    "y": [10],
                    "x": [100],
                },
                attrs={
                    "constellation_id": "S2",
                },
            )
        
    return  Masking(prod=prod)

@pytest.fixture
def masking_omnicloud():
    prod = xr.Dataset(
        data_vars={
            "bands": (
                ("wl", "y", "x"),
                np.arange(3072, dtype=np.float32).reshape(3, 32, 32),
            ),
            "ndwi": (
                ("y", "x"),
                np.zeros((32, 32), dtype=np.float32),
            ),
        },
        coords={
            "wl": [550, 665, 842],
            "y": np.arange(32, dtype=np.uint32),
            "x": np.arange(32, dtype=np.uint32),
        },
        attrs={
            "constellation_id": "S2",
        },
    )

    return Masking(prod=prod, tile_size=32)

# ============================================================
# Default configuration
# ============================================================

def test_default_configuration():
    settings = Settings()
    
    assert settings.bvis == 490
    assert settings.bnir == 842
    assert settings.bswir == 1610

    assert settings.bcirrus is None
    assert settings.bwv == 945

    assert settings.swir_threshold == 0.2
    assert settings.ndwi_threshold == 0.0

    assert settings.OCM_resolution == 60
    assert settings.OCM_inference_device == "cpu"

    assert settings.number_of_flags == 32

# ============================================================
# _set
# ============================================================    

@pytest.mark.parametrize(
    "flags, bit, cond, expected",
    [
        (np.array([0,0,0]), 2, [True, False, True], np.array([4,0,4]) ),
        (np.array([1,2,8]), 2, [True, False, True], np.array([5,2,12]) ),
        (np.array([1,2,8]), 2, [False, False, False], np.array([1,2,8]) ),
    ],
)
def test_set(flags, bit, cond, expected):
   _set(flags, bit, cond)
   np.testing.assert_array_equal(flags, expected)

# ============================================================
# _chunks
# ============================================================

@pytest.mark.parametrize(
    "n, size, minimum, expected",
    [
        (10, 5, 0, (5,5)),
        (11,5,0, (5,5,1)),
        (11,5,2,(6,5)),
        (21,5,2,(5,5,6,5))
    ]
)
def test_chunks(n, size, minimum, expected):
    assert _chunks(n,size,minimum) == expected

# ============================================================
#  _make_block_fn
# ============================================================

def test_make_block_fn_with_cube():
    cube = np.zeros((3, 4, 5), dtype=np.float32)

    step = Mock()
    step.fn.side_effect = lambda tile: tile.f.__setitem__(
        (slice(None), slice(None)), 1
    )

    block_fn = _make_block_fn(
        steps=[step],
        slot=0,
        has_cube=True,
        aux_names=[],
    )

    with patch("grs.mask._Tile", wraps=_Tile) as mock_tile:
        result = block_fn(cube)

    assert result.shape == (4, 5)
    assert result.dtype == np.uint32
    step.fn.assert_called_once()
    np.testing.assert_array_equal(
        result,
        np.ones((4, 5), dtype=np.uint32),
    )

# ============================================================
#  Masking init
# ============================================================

@pytest.mark.parametrize(
    "constellation_id",
    ["S2", "L8"]
)
def test_masking_init(constellation_id):
    obj =  xr.Dataset(
        data_vars={
            "var1": (
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
        attrs={
            "constellation_id": constellation_id
        }
    )
    masking = Masking(prod=obj)
    
    assert masking.prod == obj
    assert masking.tile_size == 1024
    
    if(constellation_id == "S2"):
        assert masking.sentinel2 == True
        assert masking.bcirrus == 1375
    else:
        assert masking.sentinel2 == False
        assert masking.bcirrus == 1370
    np.testing.assert_array_equal(
            masking.flags,
            np.zeros((2,2), dtype=np.uint32)
        )

# ============================================================
#  _register
# ============================================================
    
def test_register(masking):
    masking = masking
    masking._register(bit=2, name="Test bit", description="Unit testing bit")
    
    assert masking.flag_names[2] == "Test bit"
    assert masking.flag_descriptions[2] == "Unit testing bit"

# ============================================================
#  _wl_index
# ============================================================

@pytest.mark.parametrize(
    "wl, nearest, expected",
    [
        (665, False, 1),
        (560, True, 0),
        (800, True, 2),
    ]
)
def test_wl_index(masking, wl, nearest, expected):
    assert masking._wl_index(wl=wl, nearest=nearest) == expected

# ============================================================
#  _require_name
# ============================================================

def test_require_name_exists(masking):
    result = masking._require("ndwi")
    xr.testing.assert_equal(result, masking.prod["ndwi"])
    
def test_require_raises_keyerror_if_variable_missing(masking):
    with pytest.raises(
        KeyError,
        match="the dataset has no variable 'missing', needed by this mask",
    ):
        masking._require("missing")
 
# ============================================================
#  Tests _flags_lazy
# ============================================================        

class TestFlagsLazy:
    
    def test_returns_dask_array(self,masking):
        def set_flags_from_band_count(tile):
                tile.f[:] = tile.cube.shape[0]
            
        step = _Step(
            fn=set_flags_from_band_count,
            bands=[0],
            halo=0,
            aux=[],
        )
        
        result = masking._flags_lazy(steps=[step])
        assert isinstance(result, dsa.Array)
        assert result.shape == (4,4)
        assert result.dtype == np.uint32
        computed = result.compute()
        np.testing.assert_array_equal(result, np.ones((4,4)))

    def test_uses_all_bands_when_needed(self,masking):
        def set_flags_from_band_count(tile):
            tile.f[:] = tile.cube.shape[0]
            
        step = _Step(
            fn=set_flags_from_band_count,
            bands=None,
            halo=0,
            aux=[],
        )
        
        result = masking._flags_lazy(steps=[step]).compute()
        np.testing.assert_array_equal(result, np.full((4, 4), 3, dtype=np.uint32))

    def test_raises_error_when_halo_too_large(self,masking):
        def set_flags_from_band_count(tile):
                tile.f[:] = tile.cube.shape[0]
                
        step = _Step(
            fn=set_flags_from_band_count,
            bands=[0],
            halo=5,
            aux=[],
        )
        with pytest.raises(
                ValueError,
                match= re.escape("tile_size (2) must be at least the halo (5 pixels)"),
        ):
            masking._flags_lazy(steps=[step])

    def test_nominal_with_halo(self,masking):
        def flag_neighbors_of_99(tile):
            src = tile.cube[0]

            tile.f[:] = 0
            tile.f[:-1, :] |= (src[1:, :] == 99).astype(np.uint32)
            tile.f[1:, :] |= (src[:-1, :] == 99).astype(np.uint32)
            tile.f[:, :-1] |= (src[:, 1:] == 99).astype(np.uint32)
            tile.f[:, 1:] |= (src[:, :-1] == 99).astype(np.uint32)
        
        bands = np.zeros((3, 4, 4), dtype=np.float32)

        bands[0, 2, 2] = 99

        masking.prod["bands"] = (
            ("wl", "y", "x"),
            bands,
        )

        step = _Step(
            fn=flag_neighbors_of_99,
            bands=[0],
            halo=1,
            aux=[],
        )

        result = masking._flags_lazy([step]).compute()

        expected = np.zeros((4, 4), dtype=np.uint32)
        expected[1, 2] = 1
        expected[2, 1] = 1
        expected[2, 3] = 1
        expected[3, 2] = 1

        np.testing.assert_array_equal(result, expected)



# ============================================================
#  _pixel_size, _ocm_factor
# ============================================================        

def test_pixel_size(masking, masking_small_raster):
    assert masking._pixel_size() == 20
    assert masking_small_raster._pixel_size() is None
    
def test_ocm_factor(masking, masking_small_raster):
    #OCM resolution is 60
    assert masking._ocm_factor() == 3
    assert masking_small_raster._ocm_factor() == 1

# ============================================================
#  Tests _get_omnicloudmask
# ============================================================   

class TestGetOmnicloudmask:

    def test_nominal(self,masking_omnicloud):
        masking = masking_omnicloud

        ny, nx = masking.prod.sizes["y"], masking.prod.sizes["x"]

        masking._ocm_factor = lambda: 1

        # Masque synthétique :
        # 0 = Clear
        # 1 = Thick Cloud
        # 2 = Thin Cloud
        # 3 = Cloud Shadow
        coarse = np.zeros((ny, nx), dtype=np.uint8)
        coarse[0:4, 0:4] = 1
        coarse[4:8, 4:8] = 2
        coarse[8:12, 8:12] = 3

        with patch(
            "omnicloudmask.predict_from_array",
            return_value=[coarse],
        ) as mock_predict:

            counts = masking._get_omnicloudmask()

        mock_predict.assert_called_once()

        np.testing.assert_array_equal(masking.omnimask, coarse)

        assert counts == [16, 16, 16]

        # Vérifier les flags
        assert np.all((masking.flags[0:4, 0:4] & (1 << 11)) != 0)
        assert np.all((masking.flags[4:8, 4:8] & (1 << 12)) != 0)
        assert np.all((masking.flags[8:12, 8:12] & (1 << 13)) != 0)
        
    def test_coarsens_input(self,masking_omnicloud):
        masking = masking_omnicloud
        masking._ocm_factor = lambda: 2

        def fake_predict(array, **kwargs):
            assert array.shape == (3, 16, 16)
            return [np.zeros((16, 16), dtype=np.uint8)]

        with patch(
            "omnicloudmask.predict_from_array",
            side_effect=fake_predict,
        ):
            counts = masking._get_omnicloudmask()

        assert counts == [0, 0, 0]
        assert masking.omnimask.shape == (32, 32)

    def test_cpu_device_by_default(self, masking_omnicloud):
        masking = masking_omnicloud
        masking._ocm_factor = lambda: 1
        ny, nx = masking.prod.sizes["y"], masking.prod.sizes["x"]
        with patch("omnicloudmask.predict_from_array", return_value=[np.zeros((ny, nx), np.uint8)]) as mock_predict:
            masking._get_omnicloudmask()
        assert mock_predict.call_args.kwargs["inference_device"] == "cpu"
        assert mock_predict.call_args.kwargs["mosaic_device"] == "cpu"
        assert masking.OCM_device_used == "cpu"

    def test_cuda_failure_falls_back_to_cpu(self, masking_omnicloud, caplog):
        # e.g. out of GPU memory when several runs share the GPU: same mask, computed on the CPU
        masking = masking_omnicloud
        masking._ocm_factor = lambda: 1
        masking.OCM_inference_device = masking.OCM_mosaic_device = "cuda"
        ny, nx = masking.prod.sizes["y"], masking.prod.sizes["x"]
        coarse = np.zeros((ny, nx), dtype=np.uint8)
        coarse[0:4, 0:4] = 1
        devices = []

        def fake_predict(array, **kwargs):
            devices.append(kwargs["inference_device"])
            if kwargs["inference_device"] != "cpu":
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return [coarse]

        with patch("omnicloudmask.predict_from_array", side_effect=fake_predict):
            counts = masking._get_omnicloudmask()
        assert devices == ["cuda", "cpu"]
        assert masking.OCM_device_used == "cpu"
        assert counts == [16, 0, 0]
        assert "run on the CPU" in caplog.text


@pytest.mark.parametrize("device, cuda, expected", [
    ("cpu", True, "cpu"), ("CUDA", False, "cuda"), ("cuda:1", True, "cuda:1"), (None, True, "cpu"),
    ("auto", True, "cuda"), ("auto", False, "cpu"),
])
def test_ocm_device(monkeypatch, device, cuda, expected):
    import torch
    from grs.mask import _ocm_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert _ocm_device(device) == expected


def test_ocm_device_of_the_configuration():
    # packaged default 'auto', passed by Process to the masking (a user configuration may change it)
    import os
    import yaml
    import grs
    from grs.grs_process import Process
    from grs.config import load_config
    with open(os.path.join(os.path.dirname(grs.__file__), 'config.yml')) as f:
        assert yaml.safe_load(f)["processor"]["ocm_device"] == "auto"
    assert Process().ocm_device == load_config()["processor"]["ocm_device"]


# ============================================================
#  Tests Statistics
# ============================================================   
class TestStatistics:
    def test_get_stats(self,masking):
        masking = masking
        masking._register(3, "test_bit", "unit testing bit")
        masking.flags[:, :] = 0
        masking.flags[:2, :] = 1 << 3
        masking.get_stats()

        assert masking.flag_stats["flag_test_bit"] == pytest.approx(0.5)

    def test_print_stats(self,masking):
        masking = masking
        masking.prod["flags"] = (
            ("y", "x"),
            np.zeros((4, 4), dtype=np.uint32),
        )

        masking._register(3, "test_bit", "Unit testing bit")
        masking._register(4, "other_bit", "Another test bit")

        pflags = masking.prod["flags"]
        pflags.attrs["flag_names"] = [
            "test_bit",
            "other_bit",
            "None",
        ]
        pflags.attrs["flag_descriptions"] = [
            "Unit testing bit",
            "Another test bit",
            "Unused flag",
        ]

        masking.prod.attrs["flag_test_bit"] = 0.5
        masking.prod.attrs["flag_other_bit"] = 0.25

        result = masking.print_stats()

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "name",
            "description",
            "bit",
            "statistics",
        ]

        assert len(result) == 2
        assert "None" not in result["name"].values

        assert result["name"].tolist() == ["test_bit", "other_bit"]
        assert result["description"].tolist() == [
            "Unit testing bit",
            "Another test bit",
        ]
        assert result["statistics"].tolist() == [0.5, 0.25]

# ============================================================
#  process
# ============================================================  

@pytest.mark.parametrize(
    "output, expected_type",
    [
        (None, type(None)),
        ("prod", xr.Dataset),
        ("flags", xr.DataArray),
    ],
)
def test_process_outputs(
    masking,
    monkeypatch,
    output,
    expected_type,
):
    masking = masking

    monkeypatch.setattr(masking, "_nodata_step", lambda: "nodata")
    monkeypatch.setattr(masking, "_water_step", lambda: "water")
    monkeypatch.setattr(masking, "_high_swir_step", lambda: "swir")
    monkeypatch.setattr(masking, "_surfwater_step", lambda: "surfwater")

    expected_flags = np.array(
        [
            [8, 0, 0, 0],
            [0, 8, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 8],
        ],
        dtype=np.uint32,
    )

    def fake_flags_lazy(steps):
        return dsa.from_array(expected_flags, chunks=(4, 4))

    monkeypatch.setattr(masking, "_flags_lazy", fake_flags_lazy)

    def fake_omnicloudmask(bitmasks):
        return [0, 0, 0]

    monkeypatch.setattr(
        masking,
        "_get_omnicloudmask",
        fake_omnicloudmask,
    )

    def fake_store_stats(bits, counts, total):
        masking.flag_stats = {}

    monkeypatch.setattr(masking, "_store_stats", fake_store_stats)

    result = masking.process(output=output)

    if output is None:
        assert result is None
        np.testing.assert_array_equal(masking.flags, expected_flags)
        np.testing.assert_array_equal(
            masking.prod["flags"].values,
            expected_flags,
        )

    elif output == "prod":
        assert result is masking.prod
        assert "flags" in result.data_vars

    else:
        assert isinstance(result, expected_type)
        assert result.dims == ("y", "x")
        assert "flag_names" in result.attrs
        np.testing.assert_array_equal(
            result.values,
            masking.flags,
        )
 
# ============================================================
#  Tests Mask creation
# ============================================================  

class TestMaskCreation:
    
    def test_mask_block(self):
        masking = Masking.__new__(Masking)
        block = np.array([[1,2],[3,4]])
        tomask = (2,3)
        tokeep = (1,4)
        
        expected = np.array([
            [0,1],
            [1,0]
        ])
        result = masking._mask_block(block, tomask, tokeep, np.uint32)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "tomask, tokeep, expected",
        [
            ((2,3), (1,4), np.array([
                [0, 1],
                [1, 0]
            ])),
            ((), (), np.array([
                [0, 0],
                [0, 0]
            ])),
        ]
    )
    def test_create_mask(self,tomask, tokeep, expected):
        masking = Masking.__new__(Masking)
        flags = xr.DataArray([[2,4],[8,16]])
        mask_name = "test mask"
        
        result = masking.create_mask(flags, tomask=tomask, tokeep=tokeep, mask_name=mask_name)
        print(result)
        np.testing.assert_array_equal(result, expected)
        assert result.name == mask_name
# ============================================================
#  Snow and ice
# ============================================================

def _snow_masking(pixels):
    '''Masking of a 1 x n Sentinel-2 raster, one pixel per spectrum (TOA reflectance of the bands below).'''
    wl = [490, 560, 665, 842, 1375, 1610, 2190]
    cube = np.array([[p[w] for w in wl] for p in pixels], dtype=np.float32).T[:, None, :]
    prod = xr.Dataset({'bands': (('wl', 'y', 'x'), cube)},
                      coords={'wl': wl, 'y': [0], 'x': np.arange(len(pixels))},
                      attrs={'constellation_id': 'S2'})
    return Masking(prod=prod, tile_size=1024)


def _px(b490, b560, b665, b842, b1375=0.001, b1610=0.01, b2190=0.005):
    return {490: b490, 560: b560, 665: b665, 842: b842, 1375: b1375, 1610: b1610, 2190: b2190}


SNOW_CASES = [
    # name, spectrum, (snow, ice)
    ('fresh snow', _px(0.70, 0.66, 0.64, 0.62, b1610=0.06), (1, 0)),
    ('lake ice (Lake Erie)', _px(0.40, 0.36, 0.33, 0.25, b1610=0.007), (0, 1)),
    ('very turbid water (Rio de la Plata)', _px(0.12, 0.15, 0.16, 0.114, b1610=0.04), (0, 0)),
    ('clear water', _px(0.09, 0.07, 0.04, 0.02, b1610=0.005), (0, 0)),
    ('snow under cirrus', _px(0.70, 0.66, 0.64, 0.62, b1375=0.02, b1610=0.06), (0, 0)),
    ('vegetation', _px(0.05, 0.08, 0.05, 0.35, b1610=0.18), (0, 0)),
    ('cloud (bright SWIR)', _px(0.50, 0.48, 0.47, 0.46, b1610=0.35), (0, 0)),
]


def test_snow_ice_step():
    masking = _snow_masking([c[1] for c in SNOW_CASES])
    masking.snow_ice_mask()
    flags = masking.flags[0]
    for k, (name, _, (snow, ice)) in enumerate(SNOW_CASES):
        assert bool(flags[k] & (1 << 21)) == bool(snow), name
        assert bool(flags[k] & (1 << 22)) == bool(ice), name
    assert masking.flag_names[21] == 'snow' and masking.flag_names[22] == 'ice'
    assert 'NDSI(560, 1610) > 0.40' in masking.flag_descriptions[21]


def test_snow_ice_helper():
    flags = xr.DataArray(np.array([0, 1 << 21, 1 << 22, 1 << 3], dtype=np.uint32))
    np.testing.assert_array_equal(Masking.snow_ice(flags).values, [False, True, True, False])
