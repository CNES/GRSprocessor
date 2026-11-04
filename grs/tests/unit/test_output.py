import builtins
import logging
import shutil
import sys
import types

import numpy as np 
import pytest 
import xarray as xr

import grs.output as mod
from unittest.mock import MagicMock, patch
from grs.output import ( L2aProduct, _base_encoding, _clear_store, _coarsen, _compression_encoding, _consolidate, _pyramid_levels, _rrs_encoding, _coord_encoding, _package_version, _reset_encoding, _load_coords, _format_from_name, _is_remote, _update_geotransform, _var_chunking, _xarray_supports_shards, _zarr_major)
from grs.output import (NETCDF_ENGINE, _CONFLICTING_ATTRS, _NETCDF_EXTENSIONS, _ZARR_EXTENSIONS, _FORMAT_EXTENSION, PYRAMID_FACTOR, PYRAMID_MIN_SIZE, LEVEL_GROUP, ANCILLARY_GROUP, _DEFAULT_SHARDS, _DEFAULT_CHUNKS_SHARDED, _DEFAULT_CHUNKS, )

def test_format_constants(): 
    """Vérifie les constantes relatives aux formats de sortie."""

    assert NETCDF_ENGINE == "netcdf4"

    assert _CONFLICTING_ATTRS == (
        "grid_mapping",
        "scale_factor",
        "_FillValue",
        "add_offset",
    )

    assert _NETCDF_EXTENSIONS == (".nc", ".nc4", ".cdf")
    assert _ZARR_EXTENSIONS == (".zarr",)

    assert _FORMAT_EXTENSION == {
        "netcdf": ".nc",
        "zarr": ".zarr",
    }

def test_pyramid_constants(): 
    """Vérifie les paramètres de génération des pyramides."""

    assert PYRAMID_FACTOR == 2
    assert PYRAMID_MIN_SIZE == 512

    assert LEVEL_GROUP == "{:d}"
    assert ANCILLARY_GROUP == "ancillary"

def test_zarr_layout_constants(): 
    """Vérifie les paramètres de shards et de chunks Zarr."""

    assert _DEFAULT_SHARDS == {
        "y": 1024,
        "x": 1024,
        "wl": -1,
    }

    assert _DEFAULT_CHUNKS_SHARDED == {
        "y": 256,
        "x": 256,
        "wl": -1,
    }

    assert _DEFAULT_CHUNKS == {
        "y": 512,
        "x": 512,
        "wl": -1,
    }

def test_base_encoding_content(): 
    """Vérifie l'encodage des variables principales."""

    encoding = _base_encoding()

    assert encoding["aot550"] == {
        "dtype": "int16",
        "scale_factor": 0.001,
        "_FillValue": -9999,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["BRDFg"] == {
        "dtype": "int16",
        "scale_factor": 0.00001,
        "add_offset": 0.3,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["dem"] == {
        "dtype": "int16",
        "scale_factor": 1,
        "add_offset": 500,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["vza"] == {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["raa"] == {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 180,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["sza"] == {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 30,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

    assert encoding["flags"] == {
        "grid_mapping": "spatial_ref",
    }

    assert encoding["mask"] == {
        "grid_mapping": "spatial_ref",
    }

def test_base_encoding_returns_independent_dictionaries(): 
    """Vérifie que deux appels ne partagent pas leurs dictionnaires."""

    encoding1 = _base_encoding()
    encoding2 = _base_encoding()

    encoding1["aot550"]["scale_factor"] = 123

    assert encoding2["aot550"]["scale_factor"] == 0.001

def test_base_encoding_variables(): 
    """Vérifie la présence exacte des variables attendues."""

    encoding = _base_encoding()

    assert set(encoding) == {
        "aot550",
        "BRDFg",
        "dem",
        "vza",
        "raa",
        "sza",
        "flags",
        "mask",
    }

def test_rrs_encoding_content(): 
    """Vérifie l'encodage des réflectances Rrs."""

    encoding = _rrs_encoding()

    assert encoding == {
        "dtype": "int16",
        "scale_factor": 0.00001,
        "add_offset": 0.3,
        "_FillValue": -32768,
        "grid_mapping": "spatial_ref",
    }

def test_rrs_encoding_returns_independent_dictionaries(): 
    """Vérifie que chaque appel renvoie un nouveau dictionnaire."""

    encoding1 = _rrs_encoding()
    encoding2 = _rrs_encoding()

    encoding1["dtype"] = "float32"

    assert encoding2["dtype"] == "int16"

def test_coord_encoding_content(): 
    """Vérifie l'encodage des coordonnées temporelles et spatiales."""

    encoding = _coord_encoding()

    assert encoding == {
        "time": {
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "proleptic_gregorian",
            "dtype": "int64",
        },
        "spatial_ref": {
            "dtype": "int32",
        },
    }

def test_coord_encoding_returns_independent_dictionaries():
    """Vérifie l'indépendance des dictionnaires de coordonnées."""

    encoding1 = _coord_encoding()
    encoding2 = _coord_encoding()

    encoding1["time"]["dtype"] = "float64"

    assert encoding2["time"]["dtype"] == "int64"

def test_package_version_returns_installed_version(): 
    """Renvoie la version du paquet lorsque celui-ci est installé."""

    with patch(
        "importlib.metadata.version",
        return_value="1.2.3",
    ) as mock_version:

        result = _package_version("mon-paquet")

    assert result == "1.2.3"
    mock_version.assert_called_once_with("mon-paquet")

def test_package_version_returns_unknown_if_not_installed(): 
    """Renvoie 'unknown' si le paquet n'est pas installé."""

    from importlib.metadata import PackageNotFoundError

    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError,
    ):

        result = _package_version("paquet-inexistant")

    assert result == "unknown"

@pytest.fixture 
def sample_dataset(): 
    """Crée un Dataset avec des encodages préexistants."""

    ds = xr.Dataset(
        data_vars={
            "Rrs": (
                ("y", "x"),
                np.array(
                    [[0.1, 0.2], [0.3, 0.4]],
                    dtype=np.float32,
                ),
            ),
            "flags": (
                ("y", "x"),
                np.array(
                    [[0, 1], [2, 3]],
                    dtype=np.uint8,
                ),
            ),
        },
        coords={
            "x": np.array([10.0, 11.0]),
            "y": np.array([45.0, 44.0]),
            "time": np.datetime64("2024-06-15"),
        },
    )

    # Simule les encodages hérités d'un fichier d'entrée
    ds["Rrs"].encoding = {
        "dtype": "int16",
        "scale_factor": 0.0001,
        "_FillValue": -9999,
        "chunksizes": (128, 128),
    }

    ds["flags"].encoding = {
        "dtype": "uint8",
        "zlib": True,
    }

    ds["x"].encoding = {
        "dtype": "float64",
    }

    ds["y"].encoding = {
        "dtype": "float64",
    }

    ds["time"].encoding = {
        "units": "days since 2000-01-01",
        "calendar": "standard",
    }

    return ds

def test_reset_encoding_clears_all_variable_encodings(sample_dataset):
    """Vérifie que tous les encodages sont supprimés."""

    _reset_encoding(sample_dataset)

    for var in sample_dataset.variables:
        assert sample_dataset[var].encoding == {}
    

def test_reset_encoding_preserves_data(sample_dataset): 
    """Vérifie que les données restent inchangées."""

    original = sample_dataset.copy(deep=True)

    _reset_encoding(sample_dataset)

    xr.testing.assert_equal(sample_dataset, original)

def test_reset_encoding_returns_none(sample_dataset): 
    """Vérifie que la fonction modifie le Dataset en place."""

    result = _reset_encoding(sample_dataset)

    assert result is None

def test_reset_encoding_handles_dataset_without_encoding(): 
    """Vérifie le comportement avec un Dataset sans encodage explicite."""

    ds = xr.Dataset(
        {
            "temperature": (
                ("x",),
                np.array([10.0, 20.0]),
            )
        }
    )

    _reset_encoding(ds)

    for var in ds.variables:
        assert ds[var].encoding == {}

def test_load_coords_without_lazy_coordinates(): 
    """Retourne le Dataset original si aucune coordonnée scalaire n'est lazy."""

    ds = xr.Dataset(
        {"temperature": (("y", "x"), [[10.0, 20.0], [30.0, 40.0]])},
        coords={
            "x": [0, 1],
            "y": [0, 1],
            "time": np.datetime64("2024-06-15"),
        },
    )

    result = _load_coords(ds)

    assert result is ds

def test_load_coords_computes_lazy_scalar_coordinates():
    """Test that lazy scalar coordinates are computed and loaded into memory."""

    da = pytest.importorskip("dask.array")

    time_value = np.datetime64("2024-06-15", "ns")

    ds = xr.Dataset(
        {"temperature": (("y", "x"), np.ones((2, 2)))},
        coords={
            "x": [0, 1],
            "y": [0, 1],
            "time": xr.Variable(
                (),
                da.from_array(time_value, chunks=()),
            ),
        },
    )

    assert ds["time"].chunks is not None

    result = mod._load_coords(ds)

    assert result is not ds
    assert result["time"].chunks is None
    assert result["time"].shape == ()
    assert result["time"].dtype == np.dtype("datetime64[ns]")
    assert result["time"].values == time_value

    xr.testing.assert_equal(
        result["temperature"],
        ds["temperature"],
    )


def test_load_coords_does_not_compute_dimension_coordinates():
    """Test that dimension coordinates are not computed."""

    da = pytest.importorskip("dask.array")

    ds = xr.Dataset(
        {"temperature": (("y", "x"), np.ones((2, 2)))},
        coords={
            "x": [0, 1],
            "y": [0, 1],
        },
    )

    # Replace the dimension coordinate with a mocked lazy variable.
    mocked_coord = MagicMock()
    mocked_coord.chunks = ((1, 1),)

    with patch.object(
        xr.Dataset,
        "coords",
        new_callable=lambda: property(
            lambda self: {"x": mocked_coord}
        ),
    ):
        result = mod._load_coords(ds)

    assert result is ds
    mocked_coord.compute.assert_not_called()


@pytest.mark.parametrize( ("path", "expected"), [ ("product.nc", "netcdf"), ("product.NC", "netcdf"), ("product.nc4", "netcdf"), ("product.cdf", "netcdf"), ("product.zarr", "zarr"), ("product.ZARR", "zarr"), ("/tmp/product.nc", "netcdf"), ("/tmp/product.zarr/", "zarr"), ("s3://bucket/product.nc", "netcdf"), ("s3://bucket/product.zarr/", "zarr"), ("product.tif", None), ("product", None), ("", None), ], ) 
def test_format_from_name(path, expected): 
    """Détecte correctement le format selon l'extension."""
    assert _format_from_name(path) == expected

def test_format_from_name_accepts_pathlike(): 
    """Accepte un objet pathlib.Path."""

    from pathlib import Path

    assert _format_from_name(Path("product.nc")) == "netcdf"

@pytest.mark.parametrize( ("path", "expected"), [ ("product.nc", False), ("/tmp/product.zarr", False), ("./product.zarr", False), ("s3://bucket/product.zarr", True), ("gs://bucket/product.zarr", True), ("az://container/product.zarr", True), ("https://example.org/product.zarr", True), ("file:///tmp/product.zarr", False), ("", False), ], ) 
def test_is_remote(path, expected): 
    """Distingue les URL distantes des chemins locaux."""
    assert _is_remote(path) is expected

def test_is_remote_accepts_pathlike(): 
    """Accepte un objet pathlib.Path."""

    from pathlib import Path

    assert _is_remote(Path("/tmp/product.zarr")) is False

@pytest.mark.parametrize( ("version", "expected"), [ ("2.18.7", 2), ("3.0.1", 3), ("10.2.0", 10), ], ) 
def test_zarr_major_returns_major_version(version, expected): 
    """Extrait le numéro majeur de la version de Zarr."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.__version__ = version

    with patch.dict(sys.modules, {"zarr": fake_zarr}):
        assert _zarr_major() == expected
    
def test_zarr_major_returns_none_if_zarr_is_missing(): 
    """Retourne None si Zarr ne peut pas être importé."""

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("zarr not installed")
        return original_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"zarr": None}),
        patch("builtins.__import__", side_effect=fake_import),
    ):
        assert _zarr_major() is None

def test_zarr_major_returns_none_if_version_is_invalid(): 
    """Retourne None si la version ne peut pas être interprétée."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.__version__ = "invalid"

    with patch.dict(sys.modules, {"zarr": fake_zarr}):
        assert _zarr_major() is None
    
@pytest.mark.parametrize( ("source", "expected"), [ ("def encode():\n shards = None", True), ("def encode():\n chunks = None", False), ], ) 
def test_xarray_supports_shards(source, expected): 
    """Détecte la présence de la fonctionnalité shards dans le code source."""

    with patch("inspect.getsource", return_value=source):
        assert _xarray_supports_shards() is expected

def test_xarray_supports_shards_returns_false_on_error(): 
    """Retourne False si l'inspection du code source échoue."""

    with patch(
        "inspect.getsource",
        side_effect=OSError("source unavailable"),
    ):
        assert _xarray_supports_shards() is False

@pytest.mark.parametrize( ("compression", "clevel", "zarr_major"), [ (None, 5, 2), ("zstd", 5, None), ], ) 
def test_compression_encoding_disabled(compression, clevel, zarr_major): 
    """Ne configure pas la compression si elle est désactivée ou indisponible."""

    assert _compression_encoding(
        compression, clevel, zarr_major
    ) == {}

def test_compression_encoding_zarr_v3(): 
    """Configure BloscCodec avec l'API de Zarr 3."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.__path__ = []

    fake_codecs = types.ModuleType("zarr.codecs")
    fake_codec = MagicMock(name="BloscCodec")
    fake_codecs.BloscCodec = fake_codec

    with patch.dict(
        sys.modules,
        {
            "zarr": fake_zarr,
            "zarr.codecs": fake_codecs,
        },
    ):
        result = _compression_encoding("zstd", 5, 3)

    fake_codec.assert_called_once_with(
        cname="zstd",
        clevel=5,
        shuffle="shuffle",
    )

    assert result == {
        "compressors": (fake_codec.return_value,),
    }

def test_compression_encoding_zarr_v2(): 
    """Configure Blosc avec l'API de Zarr 2."""

    fake_numcodecs = types.ModuleType("numcodecs")
    fake_numcodecs.Blosc = MagicMock(name="Blosc")
    fake_numcodecs.Blosc.SHUFFLE = 1

    with patch.dict(sys.modules, {"numcodecs": fake_numcodecs}):
        result = _compression_encoding("zstd", 7, 2)

    fake_numcodecs.Blosc.assert_called_once_with(
        cname="zstd",
        clevel=7,
        shuffle=1,
    )

    assert result == {
        "compressor": fake_numcodecs.Blosc.return_value,
    }

def test_compression_encoding_returns_empty_dict_on_error(caplog): 
    """Journalise un avertissement et retourne {} si la compression échoue."""

    with (
        patch.dict(sys.modules, {"numcodecs": None}),
        caplog.at_level(logging.WARNING),
    ):
        result = _compression_encoding("invalid", 5, 2)

    assert result == {}
    assert "compression 'invalid' not available" in caplog.text

@pytest.mark.parametrize( ("dims", "dask_chunks", "chunks", "expected"), [ ( ("y", "x"), ((512, 512), (512, 512)), {"y": 256, "x": 256}, ((256, 256), (512, 512)), ), ( ("y", "x", "wl"), ((512, 512), (512, 512), (10,)), {"y": 256, "x": 256, "wl": -1}, ((256, 256, 10), (512, 512, 10)), ), ( ("y", "x"), ((100, 100), (120, 120)), {"y": 256, "x": 256}, ((100, 120), (100, 120)), ), ( ("y", "x"), ((500, 500), (300, 300)), {"y": 256, "x": 100}, ((500, 100), (500, 300)), ), ( ("y", "x"), ((512, 512), (512, 512)), {}, ((512, 512), (512, 512)), ), ( ("y", "x"), ((512, 512), (512, 512)), {"y": None, "x": -1}, ((512, 512), (512, 512)), ), ], ) 
def test_var_chunking(dims, dask_chunks, chunks, expected): 
    """Vérifie le calcul des chunks stockés et des blocs d'écriture."""

    assert _var_chunking(dims, dask_chunks, chunks) == expected

def test_var_chunking_clips_chunk_to_block(): 
    """Limite le chunk demandé à la taille du bloc."""

    result = _var_chunking(
        dims=("y",),
        dask_chunks=((128, 128),),
        chunks={"y": 512},
    )

    assert result == ((128,), (128,))

def test_clear_store_removes_existing_local_store(tmp_path):
    """Supprime un store local existant."""

    store = tmp_path / "product.zarr"
    store.mkdir()
    (store / "metadata.txt").write_text("test")

    _clear_store(str(store))

    assert not store.exists()

def test_clear_store_does_nothing_if_local_store_missing(tmp_path): 
    """Ne fait rien si le store local n'existe pas."""

    store = tmp_path / "missing.zarr"

    with patch.object(shutil, "rmtree") as mock_rmtree:
        _clear_store(str(store))

    mock_rmtree.assert_not_called()

def test_clear_store_removes_existing_remote_store(): 
    """Supprime un store distant existant via fsspec."""

    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.core = MagicMock()

    fake_fs = MagicMock()
    fake_fs.exists.return_value = True

    fake_fsspec.core.url_to_fs.return_value = (
        fake_fs,
        "bucket/product.zarr",
    )

    with patch.dict(sys.modules, {"fsspec": fake_fsspec}):
        _clear_store(
            "s3://bucket/product.zarr",
            storage_options={"anon": True},
        )

    fake_fsspec.core.url_to_fs.assert_called_once_with(
        "s3://bucket/product.zarr",
        anon=True,
    )

    fake_fs.rm.assert_called_once_with(
        "bucket/product.zarr",
        recursive=True,
    )

def test_clear_store_does_not_remove_missing_remote_store():
    """Ne supprime rien si le store distant n'existe pas."""

    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.core = MagicMock()

    fake_fs = MagicMock()
    fake_fs.exists.return_value = False

    fake_fsspec.core.url_to_fs.return_value = (
        fake_fs,
        "bucket/missing.zarr",
    )

    with patch.dict(sys.modules, {"fsspec": fake_fsspec}):
        _clear_store("s3://bucket/missing.zarr")

    fake_fs.rm.assert_not_called()

def test_consolidate_local_store(): 
    """Consolide la racine et les groupes d'un store local."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.consolidate_metadata = MagicMock()

    with patch.dict(sys.modules, {"zarr": fake_zarr}):
        _consolidate(
            "/tmp/product.zarr",
            groups=["0", "1", "ancillary"],
        )

    calls = fake_zarr.consolidate_metadata.call_args_list

    assert len(calls) == 4
    assert calls[0].args == ("/tmp/product.zarr",)
    assert calls[1].kwargs == {"path": "0"}
    assert calls[2].kwargs == {"path": "1"}
    assert calls[3].kwargs == {"path": "ancillary"}

def test_consolidate_remote_store_zarr_v2(): 
    """Utilise fsspec.get_mapper pour un store distant Zarr 2."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.consolidate_metadata = MagicMock()

    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.get_mapper = MagicMock(return_value="fake-mapper")

    with patch.dict(
        sys.modules,
        {
            "zarr": fake_zarr,
            "fsspec": fake_fsspec,
        },
    ):
        _consolidate(
            "s3://bucket/product.zarr",
            storage_options={"anon": True},
            zarr_major=2,
            groups=["0"],
        )

    fake_fsspec.get_mapper.assert_called_once_with(
        "s3://bucket/product.zarr",
        anon=True,
    )

    assert fake_zarr.consolidate_metadata.call_count == 2
    assert fake_zarr.consolidate_metadata.call_args_list[0].args == (
        "fake-mapper",
    )
    assert fake_zarr.consolidate_metadata.call_args_list[1].kwargs == {
        "path": "0",
    }

def test_consolidate_remote_store_zarr_v3():
    """Utilise FsspecStore.from_url pour un store distant Zarr 3."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.consolidate_metadata = MagicMock()

    fake_zarr.storage = MagicMock()
    fake_zarr.storage.FsspecStore.from_url.return_value = "fake-store"

    with patch.dict(sys.modules, {"zarr": fake_zarr}):
        _consolidate(
            "s3://bucket/product.zarr",
            storage_options={"anon": True},
            zarr_major=3,
            groups=["ancillary"],
        )

    fake_zarr.storage.FsspecStore.from_url.assert_called_once_with(
        "s3://bucket/product.zarr",
        storage_options={"anon": True},
    )

    assert fake_zarr.consolidate_metadata.call_count == 2
    assert fake_zarr.consolidate_metadata.call_args_list[0].args == (
        "fake-store",
    )
    assert fake_zarr.consolidate_metadata.call_args_list[1].kwargs == {
        "path": "ancillary",
    }

def test_consolidate_remote_store_without_storage_options(): 
    """Utilise des options vides si aucune option n'est fournie."""

    fake_zarr = types.ModuleType("zarr")
    fake_zarr.consolidate_metadata = MagicMock()

    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.get_mapper = MagicMock(return_value="fake-mapper")

    with patch.dict(
        sys.modules,
        {
            "zarr": fake_zarr,
            "fsspec": fake_fsspec,
        },
    ):
        _consolidate(
            "s3://bucket/product.zarr",
            zarr_major=2,
        )

    fake_fsspec.get_mapper.assert_called_once_with(
        "s3://bucket/product.zarr",
    )

# ------------------------------------------------------------------
# Tests for _pyramid_levels
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    ("ny", "nx", "min_size", "max_levels", "expected"),
    [
        (5490, 5490, 512, None, 4),
        (400, 300, 512, None, 0),
        (1024, 1024, 512, None, 1),
        (1025, 1025, 512, None, 2),
        (2048, 1024, 512, None, 2),
        (4096, 2048, 512, None, 3),
        (5490, 5490, 512, 2, 2),
        (5490, 5490, 512, 1, 1),
        (5490, 5490, 512, 10, 4),
        (1, 1000, 512, None, 0),
        (1000, 1, 512, None, 0),
        (0, 1000, 512, None, 0),
        (1000, 0, 512, None, 0),
        (512, 512, 512, None, 0),
        (513, 512, 512, None, 1),
    ],
)
def test_pyramid_levels(ny, nx, min_size, max_levels, expected):
    """Test the number of pyramid overview levels."""

    assert _pyramid_levels(
        ny,
        nx,
        min_size=min_size,
        max_levels=max_levels,
    ) == expected


def test_pyramid_levels_uses_default_min_size():
    """Test the default minimum pyramid size."""

    assert _pyramid_levels(400, 300) == 0
    assert _pyramid_levels(1024, 1024) == 1


def test_pyramid_levels_stops_at_max_levels():
    """Test that the maximum number of levels is respected."""

    assert _pyramid_levels(
        10000,
        10000,
        min_size=512,
        max_levels=2,
    ) == 2


# ------------------------------------------------------------------
# Fixtures for _coarsen
# ------------------------------------------------------------------

@pytest.fixture
def spatial_dataset():
    """Create a dataset containing continuous and discrete spatial variables."""

    return xr.Dataset(
        data_vars={
            "reflectance": (
                ("y", "x"),
                np.array(
                    [
                        [1.0, 3.0, 5.0, 7.0],
                        [5.0, 7.0, 9.0, 11.0],
                        [9.0, 11.0, 13.0, 15.0],
                        [13.0, 15.0, 17.0, 19.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            "temperature": (
                ("y", "x"),
                np.array(
                    [
                        [10.0, np.nan, 30.0, 40.0],
                        [30.0, 50.0, 50.0, 60.0],
                        [70.0, 90.0, 100.0, 110.0],
                        [90.0, 110.0, 120.0, 130.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            "flags": (
                ("y", "x"),
                np.array(
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ],
                    dtype=np.uint8,
                ),
            ),
            "mask": (
                ("y", "x"),
                np.array(
                    [
                        [True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False],
                        [False, False, True, True],
                    ],
                    dtype=bool,
                ),
            ),
        },
        coords={
            "y": [3.5, 2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
        attrs={
            "satellite": "S2A",
            "description": "Synthetic test dataset",
        },
    )


@pytest.fixture
def odd_spatial_dataset():
    """Create a spatial dataset with dimensions that are not divisible by the reduction factor."""

    return xr.Dataset(
        data_vars={
            "reflectance": (
                ("y", "x"),
                np.arange(15, dtype=np.float32).reshape(3, 5),
            ),
            "flags": (
                ("y", "x"),
                np.arange(15, dtype=np.uint8).reshape(3, 5),
            ),
        },
        coords={
            "y": [2.0, 1.0, 0.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        attrs={"description": "Odd-sized dataset"},
    )


# ------------------------------------------------------------------
# Tests for _coarsen
# ------------------------------------------------------------------

def test_coarsen_reduces_spatial_dimensions(spatial_dataset):
    """Test that spatial dimensions are reduced by the specified factor."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    assert result.sizes["y"] == 2
    assert result.sizes["x"] == 2


def test_coarsen_computes_mean_for_continuous_variables(spatial_dataset):
    """Test that continuous variables are averaged over each block."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    expected = np.array(
        [
            [4.0, 8.0],
            [12.0, 16.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result["reflectance"].values,
        expected,
    )


def test_coarsen_ignores_nan_when_computing_mean(spatial_dataset):
    """Test that NaN values are ignored when averaging continuous variables."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    expected = np.array(
        [
            [30.0, 45.0],
            [90.0, 115.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result["temperature"].values,
        expected,
    )


def test_coarsen_preserves_first_pixel_for_integer_variables(spatial_dataset):
    """Test that integer variables use the first pixel of each block."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    expected = np.array(
        [
            [1, 3],
            [9, 11],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(
        result["flags"].values,
        expected,
    )


def test_coarsen_preserves_first_pixel_for_boolean_variables(spatial_dataset):
    """Test that boolean variables use the first pixel of each block."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    expected = np.array(
        [
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(
        result["mask"].values,
        expected,
    )


def test_coarsen_preserves_dataset_attributes(spatial_dataset):
    """Test that dataset attributes are preserved."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset, factor=2)

    assert result.attrs == spatial_dataset.attrs


def test_coarsen_preserves_variable_attributes():
    """Test that continuous variable attributes are preserved."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
                {"units": "sr-1", "description": "Remote sensing reflectance"},
            )
        },
        coords={
            "y": [3.5, 2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
    )

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(ds, factor=2)

    assert result["reflectance"].attrs == {
        "units": "sr-1",
        "description": "Remote sensing reflectance",
    }


def test_coarsen_handles_incomplete_edge_blocks(odd_spatial_dataset):
    """Test that incomplete blocks at the right and bottom edges are retained."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(odd_spatial_dataset, factor=2)

    assert result.sizes["y"] == 2
    assert result.sizes["x"] == 3

    np.testing.assert_allclose(
        result["reflectance"].values,
        np.array(
            [
                [3.0, 5.0, 6.5],
                [10.5, 12.5, 14.0],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        result["flags"].values,
        np.array(
            [
                [0, 2, 4],
                [10, 12, 14],
            ],
            dtype=np.uint8,
        ),
    )


def test_coarsen_updates_geotransform(odd_spatial_dataset):
    """Test that the geotransform update function is called."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ) as mock_update:
        result = _coarsen(odd_spatial_dataset, factor=2)

    mock_update.assert_called_once()

    assert result.sizes["y"] == 2
    assert result.sizes["x"] == 3


def test_coarsen_uses_default_factor(spatial_dataset):
    """Test that the default reduction factor is applied."""

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(spatial_dataset)

    assert result.sizes["y"] == 2
    assert result.sizes["x"] == 2


def test_coarsen_preserves_non_spatial_variables():
    """Test the behavior of variables that do not have both spatial dimensions."""

    ds = xr.Dataset(
        data_vars={
            "reflectance": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
            "quality": (
                ("wl",),
                np.array([1, 2, 3], dtype=np.int16),
            ),
        },
        coords={
            "y": [3.5, 2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
            "wl": [490, 560, 842],
        },
    )

    with patch.object(
        mod,
        "_update_geotransform",
        side_effect=lambda ds: ds,
    ):
        result = _coarsen(ds, factor=2)

    assert "quality" in result
    assert result["quality"].sizes == {"wl": 3}

    np.testing.assert_array_equal(
        result["quality"].values,
        [1, 2, 3],
    )


# ------------------------------------------------------------------
# Tests for _update_geotransform
# ------------------------------------------------------------------

def test_update_geotransform_updates_transform():
    """Test that spatial dimensions and geotransform are updated."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("row", "col"),
                np.ones((2, 3), dtype=np.float32),
            )
        },
        coords={
            "row": [1.5, 0.5],
            "col": [0.5, 1.5, 2.5],
        },
    )

    expected_transform = MagicMock(name="transform")

    rio_mock = MagicMock()
    rio_mock.transform.return_value = expected_transform
    rio_mock.set_spatial_dims.return_value = ds
    rio_mock.write_transform.return_value = ds

    with patch.object(
        xr.Dataset,
        "rio",
        new_callable=lambda: property(lambda self: rio_mock),
        create=True,
    ):
        result = _update_geotransform(
            ds,
            dims=("row", "col"),
        )

    rio_mock.set_spatial_dims.assert_called_once_with(
        x_dim="col",
        y_dim="row",
    )
    rio_mock.transform.assert_called_once_with(recalc=True)
    rio_mock.write_transform.assert_called_once_with(expected_transform)

    assert result is ds


def test_update_geotransform_uses_default_dimensions():
    """Test that the default spatial dimension names are used."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.float32),
            )
        },
        coords={
            "y": [1.5, 0.5],
            "x": [0.5, 1.5],
        },
    )

    rio_mock = MagicMock()
    rio_mock.set_spatial_dims.return_value = ds
    rio_mock.transform.return_value = "fake-transform"
    rio_mock.write_transform.return_value = ds

    with patch.object(
        xr.Dataset,
        "rio",
        new_callable=lambda: property(lambda self: rio_mock),
        create=True,
    ):
        result = _update_geotransform(ds)

    rio_mock.set_spatial_dims.assert_called_once_with(
        x_dim="x",
        y_dim="y",
    )

    assert result is ds


def test_update_geotransform_returns_dataset_if_spatial_dims_fail(caplog):
    """Test that the original dataset is returned if spatial dimension setup fails."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.float32),
            )
        }
    )

    rio_mock = MagicMock()
    rio_mock.set_spatial_dims.side_effect = ValueError("invalid spatial dimensions")

    with (
        patch.object(
            xr.Dataset,
            "rio",
            new_callable=lambda: property(lambda self: rio_mock),
            create=True,
        ),
        caplog.at_level(logging.WARNING),
    ):
        result = _update_geotransform(ds)

    assert result is ds
    assert "could not update the geotransform" in caplog.text
    assert "invalid spatial dimensions" in caplog.text


def test_update_geotransform_returns_dataset_if_transform_fails(caplog):
    """Test that the dataset is returned if calculating the transform fails."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.float32),
            )
        }
    )

    rio_mock = MagicMock()
    rio_mock.set_spatial_dims.return_value = ds
    rio_mock.transform.side_effect = RuntimeError("transform unavailable")

    with (
        patch.object(
            xr.Dataset,
            "rio",
            new_callable=lambda: property(lambda self: rio_mock),
            create=True,
        ),
        caplog.at_level(logging.WARNING),
    ):
        result = _update_geotransform(ds)

    assert result is ds
    assert "could not update the geotransform" in caplog.text
    assert "transform unavailable" in caplog.text


def test_update_geotransform_returns_dataset_if_write_transform_fails(caplog):
    """Test that the dataset is returned if writing the transform fails."""

    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.float32),
            )
        }
    )

    rio_mock = MagicMock()
    rio_mock.set_spatial_dims.return_value = ds
    rio_mock.transform.return_value = "fake-transform"
    rio_mock.write_transform.side_effect = RuntimeError("write failed")

    with (
        patch.object(
            xr.Dataset,
            "rio",
            new_callable=lambda: property(lambda self: rio_mock),
            create=True,
        ),
        patch.object(
            logging,
            "warning",
        ) as mock_warning,
    ):
        result = _update_geotransform(ds)

    assert result is ds
    mock_warning.assert_called_once()
    assert "could not update the geotransform" in mock_warning.call_args.args[0]


# ------------------------------------------------------------------
# Tests for L2aProduct
# ------------------------------------------------------------------


class FakeRioAccessor:
    """Minimal replacement for the rioxarray accessor used by unit tests."""

    def __init__(self, obj):
        self._obj = obj

    def set_spatial_dims(self, x_dim="x", y_dim="y", inplace=False):
        self._obj.attrs["_x_dim"] = x_dim
        self._obj.attrs["_y_dim"] = y_dim
        return self._obj

    def write_coordinate_system(self, inplace=False):
        return self._obj

    def write_crs(self, input_crs=None, grid_mapping_name="spatial_ref", inplace=False, **kwargs):
        if input_crs is not None:
            self._obj.attrs["crs"] = str(input_crs)
        return self._obj

    def transform(self, recalc=False):
        return (0, 1, 0, 0, 0, -1)

    def write_transform(self, transform=None, grid_mapping_name="spatial_ref", inplace=False):
        if transform is not None:
            self._obj.attrs["GeoTransform"] = str(transform)
        return self._obj


@pytest.fixture(autouse=True)
def fake_rio_accessor(monkeypatch):
    """Register a minimal rio accessor without requiring rioxarray."""

    def get_rio(obj):
        return FakeRioAccessor(obj)

    monkeypatch.setattr(
        xr.Dataset,
        "rio",
        property(get_rio),
        raising=False,
    )
    monkeypatch.setattr(
        xr.DataArray,
        "rio",
        property(get_rio),
        raising=False,
    )


@pytest.fixture(autouse=True)
def disable_geotransform_update(monkeypatch):
    """Disable geotransform updates that depend on rioxarray."""

    monkeypatch.setattr(
        "grs.output._update_geotransform",
        lambda ds, dims=("y", "x"): ds,
    )


@pytest.fixture
def sample_products():
    """Create synthetic products for testing L2aProduct."""

    x = np.array([100.0, 110.0, 120.0, 130.0])
    y = np.array([200.0, 190.0, 180.0])
    wl = np.array([490, 560, 665])

    native_raster = xr.Dataset(
        {
            "vza": (
                ("wl", "y", "x"),
                np.full((3, 3, 4), 10.0, dtype=np.float32),
            ),
            "sza": (
                ("y", "x"),
                np.full((3, 4), 30.0, dtype=np.float32),
            ),
            "raa": (
                ("wl", "y", "x"),
                np.full((3, 3, 4), 20.0, dtype=np.float32),
            ),
            "flags": (
                ("y", "x"),
                np.zeros((3, 4), dtype=np.uint8),
            ),
            "wl_true": (("wl",), wl),
        },
        coords={"x": x, "y": y, "wl": wl},
        attrs={
            "satellite": "S2A",
            "acquisition_date": "2024-06-15T10:30:00.000Z",
            "test_attribute": "native_metadata",
        },
    )

    l2_prod = xr.Dataset(
        {
            "Rrs": (
                ("wl", "y", "x"),
                np.full((3, 3, 4), 0.01, dtype=np.float32),
            ),
            "aot550": (
                ("y", "x"),
                np.full((3, 4), 0.1, dtype=np.float32),
            ),
            "BRDFg": (
                ("y", "x"),
                np.full((3, 4), 0.3, dtype=np.float32),
            ),
        },
        coords={"x": x, "y": y, "wl": wl},
    )

    cams_raster = xr.Dataset(
        {
            "aot_cams": (
                ("y", "x"),
                np.full((2, 2), 0.2, dtype=np.float32),
            ),
        },
        coords={
            "x": [105.0, 125.0],
            "y": [195.0, 175.0],
        },
    )

    gas_trans = MagicMock()
    gas_trans.Tg_tot_coarse = xr.DataArray(
        np.ones((2, 2, 3), dtype=np.float32),
        dims=("x", "y", "wl"),
        coords={
            "x": [105.0, 125.0],
            "y": [195.0, 175.0],
            "wl": wl,
        },
        name="Tg_tot_coarse",
    )

    prod = MagicMock()
    prod.raster = native_raster
    prod.sunglint_threshold = 0.2
    prod.ndwi_threshold = 0.0
    prod.vis_swir_index_threshold = 0.1
    prod.hcld_threshold = 0.003
    prod.dirdata = "/tmp/grsdata"
    prod.abs_gas_file = "/tmp/abs_gas.nc"
    prod.water_vapor_transmittance_file = "/tmp/water_vapor.nc"
    prod.bcirrus = None
    prod.cirrus = None
    prod.bwv = None
    prod.wv = None

    return prod, l2_prod, cams_raster, gas_trans


@pytest.fixture
def l2a_product(sample_products):
    """Create an initialized L2aProduct."""

    prod, l2_prod, cams_raster, gas_trans = sample_products

    return L2aProduct(
        prod=prod,
        l2_prod=l2_prod,
        cams=MagicMock(raster=cams_raster),
        gas_trans=gas_trans,
    )


def test_l2a_product_initialization(sample_products):
    """Test initialization of the L2aProduct object."""

    prod, l2_prod, cams_raster, gas_trans = sample_products

    product = L2aProduct(
        prod=prod,
        l2_prod=l2_prod,
        cams=MagicMock(raster=cams_raster),
        gas_trans=gas_trans,
    )

    assert product.prod is prod
    assert product.cams.raster is cams_raster
    assert product.gas_trans is gas_trans
    assert product.dem is None
    assert product.ancillary is not None
    assert product.wl_cirrus_band == 1375
    assert product.wl_wv_band == 945
    assert product.complevel == 5


def test_construct_l2a_main_product_variables(l2a_product):
    """Test that the main product contains the expected variables."""

    expected_variables = {
        "Rrs",
        "aot550",
        "BRDFg",
        "vza",
        "sza",
        "raa",
        "flags",
        "dem",
    }

    assert expected_variables.issubset(set(l2a_product.l2_prod.data_vars))


def test_construct_l2a_metadata(l2a_product):
    """Test that native metadata and processing parameters are copied."""

    attrs = l2a_product.l2_prod.attrs

    assert attrs["satellite"] == "S2A"
    assert attrs["acquisition_date"] == "2024-06-15T10:30:00.000Z"
    assert attrs["test_attribute"] == "native_metadata"
    assert attrs["sunglint_threshold"] == "0.2"
    assert attrs["ndwi_threshold"] == "0.0"
    assert attrs["vis_swir_index_threshold"] == "0.1"
    assert attrs["hcld_threshold"] == "0.003"
    assert attrs["dirdata"] == "/tmp/grsdata"
    assert attrs["abs_gas_file"] == "/tmp/abs_gas.nc"
    assert attrs["water_vapor_transmittance_file"] == "/tmp/water_vapor.nc"
    assert "GRSdriver_version" in attrs


def test_construct_l2a_does_not_modify_native_attributes(sample_products):
    """Test that native raster attributes remain unchanged."""

    prod, l2_prod, cams_raster, gas_trans = sample_products
    original_attrs = dict(prod.raster.attrs)

    L2aProduct(
        prod=prod,
        l2_prod=l2_prod,
        cams=MagicMock(raster=cams_raster),
        gas_trans=gas_trans,
    )

    assert prod.raster.attrs == original_attrs


def test_construct_l2a_extracts_angles(l2a_product):
    """Test that geometry angles are extracted correctly."""

    ds = l2a_product.l2_prod

    np.testing.assert_allclose(ds["vza"].values, 10.0)
    np.testing.assert_allclose(ds["sza"].values, 30.0)
    np.testing.assert_allclose(ds["raa"].values, 20.0)

    assert ds["vza"].dims == ("y", "x")
    assert ds["sza"].dims == ("y", "x")
    assert ds["raa"].dims == ("y", "x")


def test_construct_l2a_creates_empty_dem_when_missing(l2a_product):
    """Test that a missing DEM is replaced with a NaN-filled array."""

    dem = l2a_product.l2_prod["dem"]

    assert dem.dtype == np.float32
    assert dem.dims == ("y", "x")
    assert bool(dem.isnull().all())


def test_construct_l2a_uses_provided_dem(sample_products):
    """Test that the provided DEM is included in the main product."""

    prod, l2_prod, cams_raster, gas_trans = sample_products

    dem = xr.DataArray(
        np.full((3, 4), 150.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": l2_prod.x, "y": l2_prod.y},
        name="dem",
    )

    product = L2aProduct(
        prod=prod,
        l2_prod=l2_prod,
        cams=MagicMock(raster=cams_raster),
        gas_trans=gas_trans,
        dem=dem,
    )

    np.testing.assert_array_equal(
        product.l2_prod["dem"].values,
        dem.values,
    )


def test_construct_l2a_ancillary_variables(l2a_product):
    """Test that ancillary variables use coarse-resolution dimensions."""

    ancillary = l2a_product.ancillary

    assert "aot_cams" in ancillary
    assert "Tg_tot_coarse" in ancillary
    assert "xc" in ancillary.dims
    assert "yc" in ancillary.dims
    assert "x" not in ancillary.dims
    assert "y" not in ancillary.dims
    assert "wl_all" in ancillary.dims


def test_scrub_attrs_removes_conflicting_attributes():
    """Test that conflicting variable attributes are removed."""

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((2, 2), dtype=np.float32),
                {
                    "grid_mapping": "spatial_ref",
                    "scale_factor": 0.00001,
                    "_FillValue": -32768,
                    "add_offset": 0.3,
                    "units": "sr-1",
                },
            ),
        }
    )

    L2aProduct._scrub_attrs(ds)

    for attr in (
        "grid_mapping",
        "scale_factor",
        "_FillValue",
        "add_offset",
    ):
        assert attr not in ds["Rrs"].attrs

    assert ds["Rrs"].attrs["units"] == "sr-1"


def test_bool_attrs_to_int_converts_dataset_and_variable_attributes():
    """Test that boolean attributes are converted to integers."""

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((2, 2)),
                {"valid": True, "invalid": False},
            ),
        },
        attrs={
            "flag_a": True,
            "flag_b": False,
            "description": "test",
        },
    )

    L2aProduct._bool_attrs_to_int(ds)

    assert ds.attrs["flag_a"] == 1
    assert ds.attrs["flag_b"] == 0
    assert type(ds.attrs["flag_a"]) is int
    assert ds["Rrs"].attrs["valid"] == 1
    assert ds["Rrs"].attrs["invalid"] == 0
    assert type(ds["Rrs"].attrs["valid"]) is int
    assert ds.attrs["description"] == "test"


def test_prepare_export_datacube(l2a_product):
    """Test preparation of the datacube export."""

    ds, encoding = l2a_product._prepare_export(snap_compliant=False)

    assert ds.attrs["metadata_profile"] == "datacube"
    assert "Rrs" in ds.data_vars
    assert "Rrs" in encoding
    assert encoding["Rrs"]["dtype"] == "int16"
    assert encoding["Rrs"]["scale_factor"] == 0.00001
    assert encoding["Rrs"]["add_offset"] == 0.3
    assert encoding["Rrs"]["_FillValue"] == -32768
    assert "aot550" in encoding
    assert "BRDFg" in encoding
    assert "grid_mapping" not in ds["Rrs"].attrs


def test_prepare_export_snap_compliant(l2a_product):
    """Test preparation of the SNAP-compatible export."""

    ds, encoding = l2a_product._prepare_export(snap_compliant=True)

    assert ds.attrs["metadata_profile"] == "beam"
    assert ds.attrs["auto_grouping"] == "Rrs:Rrs_g"
    assert "Rrs" not in ds.data_vars

    for wavelength in (490, 560, 665):
        variable = f"Rrs_{wavelength}"

        assert variable in ds.data_vars
        assert variable in encoding
        assert encoding[variable]["dtype"] == "int16"
        assert encoding[variable]["scale_factor"] == 0.00001
        assert encoding[variable]["add_offset"] == 0.3

        attrs = ds[variable].attrs
        assert attrs["Unit"] == "sr-1"
        assert attrs["units"] == "sr-1"
        assert attrs["radiation_wavelength"] == wavelength
        assert attrs["radiation_wavelength_unit"] == "nm"
        assert attrs["wavelength"] == wavelength


def test_prepare_export_does_not_modify_original_product(l2a_product):
    """Test that export preparation does not modify the original dataset."""

    original_attrs = dict(l2a_product.l2_prod.attrs)
    original_rrs_attrs = dict(l2a_product.l2_prod["Rrs"].attrs)

    l2a_product._prepare_export(snap_compliant=False)

    assert l2a_product.l2_prod.attrs == original_attrs
    assert l2a_product.l2_prod["Rrs"].attrs == original_rrs_attrs


@pytest.mark.parametrize(
    "output_path, ext, expected_basename, expected_main",
    [
        ("scene", ".nc", "scene", "scene/scene.nc"),
        ("scene.nc", ".nc", "scene", "scene.nc"),
        ("scene.zarr", ".zarr", "scene", "scene.zarr"),
        ("scene.nc", ".zarr", "scene", "scene.nc"),
    ],
)
def test_output_names(
    tmp_path,
    output_path,
    ext,
    expected_basename,
    expected_main,
):
    """Test output path resolution and directory creation."""

    path = tmp_path / output_path

    directory, basename, main = L2aProduct._output_names(
        str(path),
        ext,
    )

    assert basename == expected_basename
    assert main == str(tmp_path / expected_main)
    assert (tmp_path / directory).is_dir()



def test_output_names_without_extension(tmp_path):
    """Test legacy output path resolution without an extension."""

    output_path = tmp_path / "scene"

    directory, basename, main = L2aProduct._output_names(
        str(output_path),
        None,
    )

    assert directory == str(output_path)
    assert basename == "scene"
    assert main == str(output_path / "scene")
    assert output_path.is_dir()


def test_export_defaults_to_netcdf(l2a_product, monkeypatch):
    """Test that export defaults to NetCDF when no format is specified."""

    mock_export = MagicMock(return_value=None)
    monkeypatch.setattr(l2a_product, "export_to_netcdf", mock_export)

    result = l2a_product.export("scene")

    mock_export.assert_called_once_with("scene")
    assert result is None


@pytest.mark.parametrize(
    "path, expected_format",
    [
        ("scene.nc", "netcdf"),
        ("scene.nc4", "netcdf"),
        ("scene.cdf", "netcdf"),
        ("scene.zarr", "zarr"),
        ("scene", "netcdf"),
        ("scene.tif", "netcdf"),
    ],
)
def test_export_detects_format_from_path(
    l2a_product,
    monkeypatch,
    path,
    expected_format,
):
    """Test that export selects the format from the path extension."""

    mock_netcdf = MagicMock(return_value="netcdf_result")
    mock_zarr = MagicMock(return_value="zarr_result")

    monkeypatch.setattr(l2a_product, "export_to_netcdf", mock_netcdf)
    monkeypatch.setattr(l2a_product, "export_to_zarr", mock_zarr)

    result = l2a_product.export(path)

    if expected_format == "netcdf":
        mock_netcdf.assert_called_once_with(path)
        mock_zarr.assert_not_called()
        assert result == "netcdf_result"
    else:
        mock_zarr.assert_called_once_with(path)
        mock_netcdf.assert_not_called()
        assert result == "zarr_result"


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_export_respects_explicit_format(l2a_product, monkeypatch, fmt):
    """Test that an explicit format determines the export method."""

    mock_netcdf = MagicMock(return_value=None)
    mock_zarr = MagicMock(return_value="store_path")

    monkeypatch.setattr(l2a_product, "export_to_netcdf", mock_netcdf)
    monkeypatch.setattr(l2a_product, "export_to_zarr", mock_zarr)

    result = l2a_product.export("scene", fmt=fmt, snap_compliant=True)

    if fmt == "netcdf":
        mock_netcdf.assert_called_once_with("scene", snap_compliant=True)
        mock_zarr.assert_not_called()
        assert result is None
    else:
        mock_zarr.assert_called_once_with("scene", snap_compliant=True)
        mock_netcdf.assert_not_called()
        assert result == "store_path"


def test_export_passes_kwargs(l2a_product, monkeypatch):
    """Test that export forwards keyword arguments to the selected method."""

    mock_export = MagicMock(return_value="store_path")
    monkeypatch.setattr(l2a_product, "export_to_zarr", mock_export)

    result = l2a_product.export(
        "scene.zarr",
        chunks={"x": 128, "y": 128},
        pyramids=False,
        compression=None,
    )

    mock_export.assert_called_once_with(
        "scene.zarr",
        chunks={"x": 128, "y": 128},
        pyramids=False,
        compression=None,
    )
    assert result == "store_path"


@pytest.mark.parametrize("fmt", ["hdf5", "nc", "ZARR", "", 123])
def test_export_rejects_unknown_format(l2a_product, fmt):
    """Test that export rejects unsupported format names."""

    with pytest.raises(ValueError, match="unknown format"):
        l2a_product.export("scene", fmt=fmt)


@pytest.mark.parametrize(
    "path, fmt",
    [
        ("scene.nc", "zarr"),
        ("scene.zarr", "netcdf"),
        ("scene.nc4", "zarr"),
        ("scene.cdf", "zarr"),
    ],
)
def test_export_rejects_conflicting_format_and_extension(
    l2a_product,
    path,
    fmt,
):
    """Test that export rejects conflicting formats and extensions."""

    with pytest.raises(ValueError, match="contradicts the extension"):
        l2a_product.export(path, fmt=fmt)


def test_export_to_netcdf_rejects_remote_paths(l2a_product):
    """Test that NetCDF export rejects remote URLs."""

    with pytest.raises(ValueError, match="NetCDF can only be written to a local path"):
        l2a_product.export_to_netcdf("s3://bucket/scene.nc")


def test_export_to_netcdf_writes_main_and_ancillary_files(
    l2a_product, tmp_path, monkeypatch
):
    calls = []

    def fake_to_netcdf(self, path, *args, **kwargs):
        calls.append({
            "path": str(path),
            "args": args,
            "kwargs": kwargs,
        })

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    l2a_product.export_to_netcdf(str(tmp_path / "scene.nc"))

    assert len(calls) == 2

    main_call, ancillary_call = calls

    assert main_call["path"] == str(tmp_path / "scene.nc")
    assert ancillary_call["path"] == str(tmp_path / "scene_anc.nc")

    assert main_call["kwargs"]["engine"] == mod.NETCDF_ENGINE
    assert ancillary_call["kwargs"]["engine"] == mod.NETCDF_ENGINE

    assert main_call["kwargs"]["encoding"]
    assert ancillary_call["kwargs"]["encoding"]

    assert all(
        encoding["zlib"] is True
        and encoding["complevel"] == l2a_product.complevel
        for encoding in main_call["kwargs"]["encoding"].values()
    )

    assert all(
        encoding["zlib"] is True
        and encoding["complevel"] == l2a_product.complevel
        for encoding in ancillary_call["kwargs"]["encoding"].values()
    )



def test_export_to_netcdf_adds_compression_options(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that NetCDF encodings receive the configured compression level."""

    calls = []

    def fake_to_netcdf(self, path, *args, **kwargs):
        calls.append({
            "path": str(path),
            "encoding": kwargs.get("encoding", {}),
            "engine": kwargs.get("engine"),
        })

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)
    l2a_product.complevel = 7

    l2a_product.export_to_netcdf(str(tmp_path / "scene.nc"))
    assert len(calls) == 2

    for call in calls:
        assert call["engine"] == mod.NETCDF_ENGINE

        for encoding in call["encoding"].values():
            assert encoding["zlib"] is True
            assert encoding["complevel"] == l2a_product.complevel


def test_export_to_netcdf_removes_existing_files(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that existing NetCDF outputs are removed before writing."""

    output_path = tmp_path / "scene.nc"
    ancillary_path = tmp_path / "scene_anc.nc"

    output_path.touch()
    ancillary_path.touch()

    written_files = []

    def fake_to_netcdf(self, path, *args, **kwargs):
        written_files.append(str(path))

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    l2a_product.export_to_netcdf(str(output_path))

    assert str(output_path) in written_files
    assert str(ancillary_path) in written_files


def test_export_to_netcdf_returns_none(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that NetCDF export returns None."""

    def fake_to_netcdf(self, path, *args, **kwargs):
        pass

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    result = l2a_product.export_to_netcdf(str(tmp_path / "scene.nc"))

    assert result is None

def test_export_to_zarr_rejects_remote_paths_without_zarr_extension(
    l2a_product,
    monkeypatch,
):
    """Test that remote Zarr URLs require a .zarr extension."""

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            xr.Dataset(
                {
                    "Rrs": (
                        ("y", "x"),
                        np.ones((4, 4), dtype=np.float32),
                    ),
                },
                coords={"x": range(4), "y": range(4)},
            ),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    with pytest.raises(ValueError, match="has to end with '.zarr'"):
        l2a_product.export_to_zarr(
            "s3://bucket/scene",
            pyramids=False,
        )


def test_export_to_zarr_returns_store_path(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that Zarr export returns the resolved store path."""

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(mod, "_compression_encoding", lambda *args: {})
    monkeypatch.setattr(mod, "_reset_encoding", lambda ds: None)
    monkeypatch.setattr(mod, "_load_coords", lambda ds: ds)
    monkeypatch.setattr(mod, "_clear_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_consolidate", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        mod,
        "_xarray_supports_shards",
        lambda: False,
    )

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
        },
        coords={"x": range(4), "y": range(4)},
        attrs={"title": "test"},
    )

    ancillary = xr.Dataset(
        {
            "aot_cams": (
                ("yc", "xc"),
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        coords={"xc": range(2), "yc": range(2)},
    )

    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            ds.copy(),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    monkeypatch.setattr(l2a_product, "_scrub_attrs", lambda ds: None)

    l2a_product.ancillary = ancillary

    write_calls = []

    def fake_to_zarr(self, store, **kwargs):
        write_calls.append((store, kwargs))
        return MagicMock(name="delayed_write")

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
    monkeypatch.setattr(
        "dask.compute",
        lambda *args, **kwargs: args,
    )

    result = l2a_product.export_to_zarr(
        str(tmp_path / "scene.zarr"),
        pyramids=False,
        sharding=False,
        consolidate=False,
        compression=None,
    )

    assert result == str(tmp_path / "scene.zarr")
    assert len(write_calls) == 3

    stores = [store for store, _ in write_calls]
    assert all(store == result for store in stores)

    groups = [kwargs.get("group") for _, kwargs in write_calls]
    assert groups == [None, "0", "ancillary"]


def test_export_to_zarr_disables_pyramids(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that disabling pyramids writes only the full-resolution group."""

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(mod, "_compression_encoding", lambda *args: {})
    monkeypatch.setattr(mod, "_reset_encoding", lambda ds: None)
    monkeypatch.setattr(mod, "_load_coords", lambda ds: ds)
    monkeypatch.setattr(mod, "_clear_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_xarray_supports_shards", lambda: False)

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
        },
        coords={"x": range(4), "y": range(4)},
    )

    ancillary = xr.Dataset(
        {
            "aot_cams": (
                ("yc", "xc"),
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        coords={"xc": range(2), "yc": range(2)},
    )

    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            ds.copy(),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    monkeypatch.setattr(l2a_product, "_scrub_attrs", lambda ds: None)
    l2a_product.ancillary = ancillary

    write_calls = []

    def fake_to_zarr(self, store, **kwargs):
        write_calls.append((store, kwargs))
        return MagicMock(name="delayed_write")

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
    monkeypatch.setattr("dask.compute", lambda *args, **kwargs: args)

    l2a_product.export_to_zarr(
        str(tmp_path / "scene.zarr"),
        pyramids=False,
        sharding=False,
        consolidate=False,
        compression=None,
    )

    groups = [kwargs.get("group") for _, kwargs in write_calls]

    assert groups == [None, "0", "ancillary"]


def test_export_to_zarr_forwards_storage_options(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that storage options are forwarded to Zarr writes."""

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(mod, "_compression_encoding", lambda *args: {})
    monkeypatch.setattr(mod, "_reset_encoding", lambda ds: None)
    monkeypatch.setattr(mod, "_load_coords", lambda ds: ds)
    monkeypatch.setattr(mod, "_clear_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_xarray_supports_shards", lambda: False)

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
        },
        coords={"x": range(4), "y": range(4)},
    )

    ancillary = xr.Dataset(
        {
            "aot_cams": (
                ("yc", "xc"),
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        coords={"xc": range(2), "yc": range(2)},
    )

    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            ds.copy(),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    monkeypatch.setattr(l2a_product, "_scrub_attrs", lambda ds: None)
    l2a_product.ancillary = ancillary

    write_calls = []

    def fake_to_zarr(self, store, **kwargs):
        write_calls.append(kwargs)
        return MagicMock(name="delayed_write")

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
    monkeypatch.setattr("dask.compute", lambda *args, **kwargs: args)

    storage_options = {"anon": True}

    l2a_product.export_to_zarr(
        str(tmp_path / "scene.zarr"),
        pyramids=False,
        sharding=False,
        consolidate=False,
        compression=None,
        storage_options=storage_options,
    )

    assert len(write_calls) == 3

    for kwargs in write_calls:
        assert kwargs["storage_options"] == storage_options


def test_export_to_zarr_consolidates_metadata(
    l2a_product,
    tmp_path,
    monkeypatch,
):
    """Test that Zarr metadata consolidation is called after writing."""

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(mod, "_compression_encoding", lambda *args: {})
    monkeypatch.setattr(mod, "_reset_encoding", lambda ds: None)
    monkeypatch.setattr(mod, "_load_coords", lambda ds: ds)
    monkeypatch.setattr(mod, "_clear_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_xarray_supports_shards", lambda: False)

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
        },
        coords={"x": range(4), "y": range(4)},
    )

    ancillary = xr.Dataset(
        {
            "aot_cams": (
                ("yc", "xc"),
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        coords={"xc": range(2), "yc": range(2)},
    )

    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            ds.copy(),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    monkeypatch.setattr(l2a_product, "_scrub_attrs", lambda ds: None)
    l2a_product.ancillary = ancillary

    monkeypatch.setattr(
        xr.Dataset,
        "to_zarr",
        lambda self, store, **kwargs: MagicMock(name="delayed_write"),
    )
    monkeypatch.setattr("dask.compute", lambda *args, **kwargs: args)

    mock_consolidate = MagicMock()
    monkeypatch.setattr(mod, "_consolidate", mock_consolidate)

    store = str(tmp_path / "scene.zarr")

    result = l2a_product.export_to_zarr(
        store,
        pyramids=False,
        sharding=False,
        consolidate=True,
        compression=None,
    )

    assert result == store
    mock_consolidate.assert_called_once_with(
        store,
        None,
        2,
        ["0", "ancillary"],
    )


def test_export_to_zarr_logs_consolidation_failure(
    l2a_product,
    tmp_path,
    monkeypatch,
    caplog,
):
    """Test that metadata consolidation failures are logged without failing the export."""

    import logging

    monkeypatch.setattr(mod, "_zarr_major", lambda: 2)
    monkeypatch.setattr(mod, "_compression_encoding", lambda *args: {})
    monkeypatch.setattr(mod, "_reset_encoding", lambda ds: None)
    monkeypatch.setattr(mod, "_load_coords", lambda ds: ds)
    monkeypatch.setattr(mod, "_clear_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_xarray_supports_shards", lambda: False)

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            ),
        },
        coords={"x": range(4), "y": range(4)},
    )

    ancillary = xr.Dataset(
        {
            "aot_cams": (
                ("yc", "xc"),
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        coords={"xc": range(2), "yc": range(2)},
    )

    monkeypatch.setattr(
        l2a_product,
        "_prepare_export",
        lambda snap_compliant=False: (
            ds.copy(),
            {"Rrs": {"dtype": "int16"}},
        ),
    )

    monkeypatch.setattr(l2a_product, "_scrub_attrs", lambda ds: None)
    l2a_product.ancillary = ancillary

    monkeypatch.setattr(
        xr.Dataset,
        "to_zarr",
        lambda self, store, **kwargs: MagicMock(name="delayed_write"),
    )
    monkeypatch.setattr("dask.compute", lambda *args, **kwargs: args)

    monkeypatch.setattr(
        mod,
        "_consolidate",
        MagicMock(side_effect=RuntimeError("consolidation failed")),
    )

    with caplog.at_level(logging.WARNING):
        result = l2a_product.export_to_zarr(
            str(tmp_path / "scene.zarr"),
            pyramids=False,
            sharding=False,
            consolidate=True,
            compression=None,
        )

    assert result == str(tmp_path / "scene.zarr")
    assert "could not consolidate zarr metadata" in caplog.text
