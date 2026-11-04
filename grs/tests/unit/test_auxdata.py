import numpy as np
import pandas as pd
import pytest

from grs.auxdata import AuxData, SensorData

@pytest.fixture
def mock_grs_root(monkeypatch, tmp_path):
    """Provide a temporary GRS root directory for sensor configuration tests."""
    from grs import config

    monkeypatch.setattr(config, "grs_root", str(tmp_path), raising=False)
    return tmp_path

@pytest.fixture
def aux_data_files(monkeypatch, tmp_path):
    """Create temporary auxiliary data files."""
    sunglint_file = tmp_path / "sunglint.txt"
    rayleigh_file = tmp_path / "rayleigh.txt"

    sunglint_file.write_text(
        "wl mean\n"
        "400 1.1\n"
        "500 1.2\n"
        "600 1.3\n"
    )

    rayleigh_file.write_text(
        "\n" * 16
        + "400 0.20 0.03\n"
        + "500 0.10 0.02\n"
        + "600 0.05 0.01\n"
    )

    import grs.auxdata as auxdata_module

    monkeypatch.setattr(auxdata_module, "sunglint_eps_file", sunglint_file)
    monkeypatch.setattr(auxdata_module, "rayleigh_file", rayleigh_file)

    return sunglint_file, rayleigh_file


def test_auxdata_initialization(aux_data_files):
    """Test that auxiliary data are loaded correctly."""
    aux = AuxData()

    assert aux.rot is not None
    assert aux.sunglint_eps is not None
    assert aux.rot.size == 3
    assert aux.sunglint_eps.sizes["wl"] == 3
    assert aux.rot_pressure_ref == 101325


def test_auxdata_rayleigh_reference_pressure(aux_data_files):
    """Test the reference pressure used for Rayleigh optical thickness."""
    aux = AuxData()

    assert aux.rot_pressure_ref == 101325
    np.testing.assert_allclose(
        aux.rot.values,
        np.array([0.20, 0.10, 0.05]),
    )


def test_auxdata_wavelength_interpolation(aux_data_files):
    """Test interpolation of auxiliary data onto requested wavelengths."""
    aux = AuxData(wl=[450, 550])

    assert aux.rot.size == 2
    assert aux.sunglint_eps.sizes["wl"] == 2

    np.testing.assert_allclose(
        aux.rot.values,
        np.array([0.15, 0.075]),
    )

    np.testing.assert_allclose(
        aux.sunglint_eps.values,
        np.array([1.15, 1.25]),
    )


def test_auxdata_interpolation_at_original_wavelengths(aux_data_files):
    """Test interpolation at wavelengths present in the source data."""
    aux = AuxData(wl=[400, 600])

    np.testing.assert_allclose(
        aux.rot.values,
        np.array([0.20, 0.05]),
    )

    np.testing.assert_allclose(
        aux.sunglint_eps.values,
        np.array([1.1, 1.3]),
    )


@pytest.mark.parametrize(
    "sensor, expected_resolution, expected_bands",
    [
        ("S2A", 20, 11),
        ("S2B", 20, 11),
        ("LANDSAT_4", 30, 6),
        ("LANDSAT_5", 30, 6),
        ("LANDSAT_7", 30, 6),
        ("LANDSAT_8", 30, 8),
    ],
)
def test_sensordata_initialization(sensor, expected_resolution, expected_bands,mock_grs_root):
    """Test sensor configuration and band metadata."""
    data = SensorData(sensor)

    assert data.sensor == sensor
    assert data.resolution == expected_resolution
    assert len(data.indband) == expected_bands
    assert len(data.central_wavelength) == expected_bands
    assert len(data.rot) == expected_bands
    assert len(data.rg) == expected_bands
    assert len(data.angle_names) == expected_bands
    assert len(data.band_names) == expected_bands


@pytest.mark.parametrize(
    "sensor",
    ["S2A", "S2B", "LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
)
def test_sensordata_spectral_values_are_valid(sensor, mock_grs_root):
    """Test the validity of sensor spectral configuration values."""
    data = SensorData(sensor)

    assert np.all(np.isfinite(data.central_wavelength))
    assert np.all(np.asarray(data.central_wavelength) > 0)
    assert np.all(np.isfinite(data.rot))
    assert np.all(np.asarray(data.rot) > 0)
    assert np.all(np.isfinite(data.rg))
    assert np.all(np.asarray(data.rg) > 0)


@pytest.mark.parametrize(
    "sensor, expected_band_names",
    [
        ("S2A", ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")),
        ("S2B", ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")),
        (
            "LANDSAT_8",
            (
                "coastal_aerosol",
                "blue",
                "green",
                "panchromatic",
                "red",
                "near_infrared",
                "swir_1",
                "swir_2",
            ),
        ),
    ],
)
def test_sensordata_band_names(sensor, expected_band_names, mock_grs_root):
    """Test the configured band names for selected sensors."""
    data = SensorData(sensor)

    assert tuple(data.band_names) == expected_band_names


def test_sensordata_landsat_solar_irradiance(mock_grs_root):
    """Test solar irradiance availability for Landsat sensors."""
    for sensor in ("LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"):
        data = SensorData(sensor)

        assert data.solar_irr is not None
        assert len(data.solar_irr) == len(data.central_wavelength)


def test_sensordata_sentinel_solar_irradiance(mock_grs_root):
    """Test that Sentinel-2 solar irradiance is left unspecified."""
    for sensor in ("S2A", "S2B"):
        data = SensorData(sensor)

        assert data.solar_irr is None


def test_sensordata_sentinel_cirrus_configuration(mock_grs_root):
    """Test the cirrus configuration for Sentinel-2 sensors."""
    for sensor in ("S2A", "S2B"):
        data = SensorData(sensor)

        assert data.cirrus[0] == "B10"
        assert data.cirrus[1] > 0


def test_sensordata_landsat_cirrus_configuration(mock_grs_root):
    """Test the cirrus configuration for Landsat sensors."""
    expected = {
        "LANDSAT_4": "no",
        "LANDSAT_5": "no",
        "LANDSAT_7": "no",
        "LANDSAT_8": "cirrus",
    }

    for sensor, expected_band in expected.items():
        data = SensorData(sensor)

        assert data.cirrus[0] == expected_band


def test_sensordata_invalid_sensor(mock_grs_root):
    """Test that an unsupported sensor raises a KeyError."""
    with pytest.raises(KeyError):
        SensorData("INVALID_SENSOR")


def test_sensordata_ndwi_configuration(mock_grs_root):
    """Test the NDWI band indices and thresholds."""
    data = SensorData("S2A")

    assert data.NDWI_vis == 2
    assert data.NDWI_nir == 8
    assert data.NDWI_threshold == [-0.03, 1.0]

    assert data.NDWI_swir_nir == 8
    assert data.NDWI_swir_swir == 9
    assert data.NDWI_swir_threshold == [0.12, 2.0]


def test_sensordata_high_nir_configuration(mock_grs_root):
    """Test the high-NIR threshold configuration."""
    data = SensorData("S2A")

    assert data.high_nir[0] == 8
    assert data.high_nir[1] == pytest.approx(0.0275)


def test_sensordata_o2band_configuration(mock_grs_root):
    """Test the oxygen absorption band configuration."""
    for sensor in ("S2A", "S2B"):
        data = SensorData(sensor)

        assert data.O2band[0] == "B9"
        assert data.O2band[1] == pytest.approx(0.08)
        assert data.O2band[2] == pytest.approx(0.12)

    for sensor in ("LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"):
        data = SensorData(sensor)

        assert data.O2band is None


def test_sensordata_l1_flags_configuration(mock_grs_root):
    """Test the L1 flag names for Sentinel-2 and Landsat 8."""
    s2a = SensorData("S2A")

    assert s2a.cloud_flag == "opaque_clouds_10m"
    assert s2a.cirrus_flag == "cirrus_clouds_10m"
    assert s2a.shadow_flag == ""

    landsat8 = SensorData("LANDSAT_8")

    assert landsat8.cloud_flag == "cloud_confidence_high"
    assert landsat8.cirrus_flag == "cirrus_confidence_high"
    assert landsat8.shadow_flag == "cloud_shadow_confidence_high"
