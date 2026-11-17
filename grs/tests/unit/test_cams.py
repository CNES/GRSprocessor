"""CAMS data (grs.cams.CamsProduct) on the grid of the test L1C subset."""

import numpy as np
import pytest
import xarray as xr

from grs import CamsProduct
from grs.product import Product


@pytest.fixture(scope='module')
def prod(l1c_path):
    return Product(xr.open_dataset(l1c_path))


@pytest.fixture(scope='module')
def cams(prod, cams_path):
    cams = CamsProduct(prod.raster, cams_file=cams_path)
    cams.load()
    return cams


def test_variables(cams):
    expected = {'u10', 'v10', 't2m', 'msl', 'sp', 'tcwv', 'gtco3', 'tcno2', 'tcco', 'tc_ch4',
                'aod469', 'aod550', 'aod670', 'aod865', 'aod1240'}
    assert expected <= set(cams.variables)
    assert expected <= set(cams.raster.data_vars)


def test_grid_covers_the_image(cams, prod):
    # coarse grid of the CAMS data spanning the image (same corners as the image grid)
    assert cams.raster.x.values[[0, -1]] == pytest.approx([prod.xmin, prod.xmax])
    assert cams.raster.y.values[[0, -1]] == pytest.approx([prod.ymax, prod.ymin])
    assert cams.raster.rio.crs == prod.raster.rio.crs


def test_values_of_the_day(cams):
    # winter morning over the Camargue: mistral (northerly wind) and about 1014 hPa
    assert float(cams.raster.v10.mean()) == pytest.approx(-6.2, abs=0.5)
    assert float(cams.raster.t2m.mean()) == pytest.approx(281.86, abs=0.1)
    assert float(cams.raster.sp.mean()) == pytest.approx(101420, abs=100)
    assert float(cams.raster.msl.mean()) == pytest.approx(101707, abs=20)
    assert float(cams.raster.tcwv.mean()) == pytest.approx(10.5, abs=0.5)


def test_aod_spectrum(cams):
    aod = cams.cams_aod
    assert aod.sizes['wl'] == 5
    assert np.all(aod.values > 0)
    # aerosol optical depth decreasing with the wavelength
    spectrum = aod.mean(['x', 'y']).sortby('wl').values
    assert np.all(np.diff(spectrum) < 0)


def _grid(lon):
    return xr.Dataset({'u10': (('latitude', 'longitude'), np.tile(np.asarray(lon, float), (3, 1)))},
                      coords={'latitude': [1., 0., -1.], 'longitude': lon})


def test_longitudes_of_the_image():
    from grs.cams import _select_longitudes
    cams, lonmin, lonmax = _select_longitudes(_grid(np.arange(0, 360, 0.5)), 12.4, 12.6)
    assert (lonmin, lonmax) == (12.4, 12.6)
    assert cams.longitude.values[[0, -1]] == pytest.approx([11.5, 13.5])
    assert np.all(cams.u10.values == cams.longitude.values)


def test_longitudes_west_of_greenwich_in_0_360():
    from grs.cams import _select_longitudes
    cams, lonmin, lonmax = _select_longitudes(_grid(np.arange(0, 360, 0.5)), -5.2, -4.8)
    assert (lonmin, lonmax) == pytest.approx((354.8, 355.2))
    assert cams.longitude.values[[0, -1]] == pytest.approx([354.0, 356.0])


def test_longitudes_across_greenwich():
    # columns on both sides of 0 in -180-180 deg, in increasing order, with their data
    from grs.cams import _select_longitudes
    cams, lonmin, lonmax = _select_longitudes(_grid(np.arange(0, 360, 0.5)), -0.1, 0.1)
    assert (lonmin, lonmax) == (-0.1, 0.1)
    lon = cams.longitude.values
    assert np.all(np.diff(lon) > 0)
    assert lon[[0, -1]] == pytest.approx([-1.0, 1.0])
    assert cams.u10.values[0] == pytest.approx(lon % 360)


def test_longitudes_of_a_grid_in_180():
    from grs.cams import _select_longitudes
    cams, lonmin, lonmax = _select_longitudes(_grid(np.arange(-180, 180, 0.5)), -0.1, 0.1)
    assert cams.longitude.values[[0, -1]] == pytest.approx([-1.0, 1.0])
