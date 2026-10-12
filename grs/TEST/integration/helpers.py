from __future__ import annotations

import numpy as np
import xarray as xr

def _find_nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.abs(values - value).argmin())

def sample_product(
        product: xr.Dataset,
        variable: str,
        point: dict,
        wavelength: float | None = None,
        window: int=1,
) -> float:
    """Sample a product at the nearest pixel or over a square window.

        The reference point is expressed in the product CRS with ``x``/``y``.
        A point can alternatively be expressed as ``lon``/``lat`` when ``pyproj``
        is available.
    """

    data = product[variable]

    if wavelength is not None:
        if 'wl' not in data.dims:
            raise AssertionError(
                f"The variable {variable} doesn't have the dimension 'wl'."
            )
        data = data.sel(wl=wavelength, method="nearest")

    if "x" in point and "y" in point:
        x = float(point["x"])
        y = float(point["y"])
    elif "lon" in point and "lat" in point:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise AssertionError(
                "pyproj required to use an in situ point in lon/lat."
            ) from exc

        if product.rio.crs is None:
            raise AssertionError("The product doesn't have an exploitable CRS")

        transformer = Transformer.from_crs(
            "EPSG:4326", product.rio.crs, always_xy=True
        )
        x, y = transformer.transform(float(point["lon"]), float(point["lat"]))
    else:
        raise AssertionError("An in situ point should contains x/y or lon/lat.")

    if "x" not in data.dims or "y" not in data.dims:
        raise AssertionError(
            f'The variable {variable} can be resampled on x/y: {data.dims}'
        )

    ix = _find_nearest_index(product.x.values, x)
    iy = _find_nearest_index(product.y.values, y)

    if window <= 1:
        value = data.isel(x=ix, y=iy).values
    else:
        radius = window // 2
        subset = data.isel(
            x=slice(max(0, ix - radius), min(product.sizes["x"], ix + radius + 1)),
            y=slice(max(0, iy - radius), min(product.sizes["y"], iy + radius + 1)),
        )
        value = subset.mean(skipna=True).values

    value = np.asarray(value).squeeze()
    if value.size != 1:
        raise AssertionError(
            f"The resample of {variable} didn't produce a scalar value: {value.shape}"
        )

    value = float(value)
    if not np.isfinite(value):
        raise AssertionError(
            f"The resampled value of {variable} is NaN/Inf "
            f"on the reference point {point}."
        )
    return value


def get_hypernets_reflectance(hypernets: xr.Dataset):
    if "reflectance" not in hypernets:
        raise AssertionError(
            "The HYPERNETS file doesn't contain the 'reflectance' variable."
        )

    return hypernets["reflectance"]


def interpolate_hypernets_reflectance(
    hypernets: xr.Dataset,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """Rrs (sr-1) de HYPERNETS interpolée sur target_wavelengths.

    HYPERNETS fournit rho_w (sans dimension) : Rrs = rho_w / pi.
    Les longueurs d'onde hors de la plage mesurée valent NaN.
    """
    reflectance = get_hypernets_reflectance(hypernets)
    if "series" in reflectance.dims:
        reflectance = reflectance.median("series", skipna=True)

    wavelength = hypernets["wavelength"].values
    rrs = reflectance.values / np.pi

    valid = np.isfinite(wavelength) & np.isfinite(rrs)
    if valid.sum() < 2:
        raise AssertionError("Not enough valid HYPERNETS reflectance values.")

    return np.interp(
        target_wavelengths, wavelength[valid], rrs[valid],
        left=np.nan, right=np.nan,
    )

def get_hypernets_location(hypernets):
    lat = float(hypernets.attrs["site_latitude"])
    lon = float(hypernets.attrs["site_longitude"])

    return lon, lat