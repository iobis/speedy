import netCDF4
from speedy.config import DEFAULT_DATA_DIR, TEMPERATURE_FILENAME
import os
import numpy as np
import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import geopandas
import xarray
import geopandas as gpd
import logging


def get_indexes(lats, lons, input_lat, input_lon):
    dist_sq = (lats - input_lat) ** 2 + (lons - input_lon) ** 2
    minindex_flattened = dist_sq.argmin()
    return np.unravel_index(minindex_flattened, lats.shape)


def get_temperature(input_lon: float, input_lat: float, lonvals, latvals, temp) -> float:
    lat_index = np.abs(latvals - input_lat).argmin()
    lon_index = np.abs(lonvals - input_lon).argmin()
    temperature = temp[1, lat_index, lon_index]
    if np.ma.is_masked(temperature):
        return None
    else:
        return temperature.data


def kde_cdf(x, kde, samples):
    return kde.integrate_box_1d(-np.inf, x)


def calculate_thermal_suitability(distribution: geopandas.GeoDataFrame, thetao, resolution: int, kde_bandwidth: float = 0.5, data_dir: str = DEFAULT_DATA_DIR):
    logging.debug("Sampling temperature data")
    temperatures = []
    for coord in zip(distribution["geometry"].x, distribution["geometry"].y):
        temp_val = thetao.sel(lat=coord[1], lon=coord[0], method="nearest").item()
        if not np.isnan(temp_val):
            temperatures.append(temp_val)
    temperatures = np.array(temperatures)
    
    if len(temperatures) < 3:
        return None

    logging.debug("Calculating KDE")
    kde = gaussian_kde(temperatures, bw_method=kde_bandwidth)

    original_shape = thetao.shape
    temp_flat = thetao.values.flatten()
    valid_mask = ~np.isnan(temp_flat)
    temp_valid = temp_flat[valid_mask]
    kde_values = np.full(temp_flat.shape, np.nan)

    logging.debug("Evaluating KDE")
    kde_values[valid_mask] = kde.evaluate(temp_valid)
    kde_reshaped = kde_values.reshape(original_shape)

    suitability = xarray.DataArray(
        kde_reshaped,
        dims=thetao.dims,
        coords=thetao.coords
    )

    suitability = suitability / suitability.max()
    return suitability


def calculate_thermal_envelope(distribution: geopandas.GeoDataFrame, resolution: int, data_dir: str = DEFAULT_DATA_DIR):

    if distribution.empty:
        return None

    kde_bandwidth = 0.5
    percentiles = (1, 99)
    temperature_file = os.path.join(data_dir, TEMPERATURE_FILENAME)

    # TODO: handle missing data (coastal)
    # TODO: convert to xarray?
    f = netCDF4.Dataset(temperature_file)
    temp = f.variables["thetao_mean"]
    lat_vals = f.variables["lat"][:]
    lon_vals = f.variables["lon"][:]

    temperatures = [get_temperature(coord[0], coord[1], lon_vals, lat_vals, temp) for coord in zip(distribution["geometry"].x , distribution["geometry"].y)]
    temperatures = np.array([t.item() for t in temperatures if t is not None])

    if len(temperatures) < 3:
        return None

    kde = gaussian_kde(temperatures, bw_method=kde_bandwidth)
    resampled = kde.resample(10000)
    percentiles = (np.percentile(resampled, percentiles[0]), np.percentile(resampled, percentiles[1]))

    # mask

    xds = xarray.open_dataset(temperature_file, engine="netcdf4")
    thetao = xds["thetao_mean"].sel(time="2010-01-01")
    
    # Rescale to 1/4 resolution for faster processing (0.05° -> 0.1°)
    logging.info("Downscaling temperature data to 1/4 resolution for performance")
    thetao = thetao.coarsen(lat=2, lon=2, boundary='trim').mean()
    
    envelope = thetao.where(thetao >= percentiles[0]).where(thetao <= percentiles[1])

    df = envelope.to_dataframe().reset_index()
    df = df[df["thetao_mean"].notna()]
    df = df.h3.geo_to_h3(resolution, lat_col="lat", lng_col="lon", set_index=False)
    df["h3"] = df[f"h3_0{resolution}"]
    df = df[["h3"]].drop_duplicates()

    return df
