import netCDF4
from speedy.config import DEFAULT_DATA_DIR, TEMPERATURE_FILENAME
import os
import numpy as np
import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt


def get_indexes(lats, lons, input_lat, input_lon):
    dist_sq = (lats - input_lat) ** 2 + (lons - input_lon) ** 2
    minindex_flattened = dist_sq.argmin()
    return np.unravel_index(minindex_flattened, lats.shape)


def get_temperature(input_lon: float, input_lat: float, data_dir: str = DEFAULT_DATA_DIR) -> float:
    f = netCDF4.Dataset(os.path.join(data_dir, TEMPERATURE_FILENAME))

    temp = f.variables["thetao_mean"]
    latvals = f.variables["lat"][:]
    lonvals = f.variables["lon"][:]

    lat_index = np.abs(latvals - input_lat).argmin()
    lon_index = np.abs(lonvals - input_lon).argmin()
    temperature = temp[1, lat_index, lon_index]

    if np.ma.is_masked(temperature):
        return None
    else:
        return temperature.data


def kde_cdf(x, kde, samples):
    return kde.integrate_box_1d(-np.inf, x)


def calculate_thermal_envelope(temperatures: np.array):
    kde_bandwidth = 0.5
    percentiles = (1, 99)
    kde = gaussian_kde(temperatures, bw_method=kde_bandwidth)
    resampled = kde.resample(1000)
    percentiles = (np.percentile(resampled, percentiles[0]), np.percentile(resampled, percentiles[1]))
    print(percentiles)
