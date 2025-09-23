from speedy import Speedy
from speedy.temperature import calculate_thermal_envelope, calculate_thermal_suitability
from speedy.config import DEFAULT_DATA_DIR, MR_LDES_FILENAME, TEMPERATURE_URL, TEMPERATURE_FILENAME
import duckdb
import matplotlib.pyplot as plt
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import antimeridian
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from palettable.cartocolors.sequential import Teal_3, Magenta_5
import logging
from cProfile import Profile
from pstats import SortKey, Stats
import os
import xarray
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from threading import Lock


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


sp = Speedy(h3_resolution=7)
resolution = 3
density_max_rings = 50
density_sd = 1000
density_cutoff = 1e-8
density_plot_cutoff = 1e-4
suitability_kde_bandwidth = 2
suitability_coarsen = 4
suitability_plot_cutoff = 1e-3


def all_h3_cells(resolution: int) -> set[str]:
    cells = set()
    for r0 in h3.get_res0_cells():
        cells.update(h3.cell_to_children(r0, resolution))
    return cells


def sample_cell_mean(suitability, hcell: str) -> float:
    corners = h3.cell_to_boundary(hcell)
    vals = []
    for lat, lon in corners:
        v = suitability.sel(lat=lat, lon=lon, method="nearest").item()
        if not np.isnan(v):
            vals.append(v)
    if len(vals) == 0:
        latc, lonc = h3.cell_to_latlng(hcell)
        v = suitability.sel(lat=latc, lon=lonc, method="nearest").item()
        return v
    return float(np.mean(vals))


def get_density(dist_speedy):
    logging.debug("Calculating density")
    dist = dist_speedy.drop("geometry", axis=1)
    dist = dist.set_index("cell").h3.h3_to_parent(resolution).reset_index()
    dist["cell"] = dist[f"h3_0{resolution}"]
    dist = dist.drop(columns=f"h3_0{resolution}")
    conn = duckdb.connect()
    conn.register("dist", dist)
    dist = conn.execute("""
        select
            cell,
            max(source_obis) as source_obis,
            max(source_gbif) as source_gbif,
            sum(records) as records,
            min(min_year) as min_year,
            max(max_year) as max_year
        from dist group by cell
    """).fetchdf()
    cells = dist["cell"].tolist()
    density = sp.calculate_density(cells, max_rings=density_max_rings, sd=density_sd, density_cutoff=density_cutoff)
    density["density"] = density["density"] / density["density"].max()
    density = density.set_index("h3").h3.h3_to_geo_boundary().reset_index()
    return density


def transform_density_for_plotting(density):
    density_plot = density.copy(deep=True)
    density_plot["geometry"] = density_plot["geometry"].apply(antimeridian.fix_polygon)
    density_plot = density_plot[density_plot["density"] >= density_plot_cutoff]
    return density_plot


def get_suitability(dist_speedy, thetao):
    logging.debug("Calculating suitability")
    suitability = calculate_thermal_suitability(dist_speedy, thetao=thetao, resolution=resolution, kde_bandwidth=suitability_kde_bandwidth)
    h3_cells = sorted(all_h3_cells(resolution))
    values = [sample_cell_mean(suitability, h) for h in h3_cells]
    return pd.DataFrame({"h3": h3_cells, "suitability": values}).set_index("h3").h3.h3_to_geo_boundary().reset_index()


def transform_suitability_for_plotting(suitability):
    suitability_plot = suitability.copy(deep=True)
    suitability_plot["geometry"] = suitability_plot["geometry"].apply(antimeridian.fix_polygon)
    suitability_plot = suitability_plot[suitability_plot["suitability"] >= suitability_plot_cutoff]
    return suitability_plot


def generate_png(density_plot, suitability_plot, dist_speedy, filename):
    logging.debug("Generating PNG")
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300, subplot_kw={"projection": proj})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="black")

    den_min = float(density_plot["density"].min())
    den_max = float(density_plot["density"].max())
    den_norm = mpl.colors.Normalize(vmin=den_min, vmax=den_max)
    density_plot.plot(column="density", cmap=Teal_3.mpl_colormap, linewidth=0, alpha=0.6, ax=ax)
    den_sm = mpl.cm.ScalarMappable(cmap=Teal_3.mpl_colormap, norm=den_norm)
    den_sm._A = []
    cbar_den = fig.colorbar(den_sm, ax=ax, shrink=0.5, fraction=0.025, pad=0.04)
    cbar_den.set_label("Density")

    suit_min = float(suitability_plot["suitability"].min())
    suit_max = float(suitability_plot["suitability"].max())
    suit_norm = mpl.colors.Normalize(vmin=suit_min, vmax=suit_max)
    suitability_plot.plot(column="suitability", cmap=Magenta_5.mpl_colormap, linewidth=0, alpha=0.4, ax=ax)
    suit_sm = mpl.cm.ScalarMappable(cmap=Magenta_5.mpl_colormap, norm=suit_norm)
    suit_sm._A = []
    cbar_suit = fig.colorbar(suit_sm, ax=ax, shrink=0.5, fraction=0.025, pad=0.03)
    cbar_suit.set_label("Suitability")

    dist_points = dist_speedy.dropna(subset=["geometry"]).copy()
    if not dist_points.empty and not all(dist_points.geom_type == "Point"):
        dist_points["geometry"] = dist_points.geometry.centroid
    if not dist_points.empty:
        dist_points.plot(ax=ax, markersize=3, color="black", alpha=0.7, linewidth=0)

    ax.set_axis_off()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def process_species(aphiaid, thetao):
    output_path = f"speedy_output/edna_qc/parquet/{aphiaid}.parquet"
    output_path_png = f"speedy_output/edna_qc/images/{aphiaid}.png"

    if not os.path.exists(output_path) or not os.path.exists(output_path_png):
        logging.debug(f"Processing {aphiaid}")
        logging.debug("Reading distribution data")
        dist_speedy = sp.read_distribution_grid(aphiaid)
        density = get_density(dist_speedy)
        suitability = get_suitability(dist_speedy, thetao)

        density_selected = density[["h3", "density"]].copy()
        suitability_selected = suitability[["h3", "suitability"]].copy()
        suitability_selected = suitability_selected.dropna(subset=["suitability"])
        joined_df = density_selected.merge(suitability_selected, on="h3", how="left")
        joined_df["aphiaid"] = aphiaid
        logging.debug(f"Saving data to {output_path}")
        joined_df.to_parquet(output_path, index=False)

        # ax = suitability.plot.imshow(cmap="magma", vmin=0, vmax=1, robust=True)
        # ax.figure.savefig(f"speedy_output/thermal_{aphiaid}.png", dpi=300, bbox_inches="tight")
        # sp.export_map(f"speedy_output/map_{aphiaid}.html", density=density_plot, distribution=dist_speedy, suitability=suitability_plot, density_palette=Teal_3, suitability_palette=Magenta_5)
        density_plot = transform_density_for_plotting(density)
        suitability_plot = transform_suitability_for_plotting(suitability)
        logging.debug(f"Saving image to {output_path_png}")
        generate_png(density_plot, suitability_plot, dist_speedy, output_path_png)
        return f"Processed {aphiaid}"
    else:
        logging.debug(f"Output file {output_path} already exists, skipping data processing")
        return f"Skipped {aphiaid} (already exists)"


class ProgressTracker:
    def __init__(self, total_tasks):
        self.completed = 0
        self.total = total_tasks

    def callback(self, future):
        self.completed += 1
        percent = (self.completed / self.total) * 100
        aphiaid = future.aphiaid
        try:
            result = future.result()
            logging.info(f"✅ Completed {self.completed}/{self.total} ({percent:.1f}%) - AphiaID: {aphiaid}")
        except Exception as e:
            logging.error(f"❌ Failed {self.completed}/{self.total} ({percent:.1f}%) - AphiaID: {aphiaid}: {e}")


if __name__ == "__main__":
    aphiaids = sp.read_aphiaids()
    logging.info(f"Found {len(aphiaids)} aphiaids")
    logging.info("Opening temperature dataset")
    temperature_file = os.path.join(DEFAULT_DATA_DIR, TEMPERATURE_FILENAME)
    xds = xarray.open_dataset(temperature_file, engine="netcdf4")
    thetao = xds["thetao_mean"].sel(time="2010-01-01")
    logging.info("Downscaling temperature data")
    thetao = thetao.coarsen(lat=suitability_coarsen, lon=suitability_coarsen, boundary="exact").mean()

    process_func = functools.partial(process_species, thetao=thetao)

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        for aphiaid in aphiaids:
            future = executor.submit(process_func, aphiaid)
            future.aphiaid = aphiaid
            futures.append(future)

        progress_tracker = ProgressTracker(len(futures))
        for future in futures:
            future.add_done_callback(progress_tracker.callback)


# TODO refactor: precalculate temperature layer to h3, evaluate kde
