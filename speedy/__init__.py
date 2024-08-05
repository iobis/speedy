import re
import geopandas
import h3pandas
import logging
import pyworms
import pandas as pd
import os
import shapely
from h3pandas.util.shapely import polyfill
import h3
import duckdb
from haversine import haversine, Unit
from scipy.stats import norm
from lonboard import Map, SolidPolygonLayer, ScatterplotLayer
from lonboard.colormap import apply_continuous_cmap
from palettable.cartocolors.diverging import Temps_3
from palettable.cubehelix import classic_16, cubehelix1_16
import numpy as np
from speedy.temperature import get_temperature, calculate_thermal_envelope
from speedy.config import DEFAULT_DATA_DIR
import antimeridian


def normalize_density(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


class Speedy:

    def __init__(self, h3_resolution: int = 7, data_dir: str = DEFAULT_DATA_DIR, cache_marineregions: bool = True, cache_summary: bool = False, cache_density: bool = False, ignore_missing_wkt = True):

        self.h3_resolution = h3_resolution
        self.data_dir = data_dir
        self.cache_marineregions = cache_marineregions
        self.cache_summary = cache_summary
        self.cache_density = cache_density
        self.ignore_missing_wkt = ignore_missing_wkt

    def read_distribution_grid(self, aphiaid: int) -> geopandas.GeoDataFrame:
        logging.debug(f"Reading distribution data for https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphiaid}")
        filters = [("AphiaID", "==", aphiaid)]
        gdf = geopandas.read_parquet(os.path.join(self.data_dir, f"distributions_{self.h3_resolution}"), filters=filters)
        return gdf

    def get_worms_mrgids(self, aphiaid: int) -> list:
        res = pyworms.aphiaDistributionsByAphiaID(id=aphiaid)
        distribution = [{"mrgid": re.search("\\d+", entry["locationID"])[0], "establishmentMeans": entry["establishmentMeans"]} for entry in res]
        return distribution

    def index_geometry(self, geometry: geopandas.GeoSeries) -> pd.DataFrame:
        if isinstance(geometry, shapely.geometry.linestring.LineString):
            return None
        elif isinstance(geometry, shapely.geometry.point.Point):
            return None
        elif isinstance(geometry, shapely.geometry.polygon.Polygon):
            poly = polyfill(geometry, self.h3_resolution, geo_json=True)
            if len(poly) == 0:
                centroid = geometry.centroid
                poly = [h3.geo_to_h3(centroid.y, centroid.x, self.h3_resolution)]
            return pd.DataFrame(data={"h3": list(poly)})
        else:
            raise NotImplementedError(f"Type {type(geometry)} not supported")

    def index_wkt(self, wkt: str) -> pd.DataFrame:
        wkt = re.sub(r"<.*>", "", wkt).strip()
        if len(wkt) == 0:
            return None
        try:
            gs = geopandas.GeoSeries.from_wkt([wkt])
            # gs is length 1 but can contain a geometrycollection, so split
            assert len(gs) == 1
            if hasattr(gs[0], "geoms"):
                gs = geopandas.GeoSeries(gs[0].geoms)
        except Exception:
            logging.error(f"Error parsing WKT: {wkt_file}")
            raise

        frames = []
        for i, g in enumerate(gs):
            frame = self.index_geometry(g)
            if frame is not None:
                frames.append(frame)

        if len(frames) == 0:
            return None
        return pd.concat(frames, axis=0)

    def index_wkt_file(self, wkt_file: str) -> pd.DataFrame:
        logging.debug(f"Reading WKT file {wkt_file}")
        with open(wkt_file, "r") as f:
            wkt = f.read()
            return self.index_wkt(wkt)

    def create_indexed_mrgid(self, mrgid: int) -> None:
        logging.debug(f"Creating indexed geometry for http://marineregions.org/mrgid/{mrgid}")
        wkts_path = os.path.join(self.data_dir, "mr_wkt", mrgid)
        if not os.path.exists(wkts_path):
            raise FileNotFoundError(f"No WKT found at {wkts_path}, may be missing from the LDES export")
        frames = []
        wkt_files = os.listdir(wkts_path)
        for file in wkt_files:
            wkt_file = os.path.join(wkts_path, file)
            frame = self.index_wkt_file(wkt_file)
            if frame is not None:
                frames.append(frame)
        if len(frames) == 0:
            raise FileNotFoundError(f"No suitable geometries found for http://marineregions.org/mrgid/{mrgid}")
        df = pd.concat(frames, axis=0).drop_duplicates()
        return df

    def get_indexed_mrgid(self, mrgid: str) -> pd.DataFrame:
        logging.debug(f"Trying to read indexed MRGID {mrgid}")
        parquet_file = os.path.join(self.data_dir, f"mr_indexed_{self.h3_resolution}", f"{mrgid}.parquet")
        if self.cache_marineregions and os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
        else:
            df = self.create_indexed_mrgid(mrgid)
            os.makedirs(os.path.join(self.data_dir, f"mr_indexed_{self.h3_resolution}"), exist_ok=True)
            df.to_parquet(os.path.join(self.data_dir, f"mr_indexed_{self.h3_resolution}", f"{mrgid}.parquet"))
        return df

    def get_worms_distribution(self, aphiaid: int) -> pd.DataFrame:
        logging.debug(f"Generating WoRMS distribution for https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphiaid}")
        shapes = self.get_worms_mrgids(aphiaid)
        frames = []
        for shape in shapes:
            mrgid = shape["mrgid"]
            try:
                df = self.get_indexed_mrgid(mrgid)
                df["mrgid"] = mrgid
                df["establishmentMeans"] = shape["establishmentMeans"]
                frames.append(df)
            except FileNotFoundError as e:
                if self.ignore_missing_wkt:
                    logging.error(e)
                else:
                    raise
        return pd.concat(frames, axis=0)

    def set_establishmentmeans(self, df):
        df["establishmentMeans"] = None
        df.loc[df["native"].notna() & df["native"], "establishmentMeans"] = "native"
        df.loc[df["introduced"].notna() & df["introduced"], "establishmentMeans"] = "introduced"

    def resample(self, df: pd.DataFrame, resolution: int) -> pd.DataFrame:

        logging.info(f"Resampling to resolution {resolution}")

        df = df.set_index("h3").h3.h3_to_parent(resolution).reset_index()
        df["h3"] = df[f"h3_0{resolution}"]
        df = df.drop(columns=f"h3_0{resolution}")

        resampled = self.summarize_h3_pandas(df)
        self.set_establishmentmeans(resampled)

        return resampled

    def summarize_h3_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("h3").agg({
            "source_obis": "max",
            "source_gbif": "max",
            "records": "sum",
            "min_year": "min",
            "max_year": "max",
            "native": "max",
            "introduced": "max",
            "uncertain": "max"
        }).reset_index()

    def summarize_h3_duckdb(self, df: pd.DataFrame) -> pd.DataFrame:
        conn = duckdb.connect()
        conn.register("dist_data", df)
        summ = conn.execute("""
            select
                h3,
                max(source_obis) as source_obis,
                max(source_gbif) as source_gbif,
                sum(records) as records,
                min(min_year) as min_year,
                max(max_year) as max_year,
                max(native) as native,
                max(introduced) as introduced,
                max(uncertain) as uncertain
            from dist_data group by h3
        """).fetchdf()
        return summ

    def create_density(self, aphiaid: int, resolution: int, max_rings: int = 50, sd: float = 1000, density_cutoff: float = 1e-10):
        summary = self.get_summary(aphiaid, resolution, as_geopandas=False)
        summary = summary[(summary["source_gbif"] == True) | (summary["source_obis"] == True)]

        cells = summary["h3"].tolist()
        densities = dict()

        for cell in cells:
            cell_latlon = h3.h3_to_geo(cell)
            continue_cell = True
            for i in range(max_rings):
                ring = h3.k_ring(cell, i)
                for ring_cell in ring:
                    ring_cell_latlon = h3.h3_to_geo(ring_cell)
                    distance = haversine(cell_latlon, ring_cell_latlon, unit=Unit.KILOMETERS)
                    density = norm.pdf(distance, loc=0, scale=sd)
                    if ring_cell in densities:
                        densities[ring_cell] = densities[ring_cell] + density
                    else:
                        densities[ring_cell] = density
                    if density < density_cutoff:
                        continue_cell = False
                if not continue_cell:
                    break

        normalized_densities = normalize_density(densities)
        df = pd.DataFrame(list(normalized_densities.items()), columns=["h3", "density"])

        df_sorted = df.sort_values(by="density", ascending=False).reset_index(drop=True)
        df_sorted["cumulative_sum"] = df_sorted["density"].cumsum()
        df_sorted["percentile"] = (1 - df_sorted["cumulative_sum"])
        df = df.merge(df_sorted[["h3", "percentile"]], on="h3", how="left")

        return df

    def get_density(self, aphiaid: int, resolution: int, max_rings: int = 50, sd: float = 1000, density_cutoff: float = 1e-10, as_geopandas: bool = True) -> pd.DataFrame:
        parquet_file = os.path.join(self.data_dir, f"density_{resolution}", f"{aphiaid}.parquet")
        if self.cache_density and os.path.exists(parquet_file):
            density = pd.read_parquet(parquet_file)
        else:
            density = self.create_density(aphiaid, resolution, max_rings, sd, density_cutoff)
            os.makedirs(os.path.join(self.data_dir, f"density_{resolution}"), exist_ok=True)
            density.to_parquet(parquet_file)
        if as_geopandas:
            density = density.set_index("h3").h3.h3_to_geo_boundary()
        return density

    def get_thermal_envelope(self, aphiaid: int):
        distribution = self.read_distribution_grid(aphiaid)
        envelope = calculate_thermal_envelope(distribution, self.data_dir)
        return envelope

    def create_summary(self, aphiaid: int, resolution: int):

        # get OBIS/GBIF data

        distribution_grid = self.read_distribution_grid(aphiaid).filter([f"h3_0{self.h3_resolution}", "source_obis", "source_gbif", "min_year", "max_year", "records"]).rename({f"h3_0{self.h3_resolution}": "h3"}, axis=1)
        distribution_grid["records"] = distribution_grid["records"].astype("Int64")

        # get WoRMS distribution

        worms_grid = self.get_worms_distribution(aphiaid)

        worms_grid["native"] = worms_grid["establishmentMeans"].str.startswith("Native")
        worms_grid["introduced"] = worms_grid["establishmentMeans"].str.startswith("Alien")
        worms_grid["uncertain"] = worms_grid["establishmentMeans"].str.startswith("Origin")

        worms_grid.fillna(value={
            "native": False,
            "introduced": False,
            "uncertain": False
        }, inplace=True)

        # create summary

        logging.info("Merging distribution and WoRMS layers")

        merged = pd.concat([distribution_grid, worms_grid])
        merged = self.summarize_h3_duckdb(merged)
        self.set_establishmentmeans(merged)

        # resample

        if resolution is not None and resolution < self.h3_resolution:
            merged = self.resample(merged, resolution)

        return merged

    def get_summary(self, aphiaid: int, resolution: int, as_geopandas: bool = True) -> pd.DataFrame:
        parquet_file = os.path.join(self.data_dir, f"summary_{resolution}", f"{aphiaid}.parquet")
        if self.cache_summary and os.path.exists(parquet_file):
            summary = pd.read_parquet(parquet_file)
        else:
            summary = self.create_summary(aphiaid, resolution)
            os.makedirs(os.path.join(self.data_dir, f"summary_{resolution}"), exist_ok=True)
            summary.to_parquet(parquet_file)
        if as_geopandas:
            summary = summary.set_index("h3").h3.h3_to_geo_boundary()
        return summary

    def create_summary_layer(self, gdf: geopandas.GeoDataFrame) -> SolidPolygonLayer:
        indexes_to_fix = list(gdf.cx[178:180, -90:90].index) + list(gdf.cx[-180:-178, -90:90].index)
        gdf.loc[indexes_to_fix, "geometry"] = gdf.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
        em = gdf["establishmentMeans"].fillna("none")
        color_map = {
            "native": [171, 196, 147],
            "introduced": [245, 66, 93],
            "uncertain": [202, 117, 255],
            "none": [237, 167, 69]
        }
        colors = np.array(em.map(color_map).values.tolist()).astype("uint8")
        polygon_layer = SolidPolygonLayer.from_geopandas(
            gdf,
            get_fill_color=colors,
            opacity=0.3
        )
        return polygon_layer

    def create_density_layer(self, gdf: geopandas.GeoDataFrame) -> SolidPolygonLayer:
        indexes_to_fix = list(gdf.cx[178:180, -90:90].index) + list(gdf.cx[-180:-178, -90:90].index)
        gdf.loc[indexes_to_fix, "geometry"] = gdf.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
        gdf = gdf[gdf["percentile"] >= 0.01]
        layer = SolidPolygonLayer.from_geopandas(
            gdf,
            opacity=1
        )
        normalized_density = gdf["density"] / gdf["density"].max()
        layer.get_fill_color = apply_continuous_cmap(normalized_density, Temps_3, alpha=0.4)
        return layer

    def create_distribution_layer(self, gdf: geopandas.GeoDataFrame) -> ScatterplotLayer:
        layer = ScatterplotLayer.from_geopandas(
            gdf,
            get_line_color=[0, 0, 0],
            opacity=1,
            radius_min_pixels=3,
            radius_max_pixels=3,
            stroked=True,
            filled=False,
            line_width_min_pixels=1,
            line_width_max_pixels=1
        )
        return layer

    def create_envelope_layer(self, gdf: geopandas.GeoDataFrame) -> SolidPolygonLayer:
        indexes_to_fix = list(gdf.cx[178:180, -90:90].index) + list(gdf.cx[-180:-178, -90:90].index)
        gdf.loc[indexes_to_fix, "geometry"] = gdf.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
        layer = SolidPolygonLayer.from_geopandas(
            gdf,
            get_fill_color=[146, 181, 85],
            opacity=0.3
        )
        return layer

    def export_summary(self, df: pd.DataFrame, path) -> None:
        df.set_index("h3").h3.h3_to_geo_boundary().to_file(path, driver="GPKG")

    def export_density(self, df: pd.DataFrame, path) -> None:
        df.set_index("h3").h3.h3_to_geo_boundary().to_file(path, driver="GPKG")

    def export_map(self, path: str, summary: geopandas.GeoDataFrame = None, density: geopandas.GeoDataFrame = None, distribution: geopandas.GeoDataFrame = None, envelope: geopandas.GeoDataFrame = None) -> None:
        layers = []
        if density is not None:
            layer = self.create_density_layer(density)
            layers.append(layer)
        if summary is not None:
            layer = self.create_summary_layer(summary)
            layers.append(layer)
        if distribution is not None:
            layer = self.create_distribution_layer(distribution)
            layers.append(layer)
        if envelope is not None:
            layer = self.create_envelope_layer(envelope)
            layers.append(layer)
        map = Map(layers)
        with open(path, "w") as f:
            f.write(map.as_html().data)
