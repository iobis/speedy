import re
import geopandas
import h3pandas
import logging
import pyworms
import pandas as pd
import os
import hashlib
from rdflib import Graph
import shapely
from h3pandas.util.shapely import polyfill
import h3
import duckdb
from haversine import haversine, Unit
from scipy.stats import norm
import geopandas as gpd
from lonboard import Map, SolidPolygonLayer
from lonboard.colormap import apply_continuous_cmap
from palettable.cartocolors.diverging import Temps_3
import numpy as np


def normalize_density(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


class Speedy:

    def __init__(self, h3_resolution: int = 7, data_dir: str = "speedy_data", cache_marineregions: bool = True, cache_summary: bool = False, cache_density: bool = False, ignore_missing_wkt = True):

        self.h3_resolution = h3_resolution
        self.data_dir = data_dir
        self.cache_marineregions = cache_marineregions
        self.cache_summary = cache_summary
        self.cache_density = cache_density
        self.ignore_missing_wkt = ignore_missing_wkt

        self.prefixes = """
            @prefix tree: <https://w3id.org/tree#> .
            @prefix ldes: <https://w3id.org/ldes#> .
            @prefix dc: <http://purl.org/dc/terms/> .
            @prefix sh: <http://www.w3.org/ns/shacl#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
            @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
            @prefix gsp: <http://www.opengis.net/ont/geosparql#> .
            @prefix dcat: <http://www.w3.org/ns/dcat#> .
            @prefix mr: <http://marineregions.org/ns/ontology#> .
            @prefix schema: <https://schema.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix mrt: <http://marineregions.org/ns/placetypes#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix prov: <http://www.w3.org/ns/prov#> .
        """

    def create_record(self) -> str:
        return self.prefixes + "\n"

    def parse_ldes(self, predicate: str):

        with open(os.path.join(self.data_dir, "MRGID-LDES-export-geometries.ttl"), "r") as file:
            record = self.create_record()

            for line in file:
                line = line.strip()
                record = record + "\n" + line

                if line.endswith(" ."):
                    g = Graph()
                    ttl = g.parse(data=record, format="ttl")
                    for stmt in ttl:
                        if str(stmt[1]) == predicate:
                            yield stmt

                    record = self.create_record()

    def prepare_mr_wkt(self) -> None:

        # first pass to get mrgid geometry mapping

        logging.info("Collecting all geometry URIs")
        geometry_mrgids = dict()

        for stmt in self.parse_ldes("http://marineregions.org/ns/ontology#hasGeometry"):
            uri = str(stmt[0])
            geometry = str(stmt[2])
            geometry_mrgids[geometry] = uri

        # second pass

        logging.info("Exporting all gemetries by MRGID")

        for stmt in self.parse_ldes("http://www.opengis.net/ont/geosparql#asWKT"):

            uri = str(stmt[0])
            wkt = str(stmt[2])

            mrgid = re.search("[0-9]+", geometry_mrgids[uri]).group()
            geometry_hash = hashlib.md5(uri.encode()).hexdigest()

            if len(wkt.strip()) == 0:
                logging.debug(f"Empty geometry: {uri}")
                continue

            output_path = os.path.join(self.data_dir, "mr_wkt", mrgid)
            output_file = os.path.join(self.data_dir, "mr_wkt", mrgid, f"{geometry_hash}.txt")
            if not os.path.exists(output_path):
                os.makedirs(os.path.join(output_path))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(wkt)

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
        df = df.merge(df_sorted[["density", "percentile"]], on="density", how="left")

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
        # fix for dateline wrapping
        offending_cells = list(gdf.cx[179:180, -90:90].index) + list(gdf.cx[-180:-179, -90:90].index)
        gdf = gdf.loc[gdf.index.difference(offending_cells), :]
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
        offending = list(gdf.cx[178:180, -90:90].index) + list(gdf.cx[-180:-178, -90:90].index)
        gdf = gdf.loc[gdf.index.difference(offending), :]
        layer = SolidPolygonLayer.from_geopandas(
            gdf,
            opacity=1
        )
        normalized_density = gdf["density"] / gdf["density"].max()
        layer.get_fill_color = apply_continuous_cmap(normalized_density, Temps_3, alpha=0.4)
        return layer

    def export_summary(self, df: pd.DataFrame, path) -> None:
        df.set_index("h3").h3.h3_to_geo_boundary().to_file(path, driver="GPKG")

    def export_density(self, df: pd.DataFrame, path) -> None:
        df.set_index("h3").h3.h3_to_geo_boundary().to_file(path, driver="GPKG")

    def export_map(self, path: str, summary: geopandas.GeoDataFrame = None, density: geopandas.GeoDataFrame = None) -> None:
        layers = []
        if density is not None:
            layer = self.create_density_layer(density)
            layers.append(layer)
        if summary is not None:
            layer = self.create_summary_layer(summary)
            layers.append(layer)
        map = Map(layers)
        with open(path, "w") as f:
            f.write(map.as_html().data)
