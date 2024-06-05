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


class Speedy:

    def __init__(self, mr_ldes_path: str, h3_resolution: int = 7):
        self.mr_ldes_path = mr_ldes_path
        self.h3_resolution = h3_resolution
        with open("data/prefixes.ttl") as f:
            self.prefixes = f.read()

    def create_record(self) -> str:
        return self.prefixes + "\n"

    def parse_ldes(self, predicate: str):

        with open(self.mr_ldes_path, "r") as file:
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

            output_path = os.path.join("data", "mr_wkt", mrgid)
            output_file = os.path.join("data", "mr_wkt", mrgid, f"{geometry_hash}.txt")
            if not os.path.exists(output_path):
                os.makedirs(os.path.join(output_path))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(wkt)

    def read_distribution_grid(self, aphiaid: int) -> geopandas.GeoDataFrame:
        logging.debug(f"Reading distribution data for https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphiaid}")
        filters = [("AphiaID", "==", aphiaid)]
        gdf = geopandas.read_parquet("data/h3_7/", filters=filters)
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
        wkts_path = os.path.join("data", "mr_wkt", mrgid)
        if not os.path.exists(wkts_path):
            raise FileNotFoundError(f"No WKT found for http://marineregions.org/mrgid/{mrgid}")
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
        df.to_parquet(os.path.join("data", "mr_indexed", f"{mrgid}.parquet"))
        return df

    def read_indexed_mrgid(self, mrgid: str) -> pd.DataFrame:
        logging.debug(f"Trying to read indexed MRGID {mrgid}")
        parquet_file = os.path.join("data", "mr_indexed", f"{mrgid}.parquet")
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"No indexed geometry found for http://marineregions.org/mrgid/{mrgid}")
        return pd.read_parquet(parquet_file)

    def get_worms_distribution(self, aphiaid: int) -> pd.DataFrame:
        logging.debug(f"Generating WoRMS distribution for https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphiaid}")
        shapes = self.get_worms_mrgids(aphiaid)
        frames = []
        for shape in shapes:
            mrgid = shape["mrgid"]
            try:
                df = self.read_indexed_mrgid(mrgid)
            except FileNotFoundError as read_error:
                logging.error(read_error)
                try:
                    df = self.create_indexed_mrgid(mrgid)
                except FileNotFoundError as create_error:
                    logging.error(create_error)
                    continue
            df["mrgid"] = mrgid
            df["establishmentMeans"] = shape["establishmentMeans"]
            frames.append(df)
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
        import duckdb
        conn = duckdb.connect()
        conn.register("data", df)
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
            from data group by h3
        """).fetchdf()
        return summ

    def create_summary(self, aphiaid: int, resolution: int = 7):

        # get OBIS/GBIF data

        distribution_grid = self.read_distribution_grid(aphiaid).filter(["h3_07", "source_obis", "source_gbif", "min_year", "max_year", "records"]).rename({"h3_07": "h3"}, axis=1)
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

        if resolution < 7:
            merged = self.resample(merged, resolution)

        return merged

    def get_summary(self, aphiaid: int, resolution: int = 7, cached=False) -> pd.DataFrame:
        parquet_file = os.path.join("data", "aphia_indexed", f"{aphiaid}.parquet")
        if cached and os.path.exists(parquet_file):
            summary = pd.read_parquet(parquet_file)
        else:
            summary = self.create_summary(aphiaid, resolution)
            summary.to_parquet(parquet_file)
        return summary
