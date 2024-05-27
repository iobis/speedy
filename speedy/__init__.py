import re
import geopandas
import h3pandas
import logging
import pyworms
import pandas as pd
import os
import hashlib
from rdflib import Graph


class Speedy:

    def __init__(self, mr_ldes_path: str):
        self.mr_ldes_path = mr_ldes_path

    def parse_ldes(self, predicate: str):

        with open("data/prefixes.ttl") as f:
            prefixes = f.read()

        with open(self.mr_ldes_path, "r") as file:
            record = None
            for line in file:
                line = line.strip()

                if record is None and line.startswith("<") and line.endswith(">"):
                    record = prefixes + "\n" + line
                elif record is not None:
                    record = record + "\n" + line
                    if line.endswith(" ."):
                        g = Graph()
                        ttl = g.parse(data=record, format="ttl")
                        for stmt in ttl:
                            if str(stmt[1]) == predicate:
                                yield stmt

                        record = None

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
        filters = [("AphiaID", "==", aphiaid)]
        gdf = geopandas.read_parquet("data/h3_7/", filters=filters)
        return gdf

    def get_worms_mrgids(self, aphiaid: int) -> list:
        res = pyworms.aphiaDistributionsByAphiaID(id=aphiaid)
        distribution = [{"mrgid": re.search("\\d+", entry["locationID"])[0], "establishmentMeans": entry["establishmentMeans"]} for entry in res]
        return distribution

    def get_worms_distribution(self, aphiaid: int) -> pd.DataFrame:
        shapes = self.get_worms_mrgids(aphiaid)
        for shape in shapes:
            wkts_path = os.path.join("data", "mr_wkt", shape["mrgid"])
            for file in os.listdir(wkts_path):
                with open(os.path.join(wkts_path, file), "r") as f:
                    wkt = f.read()
                    try:
                        gs = geopandas.GeoSeries.from_wkt([wkt])
                        # gs is length 1 but can contain a geometrycollection, so split
                        assert len(gs) == 1
                        if hasattr(gs[0], "geoms"):
                            gs = geopandas.GeoSeries(gs[0].geoms)
                    except Exception:
                        logging.error(f"Error parsing WKT: {file}")
                        raise
                    gdf = geopandas.GeoDataFrame(geometry=gs, crs="EPSG:4326")
                    pass

    def create_summary(self, aphiaid: int):

        # get OBIS/GBIF data

        distribution_grid = self.read_distribution_grid(aphiaid)

        # get MRGIDs

        worms_grid = self.get_worms_distribution(aphiaid)

        # get MR geometries
        # index MR geometries
        # optional: create thermal range
        # create summary





    # def run(self, skip: int = 0):

    #     geometry_mrgids = dict()
    #     for record in self.parse_ldes():
    #         if "geometries" in record:
    #             uri = record["uri"]
    #             geometries = record["geometries"]
    #             for geometry in geometries:
    #                 geometry_mrgids[geometry] = uri

    #     connection = sqlite3.connect("temp_geometries.db")
    #     cursor = connection.cursor()
    #     if skip == 0:
    #         cursor.execute("drop table if exists geometries")
    #     cursor.execute("create table if not exists geometries (mrgid text, h3 text)")

    #     for i, record in enumerate(self.parse_ldes()):

    #         logging.info(f"Processing record {i}")

    #         if i < skip:
    #             continue

    #         if "wkt" in record:
    #             geometry = record["uri"]
    #             wkt = record["wkt"]
    #             mrgid = re.search("[0-9]+", geometry_mrgids[geometry]).group()
    #             if len(wkt.strip()) == 0:
    #                 logging.error(f"Empty geometry: {geometry}")
    #                 continue
    #             try:
    #                 gs = geopandas.GeoSeries.from_wkt([wkt])

    #                 # gs is length 1 but can contain a geometrycollection, so split
    #                 assert len(gs) == 1
    #                 if hasattr(gs[0], "geoms"):
    #                     gs = geopandas.GeoSeries(gs[0].geoms)

    #             except Exception:
    #                 logging.error(f"Error parsing geometry: {geometry}")
    #                 with open("temp_error.txt", "w") as errorfile:
    #                     errorfile.write(f"{geometry}\n{wkt}")
    #                 raise
    #             gdf = geopandas.GeoDataFrame(geometry=gs, crs="EPSG:4326")
    #             # remove linestrings
    #             gdf = gdf[(gdf.geometry.type != "LineString") & (gdf.geometry.type != "Point")]
    #             if len(gdf) > 0:
    #                 # handle row by row
    #                 for index, row in gdf.iterrows():
    #                     try:
    #                         h3s = list(row.to_frame().transpose().h3.polyfill_resample(self.h3_resolution).index)
    #                         logging.info(f"Inserting {len(h3s)} H3 cells")
    #                         for h3 in h3s:
    #                             cursor.execute(f"insert into geometries values ('{mrgid}', '{h3}')")
    #                     except MemoryError:
    #                         # happens for POLYGON ((-63.33333015 39.00000191000004, -63.33329964 39.00000191000004, -63.33329964 41.00000191000004, -63.33329964 41.50000191000002, -63.33333015 41.50000191000002, -63.33333015 41.00000191000004, -63.33333015 39.00000191000004))
    #                         logging.error("MemoryError for {geometry}")

    #             connection.commit()
