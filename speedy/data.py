import re
import logging
import os
import hashlib
from rdflib import Graph


prefixes = """
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


def create_record() -> str:
    return prefixes + "\n"


def parse_ldes(data_dir: str = "speedy_data", predicate: str = None):

    with open(os.path.join(data_dir, "MRGID-LDES-export-geometries.ttl"), "r") as file:
        record = create_record()

        for line in file:
            line = line.strip()
            record = record + "\n" + line

            if line.endswith(" ."):
                g = Graph()
                ttl = g.parse(data=record, format="ttl")
                for stmt in ttl:
                    if str(stmt[1]) == predicate:
                        yield stmt

                record = create_record()


def prepare_mr_wkt(data_dir: str = "speedy_data") -> None:

    # first pass to get mrgid geometry mapping

    logging.info("Collecting all geometry URIs")
    geometry_mrgids = dict()

    for stmt in parse_ldes("http://marineregions.org/ns/ontology#hasGeometry"):
        uri = str(stmt[0])
        geometry = str(stmt[2])
        geometry_mrgids[geometry] = uri

    # second pass

    logging.info("Exporting all gemetries by MRGID")

    for stmt in parse_ldes("http://www.opengis.net/ont/geosparql#asWKT"):

        uri = str(stmt[0])
        wkt = str(stmt[2])

        mrgid = re.search("[0-9]+", geometry_mrgids[uri]).group()
        geometry_hash = hashlib.md5(uri.encode()).hexdigest()

        if len(wkt.strip()) == 0:
            logging.debug(f"Empty geometry: {uri}")
            continue

        output_path = os.path.join(data_dir, "mr_wkt", mrgid)
        output_file = os.path.join(data_dir, "mr_wkt", mrgid, f"{geometry_hash}.txt")
        if not os.path.exists(output_path):
            os.makedirs(os.path.join(output_path))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(wkt)
