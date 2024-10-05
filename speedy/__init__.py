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

    def __init__(self, h3_resolution: int = 7, data_dir: str = DEFAULT_DATA_DIR, cache_marineregions: bool = True, cache_summary: bool = False, cache_density: bool = False, cache_envelope: bool = False, ignore_missing_wkt = True):

        self.h3_resolution = h3_resolution
        self.data_dir = data_dir
        self.cache_marineregions = cache_marineregions
        self.cache_summary = cache_summary
        self.cache_density = cache_density
        self.cache_envelope = cache_envelope
        self.ignore_missing_wkt = ignore_missing_wkt

    def read_distribution_grid(self, aphiaid: int) -> geopandas.GeoDataFrame:
        logging.debug(f"Reading distribution data for https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphiaid}")
        filters = [("AphiaID", "==", aphiaid)]
        gdf = geopandas.read_parquet(os.path.join(self.data_dir, f"distributions_{self.h3_resolution}"), filters=filters)
        return gdf

    def get_worms_mrgids(self, aphiaid: int) -> list:
        res = pyworms.aphiaDistributionsByAphiaID(id=aphiaid)
        if res is None:
            return []
        distribution = [{
            "mrgid": re.search("\\d+", entry["locationID"])[0],
            "establishmentMeans": entry["establishmentMeans"],
            "invasiveness": entry["invasiveness"]
        } for entry in res]
        return distribution

    def index_geometry(self, geometry: geopandas.GeoSeries) -> pd.DataFrame:
        if isinstance(geometry, shapely.geometry.linestring.LineString):
            return None
        elif isinstance(geometry, shapely.geometry.point.Point):
            return None
        elif isinstance(geometry, shapely.geometry.polygon.Polygon):
            try:
                poly = polyfill(geometry, self.h3_resolution, geo_json=True)
                if len(poly) == 0:
                    centroid = geometry.centroid
                    poly = [h3.geo_to_h3(centroid.y, centroid.x, self.h3_resolution)]
                return pd.DataFrame(data={"h3": list(poly)})
            except MemoryError:
                # temporary fix for https://marineregions.org/gazetteer.php?p=details&id=48569
                # <http://www.opengis.net/def/crs/OGC/1.3/CRS84> GEOMETRYCOLLECTION (POLYGON ((179.54185435485718 -42.750973616068286, 179.52284236999958 36.137726310000126, 179.95165483468224 36.086524826508438, 179.95658808999977 61.110493879999794, 178.7245159699998 60.964906710000072, 177.1861967199996 60.552286190000174, 175.71354806999992 60.179142710000022, 174.37755402999997 59.507824820000074, 172.84587653000031 58.472661900000148, 171.7619381099999 57.56352679000004, 170.60340407999945 55.841479220000132, 169.45113835000029 54.3280251700002, 168.10318916000014 52.76616906, 166.65142067000019 51.792318250000143, 164.38009795999946 50.038390289999938, 163.3120997400006 49.289446639999952, 161.72670678000031 47.924602350000022, 160.91076417999972 47.270179560000123, 159.70478300000025 46.332194189999967, 158.63279973000004 44.925216150000047, 157.42681855000058 43.719234970000151, 155.61784678000009 42.647251700000162, 154.34486664000019 41.441270519999847, 152.87088963999963 40.369287239999792, 152.40189696000044 39.4983008399999, 151.14566656999969 38.627314429999942, 149.87268642999979 37.622330110000078, 149.60469060999998 37.019339519999924, 148.53270733999977 35.746359379999838, 147.86271779999993 35.07636983999997, 146.45573974999948 34.473379249999816, 146.12074497999956 33.468394929999953, 145.04876171000063 32.061416890000032, 143.97279339000036 31.501888399999945, 142.77079725999968 30.788436749999946, 141.8328118899997 29.850451389999854, 141.22982129999997 29.247460800000123, 139.95684117000039 28.778468119999864, 139.01885580000041 28.443473349999934, 137.81287461999966 28.041479619999805, 136.13790075999972 27.8404827600002, 136.07090180999961 25.56251829999983, 136.33889762000038 24.155540259999906, 136.40589657999953 22.145571629999825, 136.47289552999965 17.053651079999973, 135.5516599100001 16.517659450000007, 135.41766199999955 15.110681400000024, 135.08266722999963 13.770702309999807, 134.68067350000055 12.832716950000142, 134.01068395999943 11.760733680000151, 132.87170173000007 11.291740999999893, 132.00071532000052 10.085759820000002, 131.46472368999994 9.415770270000074, 132.4027090499996 8.4777849099999845, 136.48964526999976 8.8797786400001115, 138.96860659000041 9.0137765500000118, 144.86451457999976 9.0807755000001418, 148.68345499000043 8.946777589999817, 151.29641421000002 8.209789090000184, 152.90438911999985 7.94179328000002, 155.04835566000025 7.80779537000012, 155.90259232999969 7.338802679999799, 156.77357874000052 7.0708068700000615, 158.04655888000042 7.0708068700000615, 158.51555155999961 6.6688131399999344, 159.18554109999945 6.0658225500002017, 160.6595181 5.0608382299999173, 161.99949719000003 3.519862280000094, 162.73648569000031 2.1128842400001715, 163.60747209999985 0.50490933000057836, 164.00946583000021 -3.11303420999952, 163.67447104999997 -5.9269902999994288, 162.93748254999969 -7.2669693899996455, 162.53548882999965 -8.8749442999996617, 161.99949719000003 -10.281922339999584, 161.46350555999945 -11.688900389999571, 160.86051495999939 -13.095878429999493, 160.05652751000028 -13.966864839999452, 159.78853169000047 -14.904850199999544, 159.25254005999989 -16.043832429999728, 159.11854215000062 -17.383811519999519, 158.98454424000008 -18.120800019999578, 158.85054632999953 -23.480716379999592, 159.31953901 -24.418701739999683, 160.6595181 -26.093675599999408, 161.53050450999956 -27.0316609599995, 162.40149092000038 -27.433654689999624, 163.80846896000051 -28.036645279999782, 166.01943446000007 -28.907631689999739, 168.23039995999963 -29.979614959999733, 171.36473896999945 -30.51959164999959, 174.09841259999979 -31.787258379999692, 176.43939095999983 -33.333547729999438, 178.42729646000024 -35.417157069999718, 179.16029991000045 -36.745181009999584, 179.18022515999951 -38.280843569999554, 179.12650970000016 -39.426467539999678, 179.00048188999975 -42.56479159999968, 179.54185435485718 -42.750973616068286)), POLYGON ((180 -42.9053624699995, 180 36.080752269999955, 179.95165483468224 36.086524826508438, 179.93609391999968 -42.846314119999654, 180 -42.9053624699995)), LINESTRING (180 -42.9053624699995, 180 -42.908533309999541), LINESTRING (-179.99999999999872 -42.9053624699995, -179.99999999999872 -42.908533309999541), POLYGON ((180 -57.567712899999705, 180 -42.908533309999541, 179.54185435485718 -42.750973616068286, 179.54542432000039 -57.564256449999618, 180 -57.567712899999705)), POLYGON ((-72.228847669999112 -58.387171659999765, -70.955867539998252 -58.253173749999441, -71.42486021999872 -57.91817897999951, -72.362845579998378 -56.84619570999952, -73.032835129998546 -55.841211389999657, -74.439813169998672 -54.635230209999769, -74.573811079999217 -53.898241709999709, -75.109802719998839 -52.759259479999528, -76.315783899998308 -51.2182835299997, -77.572014289999061 -49.14131593999943, -78.108005929998683 -47.198346259999475, -79.313987109998152 -44.250392269999729, -80.184973519998977 -43.446404809999478, -80.586967249999333 -41.4364361799994, -80.988960969998089 -38.689479039999618, -81.189957839999067 -36.009520859999611, -82.060944239998292 -35.004536549999393, -82.513187189998973 -32.190580459999481, -82.714184049998352 -27.299656779999722, -82.647185099998239 -23.949709059999428, -82.781182999998464 -20.800758199999585, -82.8481819599989 -18.790789559999443, -83.317174639999365 -15.6418386999996, -83.451172549998631 -12.358889929999435, -83.719168369998442 -9.61193279999972, -84.322158959998177 -5.4579976199995963, -84.724152679998213 -3.1800331699997137, -85.595139089999037 -0.96906766999953864, -87.13611504999875 1.7108905100000451, -88.074100409998408 3.519862280000094, -89.414079499998422 5.4628319600000435, -90.553061719998738 7.2048047799999617, -91.960039769999185 8.8127796799999167, -93.8360104899985 10.822748320000061, -95.6449822699993 12.229726359999983, -96.984961349999 13.50270650000007, -98.2579414899989 14.105697089999802, -98.860932079998634 14.775686630000092, -100.602904899999 16.584658400000141, -101.94288398999902 17.254647950000066, -105.7953238699983 19.532612399999948, -108.40828308999917 21.408583130000192, -109.51528091999833 22.675772409999936, -112.16022453999813 22.748562219999986, -114.94750962999817 22.19615292999984, -119.86510430999829 22.145571629999825, -121.80807398999931 22.144243280000165, -126.43100184999909 22.614564310000084, -129.5129537599982 23.217554899999818, -132.39390879999888 23.418551759999851, -141.97475928999836 25.696516210000155, -146.39669027999844 26.701500530000015, -149.81363695999875 27.639485890000106, -152.69459200999847 28.309475440000032, -155.91054181999908 29.381458710000022, -159.39448744999822 30.118447210000081, -162.34244144999934 31.391427340000103, -166.09438289999829 32.061416890000032, -168.97533793999898 33.535393890000151, -171.18630343999854 34.071385520000113, -173.19627207999872 34.473379249999816, -174.60325011999885 35.3443656600002, -176.66959090999853 35.679360430000131, -176.07722711999813 35.61236148, -179.99999999999872 36.080752269999955, -179.99999999999872 -42.9053624699995, -179.99453378999888 -42.910413179999516, -179.99999999999872 -42.908533309999541, -179.99999999999872 -57.567712899999705, -72.228847669999112 -58.387171659999765)))
                return None
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
            logging.debug(f"Reading {parquet_file}")
            df = pd.read_parquet(parquet_file)
        else:
            logging.debug(f"No indexed MRGID {mrgid} found")
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
                df["invasiveness"] = shape["invasiveness"]
                frames.append(df)
            except FileNotFoundError as e:
                if self.ignore_missing_wkt:
                    logging.error(e)
                else:
                    raise
        if len(frames) == 0:
            return None
        return pd.concat(frames, axis=0)

    def set_establishmentmeans(self, df):
        df["establishmentMeans"] = None
        df.loc[df["establishmentMeans_native"].notna() & df["establishmentMeans_native"], "establishmentMeans"] = "native"
        df.loc[df["establishmentMeans_introduced"].notna() & df["establishmentMeans_introduced"], "establishmentMeans"] = "introduced"

    def set_invasiveness(self, df):
        df["invasiveness"] = None
        df.loc[df["invasiveness_invasive"].notna() & df["invasiveness_invasive"], "invasiveness"] = "invasive"
        df.loc[df["invasiveness_concern"].notna() & df["invasiveness_concern"], "invasiveness"] = "concern"

    def resample(self, df: pd.DataFrame, resolution: int) -> pd.DataFrame:

        logging.debug(f"Resampling to resolution {resolution}")

        df = df.set_index("h3").h3.h3_to_parent(resolution).reset_index()
        df["h3"] = df[f"h3_0{resolution}"]
        df = df.drop(columns=f"h3_0{resolution}")

        resampled = self.summarize_h3_pandas(df)
        self.set_establishmentmeans(resampled)
        self.set_invasiveness(resampled)

        return resampled

    def summarize_h3_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("h3").agg({
            "source_obis": "max",
            "source_gbif": "max",
            "records": "sum",
            "min_year": "min",
            "max_year": "max",
            "establishmentMeans_native": "max",
            "establishmentMeans_introduced": "max",
            "establishmentMeans_uncertain": "max",
            "invasiveness_invasive": "max",
            "invasiveness_concern": "max"
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
                max(establishmentMeans_native) as establishmentMeans_native,
                max(establishmentMeans_introduced) as establishmentMeans_introduced,
                max(establishmentMeans_uncertain) as establishmentMeans_uncertain,
                max(invasiveness_invasive) as invasiveness_invasive,
                max(invasiveness_concern) as invasiveness_concern
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

    def get_density(self, aphiaid: int, resolution: int, max_rings: int = 50, sd: float = 1000, density_cutoff: float = 1e-10, as_geopandas: bool = True, wrap_dateline: bool = True) -> pd.DataFrame:
        parquet_file = os.path.join(self.data_dir, f"density_{resolution}", f"{aphiaid}.parquet")
        if self.cache_density and os.path.exists(parquet_file):
            density = pd.read_parquet(parquet_file)
        else:
            density = self.create_density(aphiaid, resolution, max_rings, sd, density_cutoff)
            os.makedirs(os.path.join(self.data_dir, f"density_{resolution}"), exist_ok=True)
            density.to_parquet(parquet_file)
        if as_geopandas:
            density = density.set_index("h3").h3.h3_to_geo_boundary()
            if wrap_dateline:
                indexes_to_fix = list(set(list(density.cx[178:180, -90:90].index) + list(density.cx[-180:-178, -90:90].index) + list(density.cx[-180:180, 80:90].index) + list(density.cx[-180:180, -90:-80].index)))
                density.loc[indexes_to_fix, "geometry"] = density.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
        return density

    def get_thermal_envelope(self, aphiaid: int, resolution: int, as_geopandas: bool = True, wrap_dateline: bool = True, dissolve: bool = False) -> geopandas.GeoDataFrame:
        parquet_file = os.path.join(self.data_dir, f"envelope_{resolution}", f"{aphiaid}.parquet")
        if self.cache_envelope and os.path.exists(parquet_file):
            envelope = pd.read_parquet(parquet_file)
        else:
            distribution = self.read_distribution_grid(aphiaid)
            envelope = calculate_thermal_envelope(distribution, resolution, self.data_dir)
            if envelope is not None:
                os.makedirs(os.path.join(self.data_dir, f"envelope_{resolution}"), exist_ok=True)
                envelope.to_parquet(parquet_file)
            else:
                logging.warn(f"Distribution too limited for {aphiaid}, can't calculate envelope")
        if as_geopandas and envelope is not None:
            envelope = envelope.set_index("h3").h3.h3_to_geo_boundary()
            if wrap_dateline:
                indexes_to_fix = list(set(list(envelope.cx[178:180, -90:90].index) + list(envelope.cx[-180:-178, -90:90].index) + list(envelope.cx[-180:180, 80:90].index) + list(envelope.cx[-180:180, -90:-80].index)))
                envelope.loc[indexes_to_fix, "geometry"] = envelope.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
            if dissolve:
                envelope["thermal_envelope"] = True
                envelope = envelope.dissolve(by="thermal_envelope")
        return envelope

    def create_summary(self, aphiaid: int, resolution: int):

        layers = []

        # get OBIS/GBIF data

        distribution_grid = self.read_distribution_grid(aphiaid).filter([f"h3_0{self.h3_resolution}", "source_obis", "source_gbif", "min_year", "max_year", "records"]).rename({f"h3_0{self.h3_resolution}": "h3"}, axis=1)
        layers.append(distribution_grid)

        # get WoRMS distribution

        worms_grid = self.get_worms_distribution(aphiaid)

        if worms_grid is not None:
            worms_grid["establishmentMeans_native"] = worms_grid["establishmentMeans"].str.startswith("Native")
            worms_grid["establishmentMeans_introduced"] = worms_grid["establishmentMeans"].str.startswith("Alien")
            worms_grid["establishmentMeans_uncertain"] = worms_grid["establishmentMeans"].str.startswith("Origin")

            worms_grid["invasiveness_invasive"] = worms_grid["invasiveness"] == "Invasive"
            worms_grid["invasiveness_concern"] = worms_grid["invasiveness"] == "Of concern"

            worms_grid.fillna(value={
                "establishmentMeans_native": False,
                "establishmentMeans_introduced": False,
                "establishmentMeans_uncertain": False,
                "invasiveness_invasive": False,
                "invasiveness_concern": False
            }, inplace=True)

            layers.append(worms_grid)

        else:
            layers.append(pd.DataFrame({col: pd.Series(dtype=bool) for col in ["establishmentMeans_native", "establishmentMeans_introduced", "establishmentMeans_uncertain", "invasiveness_invasive", "invasiveness_concern"]}))

        # create summary

        logging.debug("Merging distribution and WoRMS layers")

        merged = pd.concat(layers)
        merged = self.summarize_h3_duckdb(merged)
        self.set_establishmentmeans(merged)
        self.set_invasiveness(merged)

        # resample

        if resolution is not None and resolution < self.h3_resolution:
            merged = self.resample(merged, resolution)

        # fix types

        merged["records"] = merged["records"].astype("Int64")
        merged["h3"] = merged["h3"].astype("string[pyarrow]")
        merged["source_obis"] = merged["source_obis"].astype("bool")
        merged["source_gbif"] = merged["source_gbif"].astype("bool")
        merged["establishmentMeans_native"] = merged["establishmentMeans_native"].astype("bool")
        merged["establishmentMeans_introduced"] = merged["establishmentMeans_introduced"].astype("bool")
        merged["establishmentMeans_uncertain"] = merged["establishmentMeans_uncertain"].astype("bool")
        merged["invasiveness_invasive"] = merged["invasiveness_invasive"].astype("bool")
        merged["invasiveness_concern"] = merged["invasiveness_concern"].astype("bool")
        merged["establishmentMeans"] = merged["establishmentMeans"].astype("string[pyarrow]")
        merged["invasiveness"] = merged["invasiveness"].astype("string[pyarrow]")

        return merged

    def get_summary(self, aphiaid: int, resolution: int, as_geopandas: bool = True, wrap_dateline: bool = True, dissolve: bool = False, tolerance: float = 0.001) -> pd.DataFrame:
        parquet_file = os.path.join(self.data_dir, f"summary_{resolution}", f"{aphiaid}.parquet")
        if self.cache_summary and os.path.exists(parquet_file):
            summary = pd.read_parquet(parquet_file)
        else:
            summary = self.create_summary(aphiaid, resolution)
            os.makedirs(os.path.join(self.data_dir, f"summary_{resolution}"), exist_ok=True)
            summary.to_parquet(parquet_file)
        if as_geopandas:
            summary = summary.set_index("h3").h3.h3_to_geo_boundary()
            if wrap_dateline:
                indexes_to_fix = list(set(list(summary.cx[178:180, -90:90].index) + list(summary.cx[-180:-178, -90:90].index) + list(summary.cx[-180:180, 80:90].index) + list(summary.cx[-180:180, -90:-80].index)))
                summary.loc[indexes_to_fix, "geometry"] = summary.loc[indexes_to_fix, "geometry"].apply(antimeridian.fix_polygon)
            if dissolve:
                summary = summary.dissolve(by=["source_obis", "source_gbif", "establishmentMeans_native", "establishmentMeans_introduced", "establishmentMeans_uncertain", "invasiveness_invasive", "invasiveness_concern", "establishmentMeans", "invasiveness"], as_index=False, dropna=False)[["source_obis", "source_gbif", "establishmentMeans_native", "establishmentMeans_introduced", "establishmentMeans_uncertain", "invasiveness_invasive", "invasiveness_concern", "establishmentMeans", "invasiveness", "geometry"]]
        summary.geometry = summary.geometry.simplify(0.001, preserve_topology=True).set_precision(grid_size=tolerance)
        return summary

    def create_summary_layer(self, gdf: geopandas.GeoDataFrame) -> SolidPolygonLayer:
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
        if envelope is not None:
            layer = self.create_envelope_layer(envelope)
            layers.append(layer)
        if density is not None:
            layer = self.create_density_layer(density)
            layers.append(layer)
        if summary is not None:
            layer = self.create_summary_layer(summary)
            layers.append(layer)
        if distribution is not None:
            layer = self.create_distribution_layer(distribution)
            layers.append(layer)
        map = Map(layers)
        with open(path, "w") as f:
            f.write(map.as_html().data)
