from speedy import Speedy
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    sp = Speedy(mr_ldes_path="data/MRGID-LDES-export-geometries.ttl", h3_resolution=7)
    # sp.prepare_mr_wkt()

    aphiaids = [159559] # 386513, 107451
    for aphiaid in aphiaids:
        logging.info(f"Processing AphiaID {aphiaid}")

        summary = sp.get_summary(aphiaid, resolution=5, cached=True)
        summary.to_parquet(f"~/Desktop/temp/summaries/summary_{aphiaid}.parquet")

        summary_point = summary.set_index("h3").h3.h3_to_geo()
        summary_point.to_parquet(f"~/Desktop/temp/summaries/summary_{aphiaid}_point.geoparquet")

        summary_poly = summary.set_index("h3").h3.h3_to_geo_boundary()
        summary_poly.to_parquet(f"~/Desktop/temp/summaries/summary_{aphiaid}_poly.geoparquet")

        logging.info("Finished")


if __name__ == "__main__":
    main()
