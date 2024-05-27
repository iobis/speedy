from speedy import Speedy
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    sp = Speedy(mr_ldes_path="data/MRGID-LDES-export-geometries.ttl")
    sp.prepare_mr_wkt()
    # sp.create_summary(141433)


if __name__ == "__main__":
    main()
