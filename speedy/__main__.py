from speedy import Speedy
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():

    sp = Speedy(
        h3_resolution=7,
        cache_marineregions=True,
        cache_summary=True,
        cache_density=False,
        ignore_missing_wkt=True
    )

    aphiaids = [212506]#, 386513, 107451]

    for aphiaid in aphiaids:

        logging.info(f"Creating summary for AphiaID {aphiaid}")
        summary = sp.get_summary(aphiaid, resolution=5)

        logging.info(f"Creating density for AphiaID {aphiaid}")
        density = sp.get_density(aphiaid, resolution=3, sd=1000)

        sp.export_map(f"/Users/pieter/Desktop/speedy_output/map_{aphiaid}.html", summary, density)


if __name__ == "__main__":
    main()
