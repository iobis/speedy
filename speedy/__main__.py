from speedy import Speedy
from speedy.data import prepare_mr_wkt, download_temperature
from speedy.temperature import get_temperature
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():

    # prepare_mr_wkt()
    # download_temperature()

    sp = Speedy(
        h3_resolution=7,
        cache_marineregions=True,
        cache_summary=True,
        cache_density=True,
        ignore_missing_wkt=True
    )

    aphiaids = [
        107451,
        212506,
        386513,
    ]

    for aphiaid in aphiaids:

        logging.info(f"Creating summary for AphiaID {aphiaid}")
        summary = sp.get_summary(aphiaid, resolution=5)

        logging.info(f"Creating density for AphiaID {aphiaid}")
        density = sp.get_density(aphiaid, resolution=3, sd=1000)

        logging.info(f"Fetching points for AphiaID {aphiaid}")
        distribution = sp.read_distribution_grid(aphiaid)

        logging.info(f"Fetching points for AphiaID {aphiaid}")
        envelope = sp.get_thermal_envelope(aphiaid)

        sp.export_map(f"../../speedy_output/map_{aphiaid}.html", summary, density, distribution, envelope)


if __name__ == "__main__":
    main()
