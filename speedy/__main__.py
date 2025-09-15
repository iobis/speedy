from speedy import Speedy
from speedy.data import prepare_mr_wkt, download_temperature
from speedy.temperature import get_temperature
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


def main():

    # prepare_mr_wkt()
    # download_temperature()

    sp = Speedy(
        h3_resolution=7,
        cache_marineregions=True,
        cache_summary=False,
        cache_density=True,
        cache_envelope=True,
        ignore_missing_wkt=True
    )

    aphiaids = [
        107119,
        # 212506,
        # 107451
    ]

    for aphiaid in aphiaids:

        # logging.info(f"Creating summary for AphiaID {aphiaid}")
        summary = sp.get_summary(aphiaid, resolution=5, dissolve=True)
        summary.to_file(f"/Users/pieter/Desktop/werk/speedy/speedy_output/summary_{aphiaid}.geojson", driver="GeoJSON")
        summary.to_file(f"/Users/pieter/Desktop/werk/speedy-maps/summary_{aphiaid}.geojson", driver="GeoJSON")

        # logging.info(f"Creating density for AphiaID {aphiaid}")
        # density = sp.get_density(aphiaid, resolution=3, sd=1000)
        # density.to_file(f"../../speedy_output/density_{aphiaid}.geojson", driver="GeoJSON")

        # logging.info(f"Creating points for AphiaID {aphiaid}")
        # distribution = sp.read_distribution_grid(aphiaid)
        # distribution.to_file(f"../../speedy_output/distribution_{aphiaid}.geojson", driver="GeoJSON")

        # logging.info(f"Creating thermal envelope for AphiaID {aphiaid}")
        # envelope = sp.get_thermal_envelope(aphiaid, resolution=5, dissolve=True)
        # envelope.to_file(f"../../speedy_output/envelope_{aphiaid}.geojson", driver="GeoJSON")

        # sp.export_map(f"../../speedy_output/map_{aphiaid}.html", summary, density, distribution, envelope)


if __name__ == "__main__":
    main()
