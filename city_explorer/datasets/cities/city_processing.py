"""Convert the cities CSV to a geojson file.

The json file should have the following structure:

{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [long, lat],
            },
            "properties": {
                "key_a": "value_a",
                "key_b": "value_b"
            },
        },
    ],
}

NOTE: Anything in the properties object is easily accessible by javascript.
"""
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype


def csv_to_geojson():
    """Convert the USA Cities file to a geojson file.

    NOTE: I had to manually go into the file and update two NANs for zip.

    """
    df_uscities = pd.read_csv("uscities.csv")

    # Convert zips to a list
    df_uscities["zips"] = df_uscities["zips"].str.split(" ")

    # Compute the geo json file
    geo_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [city_record.get("lng"), city_record.get("lat")],
                },
                "properties": city_record,
            }
            for city_record in df_uscities.to_dict(orient="records")
        ],
    }

    return geo_json


if __name__ == "__main__":

    # TODO: Update zip nans to empty list
    with open("us_cities.json", "w") as file:
        json.dump(csv_to_geojson(), file)
