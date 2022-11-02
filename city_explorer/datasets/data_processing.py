"""Module for processing all datasets."""
import os

import pandas as pd


# Definitions for filepaths to datasets
USCITIES_FILE = os.path.join(os.path.dirname(__file__), "cities/uscities.csv")
INCOME_MICRODATA_FILE = os.path.join(
    os.path.dirname(__file__), "income/income_microdata.csv"
)
RENT_MICRODATA_FILE = os.path.join(os.path.dirname(__file__), "rent/rent_microdata.csv")
LABOR_SHED_FILE = os.path.join(os.path_dirname(__file__), "geo_regions/labor_shed.csv")
DEMOGRAPHIC_FILE = os.path.join(
    os.path.dirname(__file__), "demographic/nhgis0002_ds249_20205_county.csv"
)


def load_uscities() -> pd.DataFrame:
    """Load and process us cities data."""
    df_uscities = pd.read_csv(USCITIES_FILE)

    # Convert zips to a list
    df_uscities["zips"] = df_uscities["zips"].str.split(" ")

    return df_uscities


def load_age_and_gender_data() -> pd.DataFrame:
    """Load and process demographic dataset."""
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE)

    # County fips is STATE_CODE + COUNTY_CODE (always 3 digits)
    df_demographic["county_fips"] = (
        df_demographic["STATEA"].astype(str)
        + df_demographic["COUNTYA"]
        .astype(str)
        .str.pad(
            width=3,
            side="left",
            fillchar="0",
        )
    ).astype(int)

    column_mappings = {
        "AMPKE001": "Total",
        "AMPKE002": "Male",
        "AMPKE003": "Male: Under 5 years",
        "AMPKE004": "Male: 5 to 9 years",
        "AMPKE005": "Male: 10 to 14 years",
        "AMPKE006": "Male: 15 to 17 years",
        "AMPKE007": "Male: 18 and 19 years",
        "AMPKE008": "Male: 20 years",
        "AMPKE009": "Male: 21 years",
        "AMPKE010": "Male: 22 to 24 years",
        "AMPKE011": "Male: 25 to 29 years",
        "AMPKE012": "Male: 30 to 34 years",
        "AMPKE013": "Male: 35 to 39 years",
        "AMPKE014": "Male: 40 to 44 years",
        "AMPKE015": "Male: 45 to 49 years",
        "AMPKE016": "Male: 50 to 54 years",
        "AMPKE017": "Male: 55 to 59 years",
        "AMPKE018": "Male: 60 and 61 years",
        "AMPKE019": "Male: 62 to 64 years",
        "AMPKE020": "Male: 65 and 66 years",
        "AMPKE021": "Male: 67 to 69 years",
        "AMPKE022": "Male: 70 to 74 years",
        "AMPKE023": "Male: 75 to 79 years",
        "AMPKE024": "Male: 80 to 84 years",
        "AMPKE025": "Male: 85 years and over",
        "AMPKE026": "Female",
        "AMPKE027": "Female: Under 5 years",
        "AMPKE028": "Female: 5 to 9 years",
        "AMPKE029": "Female: 10 to 14 years",
        "AMPKE030": "Female: 15 to 17 years",
        "AMPKE031": "Female: 18 and 19 years",
        "AMPKE032": "Female: 20 years",
        "AMPKE033": "Female: 21 years",
        "AMPKE034": "Female: 22 to 24 years",
        "AMPKE035": "Female: 25 to 29 years",
        "AMPKE036": "Female: 30 to 34 years",
        "AMPKE037": "Female: 35 to 39 years",
        "AMPKE038": "Female: 40 to 44 years",
        "AMPKE039": "Female: 45 to 49 years",
        "AMPKE040": "Female: 50 to 54 years",
        "AMPKE041": "Female: 55 to 59 years",
        "AMPKE042": "Female: 60 and 61 years",
        "AMPKE043": "Female: 62 to 64 years",
        "AMPKE044": "Female: 65 and 66 years",
        "AMPKE045": "Female: 67 to 69 years",
        "AMPKE046": "Female: 70 to 74 years",
        "AMPKE047": "Female: 75 to 79 years",
        "AMPKE048": "Female: 80 to 84 years",
        "AMPKE049": "Female: 85 years and over",
    }

    df_demographic = df_demographic.rename(columns=column_mappings)

    # Relevant age variables
    df_demographic["count_under_10"] = (
        df_demographic["Male: Under 5 years"]
        + df_demographic["Male: 5 to 9 years"]
        + df_demographic["Female: Under 5 years"]
        + df_demographic["Female: 5 to 9 years"]
    )

    df_demographic["count_10_to_20"] = (
        df_demographic["Male: 10 to 14 years"]
        + df_demographic["Male: 15 to 17 years"]
        + df_demographic["Male: 18 and 19 years"]
        + df_demographic["Female: 10 to 14 years"]
        + df_demographic["Female: 15 to 17 years"]
        + df_demographic["Female: 18 and 19 years"]
    )

    df_demographic["count_20_to_30"] = (
        df_demographic["Male: 20 years"]
        + df_demographic["Male: 21 years"]
        + df_demographic["Male: 22 to 24 years"]
        + df_demographic["Male: 25 to 29 years"]
        + df_demographic["Female: 20 years"]
        + df_demographic["Female: 21 years"]
        + df_demographic["Female: 22 to 24 years"]
        + df_demographic["Female: 25 to 29 years"]
    )

    df_demographic["count_30_to_50"] = (
        df_demographic["Male: 30 to 34 years"]
        + df_demographic["Male: 35 to 39 years"]
        + df_demographic["Male: 40 to 44 years"]
        + df_demographic["Male: 45 to 49 years"]
        + df_demographic["Female: 30 to 34 years"]
        + df_demographic["Female: 35 to 39 years"]
        + df_demographic["Female: 40 to 44 years"]
        + df_demographic["Female: 45 to 49 years"]
    )

    df_demographic["count_50_to_65"] = (
        df_demographic["Male: 50 to 54 years"]
        + df_demographic["Male: 55 to 59 years"]
        + df_demographic["Male: 60 and 61 years"]
        + df_demographic["Male: 62 to 64 years"]
        + df_demographic["Female: 50 to 54 years"]
        + df_demographic["Female: 55 to 59 years"]
        + df_demographic["Female: 60 and 61 years"]
        + df_demographic["Female: 62 to 64 years"]
    )

    df_demographic["count_over_65"] = (
        df_demographic["Male: 65 and 66 years"]
        + df_demographic["Male: 67 to 69 years"]
        + df_demographic["Male: 70 to 74 years"]
        + df_demographic["Male: 75 to 79 years"]
        + df_demographic["Male: 80 to 84 years"]
        + df_demographic["Male: 85 years and over"]
        + df_demographic["Female: 65 and 66 years"]
        + df_demographic["Female: 67 to 69 years"]
        + df_demographic["Female: 70 to 74 years"]
        + df_demographic["Female: 75 to 79 years"]
        + df_demographic["Female: 80 to 84 years"]
        + df_demographic["Female: 85 years and over"]
    )

    # Compute relevant percentages
    df_demographic["percent_male"] = df_demographic["Male"] / df_demographic["Total"]
    df_demographic["percent_female"] = (
        df_demographic["Female"] / df_demographic["Total"]
    )
    df_demographic["percent_under_10"] = (
        df_demographic["count_under_10"] / df_demographic["Total"]
    )
    df_demographic["percent_10_to_20"] = (
        df_demographic["count_10_to_20"] / df_demographic["Total"]
    )
    df_demographic["percent_20_to_30"] = (
        df_demographic["count_20_to_30"] / df_demographic["Total"]
    )
    df_demographic["percent_30_to_50"] = (
        df_demographic["count_30_to_50"] / df_demographic["Total"]
    )
    df_demographic["percent_50_to_65"] = (
        df_demographic["count_50_to_65"] / df_demographic["Total"]
    )
    df_demographic["percent_over_65"] = (
        df_demographic["count_over_65"] / df_demographic["Total"]
    )

    columns_to_keep = [
        "county_fips",
        "percent_male",
        "percent_female",
        "percent_under_10",  # Children
        "percent_10_to_20",  # Teens
        "percent_20_to_30",  # Young adults
        "percent_30_to_50",  # Middle Aged
        "percent_50_to_65",  # Older
        "percent_over_65",  # Retired
    ]
    return df_demographic[columns_to_keep]


def load_income_microdata() -> pd.DataFrame:
    """Load microdata for income."""
    df_income_microdata = pd.read_csv(INCOME_MICRODATA_FILE)

    return df_income_microdata


def load_rent_microdata() -> pd.DataFrame:
    """Load microdata for rent."""
    df_rent_microdata = pd.read_csv(RENT_MICRODATA_FILE)

    return df_rent_microdata


def load_labor_shed() -> pd.DataFrame:
    """Load Labor Shed Delineation Data"""

    df_labor_shed = pd.read_csv(LABOR_SHED_FILE)

    return df_labor_shed
