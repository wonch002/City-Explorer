"""Module for processing all datasets."""

import os

import pandas as pd


# Definitions for
USCITIES_FILE = os.path.join(os.path.dirname(__file__), "cities/uscities.csv")
INCOME_MICRODATA_FILE = os.path.join(
    os.path.dirname(__file__), "income/income_microdata.csv"
)
RENT_MICRODATA_FILE = os.path.join(os.path.dirname(__file__), "rent/rent_microdata.csv")


def load_uscities() -> pd.DataFrame:
    """Load and process us cities data."""
    df_uscities = pd.read_csv(USCITIES_FILE)

    # Convert zips to a list
    df_uscities["zips"] = df_uscities["zips"].str.split(" ")

    return df_uscities


def load_income_microdata() -> pd.DataFrame:
    """Load microdata for income."""
    df_income_microdata = pd.read_csv(INCOME_MICRODATA_FILE)

    return df_income_microdata


def load_rent_microdata() -> pd.DataFrame:
    """Load microdata for rent."""
    df_rent_microdata = pd.read_csv(RENT_MICRODATA_FILE)

    return df_rent_microdata
