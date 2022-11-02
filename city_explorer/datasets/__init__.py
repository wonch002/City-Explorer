"""Module to contain all data loaders specific to each dataset."""
import os


CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")

from .data_processing import (
    load_uscities,
    load_income_microdata,
    load_rent_microdata,
    load_age_and_gender_data,
)
