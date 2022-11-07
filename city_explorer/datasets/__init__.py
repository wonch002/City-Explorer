"""Module to contain all data loaders specific to each dataset."""
import os


CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")

from .data_processing import (
    load_uscities,
    load_income,
    load_rent,
    load_house_prices,
    load_labor_shed,
    load_age_and_gender_data,
)

from .data_processing import unique_occupations
