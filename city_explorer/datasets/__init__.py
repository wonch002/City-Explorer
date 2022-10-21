"""File to contain a python reference to all filepaths."""
import os


CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")

from .data_processing import (
    load_uscities as uscities,
    load_income_microdata as income_microdata,
    load_rent_microdata as rent_microdata,
)
