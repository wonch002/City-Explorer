"""Module to contain all data loaders specific to each dataset."""
import os
import shutil
from typing import Callable
import functools
import logging
from joblib import Parallel, delayed

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "cache")
DATA_CACHE_FOLDER = os.path.join(CACHE_FOLDER, "data")

from . import data_processing
from .data_processing import unique_occupations


class CachedData:
    """A class which wraps a data_loader function and enables datacaching."""

    def __init__(
        self,
        func: Callable,
        data_cache_filepath: str,
    ):
        """Initialize a CachedData object cached datset."""
        self.func = func
        self.data_cache_filepath = os.path.join(DATA_CACHE_FOLDER, data_cache_filepath)

    def _datacache_filepath(self, args, kws):
        """Compute the datacache filepath"""
        data_cache_filepath = self.data_cache_filepath

        occupation_title: str = kws.get("occupation_title", None)
        if occupation_title is not None:
            occupation_title = "".join(
                char if char.isalnum() else "_" for char in occupation_title
            ).lower()
            data_cache_filepath += "__" + occupation_title

        data_cache_filepath += ".parquet"
        return data_cache_filepath

    def __call__(self, reset_cache: bool = False, *args, **kws):
        """Call the function and load the dataset."""
        data_cache_filepath = self._datacache_filepath(args=args, kws=kws)

        if reset_cache or not os.path.exists(data_cache_filepath):
            msg = (
                f"Loading data for {self.data_cache_filepath}"
                + f" with the following args and kws {args}, {kws}"
            )
            logger.info(msg)

            df: pd.DataFrame = self.func(*args, **kws)
            if not os.path.exists(DATA_CACHE_FOLDER):
                os.makedirs(DATA_CACHE_FOLDER)
            df.to_parquet(data_cache_filepath)
        else:
            logger.info("Reading data from cache: %s", data_cache_filepath)
            df = pd.read_parquet(data_cache_filepath)

        return df


load_uscities = CachedData(
    func=data_processing.load_uscities,
    data_cache_filepath="us_cities",
)
load_income = CachedData(
    func=data_processing.load_income,
    data_cache_filepath="income",
)
load_rent = CachedData(
    func=data_processing.load_rent,
    data_cache_filepath="rent",
)
load_house_prices = CachedData(
    func=data_processing.load_house_prices,
    data_cache_filepath="house_prices",
)
load_labor_shed = CachedData(
    func=data_processing.load_labor_shed,
    data_cache_filepath="labor_shed",
)
load_age_and_gender_data = CachedData(
    func=data_processing.load_age_and_gender_data,
    data_cache_filepath="age_and_gender_data",
)
load_climate_data = CachedData(
    func=data_processing.load_climate_data,
    data_cache_filepath="climate",
)
load_education = CachedData(
    func=data_processing.load_education,
    data_cache_filepath="education",
)
load_political = CachedData(
    func=data_processing.load_political,
    data_cache_filepath="political",
)


def reset_cache(n_jobs: int = 1):
    """Reset the cached datasets.

    This should be called everytime an update is made.

    NOTE: This function takes ~13 minutes to run with 16 jobs on Cameron's computer.
    """
    funcs = [
        load_uscities,
        load_rent,
        load_house_prices,
        load_labor_shed,
        load_age_and_gender_data,
        load_climate_data,
        load_education,
        load_political,
    ]
    income_funcs = [
        functools.partial(load_income, occupation_title=occupation)
        for occupation in unique_occupations()
    ]

    if os.path.exists(DATA_CACHE_FOLDER):
        shutil.rmtree(DATA_CACHE_FOLDER)

    Parallel(n_jobs=n_jobs)(
        delayed(func)(reset_cache=True) for func in funcs + income_funcs
    )
