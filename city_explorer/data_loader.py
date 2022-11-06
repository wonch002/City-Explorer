"""Tools for returning prepared datasets."""
import os
import logging

import datasets
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_data(data_cache_filepath: str = None) -> pd.DataFrame:
    """Load all available data into a single DataFrame.

    Parameters
    ----------
    data_cache_filepath : str, optional
        The name of a file to save/load your data extract from. Default behavior is
        None.

    Returns
    -------
    pd.DataFrame
        Dataset with all available data.
    """
    if data_cache_filepath is not None:
        data_cache_filepath = os.path.join(
            datasets.CACHE_FOLDER,
            data_cache_filepath,
        )

    if data_cache_filepath is not None and os.path.exists(data_cache_filepath):
        logger.info("Reading data from cache: %s", data_cache_filepath)
        df_input = pd.read_parquet(data_cache_filepath)
    else:
        # Load all data
        df_uscities = datasets.load_uscities()
        df_laborshed = datasets.load_labor_shed()
        df_age_and_gender = datasets.load_age_and_gender_data()
        df_rent = datasets.load_rent()
        df_house_prices = datasets.load_house_prices()

        # Merge all datasets together
        df_input = df_uscities.copy()
        df_input = df_input.merge(
            right=df_laborshed,
            left_on="county_fips",
            right_on="FIPS",
            how="inner",
        )
        df_input = df_input.merge(
            right=df_age_and_gender,
            left_on="county_fips",
            right_on="county_fips",
            how="inner",
        )
        df_input = df_input.merge(
            right=df_rent,
            left_on="county_fips",
            right_on="county_fips",
            how="inner",
        )
        df_input = df_input.merge(
            right=df_house_prices,
            left_on="county_fips",
            right_on="county_fips",
            how="inner",
        )

        if data_cache_filepath is not None:

            dirname = os.path.dirname(data_cache_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            df_input.to_parquet(data_cache_filepath)

    return df_input
