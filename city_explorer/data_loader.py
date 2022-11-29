"""Tools for returning prepared datasets."""
import os
import logging

import datasets
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_data(
    occupation_title: str = "All Occupations",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load all available data into a single DataFrame.

    Parameters
    ----------
    occupation_title : str, optional
        The title of the occupation to load the income data for. Default is
        'All Occupations'. Pass "help" to display all possible values.

    use_cache : bool, optional
        Whether to use a cached dataset or not. Default behavior is True.

    Returns
    -------
    pd.DataFrame
        Dataset with all available data.
    """
    if occupation_title == "help":
        unique_occupations = datasets.unique_occupations(format=True)
        formatted_help_msg = "\n" + "\n".join(unique_occupations)
        logger.info(formatted_help_msg)
        return
    else:
        # Load all data

        reset_cache = not use_cache
        df_uscities = datasets.load_uscities(reset_cache=reset_cache)
        df_laborshed = datasets.load_labor_shed(reset_cache=reset_cache)
        df_age_and_gender = datasets.load_age_and_gender_data(reset_cache=reset_cache)
        df_rent = datasets.load_rent(reset_cache=reset_cache)
        df_income = datasets.load_income(
            reset_cache=reset_cache,
            occupation_title=occupation_title,
        )
        df_house_prices = datasets.load_house_prices(reset_cache=reset_cache)
        df_climate = datasets.load_climate_data(reset_cache=reset_cache)
        df_political = datasets.load_political()
        df_education = datasets.load_education()

        # Merge all datasets together
        df_input = df_uscities.copy()
        df_input = df_input.merge(
            right=df_climate,
            left_on=["city", "state_id"],
            right_on=["city", "state_id"],
            how="inner",
        )
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
            right=df_income,
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
        df_input = df_input.merge(
            right=df_education, left_on="county_fips", right_on="FIPS", how="inner"
        )
        df_input = df_input.merge(
            right=df_political,
            left_on="county_fips",
            right_on="county_fips",
            how="inner",
        )

    return df_input
