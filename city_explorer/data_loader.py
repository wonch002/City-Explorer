"""Tools for loading the datasets."""
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
        df_uscities = datasets.load_uscities()

        # Merge all datasets together
        df_input = df_uscities.copy()

        if data_cache_filepath is not None:

            dirname = os.path.dirname(data_cache_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            df_input.to_parquet(data_cache_filepath)

    return df_input
