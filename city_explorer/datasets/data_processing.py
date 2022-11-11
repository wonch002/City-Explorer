"""Module for processing all datasets."""
import os

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Definitions for filepaths to datasets
DATAPATH = os.path.join(os.path.dirname(__file__), "data")

# All uscities
USCITIES_FILE = os.path.join(DATAPATH, "uscities.csv")

# Income
CBSA_TO_COUNTYFIPS_FILE = os.path.join(DATAPATH, "cbsa_to_countyfips.csv")
NECTA_TO_COUNTYFIPS_FILE = os.path.join(DATAPATH, "necta_to_countyfips.csv")
INCOME_FILE = os.path.join(DATAPATH, "MSA_M2021_dl.csv")

# Rental
RENT_FILE = os.path.join(DATAPATH, "FY2023_FMR_50_county.csv")
HOUSE_PRICES_FILE = os.path.join(DATAPATH, "house_prices.csv")

# Laborshed
LABOR_SHED_FILE = os.path.join(DATAPATH, "labor_shed.csv")

# Demographic
DEMOGRAPHIC_FILE = os.path.join(DATAPATH, "nhgis0002_ds249_20205_county.csv")

# Climate
CLIMATE_FILE = os.path.join(DATAPATH, "climate.csv")



def _feature_county_fips(
        df: pd.DataFrame,
        state_code_col: str,
        county_code_col: str,
) -> pd.Series:
    """Compute the county fips code from the dataframe.
    County fips is the state code + the county code. The county code must always be
    three digits, so we pad with zero.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe to process.
    state_code_col : str
        The column name that represents the state code.
    county_code_col : str
        The column name that represents the county code.
    Returns
    -------
    pd.Series
        A pandas series with the county code
    Example
    -------
    If the 'state code` is 6 and the `county code` is 75, then the county fips is 6075.
    >>> df["county_fips"] = _feature_county_fips(
            df=df, state_code_col="state_code", county_code_col="county_code"
        )
    """
    county_fips = (
        (
                df[state_code_col].astype(str)
                + df[county_code_col]
                .astype(str)
                .str.pad(
            width=3,
            side="left",
            fillchar="0",
        )
        )
            .astype(int)
            .rename("county_fips")
    )

    return county_fips


def _county_coordinates():
    """Return the longitude and latitude coordinates of each county."""
    df_uscities = load_uscities()
    # Group each city by their county and get the center point (long and lat)
    # https://laracasts.com/discuss/channels/laravel/calculating-center-point-using-geo-latitude-and-longitude-values
    county_coordinates = (
        df_uscities.groupby("county_fips").mean()[["lng", "lat"]].reset_index()
    )

    return county_coordinates


def load_uscities() -> pd.DataFrame:
    """Load and process us cities data."""
    df_uscities = pd.read_csv(USCITIES_FILE)

    # Convert zips to a list
    df_uscities["zips"] = df_uscities["zips"].str.split(" ")

    return df_uscities

def load_climate_date() -> pd.DataFrame:
    """Load and process us climate data."""
    df = pd.read_csv(CLIMATE_FILE)

    df = df[~ df.Jan_max_temp == 999]

    return df


def load_age_and_gender_data() -> pd.DataFrame:
    """Load and process demographic dataset."""
    df_demographic = pd.read_csv(DEMOGRAPHIC_FILE)

    # County fips is STATE_CODE + COUNTY_CODE (always 3 digits)
    df_demographic["county_fips"] = _feature_county_fips(
        df=df_demographic, state_code_col="STATEA", county_code_col="COUNTYA"
    )

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


def unique_occupations(format: bool = False) -> np.ndarray:
    """Return all unique occuptions."""
    df_income = pd.read_csv(INCOME_FILE)
    unique_occupations = df_income[["OCC_TITLE", "OCC_CODE"]].drop_duplicates()
    unique_occupations["OCC_CODE"] = (
        unique_occupations["OCC_CODE"].str.replace("-", "").astype(int)
    )
    unique_occupations = unique_occupations.sort_values("OCC_CODE")

    if format:
        unique_occupations = unique_occupations.to_numpy()
        occupations = []
        for occupation in unique_occupations:
            if occupation[1] == 0:
                occupations.append(occupation[0])
            elif str(occupation[1]).endswith("0000"):
                occupations.append("\t" + occupation[0])
            else:
                occupations.append("\t\t" + occupation[0])

    else:
        occupations = unique_occupations["OCC_TITLE"].to_numpy()

    return occupations


def load_income(occupation_title: str = "All Occupations") -> pd.DataFrame:
    """Load and process dataset for income.
    Processing of income data includes imputing missing income with the average income
    values of the three nearest counties.
    Parameters
    ----------
    occupation_title : str, optional
        The title of the occupation to load the income data for. Default is
        'All Occupations'.
    Returns
    -------
    pd.DataFrame
        A dataframe with associated income data at the county level.
    """
    # Load mapping between cbsa code and county fips
    df_cbsa_to_county_mapping = pd.read_csv(CBSA_TO_COUNTYFIPS_FILE).rename(
        columns={"CBSA Code": "msa_code"}
    )[["msa_code", "FIPS State Code", "FIPS County Code"]]
    df_necta_to_county_mapping = pd.read_csv(NECTA_TO_COUNTYFIPS_FILE).rename(
        columns={"NECTA Code": "msa_code"}
    )[["msa_code", "FIPS State Code", "FIPS County Code"]]

    df_county_mapping = (
        pd.concat([df_cbsa_to_county_mapping, df_necta_to_county_mapping])
            .dropna()
            .astype(int)
    )

    df_county_mapping["county_fips"] = _feature_county_fips(
        df=df_county_mapping,
        state_code_col="FIPS State Code",
        county_code_col="FIPS County Code",
    )

    # Load income dataset and merge country fips
    df_income = pd.read_csv(INCOME_FILE)

    # Filter on the occupation title
    all_occupations = unique_occupations()

    if occupation_title not in all_occupations:
        formatted_occupations = unique_occupations(format=True)
        formatted_help_msg = "\n" + "\n".join(formatted_occupations)
        raise ValueError(
            f"`{occupation_title}` is not a valid occupation. "
            + "Please select an occupation from the following list:\n"
            + f"\t{formatted_help_msg}"
        )

    # Filter to the requested occupation
    is_occupation = df_income["OCC_TITLE"] == occupation_title
    df_income_filtered = df_income.loc[is_occupation]
    df_income_filtered = df_income_filtered.merge(
        df_county_mapping,
        left_on="AREA",
        right_on="msa_code",
        how="inner",
    )

    columns_to_keep = [
        "county_fips",
        "H_MEAN",
        "A_MEAN",
        "H_PCT10",
        "H_PCT25",
        "H_MEDIAN",
        "H_PCT75",
        "H_PCT90",
        "A_PCT10",
        "A_PCT25",
        "A_MEDIAN",
        "A_PCT75",
        "A_PCT90",
    ]

    df_income_filtered = df_income_filtered[columns_to_keep]

    # Process all columns
    for col in columns_to_keep:
        if col != "county_fips":
            # *  = indicates that a wage estimate is not available
            # **  = indicates that an employment estimate is not available
            df_income_filtered = df_income_filtered[
                ~df_income_filtered[col].isin(["*", "**"])
            ]

            # "#  = indicates a wage equal to or greater than $100.00 per hour or
            # $208,000 per year ",,,,
            replace_value = "100.00" if col.startswith("H") else "208,000"
            df_income_filtered[col] = (
                df_income_filtered[col].astype(str).replace("#", replace_value)
            )

            df_income_filtered[col] = (
                df_income_filtered[col].str.replace(",", "").astype(float)
            )

    # Collapse duplicated counties into their mean. This happens b/c the mapping from
    # msa_code -> county_fips is not neccessarily 1:1
    df_income_filtered = df_income_filtered.groupby("county_fips").mean().reset_index()

    # Impute missing income with the average of the 3 nearest counties
    df_county_coordinates = _county_coordinates()
    income_is_known = df_county_coordinates["county_fips"].isin(
        df_income_filtered["county_fips"]
    )
    df_coords_income_known = df_county_coordinates.loc[income_is_known].reset_index()
    df_coords_income_not_known = df_county_coordinates.loc[
        ~income_is_known
    ].reset_index()

    # Fit a nearestneighbors model with our known income, so we can look them up
    neighbors_model = NearestNeighbors(n_neighbors=3, metric="haversine")
    neighbors_model.fit(df_coords_income_known[["lat", "lng"]])

    # For all unknown incomes, impute a new income
    _, indices = neighbors_model.kneighbors(df_coords_income_not_known[["lat", "lng"]])
    all_imputed_incomes = []
    for i, matching_indices in enumerate(indices):
        matching_counties = df_coords_income_known.iloc[matching_indices]["county_fips"]
        imputed_income = (
            df_income_filtered[
                df_income_filtered["county_fips"].isin(matching_counties)
            ]
                .drop(["county_fips"], axis=1)
                .mean()
        ).to_dict()

        imputed_income["county_fips"] = int(
            df_coords_income_not_known.iloc[i]["county_fips"]
        )

        all_imputed_incomes.append(imputed_income)
    df_all_imputed_incomes = pd.DataFrame(all_imputed_incomes)

    # Finally, combine the imputed incomes with the actual incomes
    df_income_combined = pd.concat([df_income_filtered, df_all_imputed_incomes])

    return df_income_combined


def load_rent() -> pd.DataFrame:
    """Load rent dataset."""
    df_rent = pd.read_csv(RENT_FILE)

    df_rent["county_fips"] = _feature_county_fips(
        df_rent, state_code_col="state_code", county_code_col="county_code"
    )

    # Compute average rent. These values are the 50th percentile, so we are taking the
    # average of the four medians.
    df_rent["rent_50_avg"] = df_rent[
        [
            "rent_50_0",
            "rent_50_1",
            "rent_50_2",
            "rent_50_3",
            "rent_50_4",
        ]
    ].mean(axis=1)

    columns_to_keep = [
        "county_fips",
        "rent_50_avg",
        "rent_50_0",
        "rent_50_1",
        "rent_50_2",
        "rent_50_3",
        "rent_50_4",
    ]

    return df_rent[columns_to_keep]


def load_house_prices() -> pd.DataFrame:
    """Load and process dataset for house prices."""
    df_house_prices = pd.read_csv(HOUSE_PRICES_FILE)

    column_mapping = {
        "Full County Number": "county_fips",
        "Median Home Price 5year 2020": "home_price_5yr_median",
        "Q1 2022": "home_price_2022_q1_median",
    }
    df_house_prices = df_house_prices.rename(columns=column_mapping)

    # Convert current home prices to a float
    df_house_prices["home_price_2022_q1_median"] = (
        df_house_prices["home_price_2022_q1_median"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
    )

    return df_house_prices[column_mapping.values()].dropna()


def load_labor_shed() -> pd.DataFrame:
    """Load Labor Shed Delineation Data"""

    df_labor_shed = pd.read_csv(LABOR_SHED_FILE)

    return df_labor_shed