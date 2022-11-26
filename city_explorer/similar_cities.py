"""Contains tools and functions for computing similar cities."""
# standard
from typing import Callable, Dict, List

# external
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# internal
import data_loader

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


class TransformerPandasSupportMixin:
    """This is a simple mixin which adds Pandas support to our transformers."""

    @staticmethod
    def _preprocess(X, y=None):
        """Apply preprocessing steps."""
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1, 1)

        return X, y

    @staticmethod
    def _postprocess(x, x_transformed):
        """Post processing steps to convert back to a pandas objecta¸™dx."""
        if isinstance(x, pd.Series):
            return pd.Series(x_transformed[:, 0], x.index)
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(x_transformed, x.index, columns=x.columns)

        return x_transformed

    def fit(self, X, y=None, sample_weight=None):
        """Apply preprocessing steps before fitting."""
        _X, _y = self._preprocess(X, y)
        return super().fit(_X, _y, sample_weight)

    def fit_transform(self, X, y=None, **fit_params):
        """Apply preprocessing steps before fitting, then apply postprocessing."""
        _X, _y = self._preprocess(X, y)
        x_transformed = super().fit_transform(_X, _y, **fit_params)
        return self._postprocess(X, x_transformed)

    def transform(self, X, copy=None):
        """Apply preprocessing and postprocessing for transforming."""
        _X, _ = self._preprocess(X)
        x_transformed = super().transform(_X, copy)
        return self._postprocess(X, x_transformed)


class StandardScaler(TransformerPandasSupportMixin, StandardScaler):
    """Adding pandas support to StandardScaler."""


class SimilarCities:
    """A class to predict similar cities."""

    def __init__(
        self,
        similarity_func: Callable = euclidean_distances,
        scaler: BaseEstimator = StandardScaler,
        feature_weights: Dict[str, float] = ModuleNotFoundError,
    ):
        """Initialize the SimilarCities object.

        Parameters
        ----------
        similarity_func : Callable
            A callable which accepts X and Y. Returns a similarity matrix.

        scaler : BaseEstimator
            A scaler which describes how to normalize the dataset.

        feature_weights : Dict[str, float]
            A dictionary which maps a feature to its corresponding weight.
        """
        self.similarity_func = similarity_func
        self.scaler = scaler()
        self.feature_weights = feature_weights

    def get_features(self, data: pd.DataFrame):
        """Return the subset of features that will be used in the similar city metric."""

        if self.feature_weights is None:
            feature_names = data.select_dtypes("number").columns
        else:
            feature_names = list(self.feature_weights.keys())

        df_features = data[feature_names]

        # Drop id column if it exists
        if "id" in df_features:
            df_features = df_features.drop(["id"], axis=1)

        # Merge id back in and set as the index
        df_features = df_features.merge(data["id"], left_index=True, right_index=True)
        df_features = df_features.set_index("id")

        # df_features = df_features.dropna(axis=1).copy()
        return df_features

    def _apply_feature_weights(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature weights to the transformed dataset."""
        if self.feature_weights is None:
            return df_transformed

        # Apply feature weighting
        for feature_name, feature_weight in self.feature_weights.items():
            df_transformed[feature_name] *= feature_weight

        return df_transformed

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        df_features = self.get_features(data=data)
        df_transformed = self.scaler.transform(df_features)
        df_transformed = self._apply_feature_weights(df_transformed=df_transformed)

        return df_transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        df_features = self.get_features(data=data)
        df_transformed = self.scaler.fit_transform(df_features)
        df_transformed = self._apply_feature_weights(df_transformed=df_transformed)

        return df_transformed

    def fit(self, data: pd.DataFrame):
        """Fit the scaler."""
        df_features = self.get_features(data=data)
        self.scaler.fit(df_features)
        return self

    def predict(self, data: pd.DataFrame, city_id: int):
        """Return the list of similar cities."""
        df_transformed = self.transform(data=data)
        df_compare = df_transformed.loc[city_id].to_frame().T
        _result = self.similarity_func(X=df_transformed, Y=df_compare)[:, 0]
        result = pd.Series(
            _result, index=data["id"], name="similarity_score"
        ).sort_values()

        return result


def get_feature_weights(*sliders):
    """Compute feature weights given the slider inputs."""
    # TODO:
    return dict(
        population=1,
        home_price_5yr_median=10,  # 10x as important
        A_MEDIAN=1,
    )


def predict_similar_cities(
    city_id: int,
    occupation_title: str,
    *sliders: List[float],  # TODO
    limit: int = None,
) -> pd.Series:
    """Compute similar cities based on the given criteria.

    Parameters
    ----------
    city_id : int
        The id of the selected city.

    occupation_title : str
        The name of the selected occupation. On the frontend, the default should be
        `All Occupations`

    *sliders : List[float]
        A list of slider values which correspond the users importances of each feature.

    limit : int, optional
        The number of similar cities to show. Default is no limit.

    Returns
    -------
    pd.Series
        An ordered series which contains the similairty score for each city based on
        the specified criteria.

        Index : int
            city_id
        Value : float
            Similarity score
    """
    df_input = data_loader.load_input_data(occupation_title=occupation_title)
    feature_weights = get_feature_weights(*sliders)
    similar_cities_estimator = SimilarCities(feature_weights=feature_weights)
    similar_cities_estimator.fit(df_input)

    similar_cities = similar_cities_estimator.predict(data=df_input, city_id=city_id)
    if limit is not None:
        similar_cities = similar_cities.iloc[:limit]

    return similar_cities.copy()
