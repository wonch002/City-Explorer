"""Contains tools and functions for computing similar cities."""
# external
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# internal
# from .data_loader import load_input_data

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


def get_scaled_df(df, subset=None):
    """Compute and scaled dataframe."""
    if subset is None:
        subset = (
            df._get_numeric_data().columns
        )  # select_dtypes(include=numerics).columnns

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[subset])

    return df_scaled


class SimilarCities:
    def __init__(
        self,
        similarity_metric: str = "euclidean",
        scaler: BaseEstimator = StandardScaler,
    ):
        self.similarity_metric = similarity_metric
        self.scaler = scaler()

    def get_features(self, data: pd.DataFrame):
        """Return the subset of features that will be used in the similar city metric."""
        subset = data._get_numeric_data().columns
        df_features = data[subset]
        df_features = df_features.dropna(axis=1).copy()

        df_features = df_features.set_index("id")
        return df_features

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        df_features = self.get_features(data=data)
        df_transformed = self.scaler.transform(df_features)

        return df_transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        df_features = self.get_features(data=data)
        df_transformed = self.scaler.fit_transform(df_features)

        return df_transformed

    def fit(self, data: pd.DataFrame):
        """Fit the scaler."""
        df_features = self.get_features(data=data)
        self.scaler.fit(df_features)
        return self

    def predict(self, data: pd.DataFrame, city_id: int):
        """Return the list of similar cities."""
        df_transformed = self.transform(data=data)
        df_compare = 
        if self.similarity_metric == "euclidean":
            return euclidean_distances(X=df_transformed)
        else:
            raise NotImplementedError(f"{self.similarity_metric} is not supported.")
