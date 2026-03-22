import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer


class CustomQuantileTransformer(BaseEstimator, TransformerMixin):
    # adapted from pytabkit
    def __init__(
        self,
        random_state=None,
        n_quantiles=1000,
        subsample=1_000_000_000,
        output_distribution="normal",
    ):
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        normalizer.fit(X)
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.normalizer_.transform(X)



class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    # encodes missing and unknown values to a value one larger than the known values
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        self.encoder_.fit(X)
        self.cardinalities_ = [len(cats) for cats in self.encoder_.categories_]

        return self

    def transform(self, X):
        check_is_fitted(self, ["encoder_", "cardinalities_"])

        X = pd.DataFrame(X)
        X_enc = self.encoder_.transform(X)

        for col_idx, cardinality in enumerate(self.cardinalities_):
            mask = np.isnan(X_enc[:, col_idx])
            X_enc[mask, col_idx] = cardinality # map missing/unknown to last index

        return X_enc.astype(int)

    def get_cardinalities(self):
        check_is_fitted(self, ["cardinalities_"])
        return self.cardinalities_