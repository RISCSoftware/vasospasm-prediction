from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class Cols:
    age_diagnose = "age_diagnose"
    sex = "sex"
    hunt_hess = "hunt_hess"
    fisher = "fisher"
    cns_infection = "cns_infection"
    EVD = "EVD"
    shunt = "shunt"
    aneurysm = "aneurysm"
    aneurysm_diameter = "aneurysm_diameter"
    aneurysm_height = "aneurysm_height"
    aneurysm_location = "aneurysm_location"
    aneurysm_location_cat = "aneurysm_location_cat"
    aneurysm_treatment = "aneurysm_treatment"

    # The following features were experimental and aren't used in the final model
    aneurysm_area = "aneurysm_area"
    aneurysm_len_sum = "aneurysm_len_sum"


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column_transformer: ColumnTransformer,
        features_to_drop: list[str] | None = None,
    ):
        if features_to_drop is None:
            features_to_drop = []

        self.column_transformer = column_transformer
        self.new_features = None
        self.drop_features = features_to_drop

        self.categorical_features = [x for t in column_transformer.transformers for x in t[-1]]
        self.drop_original_features = False

    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        new_features = self.column_transformer.get_feature_names_out()
        new_features = [f for f in new_features if f not in self.drop_features]
        self.new_features = new_features
        return self

    def transform(self, X, y=None):
        X_new = self.column_transformer.transform(X)
        X_new = pd.DataFrame(X_new, columns=self.column_transformer.get_feature_names_out(), index=X.index)
        X_new = X_new[self.new_features]

        if self.drop_original_features:
            features = [f for f in X.columns if f not in self.categorical_features]
            X = X[features]

        X = pd.concat([X, X_new], axis=1)
        if y is not None:
            return X, y
        return X


def get_sex_feature_name(feature: str, category: str) -> str:
    return f"{category}"


def get_location_cat_feature_name(feature: str, category: str) -> str:
    return f"location_{category}"


def get_treatment_feature_name(feature: str, category: str) -> str:
    return f"aneurysm_{category}"


def get_categorical_columns(features: list[str]) -> list[str]:
    categorical_columns = [Cols.sex, Cols.aneurysm_location_cat, Cols.aneurysm_treatment]
    categorical_columns = [c for c in categorical_columns if c in features]
    return categorical_columns


def get_one_hot_encoder(
    categorical_columns: list[str],
    features_to_drop: list[str] | None = None,
) -> CustomOneHotEncoder:
    if features_to_drop is None:
        features_to_drop = []
    ct_list = []
    oh_kwargs = dict(handle_unknown="ignore", dtype=int, sparse_output=False)
    if Cols.sex in categorical_columns:
        fnc = get_sex_feature_name
        sex_encoder = OneHotEncoder(feature_name_combiner=fnc, **oh_kwargs)
        ct_list.append(("sex_encoder", sex_encoder, [Cols.sex]))
    if Cols.aneurysm_location_cat in categorical_columns:
        fnc = get_location_cat_feature_name
        location_encoder = OneHotEncoder(feature_name_combiner=fnc, **oh_kwargs)
        ct_list.append(("location_encoder", location_encoder, [Cols.aneurysm_location_cat]))
    if Cols.aneurysm_treatment in categorical_columns:
        fnc = get_treatment_feature_name
        treatment_encoder = OneHotEncoder(feature_name_combiner=fnc, **oh_kwargs)
        ct_list.append(("treatment_encoder", treatment_encoder, [Cols.aneurysm_treatment]))

    one_hot_encoder = CustomOneHotEncoder(
        ColumnTransformer(
            ct_list,
            remainder="drop",
            verbose_feature_names_out=False,
        ),
        features_to_drop=features_to_drop,
    )
    return one_hot_encoder


def process_categorical_columns(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy(deep=False)

    categorical_columns = get_categorical_columns(features)
    df[categorical_columns] = df[categorical_columns].astype("category")

    return df, features


def process_aneurysm_columns(
    df: pd.DataFrame,
    aneurysm_location_mapper: Callable,
    impute=True,
) -> pd.DataFrame:
    df = df.copy(deep=True)

    # Set aneurysm location category
    df[Cols.aneurysm_location_cat] = df[Cols.aneurysm_location].map(aneurysm_location_mapper)
    df[Cols.aneurysm_location_cat] = df[Cols.aneurysm_location_cat].astype("category")

    # Set aneurysm area
    df[Cols.aneurysm_area] = df[Cols.aneurysm_diameter] * df[Cols.aneurysm_height]

    # Set sum of lengths
    df[Cols.aneurysm_len_sum] = df[Cols.aneurysm_diameter] + df[Cols.aneurysm_height]

    aneurysm_mask = df[Cols.aneurysm] == True

    if impute:
        # # Impute aneurysm_diameter and aneurysm_height
        filler = np.nanmean(df.loc[aneurysm_mask, Cols.aneurysm_diameter])
        df.loc[aneurysm_mask, Cols.aneurysm_diameter] = df.loc[aneurysm_mask, Cols.aneurysm_diameter].fillna(filler)
        filler = np.nanmean(df.loc[aneurysm_mask, Cols.aneurysm_height])
        df.loc[aneurysm_mask, Cols.aneurysm_height] = df.loc[aneurysm_mask, Cols.aneurysm_height].fillna(filler)

        # Impute aneurysm area
        filler = np.nanmean(df.loc[aneurysm_mask, Cols.aneurysm_area])
        df.loc[aneurysm_mask, Cols.aneurysm_area] = df.loc[aneurysm_mask, Cols.aneurysm_area].fillna(filler)

        # Impute sum of lengths
        filler = np.nanmean(df.loc[aneurysm_mask, Cols.aneurysm_len_sum])
        df.loc[aneurysm_mask, Cols.aneurysm_len_sum] = df.loc[aneurysm_mask, Cols.aneurysm_len_sum].fillna(filler)

    # Set lengths and area to zero for non-aneurysmatic cases
    cols = [Cols.aneurysm_diameter, Cols.aneurysm_height, Cols.aneurysm_area, Cols.aneurysm_len_sum]
    for col in cols:
        df.loc[~aneurysm_mask, col] = df.loc[~aneurysm_mask, col].fillna(0)
        df[col] = df[col].astype(float)

    return df


class AneurysmLocationMapper0:
    categories = [
        ["MCA"],
        ["ACA", "ACom"],
        ["ICA", "PCom"],
        ["PCA", "AICA", "BA", "PICA", "SCA", "VA"],
    ]

    cat_features = ["_".join(c) for c in categories]
    cat_code = "_".join([f"[{f}]" for f in cat_features])

    @staticmethod
    def get(location_str):
        if not isinstance(location_str, str) or len(location_str) == 0:
            return None

        for i, cat in enumerate(AneurysmLocationMapper0.categories):
            if location_str in cat:
                return AneurysmLocationMapper0.cat_features[i]


class AneurysmLocationMapper1:
    cat_features = [
        "ACom",
        "notACom",
    ]
    cat_code = "ACom_rest"

    @staticmethod
    def get(location_str):
        if not isinstance(location_str, str) or len(location_str) == 0:
            return None

        if location_str == "ACom":
            return AneurysmLocationMapper1.cat_features[0]
        else:
            return AneurysmLocationMapper1.cat_features[1]


class AneurysmLocationMapper2:
    cat_features = []
    cat_code = "drop"

    @staticmethod
    def get(location_str):
        return None


def process_dataset(
    df: pd.DataFrame,
    features: list[str],
    aneurysm_location_mapper: type,
    only_with_aneurysm: bool,
    encode_one_hot: bool,
    impute: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if features is None:
        raise ValueError("features must be provided.")

    if only_with_aneurysm:
        df = df.loc[df[Cols.aneurysm] == True]

    df = process_aneurysm_columns(df, aneurysm_location_mapper.get, impute=impute)

    df, features = process_categorical_columns(df, features)

    if encode_one_hot:
        df, features, _ = one_hot_encode_dataset(df, features)

    return df, features


def one_hot_encode_dataset(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str], CustomOneHotEncoder]:
    features = list(features).copy()

    categorical_columns = get_categorical_columns(features)

    # Determine which category to drop for each categorical column
    features_to_drop = []

    for col in categorical_columns:
        if col == Cols.sex:
            fnc = get_sex_feature_name
            drop_candidates = ["male"]
        elif col == Cols.aneurysm_location_cat:
            fnc = get_location_cat_feature_name
            drop_candidates = [float("nan"), "notACom"]
        elif col == Cols.aneurysm_treatment:
            fnc = get_treatment_feature_name
            drop_candidates = [float("nan"), "untreated"]
        else:
            drop_candidates = df[col].unique().tolist()
            fnc = lambda c, v: f"{c}_{v}"

        unique = df[col].unique()
        # Filter drop candidates to those that are present in the column
        drop_candidates = [c for c in drop_candidates if c in unique]
        if not len(drop_candidates):
            # Revert to the first unique value if no candidates found
            drop_candidates = unique
        drop = fnc(col, drop_candidates[0])
        features_to_drop.append(drop)

    one_hot_encoder = get_one_hot_encoder(categorical_columns, features_to_drop)

    df = one_hot_encoder.fit_transform(df)

    new_features = one_hot_encoder.new_features

    assert len(set(features).intersection(new_features)) == 0, "Feature name collision!"
    features.extend(new_features)
    features = [f for f in features if f not in categorical_columns]

    return df, features, one_hot_encoder
