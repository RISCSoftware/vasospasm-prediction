import pandas as pd


def _drop_columns(X: pd.DataFrame, columns_to_drop: list[str]):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame.")
    return X.drop(columns=[c for c in columns_to_drop if c in X.columns])
