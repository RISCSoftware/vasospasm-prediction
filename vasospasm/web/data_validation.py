import pandas as pd

from vasospasm.const import Cols

value_instances = {
    Cols.sex: ["male", "female"],
    Cols.hunt_hess: [1, 2, 3, 4, 5],
    Cols.fisher: [1, 2, 3, 4],
    Cols.cns_infection: [0, 1],
    Cols.EVD: [0, 1],
    Cols.aneurysm_treatment: ["untreated", "clip", "coil", None],
    Cols.aneurysm_location: ["AICA", "SCA", "BA", "ICA", "PICA", "ACom", "ACA", "PCom", "VA", "MCA", "PCA", None],
    Cols.aneurysm: [0, 1],
}
value_ranges = {
    Cols.age_diagnose: [0, 100],
    Cols.aneurysm_diameter: [0, 25],
    Cols.aneurysm_height: [0, 25],
}
is_nullable = {
    Cols.aneurysm_diameter: True,
    Cols.aneurysm_height: True,
}


def validate_data(df: pd.DataFrame):
    validation_results = {}

    # Validate instances
    for col, valid_values in value_instances.items():
        if col in df.columns:
            validation_results[f"{col}"] = bool(df[col].isin(valid_values).all())

    # Validate ranges
    for col, (min_val, max_val) in value_ranges.items():
        if col in df.columns:
            range_ok = df[col].between(min_val, max_val)
            if is_nullable.get(col, False):
                range_ok = range_ok | df[col].isnull()
            validation_results[f"{col}"] = range_ok.all()

    # Validate all or none aneurysm-related columns are None
    aneurysm_columns = [Cols.aneurysm_location, Cols.aneurysm_diameter, Cols.aneurysm_height, Cols.aneurysm_treatment]
    aneurysm_columns_present = [c for c in df.columns if c in aneurysm_columns]
    if aneurysm_columns_present:
        df_aneurysm = df[aneurysm_columns_present]
        validation_results["aneurysm_columns_all_or_none"] = (
            df_aneurysm.isnull().all(axis=1).equals(df_aneurysm.isnull().any(axis=1))
        )

    return validation_results
