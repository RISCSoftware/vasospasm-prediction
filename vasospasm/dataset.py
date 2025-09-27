import pandas as pd

from common.dataset import AneurysmLocationMapper1, process_dataset
from vasospasm import config
from vasospasm.const import Cols, vasospasm_features


def get_dataset(
    csv: str = config.paths.dataset_csv,
    features: list[str] | None = None,
    aneurysm_location_mapper=AneurysmLocationMapper1,
    only_with_aneurysm: bool = False,
    encode_one_hot: bool = True,
    impute: bool = True,
) -> tuple[pd.DataFrame, list[str], str]:
    if features is None:
        features = vasospasm_features.copy()

    df = pd.read_csv(csv)
    df, features = process_dataset(df, features, aneurysm_location_mapper, only_with_aneurysm, encode_one_hot, impute)
    target = Cols.vasospasm

    return df, features, target


def demo():
    df, features, target = get_dataset()
    df.to_csv(config.paths.data / "processed_vasospasm.csv", index=False)


if __name__ == "__main__":
    demo()
