import pandas as pd

from common.classifier_pipelines import get_classifier_pipeline
from common.helper import set_global_seeds


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    clf_code: str,
    inner_split: any,
    random_state: int = 0,
    n_jobs: int = 10,
):
    set_global_seeds(random_state)

    optim_scoring = "balanced_accuracy"

    x = df[feature_cols]
    y = df[target_col]

    search_kwargs = dict(
        scoring=optim_scoring,
        cv=inner_split,
        verbose=1,
        n_jobs=n_jobs,
    )

    class_ratio = y.value_counts(normalize=True)
    class_ratio = class_ratio.iloc[0] / class_ratio.iloc[1]

    gridsearch_clf = get_classifier_pipeline(
        clf_code,
        random_state,
        search_kwargs,
        class_ratio,
        auto_class_ratio=True,
    )

    gridsearch_clf.fit(x, y)
    clf_pipeline = gridsearch_clf.best_estimator_

    return clf_pipeline


def get_model_filename(
    clf_code: str,
    aneurysm_location_mapper: any,
    only_with_aneurysm: bool,
) -> str:
    filename = clf_code
    if only_with_aneurysm:
        filename += "_only-with-aneurysm"
    filename += ".pkl"
    return filename
