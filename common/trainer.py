import pandas as pd
from sklearn.model_selection import cross_validate

from common.classifier_pipelines import get_classifier_pipeline
from common.helper import set_global_seeds
from common.metrics import get_feature_importance, get_fpr_tpr_precision_recall, get_multi_scoring, get_scores_stats


def train_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    clf_code: str,
    split: any,
    inner_split: any,
    random_state: int = 0,
    return_estimators: bool = False,
    return_curves: bool = False,
    return_feature_importance: bool = False,
    n_jobs: int = 10,
):
    set_global_seeds(random_state)

    need_estimators = return_estimators or return_curves

    multi_scoring = get_multi_scoring()
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
    class_ratio = class_ratio.iloc[0] / class_ratio.iloc[1]  # BUG?

    gridsearch_clf = get_classifier_pipeline(
        clf_code,
        random_state,
        search_kwargs,
        class_ratio,
        auto_class_ratio=True,
    )
    metric_scores = cross_validate(
        gridsearch_clf,
        X=x,
        y=y,
        cv=split,
        scoring=multi_scoring,
        return_train_score=True,
        return_estimator=need_estimators,
        n_jobs=1,
    )

    if need_estimators:
        estimators = metric_scores["estimator"]

    del metric_scores["estimator"]

    metric_score_stats = get_scores_stats(metric_scores)
    ret = metric_scores, metric_score_stats

    if return_estimators:
        ret = (*ret, estimators)

    if return_curves:
        curves = get_fpr_tpr_precision_recall(estimators, x, y, split)
        ret = (*ret, curves)

    if return_feature_importance:
        feature_importance = get_feature_importance(estimators, x, y, cv=split, scoring=optim_scoring)
        feature_importance["features"] = feature_cols
        ret = (*ret, feature_importance)

    return ret
