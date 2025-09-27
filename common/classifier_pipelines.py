import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ClfCode:
    tree_classifier = "tree_classifier"
    random_forest = "random_forest"
    gbt = "gbt"
    xgbt = "xgbt"
    logistic_regression = "logistic_regression"
    linear_svc = "linear_svc"
    nonlinear_svc = "nonlinear_svc"
    knn = "knn"

    @classmethod
    def get_all(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(k) and isinstance(v, str)]


def normalize_class_weights(
    weights: dict,
    class_counts: dict,
):
    """
    >>> normalize_class_weights(weights = {0: 1., 1: 4.}, class_counts={0: 400, 1: 100})
    {0: 0.625, 1: 2.5}
    """
    weights = pd.Series(weights)
    class_counts = pd.Series(class_counts)
    factor = np.sum(class_counts) / np.sum(weights * class_counts)

    normalized_weights = (factor * weights).to_dict()

    return normalized_weights


def get_default_class_weight_range():
    """
    >>> get_default_class_weight_range()
    """
    minf, maxf, step = 3.0, 9.0, 10
    r = np.logspace(np.log10(minf), np.log10(maxf), step)

    weight_sets = [{0: 1.0, 1: x} for x in r]
    class_counts = {0: 400, 1: 100}
    normalized_weight_sets = [normalize_class_weights(w, class_counts) for w in weight_sets]
    return normalized_weight_sets


def get_classifier_pipeline(
    clf_code: str,
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
    auto_class_ratio: bool = True,
):
    """

    :param clf_code:
    :param random_state:
    :param search_kwargs:
    :param class_ratio: The number of negative samples divided by the number of positive samples
    :param auto_class_ratio: If True, class_ratio is ignored and "balanced" is used
    :return:
    """
    if class_ratio is None and not auto_class_ratio:
        raise ValueError("class_ratio must be provided if auto_class_ratio is False")

    if auto_class_ratio:
        class_ratio_or_none = None
    else:
        class_ratio_or_none = class_ratio

    if clf_code == ClfCode.tree_classifier:
        return get_tree_classifier_pipeline(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.nonlinear_svc:
        return get_nonlinear_svm_classifier(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.random_forest:
        return get_random_forest_classifier(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.gbt:
        return get_gbt_classifier(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.xgbt:
        return get_xgboost_gbt_classifier(random_state, search_kwargs, class_ratio)
    elif clf_code == ClfCode.knn:
        return get_knn_classifier(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.logistic_regression:
        return get_logistic_regression_classifier(random_state, search_kwargs, class_ratio_or_none)
    elif clf_code == ClfCode.linear_svc:
        return get_linear_svm_classifier(random_state, search_kwargs, class_ratio_or_none)
    else:
        raise ValueError(f"'{clf_code}' is not a proper method")


def get_default_cv_split():
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)


def ensure_search_kwargs(search_kwargs: dict):
    if search_kwargs is None:
        search_kwargs = dict(
            scoring="balanced_accuracy",
            cv=get_default_cv_split(),
            verbose=1,
            n_jobs=-1,
        )
    return search_kwargs


def get_class_weight(class_ratio: float):
    if class_ratio is None:
        return "balanced"

    pseudo_occs = np.array([class_ratio, 1])

    # n_samples / (n_classes * np.bincount(y))
    class_weight = pseudo_occs.sum() / (2 * pseudo_occs)
    class_weight = {i: w for i, w in enumerate(class_weight)}

    return class_weight


def get_tree_classifier_pipeline(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    classifier = DecisionTreeClassifier(
        random_state=random_state,
        class_weight=class_weight,
    )
    pipeline = Pipeline(
        [
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        # clf__criterion=["gini", "entropy"],
        clf__splitter=["best", "random"],
        clf__max_features=["log2", "sqrt", None],
        clf__max_depth=range(2, 6),
        clf__min_samples_leaf=[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        # clf__min_samples_leaf=[0.03, 0.05, 0.07, 0.1, 0.15],
        # clf__min_samples_leaf=range(3, 8),
        # clf__min_samples_split=range(2, 6),
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv


def get_random_forest_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    classifier = RandomForestClassifier(
        random_state=random_state,
        criterion="gini",
        class_weight=class_weight,
    )
    pipeline = Pipeline(
        [
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        clf__n_estimators=[10, 30, 100, 300],
        clf__max_depth=range(2, 6),
        # clf__min_samples_leaf=[0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        clf__min_samples_leaf=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        # clf__min_samples_leaf=[0.03, 0.05, 0.07, 0.1, 0.15],
        clf__class_weight=["balanced", "balanced_subsample"],
        clf__bootstrap=[True, False],
        clf__max_features=["log2", "sqrt", None],
    )
    search_cv = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        **search_kwargs,
        n_iter=100,
    )
    return search_cv


def get_gbt_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    # classifier = GradientBoostingClassifier(
    classifier = HistGradientBoostingClassifier(
        class_weight=class_weight,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        clf__learning_rate=(0.001, 0.003, 0.01, 0.03, 0.1),
        # clf__max_depth=(3, 5, 7, 10, None),
        clf__max_depth=range(2, 6),
        # clf__max_leaf_nodes=(3, 10, 30),
        clf__min_samples_leaf=[5, 10, 20, 50],
        clf__l2_regularization=[0, 0.01, 0.1, 1],
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv


def get_xgboost_gbt_classifier(random_state: int = 0, search_kwargs: dict = None, class_ratio: float = None):
    if class_ratio is None:
        raise ValueError("class_ratio must be provided for XGBoost")

    search_kwargs = ensure_search_kwargs(search_kwargs)

    # classifier = GradientBoostingClassifier(
    classifier = XGBClassifier(
        scale_pos_weight=class_ratio,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        # clf__scale_pos_weight=np.linspace(0.8, 1.4, 7) * class_ratio,
        clf__max_depth=range(2, 5),
        clf__learning_rate=[0.001, 0.003, 0.01],
        clf__subsample=[0.5, 0.7, 1],
        clf__min_child_weight=[1, 3, 10],
        #
        # clf__n_estimators=[30, 100],
        # clf__gamma=[0, 1, 2, 3, 5],
        # clf__reg_lambda=[0.01, 0.1, 1],
    )
    # search_cv = GridSearchCV(
    search_cv = RandomizedSearchCV(
        pipeline,
        # param_grid=param_grid,
        param_distributions=param_grid,
        random_state=random_state,
        n_iter=100,
        **search_kwargs,
    )
    return search_cv


def get_logistic_regression_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    classifier = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        max_iter=1000,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        clf__C=np.logspace(-4, 2, 13),
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv


def get_linear_svm_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    classifier = SVC(
        class_weight=class_weight,
        kernel="linear",
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        clf__C=np.logspace(-4, 1, 7),
        clf__gamma=["scale", "auto"],
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv


def get_nonlinear_svm_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)
    class_weight = get_class_weight(class_ratio)

    classifier = SVC(
        random_state=random_state,
        class_weight=class_weight,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        clf__kernel=["poly", "rbf", "sigmoid"],
        clf__C=np.logspace(-4, 1, 7),
        clf__gamma=["scale", "auto"],
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv


def get_knn_classifier(
    random_state: int = 0,
    search_kwargs: dict = None,
    class_ratio: float = None,
):
    search_kwargs = ensure_search_kwargs(search_kwargs)

    classifier = KNeighborsClassifier()
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", classifier),
        ]
    )
    param_grid = dict(
        smote__k_neighbors=[5, 7, 10, 15, 20, 30],
        clf__n_neighbors=[5, 7, 10, 15, 20, 30],
        clf__weights=["uniform", "distance"],
    )
    search_cv = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        **search_kwargs,
    )
    return search_cv
