import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def pr_auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def get_multi_scoring():
    neg_label = 0

    specificity = make_scorer(recall_score, pos_label=neg_label)
    npv = make_scorer(precision_score, pos_label=neg_label)
    pr_auc = make_scorer(pr_auc_score, needs_threshold=True)

    scoring = {
        # "acc": "accuracy",
        "balanced_acc": "balanced_accuracy",
        "sensitivity": "recall",  # tpr, recall
        "specificity": specificity,  # tnr
        "precision": "precision",  # ppv
        "npv": npv,
        "roc_auc": "roc_auc",
        "pr_auc": pr_auc,
        "ap": "average_precision",
        "f1": "f1",
    }
    return scoring


def get_scores_stats(scores):
    score_stats = {}
    for score_name, score in scores.items():
        mean = score.mean()
        sem = stats.sem(score)

        # Compute the 95% confidence interval error bound
        ci95eb = sem * stats.t.ppf((1 + 0.95) / 2.0, len(score) - 1)
        # ci95 = stats.t.interval(0.95, len(score) - 1, loc=mean, scale=sem)
        # ci95eb = (ci95[1] - ci95[0]) / 2

        score_stats[f"{score_name}_mean"] = mean
        score_stats[f"{score_name}_std"] = score.std(ddof=1)
        score_stats[f"{score_name}_sem"] = sem
        score_stats[f"{score_name}_ci95eb"] = ci95eb

    return score_stats


def get_feature_importance(
    classifiers: list[ClassifierMixin],
    x: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | np.ndarray,
    cv,
    scoring: str = "balanced_accuracy",
    permutation_importance_repeats: int = 0,
) -> dict:
    assert len(classifiers) == cv.get_n_splits()

    permutation_fi = []
    impurity_fi = []

    coefficients = []
    normalized_coefficients = []

    intercepts = []
    normalized_intercepts = []

    for pipeline_or_clf, (_, test) in zip(classifiers, cv.split(x, y)):
        x_test = x.iloc[test]
        y_test = y.iloc[test]

        if permutation_importance_repeats > 0:
            # Permutation feature importance
            res = permutation_importance(
                pipeline_or_clf,
                x_test,
                y_test,
                scoring=scoring,
                n_repeats=permutation_importance_repeats,
            )
            vals = res.importances_mean
            permutation_fi.append(vals)

        clf = pipeline_or_clf
        pipeline = None
        if isinstance(clf, GridSearchCV | RandomizedSearchCV):
            clf = clf.best_estimator_
        if isinstance(clf, Pipeline):
            pipeline = clf
            clf = pipeline["clf"]

        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
            impurity_fi.append(vals)

        if hasattr(clf, "coef_"):
            ncoef = clf.coef_.reshape(-1)
            normalized_coefficients.append(ncoef)

            if "scaler" in pipeline.named_steps:
                scaler = pipeline.named_steps["scaler"]
                coef = ncoef / scaler.scale_
                coefficients.append(coef)

            if hasattr(clf, "intercept_"):
                nintercept = clf.intercept_
                normalized_intercepts.append(nintercept)

                if "scaler" in pipeline.named_steps:
                    scaler = pipeline.named_steps["scaler"]
                    intercept = np.dot(-scaler.mean_ / scaler.scale_, ncoef) + nintercept
                    intercepts.append(intercept)

                    # logit = intercept + sum_i ((x_i - m_i) / s_i) * c_i
                    # actual_coef_i = c_i / s_i
                    # actual_intercept = intercept - sum_i m_i * c_i / s_i

                    # Uncomment to verify correctness of intercept and coefficients

                    # def sigmoid(x):
                    #     return 1 / (1 + np.exp(-x))
                    #
                    # x = x_test.iloc[0].values
                    # y1 = sigmoid(np.dot(x, coef) + intercept)
                    # y2 = pipeline_or_clf.predict_proba([x])[:, 1]
                    # assert np.isclose(y1, y2)

    res = {
        "features": x.columns.tolist(),
    }
    if len(permutation_fi):
        res["permutation_feature_importance"] = permutation_fi
    if len(impurity_fi):
        res["impurity_feature_importance"] = impurity_fi
    if len(normalized_coefficients):
        res["normalized_coefficients"] = normalized_coefficients
    if len(coefficients):
        res["coefficients"] = coefficients
    if len(normalized_intercepts):
        res["normalized_intercepts"] = normalized_intercepts
    if len(intercepts):
        res["intercepts"] = intercepts

    return res


def get_pool_aggregated_fpr_tpr_precision_recall(
    y_true_y_pred_list: list[tuple[np.ndarray, np.ndarray]],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes pooled FPR, TPR, precision, and recall from a list of (y_true, y_pred) tuples.

    Args:
        y_true_y_pred_list: List of tuples, where each tuple contains the true labels and predicted scores.
        **kwargs: Additional keyword arguments forwarded to sklearn.metrics.roc_curve and precision_recall_curve.

    Returns:
        A tuple (fpr, tpr, precision, recall), where fpr and tpr are the false positive rate and true positive rate
        arrays, and precision and recall are the precision-recall curve arrays.
    """
    y_true_pooled = np.concatenate([t[0] for t in y_true_y_pred_list])
    y_score_pooled = np.concatenate([t[1] for t in y_true_y_pred_list])

    # TPR over FPR
    fpr, tpr, _ = roc_curve(
        y_true_pooled,
        y_score_pooled,
        **kwargs,
    )

    # Note: precision over recall, not recall over precision
    precision, recall, _ = precision_recall_curve(
        y_true_pooled,
        y_score_pooled,
        **kwargs,
    )

    return fpr, tpr, precision, recall


def get_mean_aggregated_fpr_tpr_precision_recall(
    y_true_y_pred_list: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 10001,
    **kwargs,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the mean and standard deviation of ROC and Precision-Recall curves
    over multiple (y_true, y_pred) pairs.

    Args:
        y_true_y_pred_list: List of tuples, each containing (y_true, y_pred) arrays.
        n_points: Number of points to interpolate the curves on the [0, 1] grid.
        **kwargs: Additional keyword arguments forwarded to sklearn.metrics.roc_curve and precision_recall_curve.

    Returns:
        roc_tuple: Tuple of (fpr, mean_tpr, std_tpr).
        pr_tuple: Tuple of (recall, mean_precision, std_precision).
    """

    fpr_grid = np.linspace(0.0, 1.0, n_points)
    recall_grid = np.linspace(0.0, 1.0, n_points)

    tpr_matrix: list[np.ndarray] = []
    precision_matrix: list[np.ndarray] = []

    for y_true, y_pred in y_true_y_pred_list:
        if np.isnan(y_true).any():
            raise ValueError("y_true contains NaN values.")
        if np.isnan(y_pred).any():
            raise ValueError("y_pred contains NaN values.")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_matrix.append(tpr_interp)

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_pred, **kwargs)
        recall, precision = recall[::-1], precision[::-1]
        precision_interp = np.interp(recall_grid, recall, precision)
        precision_matrix.append(precision_interp)

    tpr_matrix = np.asarray(tpr_matrix)
    precision_matrix = np.asarray(precision_matrix)

    mean_tpr = tpr_matrix.mean(axis=0)
    std_tpr = tpr_matrix.std(axis=0)

    mean_precision = precision_matrix.mean(axis=0)
    std_precision = precision_matrix.std(axis=0)

    roc_tuple = (fpr_grid, mean_tpr, std_tpr)
    pr_tuple = (recall_grid, mean_precision, std_precision)

    return roc_tuple, pr_tuple


def get_fpr_tpr_precision_recall(
    classifiers: list[Pipeline],
    x: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | np.ndarray,
    cv,
) -> dict:
    pos_label = 1

    y_true_list = []
    y_score_list = []

    for clf, (train, test) in zip(classifiers, cv.split(x, y)):
        if hasattr(clf, "predict_proba"):
            get_scores = lambda y: clf.predict_proba(y)[:, pos_label]
        elif hasattr(clf, "decision_function"):
            get_scores = lambda y: clf.decision_function(y) * (2 * pos_label - 1)
        else:
            raise ValueError(f"Unsupported classifier type: {type(clf)}")

        y_score = get_scores(x.iloc[test])

        y_true_list.append(np.array(y.iloc[test]))
        y_score_list.append(np.array(y_score))

    y_true_y_pred_list = list(zip(y_true_list, y_score_list))

    # DEBUG: Compute the roc curve and precision-recall curve each item of y_true_list and y_score_list
    # aucs = []
    # pr_aucs = []
    # for i, (y_t, y_s) in enumerate(zip(y_true_list, y_score_list)):
    #     # fpr, tpr, _ = roc_curve(y_t, y_s, pos_label=pos_label, drop_intermediate=False)
    #     # precision, recall, _ = precision_recall_curve(y_t, y_s, pos_label=pos_label, drop_intermediate=False)
    #     roc_auc = roc_auc_score(y_t, y_s)
    #     pr_auc = pr_auc_score(y_t, y_s)
    #     aucs.append(roc_auc)
    #     pr_aucs.append(pr_auc)
    # print(f"Mean ROC AUC: {np.mean(aucs):.4f}, Mean PR AUC: {np.mean(pr_aucs):.4f}")

    pooled_fpr, pooled_tpr, pooled_precision, pooled_recall = get_pool_aggregated_fpr_tpr_precision_recall(
        y_true_y_pred_list,
        pos_label=pos_label,
        drop_intermediate=False,
    )

    roc_tuple, pr_tuple = get_mean_aggregated_fpr_tpr_precision_recall(
        y_true_y_pred_list,
        pos_label=pos_label,
        drop_intermediate=False,
    )

    fpr_grid, mean_tpr, std_tpr = roc_tuple
    recall_grid, mean_precision, std_precision = pr_tuple

    # DEBUG
    # y_true_pooled = np.concatenate([t[0] for t in y_true_y_pred_list])
    # y_score_pooled = np.concatenate([t[1] for t in y_true_y_pred_list])
    # roc_auc = roc_auc_score(y_true_pooled, y_score_pooled)
    # pr_auc = pr_auc_score(y_true_pooled, y_score_pooled)
    # print(f"ROC AUC (pooled): {roc_auc:.4f}, PR AUC (pooled): {pr_auc:.4f}")
    # # ROC AUC from mean curve
    # mean_auc = auc(fpr_grid, mean_tpr)
    # mean_pr_auc = auc(recall_grid, mean_precision)
    # print(f"ROC AUC (from mean curve): {mean_auc:.4f}, PR AUC (from mean curve): {mean_pr_auc:.4f}")

    res_dict = {
        "y_true_y_pred": y_true_y_pred_list,
        "fpr": pooled_fpr,
        "tpr": pooled_tpr,
        "precision": pooled_precision,
        "recall": pooled_recall,
        #
        "fpr_grid": fpr_grid,
        "mean_tpr": mean_tpr,
        "mean_precision": mean_precision,
        "recall_grid": recall_grid,
    }
    return res_dict
