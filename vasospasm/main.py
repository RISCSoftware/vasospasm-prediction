import json
import logging
import re
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common.classifier_pipelines import ClfCode
from common.dataset import AneurysmLocationMapper1
from common.helper import get_best_params_of_grid_search_pipeline, ndarray_to_list
from common.trainer import train_one
from common.utils.logging_ import configure_logging
from vasospasm.config import paths
from vasospasm.dataset import get_dataset

logger = logging.getLogger(__name__)


def main(
    output_dir: Path | str,
    seed: int = 0,
    n_splits: int = 10,
    n_splits_inner: int | None = None,
    n_repeats: int = 10,
    n_jobs: int = 5,
):
    if n_splits_inner is None:
        n_splits_inner = n_splits

    only_with_aneurysm_flags = [
        False,
        True,
    ]
    aneurysm_location_mappers = [
        AneurysmLocationMapper1,
        # AneurysmLocationMapper0,
        # AneurysmLocationMapper2,
    ]
    # clf_codes = ClfCode.get_all()
    clf_codes = [
        ClfCode.tree_classifier,
        ClfCode.random_forest,
        # ClfCode.gbt,
        ClfCode.xgbt,
        ClfCode.logistic_regression,
        ClfCode.linear_svc,
        ClfCode.nonlinear_svc,
        ClfCode.knn,
    ]
    exclude_feature_sets = [
        ["location_notACom", "aneurysm_untreated"],
        # ### Feature set experiments
        # ["location_notACom", "aneurysm_untreated", "cns_infection", "female"],
        # ["location_notACom", "aneurysm_untreated", "cns_infection", "female", "aneurysm_clip", "aneurysm_coil", "aneurysm_diameter", "aneurysm_height", "location_ACom"],
    ]

    split = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    inner_split = RepeatedStratifiedKFold(n_splits=n_splits_inner, n_repeats=1, random_state=seed + 1)

    default_kwargs = dict(
        split=split,
        inner_split=inner_split,
        return_estimators=True,
        return_curves=True,
        return_feature_importance=True,
        random_state=seed,
        n_jobs=n_jobs,
    )

    run_kwargs_list = [
        dict(
            clf_code=clf_code,
            aneurysm_location_mapper=aneurysm_location_mapper,
            only_with_aneurysm=only_with_aneurysm,
            exclude_features=exclude_features,
            **default_kwargs,
        )
        for only_with_aneurysm in only_with_aneurysm_flags
        for aneurysm_location_mapper in aneurysm_location_mappers
        for clf_code in clf_codes
        for exclude_features in exclude_feature_sets
    ]

    report_rows = []
    metric_scores = []
    curves = []
    feature_importance = []

    for i_run, run_params in enumerate(run_kwargs_list):
        # Pop dataset parameters
        aneurysm_location_mapper = run_params.pop("aneurysm_location_mapper")
        only_with_aneurysm = run_params.pop("only_with_aneurysm")

        # Get dataset
        df, feature_cols, target_col = get_dataset(
            aneurysm_location_mapper=aneurysm_location_mapper,
            only_with_aneurysm=only_with_aneurysm,
        )

        # Exclude features
        exclude_features = run_params.pop("exclude_features")
        if exclude_features:
            feature_cols = [col for col in feature_cols if col not in exclude_features]
            logger.info(f"Excluding features: {exclude_features}")

        # DEBUG
        # p = output_dir / f"processed_dataset_{i_run:02d}.csv"
        # p.parent.mkdir(parents=True, exist_ok=True)
        # first_cols = ["id"] + feature_cols + [target_col]
        # col_order = first_cols + [c for c in df.columns if c not in first_cols]
        # df[col_order].to_csv(p, index=False)
        # logger.info(f"CSV exported: {p}")

        # Append dataset parameters to run_params
        run_params["df"] = df
        run_params["feature_cols"] = feature_cols
        run_params["target_col"] = target_col

        logger.info(f"clf_code: {run_params['clf_code']}")
        logger.info(f"aneurysm_location_mapper: {aneurysm_location_mapper.cat_code}")
        logger.info(f"only_with_aneurysm: {only_with_aneurysm}")

        metric_scores_, metric_score_stats, estimators, curves_, fi_ = train_one(**run_params)
        metric_scores.append(metric_scores_)
        curves.append(curves_)
        feature_importance.append(fi_)
        best_params = get_best_params_of_grid_search_pipeline(estimators)
        test_score_stats = {k: metric_score_stats[k] for k in metric_score_stats if re.match(r"test.*", k)}

        # logger.debug(f"best_params: {json.dumps(best_params, indent=4)}")

        row = {
            "clf_code": run_params["clf_code"],
            "aneurysm_location_mapper": aneurysm_location_mapper.cat_code,
            "only_with_aneurysm": only_with_aneurysm,
            # "best_params": json.dumps(best_params),
            **test_score_stats,
        }
        report_rows.append(row)

        logger.info(
            f"test_balanced_acc: {test_score_stats['test_balanced_acc_mean']:.3f} "
            f"± {test_score_stats['test_balanced_acc_ci95eb']:.3f}"
        )
        logger.info(
            f"test_roc_auc: {test_score_stats['test_roc_auc_mean']:.3f} "
            f"± {test_score_stats['test_roc_auc_ci95eb']:.3f}"
        )

    report_df = pd.DataFrame(report_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_dir / "report.csv", index=True)

    with open(output_dir / "scores.json", "w") as f:
        json.dump(metric_scores, f, indent=4, default=ndarray_to_list)

    with open(output_dir / "curves.json", "w") as f:
        json.dump(curves, f, indent=4, default=ndarray_to_list)

    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=4, default=ndarray_to_list)


if __name__ == "__main__":
    configure_logging()
    logging.getLogger("vasospasm").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.info('Running "main" function.')

    warnings.filterwarnings("ignore", ".*X has feature names, .*")
    # warnings.filterwarnings("ignore", ".*X does not have valid feature names, .*")

    main(output_dir=paths.repo / "output/vasospasm/current")
