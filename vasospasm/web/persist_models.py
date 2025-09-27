import logging
import warnings
from pathlib import Path

import joblib
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import FunctionTransformer

from common.classifier_pipelines import ClfCode
from common.dataset import AneurysmLocationMapper1, one_hot_encode_dataset
from common.utils.logging_ import configure_logging
from common.web.persist_models import get_model_filename, train_classifier
from vasospasm.config import paths
from vasospasm.dataset import get_dataset
from vasospasm.web.utils import _drop_columns

logger = logging.getLogger(__name__)


def sort_cols(cols: list[str]) -> list[str]:
    if "id" in cols:
        cols.remove("id")
        cols.insert(0, "id")
    return cols


def main(
    model_dir: Path | str,
    seed: int = 0,
    n_splits_inner: int = 10,
    n_jobs: int = 10,
):
    model_dir = Path(model_dir)

    only_with_aneurysm_flags = [
        True,
        False,
    ]
    aneurysm_location_mappers = [
        # AneurysmLocationMapper2,
        AneurysmLocationMapper1,
        # AneurysmLocationMapper0,
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
    inner_split = RepeatedStratifiedKFold(n_splits=n_splits_inner, n_repeats=1, random_state=seed + 1)

    default_kwargs = dict(
        inner_split=inner_split,
        random_state=seed,
        n_jobs=n_jobs,
        # Only relevant for only_with_aneurysm_flags=False
        exclude_features=["location_notACom", "aneurysm_untreated"],
    )

    run_kwargs_list = [
        dict(
            clf_code=clf_code,
            aneurysm_location_mapper=aneurysm_location_mapper,
            only_with_aneurysm=only_with_aneurysm,
            **default_kwargs,
        )
        for only_with_aneurysm in only_with_aneurysm_flags
        for aneurysm_location_mapper in aneurysm_location_mappers
        for clf_code in clf_codes
    ]

    for i_run, run_params in enumerate(run_kwargs_list):
        logger.info(f"clf_code: {run_params['clf_code']}")
        logger.info(f"aneurysm_location_mapper: {run_params['aneurysm_location_mapper'].cat_code}")
        logger.info(f"only_with_aneurysm: {run_params['only_with_aneurysm']}")

        # Pop dataset parameters
        aneurysm_location_mapper = run_params.pop("aneurysm_location_mapper")
        only_with_aneurysm = run_params.pop("only_with_aneurysm")

        # Get dataset
        df, feature_cols, target_col = get_dataset(
            aneurysm_location_mapper=aneurysm_location_mapper,
            only_with_aneurysm=only_with_aneurysm,
            encode_one_hot=False,
        )

        # We will need the one-hot encoder below
        df, feature_cols, one_hot_encoder = one_hot_encode_dataset(df, feature_cols)

        # Drop original categorical columns
        one_hot_encoder.drop_original_features = True

        # Exclude features
        exclude_features = run_params.pop("exclude_features")
        if exclude_features:
            feature_cols = [col for col in feature_cols if col not in exclude_features]
            logger.info(f"Excluding features: {exclude_features}")

        # DEBUG
        # p = model_dir / f"processed_dataset_{i_run:02d}.csv"
        # p.parent.mkdir(parents=True, exist_ok=True)
        # first_cols = ["id"] + feature_cols + [target_col]
        # col_order = first_cols + [c for c in df.columns if c not in first_cols]
        # df[col_order].to_csv(p, index=False)
        # logger.info(f"CSV exported: {p}")

        # Append dataset parameters to run_params
        run_params["df"] = df
        run_params["feature_cols"] = feature_cols
        run_params["target_col"] = target_col

        clf_pipeline = train_classifier(**run_params)

        clf_pipeline.steps.insert(0, ("one_hot_encoder", one_hot_encoder))

        if exclude_features:
            feature_excluder = FunctionTransformer(
                func=_drop_columns,
                kw_args={"columns_to_drop": exclude_features},
                # feature_names_out=feature_cols,  # TODO: Default param is "one-to-one", which yields wrong metadata
                validate=False,
            )
            clf_pipeline.steps.insert(1, ("feature_excluder", feature_excluder))

        filename = get_model_filename(
            run_params["clf_code"],
            aneurysm_location_mapper,
            only_with_aneurysm,
        )

        # Store model
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_pipeline, model_dir / filename, compress=1)
        logger.info(f"Model saved to {model_dir / filename}")


if __name__ == "__main__":
    configure_logging()
    logging.getLogger("vasospasm").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.info('Running "main" function.')

    warnings.filterwarnings("ignore", ".*X has feature names, .*")
    # warnings.filterwarnings("ignore", ".*X does not have valid feature names, .*")

    main(model_dir=paths.repo / "vasospasm/web/models")
