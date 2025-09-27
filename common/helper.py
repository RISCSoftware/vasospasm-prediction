import random
from datetime import datetime

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def get_best_params_of_grid_search_pipeline(estimators):
    if isinstance(estimators[0], GridSearchCV | RandomizedSearchCV):
        return [e.best_params_ for e in estimators]
    else:
        raise ValueError(f"Unsupported estimator type: {type(estimators[0])}")


def get_new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
