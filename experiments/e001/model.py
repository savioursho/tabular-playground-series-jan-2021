from typing import Optional

from sklearn.ensemble import HistGradientBoostingRegressor


def build_model(
    params_update: Optional[dict] = None,
):

    # パラメータ
    params = {
        "learning_rate": 0.05,
        "max_iter": 2000,
        "max_leaf_nodes": 63,
        "early_stopping": True,
        "scoring": "neg_mean_squared_error",
        "random_state": 0,
    }

    # パラメータの更新
    if params_update is not None:
        params = params.update(params_update)

    # モデル
    model = HistGradientBoostingRegressor(**params)

    return model
