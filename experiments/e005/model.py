from typing import Optional

import lightgbm as lgb

from src.common.util import util

logger = util.get_logger(__name__)


def build_model(
    params_update: Optional[dict] = None,
):

    # ロガー
    lgb.register_logger(logger)

    # パラメータ
    params = {
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "num_leaves": 63,
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.6,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "metric": "rmse",
        "random_state": 0,
    }

    # パラメータの更新
    if params_update is not None:
        params = params.update(params_update)

    # モデル
    model = lgb.LGBMRegressor(**params)

    return model
