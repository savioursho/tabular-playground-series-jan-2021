import gc

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, check_cv

from src.common import util

logger = util.get_logger(__name__)


# ======================================================
# CV
# ======================================================
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

# ======================================================
# 訓練する関数
# ======================================================
def train_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv=kfold,
):
    logger.info("Start Out of Fold Prediction")

    # splitter
    splitter = check_cv(cv)

    # oof予測値のデータフレームの準備
    df_oof = pd.DataFrame()
    df_oof["target"] = y
    df_oof["oof"] = np.nan
    df_oof["fold"] = np.nan

    # モデルを格納するリスト
    models = []

    # ndarrayにする
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    for num_fold, (train_index, val_index) in enumerate(splitter.split(X, y)):

        logger.debug("fold %d", num_fold)

        # model clone
        _model = clone(estimator)

        # train val の分割
        X_train, X_val = X_array[train_index], X_array[val_index]
        y_train, y_val = y_array[train_index], y_array[val_index]

        # fit
        _model.fit(X_train, y_train)

        # oof predict
        pred = _model.predict(X_val)
        df_oof.iloc[val_index, df_oof.columns.get_loc("oof")] = pred
        df_oof.iloc[val_index, df_oof.columns.get_loc("fold")] = num_fold

        # モデルをリストに格納
        models.append(_model)

        # gc
        gc.collect()

    logger.info("Finish Out of Fold Prediction.")
    return df_oof, models
