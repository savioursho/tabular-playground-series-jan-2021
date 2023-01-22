import gc
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, check_cv

from src.common.util import util

logger = util.get_logger(__name__)

# ======================================================
# 特徴量
# ======================================================

features = [
    "cont1",
    "cont2",
    "cont3",
    "cont4",
    "cont5",
    "cont6",
    "cont7",
    "cont8",
    "cont9",
    "cont10",
    "cont11",
    "cont12",
    "cont13",
    "cont14",
]

TARGET = "target"


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
    cv=kfold,
    test_run: bool = False,
):
    logger.info("Start Out of Fold Prediction")

    # load data
    df_train = util.load_data("train", test_run)

    X = df_train[features]
    y = df_train[TARGET]

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
        _model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

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
