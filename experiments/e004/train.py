import gc
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, check_cv

from src.common.fi import get_permutation_importance
from src.common.util import util
from src.features import load_features

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
skfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

# ======================================================
# 訓練する関数
# ======================================================
def train_model(
    estimator: BaseEstimator,
    cv=skfold,
    test_run: bool = False,
):
    logger.info("Start Out of Fold Prediction")

    # load data

    df_train_val = load_features(features + [TARGET], "train")

    if test_run:
        df_train_val = df_train_val.sample(n=100)

    X = df_train_val[features]
    y = df_train_val[TARGET]

    # splitter
    splitter = check_cv(cv)

    # StratifiedKFoldのためにyのbinnigしたものを作成
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_bin = pd.qcut(
        y,
        q=num_bins,
        labels=False,
    )

    # oof予測値のデータフレームの準備
    df_oof = pd.DataFrame()
    df_oof["target"] = y
    df_oof["oof"] = np.nan
    df_oof["fold"] = np.nan

    # モデルを格納するリスト
    models = []

    # permutatino_importance のデータフレームを格納するリスト
    list_df_permutation_importance = []

    # ndarrayにする
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    for num_fold, (train_index, val_index) in enumerate(splitter.split(X, y_bin)):

        logger.debug("fold %d", num_fold)
        logger.debug("train size: %d", len(train_index))
        logger.debug("val size: %d", len(val_index))

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

        # permutation importance
        logger.info("Calculating permutation importance.")
        df_permutation_importance = get_permutation_importance(
            _model,
            X_val,
            y_val,
            features,
            n_repeats=5,
            max_samples=0.5,
        )
        list_df_permutation_importance.append(df_permutation_importance)

        # モデルをリストに格納
        models.append(_model)

        # gc
        gc.collect()

    logger.info("Finish Out of Fold Prediction.")
    return df_oof, models, pd.concat(list_df_permutation_importance)
