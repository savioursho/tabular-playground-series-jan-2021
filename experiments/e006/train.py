import gc
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import _safe_indexing

from src.common.fi import get_permutation_importance
from src.common.util import util
from src.features import load_features

logger = util.get_logger(__name__)

# ======================================================
# モデル
# ======================================================


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
def get_cv(
    X: pd.DataFrame,
    y: pd.Series,
):
    skfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )

    # StratifiedKFoldのためにyのbinnigしたものを作成
    num_bins = int(np.floor(1 + np.log2(len(y))))
    y_bin = pd.qcut(
        y,
        q=num_bins,
        labels=False,
    )

    cv = skfold.split(X, y_bin)
    return cv


# ======================================================
# 訓練する関数
# ======================================================
def train_model(
    estimator: BaseEstimator,
    test_run: bool = False,
):
    logger.info("Start Out of Fold Prediction")

    # === load data ===
    df_train_val = load_features(features + [TARGET], "train")

    if test_run:
        df_train_val = df_train_val.sample(n=100)

    X = df_train_val[features]
    y = df_train_val[TARGET]

    # === cv ===
    cv = get_cv(X, y)
    splitter = check_cv(cv)

    # === oof予測値のデータフレームの準備 ===
    df_oof = pd.DataFrame()
    df_oof["target"] = y
    df_oof["oof"] = np.nan
    df_oof["fold"] = np.nan

    # === オブジェクトを格納するリスト ===
    models = []
    transformers = []

    # === permutatino_importance のデータフレームを格納するリスト ===
    list_df_permutation_importance = []

    # === ndarrayにする ===
    # X_array = np.asarray(X)
    # y_array = np.asarray(y)

    for num_fold, (train_index, val_index) in enumerate(splitter.split()):

        logger.debug("fold %d", num_fold)
        logger.debug("train size: %d", len(train_index))
        logger.debug("val size: %d", len(val_index))

        # === model clone ===
        _model = clone(estimator)

        # === train val の分割 ===
        X_train, X_val = _safe_indexing(X, train_index), _safe_indexing(X, val_index)
        y_train, y_val = _safe_indexing(y, train_index), _safe_indexing(y, val_index)

        # === 目的変数の変換 ===
        qt = QuantileTransformer(
            output_distribution="normal",
            random_state=0,
        )
        # QuantileTransformer に入れるために二次元配列にする
        y_train = np.asarray(y_train).reshape(-1, 1)
        y_val = np.asanyarray(y_val).reshape(-1, 1)

        # 変換を適用
        y_train = qt.fit_transform(y_train)
        y_val = qt.transform(y_val)

        # 1次元配列に戻す
        y_train = y_train.ravel()
        y_val = y_val.ravel()

        # === fit ===
        verbose = True
        _model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=verbose)],
        )

        # === 目的変数の逆変換 ===
        # QuantileTransformer に入れるために二次元配列にする
        y_train = np.asarray(y_train).reshape(-1, 1)
        y_val = np.asanyarray(y_val).reshape(-1, 1)

        # 逆変換を適用
        y_train = qt.inverse_transform(y_train)
        y_val = qt.inverse_transform(y_val)

        # 1次元配列に戻す
        y_train = y_train.ravel()
        y_val = y_val.ravel()

        # === oof predict ===
        pred = _model.predict(X_val)

        # 逆変換
        pred = pred.reshape(-1, 1)
        pred = qt.inverse_transform(pred)
        pred = pred.ravel()

        df_oof.iloc[val_index, df_oof.columns.get_loc("oof")] = pred
        df_oof.iloc[val_index, df_oof.columns.get_loc("fold")] = num_fold

        # === permutation importance ===
        logger.info("Calculating permutation importance.")
        df_permutation_importance = get_permutation_importance(
            _model,
            X_val,
            y_val,
            features,
            n_repeats=10,
            max_samples=0.5,
        )
        list_df_permutation_importance.append(df_permutation_importance)

        # === オブジェクトをリストに格納 ===
        models.append(_model)
        transformers.append(qt)

        # === gc ===
        gc.collect()

    logger.info("Finish Out of Fold Prediction.")
    return df_oof, models, pd.concat(list_df_permutation_importance), transformers
