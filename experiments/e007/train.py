import gc
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.pipeline import make_pipeline
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
    rte_params_update: Optional[dict] = None,
    rfr_params_update: Optional[dict] = None,
):

    # ロガー
    lgb.register_logger(logger)

    # パラメータ
    rte_params = {
        "n_estimators": 100,
        "random_state": 0,
        "n_jobs": -1,
    }
    rfr_params = {
        "n_estimators": 100,
        "max_features": 0.5,
        "max_samples": 0.5,
        "max_leaf_nodes": 63,
        "n_jobs": -1,
    }

    # パラメータの更新
    if rte_params_update is not None:
        rte_params = rte_params.update(rte_params_update)
    if rfr_params_update is not None:
        rfr_params = rfr_params.update(rfr_params_update)

    # モデル
    model = make_pipeline(
        RandomTreesEmbedding(**rte_params),
        RandomForestRegressor(**rfr_params),
    )

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

    # === permutatino_importance のデータフレームを格納するリスト ===
    list_df_permutation_importance = []

    for num_fold, (train_index, val_index) in enumerate(splitter.split()):

        logger.debug("fold %d", num_fold)
        logger.debug("train size: %d", len(train_index))
        logger.debug("val size: %d", len(val_index))

        # === model clone ===
        _model = clone(estimator)

        # === train val の分割 ===
        X_train, X_val = _safe_indexing(X, train_index), _safe_indexing(X, val_index)
        y_train, y_val = _safe_indexing(y, train_index), _safe_indexing(y, val_index)

        # === fit ===
        _model.fit(
            X_train,
            y_train,
        )

        # === oof predict ===
        pred = _model.predict(X_val)

        df_oof.iloc[val_index, df_oof.columns.get_loc("oof")] = pred
        df_oof.iloc[val_index, df_oof.columns.get_loc("fold")] = num_fold

        # === permutation importance ===
        logger.info("Calculating permutation importance.")
        df_permutation_importance = get_permutation_importance(
            _model,
            X_val,
            y_val,
            features,
            n_repeats=5,
            max_samples=0.2,
        )
        list_df_permutation_importance.append(df_permutation_importance)

        # === オブジェクトをリストに格納 ===
        models.append(_model)

        # === gc ===
        gc.collect()

    logger.info("Finish Out of Fold Prediction.")
    return df_oof, models, pd.concat(list_df_permutation_importance)
