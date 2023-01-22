#%%

# ======================================================
# インポート
# ======================================================
import os
import sys

sys.path.append("../../")
import gc

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, check_cv
from tqdm import tqdm

from src.common import util
from src.common.com_util import update_tracking
from src.kagglebook.util import Logger

# %%

# ======================================================
# 設定
# ======================================================
# テストラン
TEST_RUN = False

# ロガー設定
logger = util.get_logger(__name__)

# ID
EXPERIMENT_ID = util.get_experiment_id(__file__)

if TEST_RUN:
    RUN_ID = "test" + "-" + util.get_run_id() + "-" + EXPERIMENT_ID
else:
    RUN_ID = util.get_run_id()
    RUN_ID = RUN_ID + "-" + EXPERIMENT_ID

# ディレクトリ
RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "results",
    RUN_ID,
)

# %%
# ======================================================
# 各種定義
# ======================================================


def oof_predict(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
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

    # カラム名
    columns = X.columns.to_list()

    # プログレスバー
    pbar = tqdm(splitter.split(X, y), desc="[Out of Fold]")
    for num_fold, (train_index, val_index) in enumerate(pbar):

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


# %%
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

# ======================================================
# モデル
# ======================================================

params = {
    "n_estimators": 100,
    "max_leaf_nodes": 63,
    "random_state": 0,
    "max_features": 0.3,
    "bootstrap": True,
    "max_samples": 0.5,
}
model = RandomForestRegressor(**params)

# ======================================================
# CV
# ======================================================
splitter = KFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

# %%
# ======================================================
# データ読み込み
# ======================================================
df_train, df_test = util.load_data()

# テストランのときは100行をサンプリングして使用する
if TEST_RUN:
    df_train = df_train.sample(n=100)
    df_test = df_test.sample(n=100)

X = df_train.drop(columns=["id", "target"])
y = df_train["target"]

X_test = df_test.drop(columns=["id"])

# %%
# ======================================================
# 学習
# ======================================================

df_oof, models = oof_predict(
    model,
    X,
    y,
    splitter,
)

# %%
# ======================================================
# 予測
# ======================================================

logger.info("Prediction")

df_pred = df_test[["id"]].copy()
X_test = np.asarray(X_test)

for i, model in enumerate(models):
    df_pred[f"pred_fold_{i}"] = model.predict(X_test)


# ======================================================
# 記録
# ======================================================
logger.info("Tracking")

# スコア
# =====

oof_score = np.sqrt(
    mean_squared_error(
        df_oof["target"],
        df_oof["oof"],
    )
)
logger.info("oof score: %f", oof_score)

update_tracking(
    RUN_ID,
    "oof_score",
    oof_score,
)


# oof
# =====
OOF_DIR = os.path.join(RESULT_DIR, "oof")
os.makedirs(
    OOF_DIR,
    exist_ok=True,
)
df_oof.to_csv(
    os.path.join(OOF_DIR, "oof.csv"),
    index=False,
)

# model
# =====
MODEL_DIR = os.path.join(RESULT_DIR, "model")
os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)
for i, model in enumerate(models):
    pd.to_pickle(
        model,
        os.path.join(MODEL_DIR, f"model-fold{i}.pkl"),
    )

# predict
# =====
PRED_DIR = os.path.join(RESULT_DIR, "pred")
os.makedirs(
    PRED_DIR,
    exist_ok=True,
)
df_pred.to_csv(
    os.path.join(PRED_DIR, "prediction.csv"),
    index=False,
)

# %%
