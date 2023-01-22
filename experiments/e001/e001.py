#%%

# ======================================================
# インポート
# ======================================================
import os
import sys

sys.path.append("../../")
import gc
import shutil

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, check_cv

from src.common import util
from src.common.com_util import update_tracking

# %%


# ======================================================
# 設定
# ======================================================
# テストラン
TEST_RUN = True

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

# ======================================================
# 実験固有の定義ファイルのインポート・保存
# ======================================================
from model import build_model
from train import train_model

# 保存
# =====
DEFINITION_DIR = os.path.join(RESULT_DIR, "definitions")
os.makedirs(DEFINITION_DIR)
# モデル定義
shutil.copy("model.py", DEFINITION_DIR)
# 訓練方法定義
shutil.copy("train.py", DEFINITION_DIR)


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
model = build_model()


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

df_oof, models = train_model(
    model,
    X,
    y,
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
