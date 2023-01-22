#%%

# ======================================================
# インポート
# ======================================================
import os
import sys

sys.path.append("../../")
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common import util
from src.common.com_util import update_tracking

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

logger.debug("Start RunID: %s", RUN_ID)
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
from train import features, train_model

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
# モデル
# ======================================================
model = build_model()


# %%
# ======================================================
# 学習
# ======================================================

df_oof, models = train_model(
    model,
    test_run=TEST_RUN,
)

# %%
# ======================================================
# 予測
# ======================================================

# データ読み込み
df_test = util.load_data("test", TEST_RUN)
X_test = df_test[features]

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
