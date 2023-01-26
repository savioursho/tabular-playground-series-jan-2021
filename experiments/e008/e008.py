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
from sklearn.metrics import mean_squared_error

from src.common.fi import plot_permutation_importance
from src.common.util import util
from src.common.util.com_util import update_tracking
from src.features import load_features

# %%


# ======================================================
# 設定
# ======================================================
# テストラン
TEST_RUN = False

# ロガー設定
logger = util.get_logger(__name__)

# ======================================================
# 実験固有の定義ファイルのインポート・保存
# ======================================================
from train import build_model, features, train_model


def settings():
    # Experiment ID
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
    return RUN_ID, RESULT_DIR


RUN_ID, RESULT_DIR = settings()


# === 保存 ===
def save_definition(RESULT_DIR):
    DEFINITION_DIR = os.path.join(RESULT_DIR, "definitions")
    os.makedirs(DEFINITION_DIR)
    shutil.copy("train.py", DEFINITION_DIR)


save_definition(RESULT_DIR)


def run():
    # ======================================================
    # モデル
    # ======================================================
    preprocessor, model = build_model()

    # ======================================================
    # 学習
    # ======================================================
    with util.timer("train", logger):
        df_oof, models = train_model(
            model,
            preprocessor,
            test_run=TEST_RUN,
        )

    # ======================================================
    # 予測
    # ======================================================

    with util.timer("prediction", logger):
        # データ読み込み
        X_test = load_features(features, "test")
        if TEST_RUN:
            X_test = X_test.sample(n=100)

        df_pred = pd.DataFrame()

        for i, (preprocessor, model) in enumerate(models):
            gc.collect()
            X_test_fold = preprocessor.transform(X_test)
            pred = model.predict(X_test_fold)

            df_pred[f"pred_fold_{i}"] = pred

    return df_oof, models, df_pred


df_oof, models, df_pred = run()

#%%
# ======================================================
# 記録
# ======================================================
def record(
    df_oof,
    models,
    # df_permutation_importance,
    df_pred,
):
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

    # permutation importance
    # =====
    # PERMUTATION_IMPORTANCE_DIR = os.path.join(RESULT_DIR, "permutatino_importance")
    # os.makedirs(
    #     PERMUTATION_IMPORTANCE_DIR,
    #     exist_ok=True,
    # )
    # # csv
    # df_permutation_importance.to_csv(
    #     os.path.join(PERMUTATION_IMPORTANCE_DIR, "permutation_importance.csv"),
    #     index=False,
    # )
    # # plot
    # plot_permutation_importance(
    #     df_permutation_importance,
    #     os.path.join(PERMUTATION_IMPORTANCE_DIR, "permutation_importance.png"),
    # )

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


record(
    df_oof,
    models,
    # df_permutation_importance,
    df_pred,
)
