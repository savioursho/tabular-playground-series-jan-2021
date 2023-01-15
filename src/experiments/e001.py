import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from src import config
from src.common import update_tracking, util
from src.cv.oof import oof, save_oof
from src.fi import plot_permutation_importance

TEST_RUN = False

# EXPERIMENT ID
EXPERIMENT_ID = util.get_experiment_id(__file__)

# RUN ID
RUN_ID = util.get_run_id()
if TEST_RUN:
    RUN_ID = "test" + RUN_ID

# MODEL
MODLE_ID = "m001_hgbr"
model = util.get_model(MODLE_ID)

# DATA
df_train = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
df_test = pd.read_csv(config.RAW_DATA_DIR / "test.csv")

if TEST_RUN:
    df_train = df_train.sample(100)
    df_test = df_test.sample(100)

X_train_val = df_train.drop(columns=["target", "id"])
y_train_val = df_train["target"].values

# RUN
splitter = KFold(n_splits=5, shuffle=True, random_state=0)

df_oof, list_models, df_permutation_importance = oof(
    model,
    X_train_val,
    y_train_val,
    cv=splitter,
    return_permutation_importance=True,
)

oof_score = np.sqrt(mean_squared_error(df_oof["y_true"], df_oof["oof"]))

update_tracking(RUN_ID, "oof_score", oof_score)
update_tracking(RUN_ID, "experiment_id", EXPERIMENT_ID)

oof_file_name = "-".join([RUN_ID, EXPERIMENT_ID, MODLE_ID]) + ".csv"
save_oof(df_oof, oof_file_name)

# feature importance
plot_permutation_importance(
    df_permutation_importance,
    config.FI_FIG_DIR / ("-".join([RUN_ID, EXPERIMENT_ID, MODLE_ID]) + ".png"),
)

df_permutation_importance.to_csv(
    config.FI_DIR / ("-".join([RUN_ID, EXPERIMENT_ID, MODLE_ID]) + ".csv"), index=False
)
