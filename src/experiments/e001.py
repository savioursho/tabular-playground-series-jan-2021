import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from src import config
from src.common import update_tracking, util

# EXPERIMENT ID
EXPERIMENT_ID = util.get_experiment_id(__file__)

# RUN ID
RUN_ID = util.get_run_id()

# MODEL
MODLE_ID = "m001_hgbr"
model = util.get_model(MODLE_ID)

# DATA
df_train = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
df_test = pd.read_csv(config.RAW_DATA_DIR / "test.csv")

X_train_val = df_train.drop(columns=["target", "id"])
y_train_val = df_train["target"].values

# RUN
splitter = KFold(n_splits=5, shuffle=True, random_state=0)

y_pred = cross_val_predict(model, X_train_val, y_train_val, cv=splitter)

oof_score = np.sqrt(mean_squared_error(y_train_val, y_pred))

update_tracking(RUN_ID, "oof_score", oof_score)
update_tracking(RUN_ID, "experiment_id", EXPERIMENT_ID)
