#%%
import os
import sys

sys.path.append("../../")
from src.kagglebook import model_xgb
from src.kagglebook.runner import Runner
from src.kagglebook.util import Logger

# %%

logger = Logger()
# %%
runner = Runner(
    "test",
    model_xgb.ModelXGB,
    ["cont10"],
    {
        "random_state": 1,
        "num_round": 50,
        "early_stopping_rounds": 5,
    },
)

# %%
try:
    runner.run_train_cv()
except:
    logger.general_logger.exception("error")
# %%
try:
    runner.run_predict_cv()
except:
    logger.general_logger.exception("error")

# %%
try:
    runner.run_train_all()
except:
    logger.general_logger.exception("error")

# %%

try:
    runner.run_predict_all()
except:
    logger.general_logger.exception("error")
# %%
