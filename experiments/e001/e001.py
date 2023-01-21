#%%
import sys

sys.path.append("../../")
from src.common import util
from src.kagglebook import model_lgb
from src.kagglebook.runner import Runner
from src.kagglebook.util import Logger

# %%

logger = Logger()
RUN_ID = util.get_run_id()

# %%
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
params = {
    # init param
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.8,
    "random_state": 0,
    # fit param
    "eval_metric": "rmse",
    "stopping_rounds": 5,
}

# %%
runner = Runner(
    RUN_ID,
    model_lgb.ModelLGB,
    features,
    params,
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
