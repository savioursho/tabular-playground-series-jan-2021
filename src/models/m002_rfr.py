from sklearn.ensemble import RandomForestRegressor

from src import config
from src.common.util import dump_model, get_model_id

MODEL_ID = get_model_id(__file__)
MODLE_PATH = config.MODEL_DIR / (MODEL_ID + ".pkl")

params = {
    "random_state": 0,
}

model = RandomForestRegressor(**params)

dump_model(model, MODLE_PATH)
