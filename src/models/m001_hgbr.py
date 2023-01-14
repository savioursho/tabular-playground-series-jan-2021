from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
from src.common.util import get_model_id, dump_model
from src import config

MODEL_ID = get_model_id(__file__)
MODLE_PATH = config.MODEL_DIR / (MODEL_ID + ".pkl")

params = {
    "random_state": 0,
}

model = HistGradientBoostingRegressor(**params)

dump_model(model, MODLE_PATH)
