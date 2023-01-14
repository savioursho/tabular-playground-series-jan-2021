from src import config
import os.path
import joblib


def get_model_id(file_name: str):
    return os.path.splitext(os.path.basename(file_name))[0]


def dump_model(model, path):
    joblib.dump(model, path, compress=True)


def get_model(model_id: str):
    path = config.MODEL_DIR / (model_id + ".pkl")
    return joblib.load(path)
