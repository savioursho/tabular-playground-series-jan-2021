import os.path
from datetime import datetime

import joblib
import pandas as pd

from src import config


def get_model_id(file_name: str):
    return os.path.splitext(os.path.basename(file_name))[0]


def dump_model(model, path):
    joblib.dump(model, path, compress=True)


def get_model(model_id: str):
    path = config.MODEL_DIR / (model_id + ".pkl")
    return joblib.load(path)


def get_run_id():
    return datetime.now().strftime("%m%d_%H%M_%S")


def get_experiment_id(file_name: str):
    return os.path.splitext(os.path.basename(file_name))[0]


def load_data():
    df_train = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
    df_test = pd.read_csv(config.RAW_DATA_DIR / "test.csv")

    return df_train, df_test
