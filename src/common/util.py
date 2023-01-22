import logging
import os.path
import sys
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from typing_extensions import Literal

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


def load_data(
    train_test: Literal["train", "test", "all"] = "all",
):
    if train_test == "train":
        df_train = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
        return df_train
    elif train_test == "test":
        df_test = pd.read_csv(config.RAW_DATA_DIR / "test.csv")
        return df_test
    elif train_test == "all":
        df_train = pd.read_csv(config.RAW_DATA_DIR / "train.csv")
        df_test = pd.read_csv(config.RAW_DATA_DIR / "test.csv")
        return df_train, df_test


def get_logger(
    logger_name: str,
):
    # formatter
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    formatter = logging.Formatter(FORMAT)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # ストリームハンドラー
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    # ファイルハンドラー
    f_handler = logging.handlers.RotatingFileHandler(
        config.LOG_DIR / "experiment.log",
        encoding="utf-8",
        maxBytes=1000,
        backupCount=5,
    )
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger
