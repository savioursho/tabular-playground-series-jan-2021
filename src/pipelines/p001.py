"""
pipeline001
model type: HistGradientBoostingRegressor
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src import config
from src.common import util
from src.cv.oof import oof


def preprecess():
    pass


def generate_features():
    pass


def train(
    df_train: pd.DataFrame,
):
    #######################
    # data loading
    #######################

    X_train_val = df_train.drop(columns=["target", "id"])
    y_train_val = df_train["target"].values

    #######################
    # model definition
    #######################
    model_type = "hgbr"
    params = {
        "random_state": 0,
    }
    model = HistGradientBoostingRegressor(**params)

    #######################
    # splitter
    #######################
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)

    #######################
    # oof
    #######################
    df_oof, list_models, df_permutation_importance = oof(
        model,
        X_train_val,
        y_train_val,
        cv=splitter,
        return_permutation_importance=True,
    )
    oof_score = np.sqrt(mean_squared_error(df_oof["y_true"], df_oof["oof"]))

    #######################
    # result
    #######################
    dict_result = {
        "df_oof": df_oof,
        "list_models": list_models,
        "df_permutation_importance": df_permutation_importance,
        "oof_score": oof_score,
        "model_type": model_type,
    }

    return dict_result


def predict():
    pass
