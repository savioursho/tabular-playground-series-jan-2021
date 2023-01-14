import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from tqdm import tqdm

from src import config


def oof(model, X_train_val, y_train_val, groups=None, cv=5):
    X = np.asarray(X_train_val)
    y = np.asarray(y_train_val).ravel()

    df_oof = pd.DataFrame()
    df_oof["y_true"] = y
    df_oof["oof"] = np.nan
    df_oof["fold"] = np.nan

    cv = check_cv(cv, y, classifier=is_classifier(model))
    splits = list(cv.split(X, y, groups=groups))

    for i, (train_idx, val_idx) in enumerate(tqdm(splits, desc="[OOF]")):
        _model = clone(model)
        _model.fit(X[train_idx, :], y[train_idx])
        y_pred = _model.predict(X[val_idx, :])
        df_oof.iloc[val_idx, df_oof.columns.get_loc("oof")] = y_pred
        df_oof.iloc[val_idx, df_oof.columns.get_loc("fold")] = i

    df_oof["fold"] = df_oof["fold"].astype(np.uint8)

    return df_oof


def save_oof(df_oof: pd.DataFrame, file_name):
    path = config.OOF_DIR / file_name
    df_oof.to_csv(path, index=False)
