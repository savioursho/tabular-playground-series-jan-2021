from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from tqdm import tqdm

from src import config
from src.fi import get_permutation_importance


def oof(
    model,
    X_train_val: pd.DataFrame,
    y_train_val,
    groups=None,
    cv=5,
    return_permutation_importance: bool = False,
    permutation_importance_repeats: int = 10,
    permutation_importance_samples: Union[int, float] = 1.0,
):
    X = np.asarray(X_train_val)
    y = np.asarray(y_train_val).ravel()

    feature_names = X_train_val.columns.to_list()

    df_oof = pd.DataFrame()
    df_oof["y_true"] = y
    df_oof["oof"] = np.nan
    df_oof["fold"] = np.nan

    list_models = []
    list_df_permutation_importance = []

    cv = check_cv(cv, y, classifier=is_classifier(model))
    splits = list(cv.split(X, y, groups=groups))

    for i, (train_idx, val_idx) in enumerate(tqdm(splits, desc="[OOF]")):
        _model = clone(model)
        _model.fit(X[train_idx, :], y[train_idx])
        y_pred = _model.predict(X[val_idx, :])
        df_oof.iloc[val_idx, df_oof.columns.get_loc("oof")] = y_pred
        df_oof.iloc[val_idx, df_oof.columns.get_loc("fold")] = i
        list_models.append(_model)

        if return_permutation_importance:
            df_permutation_importance = get_permutation_importance(
                _model,
                X[val_idx, :],
                y[val_idx],
                feature_names,
                n_repeats=permutation_importance_repeats,
                max_samples=permutation_importance_samples,
            )
            list_df_permutation_importance.append(df_permutation_importance)

    df_oof["fold"] = df_oof["fold"].astype(np.uint8)

    if return_permutation_importance:
        df_permutation_importance = pd.concat(list_df_permutation_importance, axis=0)
        return df_oof, list_models, df_permutation_importance
    else:
        return df_oof, list_models


def save_oof(df_oof: pd.DataFrame, file_name):
    path = config.OOF_DIR / file_name
    df_oof.to_csv(path, index=False)
