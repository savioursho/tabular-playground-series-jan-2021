import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance


def get_permutation_importance(
    model,
    X_val,
    y_val,
    feature_names=None,
    n_repeats=30,
):
    r = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
    )

    if feature_names is None:
        feature_names = model.feature_names_in_

    df_permutation_importance = pd.DataFrame(
        r["importances"].T,
        columns=feature_names,
    )
    return df_permutation_importance


def plot_permutation_importance(df_permutation_importance, path):

    # 平均が大きい順に並び替え
    df_permutation_importance = (
        df_permutation_importance.T.assign(mean=lambda df: df.mean(axis=1))
        .sort_values("mean", ascending=False)
        .drop(columns="mean")
        .T
    )

    g = sns.catplot(
        data=df_permutation_importance.melt(),
        y="variable",
        x="value",
        kind="box",
    )

    g.savefig(path)
