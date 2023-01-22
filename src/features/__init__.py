from typing import List

import pandas as pd
from typing_extensions import Literal

from src import config
from src.common import util

logger = util.get_logger(__name__)


def load_features(
    features: List[str],
    train_test: Literal["train", "test"],
):
    logger.debug("Load features.")
    list_dfs = []
    for feature in features:
        pattern = f"{feature}_{train_test}.ftr"
        list_file = list(config.FEATURES_DATA_DIR.rglob(pattern))
        if len(list_file) == 0:
            raise Exception(f"Feature {feature} doesn't exists.")
        elif len(list_file) > 1:
            raise Exception(f"Feature {feature} might be duplicated.")
        else:
            df_feature = pd.read_feather(list_file[0])
            list_dfs.append(df_feature)

    return pd.concat(list_dfs, axis=1)


def dump_features(
    df_features: pd.DataFrame,
    train_test: Literal["train", "test"],
    dir: str,
):
    logger.debug("Dump features.")
    for col in df_features:
        path = config.FEATURES_DATA_DIR / dir / f"{col}_{train_test}.ftr"
        path.parent.mkdir(parents=True, exist_ok=True)
        df_features[[col]].to_feather(path)
