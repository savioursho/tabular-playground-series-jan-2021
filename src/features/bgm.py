import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm

from src.common.util import util
from src.features import dump_features, load_features

# ======================================
# 設定
# ======================================

logger = util.get_logger(__name__)

FILE_NAME = util.get_file_basename(__file__)


def main():
    # ======================================
    # データ読み込み
    # ======================================
    features = [
        "cont1",
        "cont2",
        "cont3",
        "cont4",
        "cont5",
        "cont6",
        "cont7",
        "cont8",
        "cont9",
        "cont10",
        "cont11",
        "cont12",
        "cont13",
        "cont14",
    ]

    df_train = load_features(
        features,
        "train",
    )
    df_test = load_features(
        features,
        "test",
    )

    # ======================================
    # 特徴量作成
    # ======================================

    # 各変数の要素数を設定
    # =====
    dict_n_components = {
        "cont1": 4,
        "cont2": 10,
        "cont3": 7,
        "cont4": 5,
        "cont5": 5,
        "cont6": 5,
        "cont7": 4,
        "cont8": 8,
        "cont9": 8,
        "cont10": 7,
        "cont11": 7,
        "cont12": 5,
        "cont13": 6,
        "cont14": 7,
    }

    # fit済みのクラスを格納する辞書
    # =====
    dict_bgm = dict()

    # 結果のデータフレーム
    # =====
    df_bgm_train = pd.DataFrame()
    df_bgm_test = pd.DataFrame()

    # trainに対して実行
    # =====
    logger.info("Creating features for train.")
    for feature in tqdm(features):
        bgm = BayesianGaussianMixture(
            n_components=dict_n_components[feature],
            max_iter=1000,
            random_state=0,
            verbose=2,
            verbose_interval=20,
        )
        df_bgm_train[feature + "_bgm"] = bgm.fit_predict(df_train[[feature]])
        dict_bgm[feature] = bgm

    # testに対して実行
    # =====
    logger.info("Creating features for test.")
    for feature in tqdm(features):
        df_bgm_test[feature + "_bgm"] = dict_bgm[feature].predict(df_test[[feature]])

    # ======================================
    # 保存
    # ======================================
    dump_features(
        df_bgm_train,
        "train",
        FILE_NAME,
    )
    dump_features(
        df_bgm_test,
        "test",
        FILE_NAME,
    )


if __name__ == "__main__":

    with util.timer("feature " + FILE_NAME, logger):
        try:
            main()
        except:
            logger.exception("Error")
