#%%
from pathlib import Path

from src.common import util
from src.features import dump_features

logger = util.get_logger(__name__)

FILE_NAME = util.get_file_basename(__file__)

# %%


def main():
    logger.info("Creating features: %s", FILE_NAME)
    # ======================================
    # データ読み込み
    # ======================================

    df_train, df_test = util.load_data()

    # ======================================
    # 特徴量一覧
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

    # ======================================
    # 保存
    # ======================================
    # train
    dump_features(
        df_train[features],
        "train",
        FILE_NAME,
    )
    # test
    dump_features(
        df_test[features],
        "test",
        FILE_NAME,
    )


# ======================================
# 実行
# ======================================
if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("Error")
