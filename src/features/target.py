from src.common.util import util
from src.features import dump_features

logger = util.get_logger(__name__)

FILE_NAME = util.get_file_basename(__file__)


def main():
    logger.info("Creating features: %s", FILE_NAME)
    # ======================================
    # データ読み込み
    # ======================================

    df_train = util.load_data("train")

    # ======================================
    # 特徴量一覧
    # ======================================

    features = ["target"]

    # ======================================
    # 保存
    # ======================================
    # train
    dump_features(
        df_train[features],
        "train",
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
