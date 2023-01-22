import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .model import Model
from .util import Util

MODEL_DIR = "./model"  # 実行ファイルと同じ階層にディレクトリを作成


class ModelLGB(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # バリデーションセットの有無
        validation = va_x is not None

        # ハイパーパラメータの設定
        params = dict(self.params)
        stopping_rounds = params.pop("stopping_rounds")
        eval_metric = params.pop("eval_metric")

        # モデルのセット
        self.model = lgb.LGBMRegressor(**params)

        # 学習
        if validation:
            self.model.fit(
                tr_x,
                tr_y,
                eval_metric=eval_metric,
                eval_set=[(va_x, va_y)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
                ],
            )

        else:
            self.model.fit(
                tr_x,
                tr_y,
            )

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self):
        model_path = os.path.join(MODEL_DIR, f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join(MODEL_DIR, f"{self.run_fold_name}.model")
        self.model = Util.load(model_path)
