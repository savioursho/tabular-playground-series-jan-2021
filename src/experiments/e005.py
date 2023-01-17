from src import config
from src.common import update_tracking, util
from src.cv.oof import save_oof
from src.fi import plot_permutation_importance
from src.pipelines import p003

TEST_RUN = False

#######################
# EXPERIMENT ID
#######################
EXPERIMENT_ID = util.get_experiment_id(__file__)

#######################
# RUN ID
#######################
RUN_ID = util.get_run_id()
if TEST_RUN:
    RUN_ID = "test" + RUN_ID

#######################
# DATA
#######################
df_train, df_test = util.load_data()
if TEST_RUN:
    df_train = df_train.sample(100)
    df_test = df_test.sample(100)

#######################
# RUN
#######################
dict_result = p003.train(df_train)

#######################
# SAVE
#######################
update_tracking(RUN_ID, "oof_score", dict_result["oof_score"])
update_tracking(RUN_ID, "experiment_id", EXPERIMENT_ID)
update_tracking(RUN_ID, "model_type", dict_result["model_type"])

oof_file_name = "-".join([RUN_ID, EXPERIMENT_ID, dict_result["model_type"]]) + ".csv"
save_oof(dict_result["df_oof"], oof_file_name)

# feature importance
plot_permutation_importance(
    dict_result["df_permutation_importance"],
    config.FI_FIG_DIR
    / ("-".join([RUN_ID, EXPERIMENT_ID, dict_result["model_type"]]) + ".png"),
)

dict_result["df_permutation_importance"].to_csv(
    config.FI_DIR
    / ("-".join([RUN_ID, EXPERIMENT_ID, dict_result["model_type"]]) + ".csv"),
    index=False,
)
