from pathlib import Path

HOME_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = HOME_DIR / "data/raw"
PROCESSED_DATA_DIR = HOME_DIR / "data/processed"
FEATURES_DATA_DIR = HOME_DIR / "data/features"

LOG_DIR = HOME_DIR / "logs"
SUBMISSION_DIR = HOME_DIR / "submissions"
OOF_DIR = HOME_DIR / "oof"
PRED_DIR = HOME_DIR / "pred"
FI_DIR = HOME_DIR / "fi"
FI_FIG_DIR = HOME_DIR / "fi_fig"
HPO_DIR = HOME_DIR / "hpo"
MODEL_DIR = HOME_DIR / "models"

TRACKING_FILE = HOME_DIR / "tracking/tracking.csv"
