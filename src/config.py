import os, sys, datetime as dt
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
TODAY = dt.date(2026, 10, 20)  # dataset reference date (never use the system clock)

def _find_dataset():
    for c in (os.getenv("DATASET_DIR"), ROOT / "Capstone_Project_Dataset_FTTT_2026",
              ROOT.parent / "Capstone_Project_Dataset_FTTT_2026", ROOT / "dataset"):
        if c and (Path(c) / "kb").is_dir():
            return Path(c).resolve()
    raise SystemExit("Dataset not found. Extract the capstone ZIP and set DATASET_DIR in .env")

DATASET = _find_dataset()
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))  # makes `from tools import lookup_order, create_ticket` work
