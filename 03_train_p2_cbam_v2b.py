"""Train the SkinSeg-YOLO26n P2-CBAM-v2B GatedFusion preset."""
import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_V2B_ARGS = [
    "--architecture", "v2b",
    "--optimizer", "AdamW",
    "--epochs", "200",
    "--name", "SkinSeg_YOLO26_P2_CBAM_v2B_GatedFusion_AdamW_E200",
]


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_p2_cbam", SCRIPT_DIR / "03_train_p2_cbam.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    train = _load_training_module()
    train.main(DEFAULT_V2B_ARGS + sys.argv[1:])


if __name__ == "__main__":
    main()
