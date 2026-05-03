import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
sys.path = [str(PYTHON_DIR)] + [path for path in sys.path if path != str(PYTHON_DIR)]
