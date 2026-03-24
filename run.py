from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.main import parse_bind_target, run


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python run.py [HOST:PORT]")

    if len(sys.argv) == 2:
        run(*parse_bind_target(sys.argv[1]))
    else:
        run()
