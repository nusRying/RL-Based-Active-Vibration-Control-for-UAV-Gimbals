import os
import sys
from pathlib import Path

if os.environ.get("RL_AVC_NO_REEXEC") != "1":
    os.environ["RL_AVC_NO_REEXEC"] = "1"
    target = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    os.execv(sys.executable, [sys.executable, "-s", target, *sys.argv[1:]])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_avc_gimbal.training.frequency_analysis import main


if __name__ == "__main__":
    main()
