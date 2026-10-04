import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gkco_dvtt.config import load_config
from gkco_dvtt.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GKCO-DVTT pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = run_pipeline(config)
    print("\nFinal metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
