"""Evaluación final: ejecutar solo después de fijar el procedimiento de entrenamiento."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/laboratorio")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/brent_univariate_v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    from oilprice.evaluation import evaluate_frozen
    from oilprice.training import select_device
    evaluate_frozen(args.run, args.prepared, select_device(args.device))


if __name__ == "__main__":
    main()
