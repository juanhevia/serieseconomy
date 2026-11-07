"""Entrena las tres redes. Ejecutar desde la raíz: python -m scripts.train_models."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/brent_univariate_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/laboratorio")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check", action="store_true", help="Comprueba el dispositivo con tensores sintéticos; no lee datos del proyecto.")
    parser.add_argument("--dry-run", action="store_true", help="Verifica los archivos y muestra el plan sin entrenar ni crear resultados.")
    args = parser.parse_args()
    try:
        from oilprice.training import (PreparedData, environment_info, run_training,
                                       seed_everything, select_device, validate_training_config)
        from oilprice.models import RecurrentRegressor
        import torch
    except ImportError as exc:
        parser.exit(2, f"Faltan dependencias de entrenamiento: {exc}. Consulta docs/entrenamiento_laboratorio.md.\n")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_training_config(config)
    device = select_device(args.device)
    if args.check:
        seed_everything(config["seed"], config["cpu_threads"])
        for name in config["architectures"]:
            model = RecurrentRegressor(name, 4).to(device)
            x = torch.ones((2, 6, 1), device=device)
            model(x).square().mean().backward()
        print(json.dumps(environment_info(device, config["cpu_threads"]), indent=2))
        print("Dispositivo y tres arquitecturas comprobados con datos sintéticos. No se ha entrenado con datos reales.")
    elif args.dry_run:
        data = PreparedData(args.prepared)
        fits = data.preflight(config)
        print(f"Archivos verificados. Plan: {fits} ajustes; hasta {config['max_epochs']} épocas por ajuste; {len(data.folds)} folds.")
        print(f"Dispositivo: {device}. Salida prevista: {args.output}. Test no leído. No se ha entrenado.")
    else:
        run_training(args.prepared, config, args.output, device)


if __name__ == "__main__":
    main()
