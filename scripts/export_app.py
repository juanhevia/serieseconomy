"""Exporta la selección de validación para inferencia; no entrena ni selecciona en test."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from oilprice.training import completed_result, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def export_bundle(run, output):
    run, output = Path(run), Path(output)
    if not (run / "completed.json").is_file():
        raise ValueError("Falta un entrenamiento terminado.")
    if output.exists():
        raise ValueError("El destino ya existe. Usa una carpeta nueva para conservar la versión anterior.")
    selections = json.loads((run / "selection_validation.json").read_text())
    if sorted(s["horizon_bdays"] for s in selections) != [5, 63, 252]:
        raise ValueError("Se requieren los tres horizontes.")
    test = pd.read_csv(run / "test_evaluation/predictions_test.csv")
    models = {}
    for selection in selections:
        job_id = selection["job_id"]
        if Path(job_id).name != job_id:
            raise ValueError("Identificador de modelo inválido.")
        job = run / "jobs" / job_id
        result = completed_result(job)
        h = selection["horizon_bdays"]
        if result is None or result["group"] != "main" or result["horizon_bdays"] != h or result["architecture"] != selection["best_network_on_validation"]:
            raise ValueError("La selección no corresponde al modelo guardado.")
        metadata = json.loads((job / "metadata.json").read_text())
        with np.load(job / "validation_predictions.npz", allow_pickle=False) as z:
            residuals = z["actual_usd"] - z["predicted_usd"]
        if not np.isfinite(residuals).all() or len(residuals) < 10:
            raise ValueError("Errores de validación inválidos.")
        lo, hi = np.quantile(residuals, [.1, .9], method="linear")
        selected_test = test[(test.architecture == metadata["architecture"]) & (test.horizon_bdays == h)]
        errors = selected_test.actual_usd - selected_test.predicted_usd
        if selected_test.empty:
            raise ValueError("Falta la evaluación del modelo seleccionado.")
        models[str(h)] = {"job_id": job_id, "architecture": metadata["architecture"],
                          "metadata": metadata, "weights": f"h{h}.pt",
                          "range": {"nominal_mass": .8, "residual_low": float(lo), "residual_high": float(hi),
                                    "validation_n": len(residuals), "test_n": len(errors),
                                    "test_coverage": float(((errors >= lo) & (errors <= hi)).mean()),
                                    "method": "Quantiles 10/90 of actual-predicted validation residuals; validation also used for selection and early stopping; no coverage guarantee."}}
    output.mkdir(parents=True)
    for item in models.values():
        shutil.copyfile(run / "jobs" / item["job_id"] / "best.pt", output / item["weights"])
    for source, target in [("test_evaluation/metrics_test.csv", "metrics_test.csv"),
                           ("test_evaluation/predictions_test.csv", "predictions_test.csv"),
                           ("comparison_validation.csv", "metrics_validation.csv")]:
        shutil.copyfile(run / source, output / target)
    manifest = {"format_version": 1, "selection_rule": "best_network_on_validation",
                "selection_sha256": file_hash(run / "selection_validation.json"),
                "training_signature": json.loads((run / "signature.json").read_text()),
                "models": models,
                "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    print(f"Modelos seleccionados y evaluación exportados a {output}. Sin entrenamiento.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/laboratorio")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/brent_v1")
    args = parser.parse_args()
    export_bundle(args.run, args.output)
