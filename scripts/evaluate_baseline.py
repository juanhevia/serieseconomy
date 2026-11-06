"""Evalúa persistencia en validación y folds; nunca lee los archivos de prueba."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oilprice.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/brent_univariate_v1")
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for record in manifest["datasets"]:
        if record["split"] != "validation":
            continue
        archive = args.prepared / record["file"]
        if sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Archivo modificado desde la preparación: {archive.name}")
        with np.load(archive, allow_pickle=False) as data:
            metrics = regression_metrics(data["y_usd"], data["persistence_usd"])
        rows.append({"group": record["group"], "horizon_bdays": record["horizon_bdays"], **metrics})
    if not rows:
        raise ValueError("El manifiesto no contiene bloques de validación.")
    results = pd.DataFrame(rows)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    results.to_csv(reports / "baseline_validation.csv", index=False)
    text = ["# Referencia ingenua: persistencia", "",
            "Predicción = último precio disponible en la fecha de origen, usando exactamente la misma información que recibirán las redes.",
            "Los objetivos son precios realmente observados. MAE y RMSE están en USD/barril; MSE en (USD/barril)².",
            f"Supuesto de disponibilidad: {manifest['config']['publication_lag_bdays']} días lunes-viernes de retraso fijo. No equivale a una reconstrucción histórica de publicaciones.",
            f"Huella de la copia fuente: `{manifest['raw_sha256']}`.", "",
            "## Validación principal", "",
            "| Horizonte (días lunes-viernes) | Muestras | MAE | RMSE | MSE |",
            "|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["group"] == "main":
            text.append(f"| {r['horizon_bdays']} | {r['n']} | {r['mae_usd']:.3f} | {r['rmse_usd']:.3f} | {r['mse_usd2']:.3f} |")
    text += ["", "## Validación walk-forward dentro de entrenamiento", "",
             "| Fold | Horizonte | Muestras | MAE | RMSE | MSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["group"] != "main":
            text.append(f"| {r['group']} | {r['horizon_bdays']} | {r['n']} | {r['mae_usd']:.3f} | {r['rmse_usd']:.3f} | {r['mse_usd2']:.3f} |")
    text += ["", "## Interpretación", "",
             "Esta tabla fija la referencia que deben superar las redes en las mismas fechas de cada horizonte. Menor MAE o RMSE significa menor error.",
             "Los horizontes tienen distintos orígenes válidos, por lo que sus cifras no son una comparación con muestra idéntica entre horizontes.",
             "Dentro de cada horizonte se mantendrán exactamente los mismos orígenes y objetivos para los cuatro modelos.",
             "Los errores de horizontes solapados no son independientes. No se calculan intervalos de confianza suponiendo independencia.",
             "No se han entrenado redes ni evaluado el bloque de prueba.", ""]
    (reports / "baseline_validation.md").write_text("\n".join(text), encoding="utf-8")
    print(results.loc[results.group == "main"].to_string(index=False))
    print("Resultados: reports/baseline_validation.md. Test no leído ni evaluado.")


if __name__ == "__main__":
    main()
