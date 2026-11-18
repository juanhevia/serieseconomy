"""Evaluación posterior al entrenamiento, sin optimización ni selección en prueba."""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from oilprice.metrics import regression_metrics
from oilprice.models import load_regressor
from oilprice.training import completed_result, file_hash, predict_scaled, write_csv, write_json


def comparison_plots(folder, horizon, dates, actual, predictions):
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(folder / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    names = list(predictions)
    rows = math.ceil(len(names) / 2)
    x = np.asarray(dates, dtype="datetime64[D]")
    for residuals, filename in [(False, f"real_predicho_h{horizon}.png"), (True, f"errores_h{horizon}.png")]:
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), squeeze=False, layout="constrained")
        for ax, name in zip(axes.ravel(), names):
            if residuals:
                ax.plot(x, predictions[name] - actual, color="#C66A26", linewidth=.9)
                ax.axhline(0, color="#444444", linewidth=.8)
                ax.set_ylabel("Predicho − real (USD/barril)")
            else:
                ax.plot(x, actual, color="#176B98", linewidth=1, label="Real")
                ax.plot(x, predictions[name], color="#C66A26", linewidth=.9, alpha=.85, label="Predicho")
                ax.set_ylabel("USD/barril")
                ax.legend(fontsize=8)
            ax.set_title(name)
            ax.set_xlabel("Fecha objetivo")
            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.grid(alpha=.2)
        for ax in axes.ravel()[len(names):]:
            ax.set_visible(False)
        title = "Errores en prueba" if residuals else "Valores reales y predichos en prueba"
        fig.suptitle(f"{title}: horizonte de {horizon} días lunes-viernes")
        fig.savefig(folder / filename, dpi=140, facecolor="white")
        plt.close(fig)


def evaluate_frozen(run, prepared, device):
    """Abre prueba solo tras verificar modelos terminados y la selección en validación."""
    run, prepared = Path(run), Path(prepared)
    if not (run / "completed.json").is_file():
        raise ValueError("El entrenamiento no está completado; no se abre prueba.")
    signature = json.loads((run / "signature.json").read_text(encoding="utf-8"))
    manifest_path = prepared / "manifest.json"
    if file_hash(manifest_path) != signature["prepared_manifest_sha256"]:
        raise ValueError("Los datos preparados difieren de los usados al entrenar.")
    if file_hash(Path(__file__).with_name("models.py")) != signature["source_sha256"]["models.py"]:
        raise ValueError("La definición de las redes ha cambiado desde el entrenamiento.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = signature["training_config"]
    selection_path = run / "selection_validation.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    with (run / "comparison_validation.csv").open(encoding="utf-8") as stream:
        comparison = list(csv.DictReader(stream))
    jobs = {}
    for row in comparison:
        if row["architecture"] == "PERSISTENCE":
            continue
        key = (row["architecture"], int(row["horizon_bdays"]))
        if key in jobs or Path(row["job_id"]).name != row["job_id"]:
            raise ValueError("Referencia a un modelo duplicada o no válida.")
        folder = run / "jobs" / row["job_id"]
        result = completed_result(folder)
        if result is None or result["group"] != "main" or (result["architecture"], result["horizon_bdays"]) != key:
            raise ValueError("El modelo no corresponde al ajuste principal terminado.")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["scaler"] != manifest["scalers"]["main"] or metadata["raw_sha256"] != manifest["raw_sha256"]:
            raise ValueError("Modelo y datos usan diferentes escaladores o copias de origen.")
        jobs[key] = (folder, metadata)
    expected = {(a, h) for a in config["architectures"] for h in config["horizons"]}
    if set(jobs) != expected:
        raise ValueError("Faltan modelos principales o hay modelos inesperados.")
    receipt = {"prepared_manifest_sha256": file_hash(manifest_path),
               "selection_validation_sha256": file_hash(selection_path),
               "models": {folder.name: file_hash(folder / "best.pt") for folder, _ in jobs.values()}}
    output = run / "test_evaluation"
    previous = output / "evaluation_signature.json"
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Ya existe una evaluación con otros modelos o selección. No se sobrescribe.")
    output.mkdir(exist_ok=True)
    write_json(previous, receipt)
    metric_rows, annual_rows, worst_rows, full_predictions = [], [], [], []
    for horizon in config["horizons"]:
        records = [r for r in manifest["datasets"] if r["group"] == "main" and r["split"] == "test" and r["horizon_bdays"] == horizon]
        if len(records) != 1 or Path(records[0]["file"]).name != records[0]["file"]:
            raise ValueError("Bloque de prueba ausente, duplicado o no válido.")
        record = records[0]
        archive_path = prepared / record["file"]
        if file_hash(archive_path) != record["sha256"]:
            raise ValueError("El archivo de prueba ha sido modificado.")
        with np.load(archive_path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in ["x", "y_usd", "persistence_usd", "origin_date", "target_date"]}
        n = record["samples"]
        if data["x"].shape != (n, manifest["config"]["window"], 1) or not np.isfinite(data["x"]).all():
            raise ValueError("Entradas de prueba inválidas.")
        if any(data[k].shape != (n,) for k in ["y_usd", "persistence_usd", "origin_date", "target_date"]):
            raise ValueError("Longitudes de prueba inconsistentes.")
        if not (data["target_date"] >= data["origin_date"]).all():
            raise ValueError("Objetivos anteriores al origen.")
        predictions = {"PERSISTENCE": data["persistence_usd"]}
        for architecture in config["architectures"]:
            folder, metadata = jobs[(architecture, horizon)]
            model = load_regressor(folder / "best.pt", metadata, device)
            scaled = predict_scaled(model, data["x"], device, config["batch_size"])
            predictions[architecture] = scaled.astype(float) * metadata["scaler"]["scale"] + metadata["scaler"]["mean"]
        baseline = regression_metrics(data["y_usd"], predictions["PERSISTENCE"])
        years = np.array([str(d)[:4] for d in data["target_date"]])
        for architecture, values in predictions.items():
            metrics = regression_metrics(data["y_usd"], values)
            improvement = 100 * (1 - metrics["rmse_usd"] / baseline["rmse_usd"]) if baseline["rmse_usd"] > 0 else None
            metric_rows.append({"architecture": architecture, "horizon_bdays": horizon, **metrics,
                                "improvement_rmse_pct": improvement})
            for year in np.unique(years):
                mask = years == year
                annual_rows.append({"architecture": architecture, "horizon_bdays": horizon, "target_year": str(year),
                                    **regression_metrics(data["y_usd"][mask], values[mask])})
            per_date = [{"architecture": architecture, "horizon_bdays": horizon,
                         "origin_date": str(data["origin_date"][i]), "target_date": str(data["target_date"][i]),
                         "actual_usd": float(data["y_usd"][i]), "predicted_usd": float(values[i]),
                         "error_usd": float(values[i] - data["y_usd"][i])} for i in range(n)]
            full_predictions.extend(per_date)
            worst_rows.extend(sorted(per_date, key=lambda r: abs(r["error_usd"]), reverse=True)[:5])
        comparison_plots(output, horizon, data["target_date"], data["y_usd"], predictions)
    write_csv(output / "metrics_test.csv", metric_rows)
    write_csv(output / "metrics_by_year.csv", annual_rows)
    write_csv(output / "largest_errors.csv", worst_rows)
    write_csv(output / "predictions_test.csv", full_predictions)
    lines = ["# Evaluación final sobre fechas no usadas para ajustar las redes", "",
             "Los pesos y escaladores permanecen fijos. Este comando no entrena ni vuelve a elegir hiperparámetros, épocas o arquitectura.",
             "En cada origen se utiliza la ventana histórica disponible bajo el supuesto de retraso de la preparación. Es una evaluación con orígenes sucesivos; no una trayectoria generada desde una única fecha.", "",
             "| Modelo | Horizonte | N | R² | MAE (USD) | MSE (USD²) | RMSE (USD) | Mejora RMSE (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        r2 = "n.a." if row["r2"] is None else f"{row['r2']:.3f}"
        improvement = "n.a." if row["improvement_rmse_pct"] is None else f"{row['improvement_rmse_pct']:.2f}"
        lines.append(f"| {row['architecture']} | {row['horizon_bdays']} | {row['n']} | {r2} | {row['mae_usd']:.3f} | {row['mse_usd2']:.3f} | {row['rmse_usd']:.3f} | {improvement} |")
    lines += ["", "R² puede ser negativo: su referencia es la media del bloque evaluado, no persistencia. No está definido con una sola observación o con valores reales constantes.",
              "Los intervalos de incertidumbre están pendientes; estos gráficos no son bandas de confianza.", "",
              "## Selección fijada antes de abrir prueba", ""]
    for row in selection:
        lines.append(f"- Horizonte {row['horizon_bdays']}: mejor red en validación {row['best_network_on_validation']}; opción preferida entonces: {row['preferred_on_validation']}.")
    lines += ["", "La selección anterior no se modifica según los resultados de prueba. Si se cambia el modelo tras observarlos, este bloque deja de ser una evaluación independiente para la nueva versión.",
              "Para analizar cuándo acierta o falla, revisar los errores por fecha, las métricas por año y los cinco errores absolutos más grandes de cada modelo y horizonte. Son descripciones, no explicaciones causales.", ""]
    for horizon in config["horizons"]:
        lines += [f"## Horizonte {horizon}", "", f"![Valores reales y predichos](real_predicho_h{horizon}.png)", "",
                  f"![Errores por fecha](errores_h{horizon}.png)", ""]
    (output / "evaluation_test.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluación final guardada en {output}. No se ha reentrenado ni cambiado la selección.")
    return metric_rows
