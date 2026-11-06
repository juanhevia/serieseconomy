"""Prepara ventanas y escaladores. Ejecutar: python -m scripts.prepare_data."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oilprice.preprocessing import fit_scaler, make_calendar, make_samples, validate_config

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preparation.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    raw_path = ROOT / config["raw_path"]
    metadata = json.loads(raw_path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_hash(raw_path) != metadata["sha256"]:
        raise ValueError("El CSV sin rellenar no coincide con su huella registrada.")
    if (metadata["series"] != "DCOILBRENTEU" or metadata["requested_start"] != config["start"]
            or metadata["requested_end"] != config["as_of"]):
        raise ValueError("La copia descargada no corresponde a la configuración.")
    raw = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date")["brent"]
    lag = config["publication_lag_bdays"]
    frame = make_calendar(raw, start=config["start"], end=config["as_of"], lag=lag,
                          fill_limit=config["max_forward_fill_bdays"])
    output = ROOT / "data/processed" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "calendar_audit.csv", date_format="%Y-%m-%d")
    manifest = {
        "config": config,
        "raw_sha256": metadata["sha256"],
        "config_sha256": sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "pipeline_sha256": file_hash(ROOT / "oilprice/preprocessing.py"),
        "calendar": "Lunes a viernes, incluyendo festivos; los objetivos no observados se excluyen.",
        "availability_assumption": f"Retraso fijo supuesto de {lag} días lunes-viernes; no reconstruye publicaciones ni revisiones históricas.",
        "scalers": {}, "datasets": [],
    }
    val_start, test_start = config["validation_start"], config["test_start"]
    end = str((pd.Timestamp(config["as_of"]) + pd.offsets.BDay(1)).date())
    groups = [("main", val_start, [
        ("train", config["start"], val_start),
        ("validation", val_start, test_start),
        ("test", test_start, end),
    ])]
    fold_starts = config["walk_forward_starts"]
    for i, (a, b) in enumerate(zip(fold_starts, [*fold_starts[1:], val_start]), 1):
        groups.append((f"fold_{i:02d}", a, [("train", config["start"], a), ("validation", a, b)]))
    for group, scaler_cutoff, blocks in groups:
        scaler = fit_scaler(frame, available_before=scaler_cutoff, lag=lag)
        manifest["scalers"][group] = scaler.to_dict()
        for split, block_start, block_end in blocks:
            for horizon in config["horizons"]:
                samples = make_samples(frame, scaler, block_start=block_start, block_end=block_end,
                                       window=config["window"], horizon=horizon, lag=lag)
                n = len(samples["y_usd"])
                if not n:
                    raise ValueError(f"No hay muestras para {group}/{split}/{horizon}.")
                filename = f"{group}_{split}_h{horizon}.npz"
                archive = output / filename
                np.savez_compressed(archive, **samples)
                manifest["datasets"].append({
                    "group": group, "split": split, "horizon_bdays": horizon,
                    "file": filename, "sha256": file_hash(archive), "samples": n,
                    "block_start": block_start, "block_end_exclusive": block_end,
                    "first_origin": str(samples["origin_date"][0]),
                    "last_origin": str(samples["origin_date"][-1]),
                    "last_target": str(samples["target_date"][-1]),
                    "x_shape": list(samples["x"].shape),
                })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# Preparación de datos: Brent univariante", "",
              f"Copia de FRED: `{config['raw_path']}`. SHA-256: `{metadata['sha256']}`.", "",
              f"Calendario de {len(frame):,} filas lunes-viernes, del {frame.index[0]:%Y-%m-%d} al {frame.index[-1]:%Y-%m-%d}.",
              f"Precios observados: {frame.brent_observed.notna().sum():,}. Fechas sin precio observado: {frame.brent_observed.isna().sum():,}.", "",
              f"Ventanas de {config['window']} filas y objetivos a {config['horizons']} días lunes-viernes. No se cuentan sesiones de mercado.",
              f"Se rellenan solo entradas con el último valor, como máximo {config['max_forward_fill_bdays']} filas; jamás se rellenan objetivos ni hacia atrás.",
              f"Se asume un retraso de publicación de {lag} días lunes-viernes. Es una hipótesis configurable, no un retraso histórico verificado.",
              "Los objetivos de cada bloque deben estar observados y disponibles, según ese supuesto, antes del inicio del siguiente bloque.", "",
              "## Muestras listas para las redes", "",
              "| Grupo | Bloque | Horizonte | Muestras | Primer origen | Último origen | Último objetivo |",
              "|---|---|---:|---:|---|---|---|"]
    for d in manifest["datasets"]:
        report.append(f"| {d['group']} | {d['split']} | {d['horizon_bdays']} | {d['samples']} | {d['first_origin']} | {d['last_origin']} | {d['last_target']} |")
    report += ["", "## Escalado", "",
               "Se usa (precio - media) / desviación estándar poblacional. Cada fold ajusta su propio escalador con precios originales disponibles antes de validación; no usa precios de validación o prueba.",
               "El mismo escalador transforma las entradas y el objetivo. Para recuperar USD/barril: predicción * scale + mean. No se recortan valores fuera del rango de entrenamiento.", "",
               "## Límites del experimento", "",
               "Esta fase prepara únicamente Brent; las otras variables del CSV exploratorio no entran todavía en las redes.",
               "La copia de FRED contiene el histórico revisado al descargarla. El retraso fijo no sustituye a las fechas reales de publicación ni a los datos disponibles en cada momento.",
               "Los resultados serán una comparación retrospectiva bajo este supuesto, no un backtest de inversión en tiempo real.",
               "El bloque test se prepara, pero no se calcula ninguna métrica de prueba en esta fase.", "",
               "Fuente: [EIA vía FRED, DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU).", ""]
    (ROOT / "reports/preparacion.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Preparados {len(manifest['datasets'])} archivos de ventanas: {output.relative_to(ROOT)}")
    print("Informe: reports/preparacion.md. Prueba reservada, sin evaluar.")


if __name__ == "__main__":
    main()
