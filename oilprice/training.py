"""Entrenamiento reproducible, selección temporal y resultados de validación.

No abre los archivos de prueba. Una tarea terminada se reutiliza tras verificar
sus huellas; una tarea interrumpida se repite desde su semilla inicial.
"""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import time

# Debe establecerse antes de inicializar CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oilprice.metrics import regression_metrics
from oilprice.models import RecurrentRegressor, load_regressor


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_training_config(config):
    for key in ("max_epochs", "patience", "batch_size", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} debe ser un entero positivo.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Semilla fuera del rango [0, 2**32).")
    for key in ("gradient_clip", "min_delta"):
        value = config[key]
        if not math.isfinite(value) or value < 0 or (key == "gradient_clip" and value == 0):
            raise ValueError(f"Valor no válido de {key}.")
    architectures = config["architectures"]
    if not architectures or len(set(architectures)) != len(architectures) or not set(architectures) <= {"RNN", "LSTM", "GRU"}:
        raise ValueError("Arquitecturas no válidas o duplicadas.")
    horizons = config["horizons"]
    if not horizons or len(set(horizons)) != len(horizons) or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Horizontes no válidos o duplicados.")
    if not config["candidates"]:
        raise ValueError("Se necesita al menos un candidato.")
    for candidate in config["candidates"]:
        if set(candidate) != {"hidden_size", "learning_rate"}:
            raise ValueError("Cada candidato debe especificar hidden_size y learning_rate.")
        if type(candidate["hidden_size"]) is not int or candidate["hidden_size"] < 1:
            raise ValueError("hidden_size debe ser un entero positivo.")
        if not math.isfinite(candidate["learning_rate"]) or candidate["learning_rate"] <= 0:
            raise ValueError("learning_rate debe ser positivo y finito.")


def select_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. Revisa el controlador y la instalación de PyTorch. No se iniciará el entrenamiento en CPU por sorpresa.")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("Selecciona explícitamente cuda o cpu.")


def seed_everything(seed, cpu_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_info(device, threads):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "cpu_threads": threads,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_scaled(model, x, device, batch_size):
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
        outputs.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs)


class PreparedData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest_path = self.folder / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = {}
        self.cache = {}
        for record in self.manifest["datasets"]:
            if record["split"] == "test":
                continue  # Ni abrir, ni comprobar la huella del archivo reservado.
            if record["split"] not in {"train", "validation"}:
                raise ValueError("Bloque inesperado en el manifiesto.")
            key = (record["group"], record["split"], record["horizon_bdays"])
            if key in self.records:
                raise ValueError(f"Bloque duplicado: {key}")
            self.records[key] = record

    @property
    def folds(self):
        return sorted({key[0] for key in self.records if key[0].startswith("fold_")})

    def load(self, group, split, horizon):
        if split not in {"train", "validation"}:
            raise ValueError("El entrenamiento solo puede abrir train y validation.")
        key = (group, split, horizon)
        if key in self.cache:
            return self.cache[key]
        record = self.records[key]
        if Path(record["file"]).name != record["file"]:
            raise ValueError("Se espera un nombre de archivo dentro del directorio preparado.")
        path = self.folder / record["file"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Archivo preparado modificado: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in ("x", "y", "y_usd", "persistence_usd", "origin_date", "target_date")}
        n = record["samples"]
        expected = (n, self.manifest["config"]["window"], 1)
        if n < 1 or data["x"].shape != expected or data["y"].shape != (n, 1):
            raise ValueError(f"Dimensiones inesperadas en {path.name}.")
        for name in ("y_usd", "persistence_usd", "origin_date", "target_date"):
            if data[name].shape != (n,):
                raise ValueError(f"Dimensión incorrecta: {name} en {path.name}.")
        for name in ("x", "y", "y_usd", "persistence_usd"):
            if not np.isfinite(data[name]).all():
                raise ValueError(f"Datos no finitos: {name} en {path.name}.")
        scaler = self.manifest["scalers"][group]
        if not math.isfinite(scaler["mean"]) or not math.isfinite(scaler["scale"]) or scaler["scale"] <= 0:
            raise ValueError("Escalador no válido.")
        if not np.allclose(data["y"].ravel() * scaler["scale"] + scaler["mean"], data["y_usd"], atol=1e-4, rtol=1e-5):
            raise ValueError("El objetivo escalado no coincide con el precio y el escalador.")
        self.cache[key] = data
        return data

    def preflight(self, config):
        if not self.folds:
            raise ValueError("Faltan folds temporales para seleccionar hiperparámetros.")
        for group in [*self.folds, "main"]:
            for horizon in config["horizons"]:
                for split in ("train", "validation"):
                    self.load(group, split, horizon)
        return len(config["architectures"]) * len(config["horizons"]) * (len(config["candidates"]) * len(self.folds) + 1)


def completed_result(folder):
    result_path = folder / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for filename in ("best.pt", "metadata.json", "history.csv", "validation_predictions.npz"):
        path = folder / filename
        if not path.exists() or file_hash(path) != result["artifacts"][filename]:
            raise ValueError(f"Resultado terminado incompleto o modificado: {path}. Usa otra carpeta de salida.")
    return result


def fit_job(data, config, architecture, horizon, candidate_index, group, output, device):
    job_id = f"{architecture.lower()}_h{horizon}_c{candidate_index:02d}_{group}"
    folder = output / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    cached = completed_result(folder)
    if cached is not None:
        print(f"Reutilizado: {job_id}", flush=True)
        return cached
    seed_everything(config["seed"], config["cpu_threads"])
    train = data.load(group, "train", horizon)
    validation = data.load(group, "validation", horizon)
    candidate = config["candidates"][candidate_index]
    model = RecurrentRegressor(architecture, candidate["hidden_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"])
    loader = DataLoader(TensorDataset(torch.from_numpy(train["x"].astype(np.float32)),
                                     torch.from_numpy(train["y"].astype(np.float32))),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    best_loss, significant_best, stale = math.inf, math.inf, 0
    best_state, best_epoch, history = None, 0, []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        train_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita en {job_id}, época {epoch}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            train_sum += loss.item() * len(x)
        prediction = predict_scaled(model, validation["x"], device, config["batch_size"])
        val_loss = float(np.mean((prediction.astype(float) - validation["y"].ravel()) ** 2))
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Validación no finita en {job_id}.")
        history.append({"epoch": epoch, "train_mse_scaled": train_sum / len(train["x"]), "validation_mse_scaled": val_loss})
        # El checkpoint siempre conserva el mínimo absoluto; min_delta solo rige la paciencia.
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if val_loss < significant_best - config["min_delta"]:
            significant_best, stale = val_loss, 0
        else:
            stale += 1
        print(f"{job_id}: época {epoch}/{config['max_epochs']}, MSE validación escalada={val_loss:.6f}", flush=True)
        if stale >= config["patience"]:
            break
    synchronize(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    scaler = data.manifest["scalers"][group]
    scaled_prediction = predict_scaled(model, validation["x"], device, config["batch_size"])
    predicted_usd = scaled_prediction.astype(float) * scaler["scale"] + scaler["mean"]
    metrics = regression_metrics(validation["y_usd"], predicted_usd)
    baseline = regression_metrics(validation["y_usd"], validation["persistence_usd"])
    metadata = {"architecture": architecture, "candidate": candidate, "input_size": 1,
                "window": data.manifest["config"]["window"], "horizon_bdays": horizon,
                "group": group, "scaler": scaler, "preparation_config": data.manifest["config"],
                "raw_sha256": data.manifest["raw_sha256"], "seed": config["seed"],
                "best_epoch": best_epoch, "torch_version": str(torch.__version__)}
    temporary_weights = folder / "best.pt.tmp"
    torch.save(best_state, temporary_weights)
    temporary_weights.replace(folder / "best.pt")
    write_json(folder / "metadata.json", metadata)
    write_csv(folder / "history.csv", history)
    temporary_predictions = folder / "validation_predictions.tmp.npz"
    np.savez_compressed(temporary_predictions, actual_usd=validation["y_usd"],
                        predicted_usd=predicted_usd, persistence_usd=validation["persistence_usd"],
                        origin_date=validation["origin_date"], target_date=validation["target_date"])
    temporary_predictions.replace(folder / "validation_predictions.npz")
    result = {"job_id": job_id, "architecture": architecture, "horizon_bdays": horizon,
              "candidate_index": candidate_index, "group": group, "metrics": metrics,
              "baseline": baseline, "parameters": sum(p.numel() for p in model.parameters()),
              "epochs_run": len(history), "best_epoch": best_epoch, "fit_seconds": seconds,
              "artifacts": {name: file_hash(folder / name) for name in
                            ("best.pt", "metadata.json", "history.csv", "validation_predictions.npz")}}
    # Solo este último archivo marca una tarea como terminada.
    write_json(folder / "result.json", result)
    return result


def run_training(prepared, config, output, device):
    validate_training_config(config)
    data = PreparedData(prepared)
    jobs = data.preflight(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device, config["cpu_threads"])
    signature = {"training_config": config, "prepared_manifest_sha256": file_hash(data.manifest_path),
                 "source_sha256": {name: file_hash(Path(__file__).parent / name)
                                   for name in ("training.py", "models.py", "metrics.py")},
                 "environment": environment}
    signature_path = output / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("La salida pertenece a otros datos, código, configuración o entorno. Usa una carpeta de salida nueva.")
    elif any(output.iterdir()):
        raise ValueError("La carpeta de salida no está vacía y no tiene una firma verificable.")
    else:
        write_json(signature_path, signature)
        write_json(output / "started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Plan: {jobs} ajustes como máximo; dispositivo {device}. El bloque de prueba no se abre.", flush=True)
    search_rows, comparisons, selections = [], [], []
    for horizon in config["horizons"]:
        validation = data.load("main", "validation", horizon)
        baseline = regression_metrics(validation["y_usd"], validation["persistence_usd"])
        comparisons.append({"architecture": "PERSISTENCE", "horizon_bdays": horizon, **baseline,
                            "improvement_rmse_pct": 0.0, "parameters": 0, "fit_seconds": 0.0,
                            "search_seconds": 0.0, "best_epoch": 0, "job_id": ""})
        neural_rows = []
        for architecture in config["architectures"]:
            candidates = []
            total_search_seconds = 0.0
            for index, candidate in enumerate(config["candidates"]):
                folds = [fit_job(data, config, architecture, horizon, index, fold, output, device) for fold in data.folds]
                count = sum(r["metrics"]["n"] for r in folds)
                pooled_rmse = math.sqrt(sum(r["metrics"]["mse_usd2"] * r["metrics"]["n"] for r in folds) / count)
                seconds = sum(r["fit_seconds"] for r in folds)
                row = {"architecture": architecture, "horizon_bdays": horizon,
                       "candidate_index": index, **candidate, "pooled_cv_rmse_usd": pooled_rmse,
                       "validation_samples": count, "search_seconds": seconds}
                search_rows.append(row)
                candidates.append(row)
                total_search_seconds += seconds
            best = min(candidates, key=lambda r: (r["pooled_cv_rmse_usd"], r["candidate_index"]))
            result = fit_job(data, config, architecture, horizon, best["candidate_index"], "main", output, device)
            improvement = 100 * (1 - result["metrics"]["rmse_usd"] / baseline["rmse_usd"]) if baseline["rmse_usd"] > 0 else None
            row = {"architecture": architecture, "horizon_bdays": horizon, **result["metrics"],
                   "improvement_rmse_pct": improvement, "parameters": result["parameters"],
                   "fit_seconds": result["fit_seconds"], "search_seconds": total_search_seconds,
                   "best_epoch": result["best_epoch"], "job_id": result["job_id"]}
            comparisons.append(row)
            neural_rows.append(row)
        best_network = min(neural_rows, key=lambda r: r["rmse_usd"])
        beats = best_network["rmse_usd"] < baseline["rmse_usd"]
        selections.append({"horizon_bdays": horizon, "best_network_on_validation": best_network["architecture"],
                           "job_id": best_network["job_id"], "beats_persistence_on_validation": beats,
                           "preferred_on_validation": best_network["architecture"] if beats else "PERSISTENCE"})
    write_csv(output / "search_results.csv", search_rows)
    write_csv(output / "comparison_validation.csv", comparisons)
    write_json(output / "selection_validation.json", selections)
    lines = ["# Comparación en validación: RNN, LSTM, GRU y persistencia", "",
             "Los hiperparámetros se seleccionan en los folds temporales por RMSE conjunto en USD/barril (ponderado por número de muestras).",
             "Después se ajusta cada red con entrenamiento principal y parada temprana en validación principal. Se restaura la época con menor MSE de validación.",
             "Estas son cifras de desarrollo y selección, no una evaluación independiente final. El bloque de prueba no se ha abierto.", "",
             "| Red | Horizonte | N | MAE (USD) | RMSE (USD) | MSE (USD²) | Mejora RMSE (%) | Parámetros | Ajuste (s) | Búsqueda (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        improvement = "n.a." if row["improvement_rmse_pct"] is None else f"{row['improvement_rmse_pct']:.2f}"
        lines.append(f"| {row['architecture']} | {row['horizon_bdays']} | {row['n']} | {row['mae_usd']:.3f} | {row['rmse_usd']:.3f} | {row['mse_usd2']:.3f} | {improvement} | {row['parameters']} | {row['fit_seconds']:.1f} | {row['search_seconds']:.1f} |")
    lines += ["", "## Lectura de los resultados", "",
              "Una mejora positiva indica menor RMSE que persistencia; una negativa indica peor resultado. No se presupone que una red gane.",
              "El tiempo de ajuste incluye entrenamiento y validación por época. La búsqueda suma todos los candidatos y folds de esa red y horizonte.",
              "Todas las redes comparten ventanas, fechas, semilla, candidatos y presupuesto máximo de épocas; el número de parámetros y la parada temprana pueden diferir.",
              "Se utiliza una sola semilla. Para estudiar estabilidad, repetir el protocolo con otras semillas y directorios nuevos, sin consultar prueba.",
              "Los horizontes tienen distintos orígenes válidos. La comparación entre modelos es justa dentro de cada horizonte.",
              "El retraso de publicación y las limitaciones de la copia histórica son los de la preparación. No se han calculado intervalos de incertidumbre.", "",
              "## Selección provisional por horizonte", ""]
    for selected in selections:
        lines.append(f"- {selected['horizon_bdays']} días: mejor red en validación {selected['best_network_on_validation']}; referencia preferida según RMSE: {selected['preferred_on_validation']}.")
    (output / "comparison_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "completed.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "max_jobs": jobs})
    print(f"Terminado. Comparación: {output / 'comparison_validation.md'}", flush=True)
    return comparisons
