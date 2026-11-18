"""Datos de inferencia e inferencia con pesos fijos. No ajusta ningún parámetro."""

from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from oilprice.preprocessing import make_calendar


def today():
    return pd.Timestamp(datetime.now(ZoneInfo("Europe/Madrid")).date())


def read_update(content, *, as_of=None):
    """CSV UTF-8 Date,brent: fechas ISO, sin duplicados y precios finitos."""
    if len(content) > 5_000_000:
        raise ValueError("El CSV supera el límite de 5 MB.")
    try:
        frame = pd.read_csv(BytesIO(content), dtype={"Date": str})
    except (ValueError, UnicodeError) as exc:
        raise ValueError("No se puede leer el CSV. Usa UTF-8, comas y columnas Date,brent.") from exc
    if set(frame.columns) != {"Date", "brent"} or frame.empty:
        raise ValueError("El CSV debe contener datos y exactamente las columnas Date,brent.")
    if not frame.Date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).all():
        raise ValueError("Las fechas deben tener formato AAAA-MM-DD.")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame.Date, format="%Y-%m-%d", errors="raise"), name="Date")
        values = pd.to_numeric(frame.brent, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Hay fechas o precios no válidos.") from exc
    if dates.has_duplicates:
        raise ValueError("El CSV contiene fechas duplicadas.")
    if (dates.dayofweek >= 5).any():
        raise ValueError("Se admiten observaciones de lunes a viernes.")
    if (dates > (today() if as_of is None else pd.Timestamp(as_of))).any():
        raise ValueError("El CSV contiene observaciones con fecha futura.")
    if not np.isfinite(values).all():
        raise ValueError("Los precios deben ser números sin huecos ni infinitos; usa punto decimal.")
    return pd.Series(values, index=dates, name="brent").sort_index()


def merge_history(history, incoming):
    if incoming.index.min() < history.index.min():
        raise ValueError("La actualización no puede preceder al inicio del histórico.")
    common = history.dropna().index.intersection(incoming.index)
    if not np.allclose(history.loc[common], incoming.loc[common], rtol=0, atol=1e-8):
        raise ValueError("El CSV modifica precios ya guardados. Revisa la fuente; solo se añaden observaciones o se completan huecos.")
    merged = history.combine_first(incoming).sort_index().rename("brent")
    return merged, int(merged.notna().sum() - history.notna().sum())


def history_csv(history):
    return history.dropna().rename_axis("Date").to_csv().encode("utf-8")


def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(history_csv(history))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_bundle(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or set(manifest["models"]) != {"5", "63", "252"}:
        raise ValueError("Paquete de modelos no compatible.")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Un archivo del modelo ha cambiado. Restaura el paquete original.")
    reference = manifest["models"]["5"]["metadata"]
    raw_path = (root / reference["preparation_config"]["raw_path"]).resolve()
    if not raw_path.is_relative_to(root.resolve()) or sha256(raw_path.read_bytes()).hexdigest() != reference["raw_sha256"]:
        raise ValueError("La copia histórica no coincide con la usada al entrenar.")
    for h, item in manifest["models"].items():
        meta = item["metadata"]
        if item["weights"] not in manifest["files"] or meta["horizon_bdays"] != int(h) or meta["group"] != "main":
            raise ValueError("Metadatos del modelo inconsistentes.")
        for key in ("scaler", "preparation_config", "raw_sha256", "window"):
            if meta[key] != reference[key]:
                raise ValueError("Los modelos no comparten la misma preparación.")
    base = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date").brent
    return manifest, base


def latest_origin(history, lag, *, as_of=None):
    limit = today() if as_of is None else pd.Timestamp(as_of)
    limit = pd.offsets.BDay().rollback(limit)
    return min(history.dropna().index.max() + pd.offsets.BDay(lag), limit)


def build_input(history, origin, metadata, *, as_of=None):
    origin = pd.Timestamp(origin)
    limit = today() if as_of is None else pd.Timestamp(as_of)
    if origin != origin.normalize() or origin.dayofweek >= 5 or origin > limit:
        raise ValueError("La fecha de predicción debe ser un día de lunes a viernes y no puede ser futura.")
    config = metadata["preparation_config"]
    before = history.loc[history.index <= origin]
    if before.empty:
        raise ValueError("No hay histórico anterior a esa fecha.")
    frame = make_calendar(before, start=config["start"], end=origin,
                          lag=config["publication_lag_bdays"], fill_limit=config["max_forward_fill_bdays"])
    window = frame.tail(metadata["window"])
    values = window.brent_available.to_numpy(dtype=float)
    if len(values) != metadata["window"] or not np.isfinite(values).all():
        raise ValueError("No hay una ventana completa de 60 días disponibles. Añade observaciones; los huecos largos no se rellenan.")
    scaler = metadata["scaler"]
    x = ((values - scaler["mean"]) / scaler["scale"]).astype(np.float32).reshape(1, -1, 1)
    if not np.isfinite(x).all():
        raise ValueError("Los precios producen una entrada no válida.")
    return x, {"origin_date": str(origin.date()), "last_observation_date": str(window.source_observation_date.iloc[-1].date()),
               "last_available_usd": float(values[-1]), "input_imputed_fraction": float(window.input_was_filled.mean())}


def forecast(history, origin, horizon, bundle, folder):
    import torch
    from oilprice.models import load_regressor
    item = bundle["models"][str(horizon)]
    meta = item["metadata"]
    x, info = build_input(history, origin, meta)
    torch.set_num_threads(2)
    model = load_regressor(Path(folder) / item["weights"], meta, "cpu")
    with torch.inference_mode():
        scaled = model(torch.from_numpy(x)).item()
    value = scaled * meta["scaler"]["scale"] + meta["scaler"]["mean"]
    if not np.isfinite(value):
        raise ValueError("La red ha producido una predicción no válida.")
    change = value - info["last_available_usd"]
    return {**info, "architecture": item["architecture"], "horizon_bdays": horizon,
            "target_date": str((pd.Timestamp(origin) + pd.offsets.BDay(horizon)).date()),
            "predicted_usd": value, "change_usd": change,
            "change_pct": 100 * change / info["last_available_usd"] if info["last_available_usd"] != 0 else None,
            "range_low_usd": value + item["range"]["residual_low"],
            "range_high_usd": value + item["range"]["residual_high"],
            "range_test_coverage": item["range"]["test_coverage"]}
