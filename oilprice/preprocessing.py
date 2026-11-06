"""Ventanas causales sobre un calendario de lunes a viernes, sin objetivos imputados."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def validate_config(config):
    for key in ("window", "publication_lag_bdays", "max_forward_fill_bdays"):
        value = config[key]
        minimum = 1 if key == "window" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} debe ser un entero >= {minimum}.")
    horizons = config["horizons"]
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Los horizontes deben ser enteros positivos.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Hay horizontes duplicados.")
    start, val, test, end = [pd.Timestamp(config[k]) for k in
                             ("start", "validation_start", "test_start", "as_of")]
    if not start < val < test < end:
        raise ValueError("Las fechas deben cumplir inicio < validación < prueba < cierre.")
    folds = [pd.Timestamp(d) for d in config["walk_forward_starts"]]
    if folds != sorted(set(folds)) or any(not start < d < val for d in folds):
        raise ValueError("Los folds deben ser crecientes y estar dentro del entrenamiento.")
    if any(d.day_of_week >= 5 for d in [val, test, *folds]):
        raise ValueError("Los cortes deben caer de lunes a viernes.")


def make_calendar(raw, *, start, end, lag, fill_limit):
    """La entrada en t solo puede proceder de una observación hasta t-lag.

    El retraso es un supuesto del experimento, no una fecha de publicación real.
    El calendario B incluye festivos. Los objetivos observados no se rellenan.
    """
    if lag < 0 or fill_limit < 0:
        raise ValueError("Retraso y límite de relleno no pueden ser negativos.")
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans:
        raise ValueError("El índice debe contener fechas válidas.")
    if raw.index.tz is not None or not raw.index.equals(raw.index.normalize()):
        raise ValueError("Se esperan fechas diarias sin zona horaria.")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Fechas duplicadas o desordenadas.")
    if (raw.index.day_of_week >= 5).any():
        raise ValueError("Hay fechas en fin de semana; revisar el calendario de origen.")
    raw = pd.to_numeric(raw, errors="raise").astype(float)
    if np.isinf(raw).any():
        raise ValueError("Hay precios infinitos.")
    if ((raw.index < pd.Timestamp(start)) | (raw.index > pd.Timestamp(end))).any():
        raise ValueError("La serie contiene fechas fuera del periodo configurado.")
    index = pd.bdate_range(start, end, name="Date")
    observed = raw.reindex(index)
    source = pd.Series(index, index=index).where(observed.notna())
    if fill_limit:
        filled = observed.ffill(limit=fill_limit)
        source = source.ffill(limit=fill_limit)
    else:
        filled = observed.copy()
    available = filled.shift(lag)
    source = source.shift(lag)
    imputed = (observed.isna() & filled.notna()).shift(lag, fill_value=False)
    frame = pd.DataFrame({
        "brent_observed": observed,
        "brent_available": available,
        "source_observation_date": source,
        "input_was_filled": imputed,
    })
    known = source.notna()
    if not (source[known] <= index[known] - pd.offsets.BDay(lag)).all():
        raise ValueError("Una entrada usa una observación posterior a su fecha permitida.")
    return frame


@dataclass(frozen=True)
class PriceScaler:
    mean: float
    scale: float
    n_observed: int
    first_observation: str
    last_observation: str
    available_before: str

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def to_dict(self):
        return asdict(self)


def fit_scaler(frame, *, available_before, lag):
    """Ajusta media y desviación solo sobre precios de entrenamiento disponibles."""
    cutoff = pd.Timestamp(available_before)
    published = frame.index + pd.offsets.BDay(lag)
    prices = frame.loc[published < cutoff, "brent_observed"].dropna()
    if prices.empty:
        raise ValueError("No hay precios disponibles para ajustar el escalador.")
    scale = float(prices.std(ddof=0))
    return PriceScaler(float(prices.mean()), scale if scale > 0 else 1.0,
                       len(prices), str(prices.index[0].date()), str(prices.index[-1].date()),
                       str(cutoff.date()))


def make_samples(frame, scaler, *, block_start, block_end, window, horizon, lag):
    """Bloque [inicio, fin). La etiqueta también debe estar disponible antes del fin.

    X tiene forma (muestras, ventana, 1). y es el precio al final del horizonte.
    Validación y prueba pueden usar historia anterior al inicio de su bloque.
    """
    if window < 1 or horizon < 1 or lag < 0:
        raise ValueError("Ventana/horizonte deben ser positivos y retraso no negativo.")
    start, end = pd.Timestamp(block_start), pd.Timestamp(block_end)
    if start >= end:
        raise ValueError("El bloque debe tener inicio anterior al fin.")
    x_values = frame.brent_available.to_numpy(dtype=float)
    targets = frame.brent_observed.to_numpy(dtype=float)
    dates = frame.index
    publication_dates = dates + pd.offsets.BDay(lag)
    imputed = frame.input_was_filled.to_numpy(dtype=float)
    windows, origins, destinations, fractions = [], [], [], []
    for i in range(window - 1, len(frame) - horizon):
        j = i + horizon
        if not start <= dates[i] < end or publication_dates[j] >= end:
            continue
        values = x_values[i - window + 1:i + 1]
        if not np.isfinite(targets[j]) or not np.isfinite(values).all():
            continue
        windows.append(values)
        origins.append(i)
        destinations.append(j)
        fractions.append(float(imputed[i - window + 1:i + 1].mean()))
    origins = np.asarray(origins, dtype=int)
    destinations = np.asarray(destinations, dtype=int)
    raw_x = np.asarray(windows, dtype=float).reshape(-1, window, 1)
    y_usd = targets[destinations]
    return {
        "x": scaler.transform(raw_x).astype(np.float32),
        "y": scaler.transform(y_usd).astype(np.float32).reshape(-1, 1),
        "y_usd": y_usd,
        "persistence_usd": x_values[origins],
        "origin_date": dates[origins].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "target_date": dates[destinations].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "last_observation_date": frame.source_observation_date.iloc[origins].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "input_imputed_fraction": np.asarray(fractions, dtype=float),
    }
