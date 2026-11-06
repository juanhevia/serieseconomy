"""Métricas en unidades originales, sin dependencias de entrenamiento."""

import numpy as np


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Se requieren vectores no vacíos de igual forma.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("No se admiten valores faltantes o infinitos en las métricas.")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    return {"n": len(error), "mae_usd": float(np.mean(np.abs(error))),
            "mse_usd2": mse, "rmse_usd": float(np.sqrt(mse))}
