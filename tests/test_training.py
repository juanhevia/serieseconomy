"""Pruebas con datos artificiales; nunca entrenan sobre el petróleo real."""

import contextlib
import copy
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Pruebas de entrenamiento opcionales: instalar PyTorch para ejecutarlas.")

import torch

from oilprice.models import RecurrentRegressor, load_regressor
from oilprice.training import (PreparedData, predict_scaled, run_training, seed_everything,
                                select_device, validate_training_config)


def toy_fixture(root):
    prepared = root / "prepared"
    prepared.mkdir()
    config = {"architectures": ["RNN", "LSTM", "GRU"], "horizons": [5],
              "candidates": [{"hidden_size": 3, "learning_rate": .01},
                             {"hidden_size": 4, "learning_rate": .005}],
              "max_epochs": 3, "patience": 2, "min_delta": 0.00001,
              "batch_size": 8, "gradient_clip": 1.0, "seed": 17, "cpu_threads": 1}
    manifest = {"config": {"window": 6, "publication_lag_bdays": 3},
                "raw_sha256": "synthetic-fixture-not-market-data", "scalers": {}, "datasets": []}
    rng = np.random.default_rng(123)
    for group in ("fold_01", "main"):
        manifest["scalers"][group] = {"mean": 10.0, "scale": 2.0}
        for split, n in (("train", 16), ("validation", 8)):
            x = rng.normal(size=(n, 6, 1)).astype(np.float32)
            y = (x[:, -1, :] * .7 + .2).astype(np.float32)
            dates = (np.datetime64("2000-01-01") + np.arange(n)).astype("U10")
            target_dates = (np.datetime64("2000-01-06") + np.arange(n)).astype("U10")
            name = f"{group}_{split}_h5.npz"
            np.savez_compressed(prepared / name, x=x, y=y, y_usd=y.ravel().astype(float) * 2 + 10,
                                persistence_usd=x[:, -1, 0].astype(float) * 2 + 10,
                                origin_date=dates, target_date=target_dates)
            manifest["datasets"].append({"group": group, "split": split, "horizon_bdays": 5,
                                         "file": name, "sha256": sha256((prepared / name).read_bytes()).hexdigest(),
                                         "samples": n})
    # Si el entrenamiento intentase abrir prueba, fallaría: este archivo no existe.
    manifest["datasets"].append({"group": "main", "split": "test", "horizon_bdays": 5,
                                 "file": "DO_NOT_OPEN_TEST.npz", "sha256": "absent", "samples": 10})
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prepared, config


class TrainingTests(unittest.TestCase):
    def test_architectures_and_saved_weights_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("RNN", "LSTM", "GRU"):
                seed_everything(17, 1)
                model = RecurrentRegressor(name, 4)
                x = torch.randn(3, 6, 1)
                result = model(x)
                self.assertEqual(tuple(result.shape), (3, 1))
                result.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                path = Path(folder) / f"{name}.pt"
                torch.save(model.state_dict(), path)
                metadata = {"architecture": name, "candidate": {"hidden_size": 4}, "input_size": 1}
                loaded = load_regressor(path, metadata)
                torch.testing.assert_close(model(x), loaded(x))

    def test_full_selection_resume_and_checkpoint_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, config = toy_fixture(root)
            output = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                rows = run_training(prepared, config, output, torch.device("cpu"))
            self.assertEqual({r["architecture"] for r in rows}, {"RNN", "LSTM", "GRU", "PERSISTENCE"})
            self.assertEqual(len(list((output / "jobs").glob("*/result.json"))), 9)
            self.assertFalse((prepared / "DO_NOT_OPEN_TEST.npz").exists())
            signatures = {str(p.relative_to(output)): p.stat().st_mtime_ns for p in (output / "jobs").glob("*/best.pt")}
            with patch("torch.optim.Adam", side_effect=AssertionError("No debe reentrenar tareas completas")), contextlib.redirect_stdout(io.StringIO()):
                resumed = run_training(prepared, config, output, torch.device("cpu"))
            self.assertEqual(rows, resumed)
            self.assertEqual(signatures, {str(p.relative_to(output)): p.stat().st_mtime_ns for p in (output / "jobs").glob("*/best.pt")})
            data = PreparedData(prepared)
            validation = data.load("main", "validation", 5)
            for row in rows:
                if row["architecture"] == "PERSISTENCE":
                    continue
                job = output / "jobs" / row["job_id"]
                metadata = json.loads((job / "metadata.json").read_text())
                model = load_regressor(job / "best.pt", metadata)
                actual_prediction = predict_scaled(model, validation["x"], torch.device("cpu"), 8).astype(float) * 2 + 10
                with np.load(job / "validation_predictions.npz", allow_pickle=False) as saved:
                    np.testing.assert_allclose(actual_prediction, saved["predicted_usd"], atol=1e-7)
                history = np.genfromtxt(job / "history.csv", delimiter=",", names=True)
                self.assertEqual(metadata["best_epoch"], int(history["epoch"][np.argmin(history["validation_mse_scaled"])]))
            changed = copy.deepcopy(config)
            changed["max_epochs"] += 1
            with self.assertRaisesRegex(ValueError, "otros datos"):
                run_training(prepared, changed, output, torch.device("cpu"))
            # Una corrupción de pesos no debe pasar inadvertida al reanudar.
            first_weight = next((output / "jobs").glob("*/best.pt"))
            first_weight.write_bytes(first_weight.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "incompleto o modificado"), contextlib.redirect_stdout(io.StringIO()):
                run_training(prepared, config, output, torch.device("cpu"))

    def test_early_stopping_and_restart_of_unfinished_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, config = toy_fixture(root)
            config.update(architectures=["RNN"], candidates=[config["candidates"][0]],
                          max_epochs=8, min_delta=1e6, patience=2)
            output = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(prepared, config, output, torch.device("cpu"))
            result_file = output / "jobs/rnn_h5_c00_main/result.json"
            result = json.loads(result_file.read_text())
            self.assertEqual(result["epochs_run"], 3)
            predictions = (result_file.parent / "validation_predictions.npz").read_bytes()
            result_file.unlink()  # Simula una interrupción antes de marcar la tarea terminada.
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(prepared, config, output, torch.device("cpu"))
            self.assertTrue(result_file.exists())
            self.assertEqual(predictions, (result_file.parent / "validation_predictions.npz").read_bytes())

    def test_missing_cuda_fails_and_test_access_is_rejected(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA no está disponible"):
                select_device("cuda")
        with tempfile.TemporaryDirectory() as temporary:
            prepared, config = toy_fixture(Path(temporary))
            data = PreparedData(prepared)
            with self.assertRaisesRegex(ValueError, "train y validation"):
                data.load("main", "test", 5)
            record = next(iter(data.records.values()))
            (prepared / record["file"]).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "modificado"):
                data.preflight(config)
            config["batch_size"] = 0
            with self.assertRaises(ValueError):
                validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
