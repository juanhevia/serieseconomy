"""Evaluación final con datos artificiales y pesos fijos; no abre el Brent real."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("La evaluación requiere PyTorch y matplotlib.")

import torch

from test_training import toy_fixture
from oilprice.evaluation import evaluate_frozen
from oilprice.metrics import regression_metrics
from oilprice.training import file_hash, run_training


def evaluation_fixture(root):
    prepared, config = toy_fixture(root)
    config.update(architectures=["RNN"], candidates=[config["candidates"][0]], max_epochs=2)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["datasets"][-1]
    n = record["samples"]
    x = np.random.default_rng(321).normal(size=(n, 6, 1)).astype(np.float32)
    dates = np.datetime64("2025-12-25") + np.arange(n)
    path = prepared / record["file"]
    np.savez_compressed(path, x=x, y_usd=(.7 * x[:, -1, 0] + .2).astype(float) * 2 + 10,
                        persistence_usd=x[:, -1, 0].astype(float) * 2 + 10,
                        origin_date=dates.astype("U10"), target_date=(dates + 5).astype("U10"))
    record["sha256"] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return prepared, config


class EvaluationTests(unittest.TestCase):
    def test_frozen_inference_metrics_plots_and_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, config = evaluation_fixture(root)
            run = root / "run"
            device = torch.device("cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(prepared, config, run, device)
            frozen_paths = list((run / "jobs").glob("*/best.pt")) + list((run / "jobs").glob("*/metadata.json"))
            frozen_paths += [run / "selection_validation.json", prepared / "manifest.json"]
            before = {p: file_hash(p) for p in frozen_paths}
            with patch("torch.optim.Adam", side_effect=AssertionError("Evaluar no debe entrenar")), contextlib.redirect_stdout(io.StringIO()):
                rows = evaluate_frozen(run, prepared, device)
            self.assertEqual(before, {p: file_hash(p) for p in frozen_paths})
            self.assertEqual({r["architecture"] for r in rows}, {"RNN", "PERSISTENCE"})
            output = run / "test_evaluation"
            for name in ("metrics_test.csv", "metrics_by_year.csv", "largest_errors.csv", "predictions_test.csv", "evaluation_test.md",
                         "real_predicho_h5.png", "errores_h5.png"):
                self.assertGreater((output / name).stat().st_size, 50)
            predictions = np.genfromtxt(output / "predictions_test.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            rnn = predictions[predictions["architecture"] == "RNN"]
            expected = regression_metrics(rnn["actual_usd"], rnn["predicted_usd"])
            for key, value in expected.items():
                self.assertAlmostEqual(next(r for r in rows if r["architecture"] == "RNN")[key], value)
            years = np.genfromtxt(output / "metrics_by_year.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            self.assertEqual(set(years["target_year"]), {2025, 2026})
            (run / "selection_validation.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ya existe una evaluación"):
                evaluate_frozen(run, prepared, device)
            (prepared / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "datos preparados difieren"):
                evaluate_frozen(run, prepared, device)

    def test_incomplete_training_cannot_open_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("numpy.load", side_effect=AssertionError("No debe abrir prueba")):
                with self.assertRaisesRegex(ValueError, "entrenamiento no está completado"):
                    evaluate_frozen(Path(temporary) / "missing-run", Path(temporary) / "missing-data", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
