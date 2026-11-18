"""Contrato de actualización y equivalencia con el experimento del laboratorio."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from oilprice.inference import (build_input, forecast, history_csv, latest_origin,
                                load_bundle, merge_history, read_update, save_history)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/brent_v1"


class UpdateTests(unittest.TestCase):
    def test_csv_validation_and_idempotent_merge(self):
        good = read_update(b"Date,brent\n2026-11-05,104.5\n2026-11-04,103\n", as_of="2026-11-19")
        self.assertTrue(good.index.is_monotonic_increasing)
        for data in [b"Date,brent\n2026-11-05,NaN\n", b"Date,brent\n2026-11-05,inf\n",
                     b"Date,brent\n2026-11-07,100\n", b"Date,brent\n2026-11-05,100\n2026-11-05,101\n",
                     b"Date,brent\n2026-11-20,100\n", b"Date,brent\n05/11/2026,100\n",
                     b"Date,brent\n2026-02-30,100\n", b"Date,brent\n2026-11-05,abc\n"]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_update(data, as_of="2026-11-19")
        base = pd.Series([102., np.nan], index=pd.to_datetime(["2026-11-03", "2026-11-04"]), name="brent")
        merged, count = merge_history(base, good)
        self.assertEqual(count, 2)
        self.assertEqual(merge_history(merged, good)[1], 0)
        with self.assertRaisesRegex(ValueError, "modifica precios"):
            merge_history(merged, good + 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            save_history(merged, path)
            restored = read_update(path.read_bytes(), as_of="2026-11-19")
            np.testing.assert_array_equal(merged.values, restored.values)
            self.assertEqual(history_csv(restored), path.read_bytes())


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "Requiere el paquete exportado")
class FrozenInferenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle, self.history = load_bundle(BUNDLE, ROOT)
        self.meta = self.bundle["models"]["5"]["metadata"]

    def test_window_is_causal_and_stale_data_are_rejected(self):
        origin = pd.Timestamp("2024-06-03")
        x, info = build_input(self.history, origin, self.meta)
        changed = self.history.copy()
        # Incluso los datos recientes que aún no estarían publicados se ignoran.
        changed.loc[changed.index > origin - pd.offsets.BDay(3)] = 99999
        x_after, info_after = build_input(changed, origin, self.meta)
        np.testing.assert_array_equal(x, x_after)
        self.assertEqual(info, info_after)
        with self.assertRaisesRegex(ValueError, "ventana completa"):
            build_input(self.history.loc[:"2024-01-01"], origin, self.meta)
        for date in ["2024-06-02", "2099-01-01", "2000-01-10"]:
            with self.assertRaises(ValueError):
                build_input(self.history, date, self.meta)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requiere PyTorch")
    def test_saved_models_reproduce_laboratory_without_fitting(self):
        expected = pd.read_csv(BUNDLE / "predictions_test.csv")
        manifest_before = (BUNDLE / "manifest.json").read_bytes()
        for h in [5, 63, 252]:
            architecture = self.bundle["models"][str(h)]["architecture"]
            row = expected[(expected.horizon_bdays == h) & (expected.architecture == architecture)].iloc[50]
            with patch("torch.optim.Adam", side_effect=AssertionError("No debe entrenar")):
                result = forecast(self.history, row.origin_date, h, self.bundle, BUNDLE)
            self.assertAlmostEqual(result["predicted_usd"], row.predicted_usd, delta=.005)
            self.assertEqual(result["target_date"], row.target_date)
            self.assertLess(result["range_low_usd"], result["range_high_usd"])
        self.assertEqual((BUNDLE / "manifest.json").read_bytes(), manifest_before)

    def test_saved_range_coverage_matches_test_without_recalibration(self):
        frame = pd.read_csv(BUNDLE / "predictions_test.csv")
        for h, item in self.bundle["models"].items():
            sample = frame[(frame.horizon_bdays == int(h)) & (frame.architecture == item["architecture"])]
            errors = sample.actual_usd - sample.predicted_usd
            band = item["range"]
            coverage = ((errors >= band["residual_low"]) & (errors <= band["residual_high"])).mean()
            self.assertAlmostEqual(coverage, band["test_coverage"])


if __name__ == "__main__":
    unittest.main()
