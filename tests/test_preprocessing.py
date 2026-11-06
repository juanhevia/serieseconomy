import unittest

import numpy as np
import pandas as pd

from oilprice.metrics import regression_metrics
from oilprice.preprocessing import fit_scaler, make_calendar, make_samples


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2020-01-01", periods=120)
        self.raw = pd.Series(np.arange(120, dtype=float) + 20, index=self.dates)

    def calendar(self, raw=None, lag=2, fill_limit=3):
        return make_calendar(self.raw if raw is None else raw,
                             start=self.dates[0], end=self.dates[-1],
                             lag=lag, fill_limit=fill_limit)

    def samples(self, frame, scaler, end=None):
        return make_samples(frame, scaler, block_start=self.dates[25],
                            block_end=self.dates[80] if end is None else end,
                            window=5, horizon=5, lag=2)

    def test_no_backward_fill_and_limit_on_long_gap(self):
        raw = self.raw.copy()
        raw.iloc[:2] = np.nan
        raw.iloc[10:17] = np.nan
        frame = self.calendar(raw, lag=2, fill_limit=3)
        self.assertTrue(frame.brent_available.iloc[:4].isna().all())
        np.testing.assert_array_equal(frame.brent_available.iloc[12:15], [29, 29, 29])
        self.assertTrue(frame.brent_available.iloc[15:19].isna().all())
        self.assertTrue(frame.brent_observed.iloc[10:17].isna().all())
        self.assertTrue(frame.input_was_filled.iloc[12:15].all())
        self.assertFalse(frame.input_was_filled.iloc[15:19].any())

    def test_target_is_observed_and_horizon_counts_weekdays(self):
        raw = self.raw.copy()
        raw.iloc[40] = np.nan
        frame = self.calendar(raw)
        scaler = fit_scaler(frame, available_before=self.dates[25], lag=2)
        batch = self.samples(frame, scaler)
        self.assertNotIn(str(self.dates[40].date()), batch["target_date"])
        self.assertEqual(batch["x"].shape[1:], (5, 1))
        for origin, target, y, last in zip(batch["origin_date"], batch["target_date"],
                                           batch["y_usd"], batch["last_observation_date"]):
            self.assertEqual(pd.Timestamp(target), pd.Timestamp(origin) + pd.offsets.BDay(5))
            self.assertEqual(y, raw.loc[target])
            self.assertLessEqual(pd.Timestamp(last), pd.Timestamp(origin) - pd.offsets.BDay(2))
            self.assertLess(pd.Timestamp(target) + pd.offsets.BDay(2), self.dates[80])
        np.testing.assert_allclose(scaler.inverse_transform(batch["y"].ravel()), batch["y_usd"], atol=1e-5)

    def test_future_prices_cannot_change_training_or_scaler(self):
        cutoff_index, lag = 80, 2
        frame = self.calendar()
        scaler = fit_scaler(frame, available_before=self.dates[cutoff_index], lag=lag)
        before = self.samples(frame, scaler)
        altered = self.raw.copy()
        # Estas observaciones solo se publicarían en validación o después.
        altered.iloc[cutoff_index - lag:] += 10000
        changed = self.calendar(altered)
        changed_scaler = fit_scaler(changed, available_before=self.dates[cutoff_index], lag=lag)
        after = self.samples(changed, changed_scaler)
        self.assertEqual(scaler, changed_scaler)
        self.assertAlmostEqual(scaler.mean, float(self.raw.iloc[:78].mean()))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_each_origin_uses_only_its_available_history(self):
        frame = self.calendar()
        scaler = fit_scaler(frame, available_before=self.dates[25], lag=2)
        before = self.samples(frame, scaler)
        altered = self.raw.copy()
        origin_index = 50
        altered.iloc[origin_index - 2 + 1:] += 500
        after = self.samples(self.calendar(altered), scaler)
        i = np.flatnonzero(before["origin_date"] == str(self.dates[origin_index].date()))[0]
        np.testing.assert_array_equal(before["x"][i], after["x"][i])
        self.assertEqual(before["persistence_usd"][i], after["persistence_usd"][i])
        self.assertNotEqual(before["y_usd"][i], after["y_usd"][i])

    def test_validation_can_use_history_before_its_block(self):
        frame = self.calendar()
        scaler = fit_scaler(frame, available_before=self.dates[25], lag=2)
        batch = self.samples(frame, scaler)
        self.assertEqual(batch["origin_date"][0], str(self.dates[25].date()))
        # Ventana: fechas 21..25, con observaciones disponibles 19..23.
        np.testing.assert_allclose(scaler.inverse_transform(batch["x"][0, :, 0]),
                                   self.raw.iloc[19:24], atol=1e-5)

    def test_empty_blocks_constant_prices_and_invalid_inputs(self):
        constant = pd.Series(50.0, index=self.dates)
        frame = self.calendar(constant)
        scaler = fit_scaler(frame, available_before=self.dates[25], lag=2)
        self.assertEqual(scaler.scale, 1.0)
        batch = make_samples(frame, scaler, block_start=self.dates[25], block_end=self.dates[27],
                             window=5, horizon=5, lag=2)
        self.assertEqual(batch["x"].shape, (0, 5, 1))
        self.assertEqual(batch["y"].shape, (0, 1))
        with self.assertRaises(ValueError):
            self.calendar(pd.concat([self.raw, self.raw.iloc[-1:]]))
        with self.assertRaises(ValueError):
            self.calendar(self.raw.iloc[::-1])
        bad = self.raw.copy()
        bad.iloc[10] = np.inf
        with self.assertRaises(ValueError):
            self.calendar(bad)

    def test_metrics_known_errors_and_rejection_of_missing_values(self):
        result = regression_metrics([1, 2, 4], [2, 2, 2])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mae_usd"], 1)
        self.assertAlmostEqual(result["mse_usd2"], 5 / 3)
        self.assertAlmostEqual(result["rmse_usd"], np.sqrt(5 / 3))
        for actual, predicted in [([], []), ([1], [1, 2]), ([1, np.nan], [1, 2])]:
            with self.assertRaises(ValueError):
                regression_metrics(actual, predicted)


if __name__ == "__main__":
    unittest.main()
