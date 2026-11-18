"""Pruebas de la interfaz con modelos guardados; sin servidor ni entrenamiento."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ["streamlit", "torch"]):
    raise unittest.SkipTest("Instalar las dependencias de la aplicación para probar su interfaz.")

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_horizons_upload_errors_and_saved_history(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"OILPRICE_HISTORY_PATH": str(Path(folder) / "history.csv")}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 3)
            for horizon in [63, 252]:
                app.selectbox[0].set_value(horizon).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.metric), 3)
            app.file_uploader[0].upload("new.csv", b"Date,brent\n2026-11-06,110\n", "text/csv").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state["history"].loc["2026-11-06"], 110)
            app.button[1].click().run()
            path = Path(folder) / "history.csv"
            self.assertTrue(path.exists())
            app.file_uploader[0].upload("bad.csv", b"Date,brent\n2026-11-06,999\n", "text/csv").run()
            app.button[0].click().run()
            self.assertTrue(app.error)
            self.assertEqual(app.session_state["history"].loc["2026-11-06"], 110)
            restored = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
            self.assertFalse(restored.exception)
            self.assertEqual(restored.session_state["history"].loc["2026-11-06"], 110)


if __name__ == "__main__":
    unittest.main()
