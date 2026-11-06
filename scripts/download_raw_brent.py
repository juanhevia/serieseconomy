"""Guarda una copia de FRED sin rellenar ni modificar el CSV exploratorio."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preparation.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["raw_path"]
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256(output.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("La copia existente no coincide con su huella; no se sobrescribe.")
        if (metadata["requested_start"] != config["start"] or metadata["requested_end"] != config["as_of"]
                or metadata["series"] != "DCOILBRENTEU"):
            raise ValueError("La copia existente no corresponde al periodo o serie solicitados.")
        print(f"Copia existente verificada: {output.relative_to(ROOT)}")
        return

    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
           f"&cosd={config['start']}&coed={config['as_of']}")
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "serieseconomy/0.1"}), timeout=45) as response:
                body = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    frame = pd.read_csv(BytesIO(body), na_values=".")
    if len(frame.columns) != 2 or "DCOILBRENTEU" not in frame:
        raise ValueError("Respuesta inesperada de FRED.")
    frame.columns = ["Date", "brent"]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["brent"] = pd.to_numeric(frame["brent"], errors="raise")
    if frame.empty or not frame.brent.notna().any() or frame.Date.duplicated().any() or not frame.Date.is_monotonic_increasing:
        raise ValueError("Serie vacía, duplicada o desordenada.")
    if not frame.Date.between(config["start"], config["as_of"]).all():
        raise ValueError("FRED devolvió fechas fuera del periodo solicitado.")
    payload = frame.to_csv(index=False, date_format="%Y-%m-%d").encode()
    metadata = {
        "series": "DCOILBRENTEU",
        "source": "U.S. Energy Information Administration via FRED",
        "url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": config["start"],
        "requested_end": config["as_of"],
        "sha256": sha256(payload).hexdigest(),
        "response_sha256": sha256(body).hexdigest(),
        "rows": len(frame),
        "observed_prices": int(frame.brent.notna().sum()),
        "transformation": "Columnas renombradas; nulos conservados; sin relleno ni recorte de extremos.",
        "vintage_warning": "Histórico descargado hoy, potencialmente revisado. No es una reconstrucción de lo publicado en cada fecha pasada."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Guardadas {len(frame)} filas, {metadata['observed_prices']} precios observados: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
