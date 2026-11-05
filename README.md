# serieseconomy

Caso práctico de predicción del precio del petróleo Brent mediante redes RNN, LSTM y GRU.

## Estado

Preparación inicial del proyecto. Los modelos, la comparativa y la aplicación todavía no están implementados.

## Materiales de la asignatura

- [Enunciado](docs/assignment/ENUNCIADO.pdf).
- [Repositorio de origen](https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction).
- `scripts/download_data.py`: descarga de datos de FRED y Yahoo Finance.
- `pyproject.toml` y `poetry.lock`: dependencias proporcionadas por la asignatura.

## Preparación del entorno

Requiere Python 3.12 o superior. Desde la carpeta del proyecto, en macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install poetry
poetry install
```

El entorno `.venv` está excluido de Git. Si ya está creado, basta con activarlo.

## Descargar los datos

Con el entorno activado y conexión a Internet:

```bash
poetry run python scripts/download_data.py
```

El resultado se guarda en `data/petroleo_dataset.csv`. Estas fuentes no necesitan el token de GitHub. El script puede continuar solo con FRED si Yahoo Finance falla; hay que revisar las columnas y los valores faltantes antes de entrenar.

## Trabajo pendiente

1. Explorar y preparar los datos sin utilizar información futura.
2. Comparar RNN, LSTM y GRU con un modelo ingenuo, usando validación temporal.
3. Evaluar horizontes de 5, 63 y 252 días hábiles y cuantificar la incertidumbre.
4. Integrar el modelo elegido en una aplicación Python y documentar su ejecución.
5. Preparar la presentación exigida en el enunciado.

No guardar tokens ni otras credenciales en el código o en este repositorio.
