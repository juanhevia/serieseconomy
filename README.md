# serieseconomy

Caso práctico de predicción del precio del petróleo Brent mediante redes RNN, LSTM y GRU.

## Estado

Preparación del entorno y análisis exploratorio completados. Los modelos, la comparativa y la aplicación todavía no están implementados.

## Análisis exploratorio

- [Cuaderno ejecutado](notebooks/01_exploracion.ipynb): código, tablas, gráficos e interpretación en español. GitHub permite leer sus resultados.
- [Informe HTML](reports/01_exploracion.html): descargar y abrir en el navegador para leerlo sin ejecutar Python.
- [Gráficos exportados](reports/figures): cobertura, histórico, variaciones, partición temporal y correlaciones.

El análisis conserva el CSV y comprueba fechas, valores faltantes y extremos. Propone bloques cronológicos 70/15/15 y muestra cómo evitar que los objetivos futuros crucen sus límites. Las correlaciones se calculan solo en entrenamiento. La copia descargada ya contiene relleno hacia delante: antes de entrenar hay que recuperar observaciones originales y definir el calendario y los retrasos de publicación.

## Materiales de la asignatura

- [Enunciado](docs/assignment/ENUNCIADO.pdf).
- [Repositorio de origen](https://github.com/UAX-VCMT-26-27/aa-case-02-oil-price-prediction).
- `scripts/download_data.py`: descarga de datos de FRED y Yahoo Finance.
- `pyproject.toml` y `poetry.lock`: dependencias iniciales de la asignatura y grupo `analysis` añadido para ejecutar el cuaderno y sus gráficos.

## Preparación del entorno

Requiere Python 3.12 o superior. Desde la carpeta del proyecto, en macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install poetry
poetry install
```

El entorno `.venv` está excluido de Git. Si ya está creado, basta con activarlo.

## Ejecutar el análisis

Con el entorno activado:

```bash
poetry run python scripts/run_analysis.py
```

El comando ejecuta todas las celdas con el Python del proyecto, actualiza sus resultados y genera el HTML y los cinco gráficos. No necesita conexión a Internet ni modifica los datos. Para editar el cuaderno en VS Code, abrir `notebooks/01_exploracion.ipynb` y seleccionar `.venv/bin/python` como entorno del kernel.

El cuaderno muestra la huella SHA-256 del CSV para identificar exactamente qué copia se analizó. Tras descargar nuevos datos, volver a ejecutar el análisis. Para experimentos comparables hay que conservar la copia y congelar las fechas de separación.

## Descargar los datos

Con el entorno activado y conexión a Internet:

```bash
poetry run python scripts/download_data.py
```

El resultado se guarda en `data/petroleo_dataset.csv`. Estas fuentes no necesitan el token de GitHub. El script puede continuar solo con FRED si Yahoo Finance falla; hay que revisar las columnas y los valores faltantes antes de entrenar.

## Trabajo pendiente

1. Recuperar observaciones sin rellenar y preparar ventanas, calendario y variables disponibles sin utilizar información futura.
2. Comparar RNN, LSTM y GRU con un modelo ingenuo, usando validación temporal.
3. Evaluar horizontes de 5, 63 y 252 días hábiles y cuantificar la incertidumbre.
4. Integrar el modelo elegido en una aplicación Python y documentar su ejecución.
5. Preparar la presentación exigida en el enunciado.

No guardar tokens ni otras credenciales en el código o en este repositorio.
