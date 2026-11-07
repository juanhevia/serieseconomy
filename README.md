# serieseconomy

Caso práctico de predicción del precio del petróleo Brent mediante redes RNN, LSTM y GRU.

## Estado

Preparación del entorno, análisis exploratorio y preparación univariante del Brent completados. La referencia ingenua está evaluada en validación y en tres folds temporales. El código de RNN, LSTM y GRU está preparado para entrenar en el laboratorio y se comprueba con datos sintéticos. **Las redes todavía no están entrenadas con los datos reales**, la prueba final sigue reservada y la aplicación está pendiente.

## Entrenamiento en el laboratorio

La [guía paso a paso](docs/entrenamiento_laboratorio.md) explica la instalación de PyTorch con GPU, las comprobaciones y cómo guardar o reanudar el trabajo.

Con el entorno configurado y la GPU disponible:

```bash
python -m scripts.train_models --device cuda --check
python -m scripts.prepare_data
python -m scripts.train_models --device cuda --dry-run
python -m scripts.train_models --device cuda --output runs/laboratorio
```

El último comando realiza la búsqueda temporal y los ajustes principales de las tres redes. Guarda pesos y resultados en `runs/laboratorio/` y no abre prueba. Si se interrumpe, repetir el mismo comando reutiliza los ajustes terminados y repite el ajuste incompleto. **La carpeta de resultados está excluida de Git: copiarla al almacenamiento personal antes de dejar el laboratorio.**

## Datos listos para entrenar

La primera versión utiliza el histórico del Brent como única entrada. Genera ventanas de 60 filas, objetivos a 5, 63 y 252 días lunes-viernes y escaladores ajustados solo en entrenamiento. Las fechas sin precio objetivo observado se excluyen.

```bash
poetry run python -m scripts.prepare_data
poetry run python -m scripts.evaluate_baseline
poetry run python -m unittest discover -s tests -v
```

Estos comandos utilizan la copia sin rellenar incluida en `data/raw/` y funcionan sin conexión después de instalar las dependencias. Las ventanas de `data/processed/` se regeneran y no se suben a GitHub.

- [Guía de preparación y uso de las ventanas](docs/preparacion_datos.md).
- [Informe de preparación y tamaños de los bloques](reports/preparacion.md).
- [Resultados de la referencia ingenua en validación](reports/baseline_validation.md).

Se supone un retraso de publicación de tres días lunes-viernes, configurable. No representa fechas de publicación históricas verificadas ni corrige revisiones posteriores de los datos. La extensión multivariante está pendiente.

## Análisis exploratorio

- [Cuaderno ejecutado](notebooks/01_exploracion.ipynb): código, tablas, gráficos e interpretación en español. GitHub permite leer sus resultados.
- [Informe HTML](reports/01_exploracion.html): descargar y abrir en el navegador para leerlo sin ejecutar Python.
- [Gráficos exportados](reports/figures): cobertura, histórico, variaciones, partición temporal y correlaciones.

El análisis conserva el CSV y comprueba fechas, valores faltantes y extremos. Propone bloques cronológicos 70/15/15 y muestra cómo evitar que los objetivos futuros crucen sus límites. Las correlaciones se calculan solo en entrenamiento. La preparación posterior recupera una copia independiente del Brent sin rellenar y conserva los cortes por fecha de esta exploración.

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

1. Ejecutar en el laboratorio el entrenamiento real de RNN, LSTM y GRU y revisar su comparación con la referencia ingenua.
2. Ampliar la preparación con variables externas y su disponibilidad documentada, comparando con la versión univariante.
3. Evaluar horizontes de 5, 63 y 252 días hábiles y cuantificar la incertidumbre.
4. Integrar el modelo elegido en una aplicación Python y documentar su ejecución.
5. Preparar la presentación exigida en el enunciado.

No guardar tokens ni otras credenciales en el código o en este repositorio.
