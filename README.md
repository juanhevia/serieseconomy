# serieseconomy

Caso práctico de predicción del precio del petróleo Brent mediante redes RNN, LSTM y GRU.

## Estado

La preparación univariante, el entrenamiento real de RNN/LSTM/GRU y la evaluación final están completados. Se ejecutaron los 63 ajustes previstos en el laboratorio. La aplicación permite predecir, mostrar un rango orientativo y actualizar/guardar datos mediante CSV sin reentrenar.

**Resultado:** los modelos elegidos en validación no mejoraron el RMSE de persistencia en prueba. La aplicación conserva esa selección y muestra esta limitación. La versión multivariante sigue pendiente; esta limitación se documenta en la presentación.

## Entrega y acceso público

- [Aplicación publicada](https://huggingface.co/spaces/juanhevia/brent-observatorio).
- [Modelos en Hugging Face](https://huggingface.co/juanhevia/brent-rnn-models): nueve redes, metadatos y resultados.
- [Presentación PDF](output/pdf/01_presentacion_brent.pdf): portada y cinco diapositivas de contenido.
- [Enlaces del proyecto PDF](output/pdf/02_enlaces_proyecto.pdf).

La demo pública ejecuta los pesos exportados en el navegador; no requiere una instalación local. Los CSV se procesan y guardan en ese navegador, sin enviarse al servidor. La ejecución original con PyTorch sigue disponible en Python. La exportación se ha comparado con las 2.737 predicciones de prueba de las redes seleccionadas: diferencia máxima de 0,000015 USD (tolerancia de 0,005 USD).

## Abrir la aplicación

Los modelos seleccionados y las tablas de resultados están incluidos en `artifacts/brent_v1`; no necesitas entrenar para abrirla.

En un entorno con PyTorch instalado:

```bash
python -m pip install -r requirements-app.txt
python app.py
```

En el Mac Intel de este proyecto ya está preparado el entorno independiente:

```bash
.venv-app/bin/python app.py
```

Abre `http://127.0.0.1:8502`. La interfaz está construida en HTML, CSS y JavaScript; Python ejecuta los modelos guardados mediante una API local.

La [guía de la aplicación](docs/aplicacion.md) explica la instalación desde cero, el formato del CSV, cómo guardar nuevos datos y las limitaciones del rango. La aplicación funciona en CPU y sin conexión después de instalar las dependencias.

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

Después de terminar y fijar el procedimiento, la evaluación final se ejecuta por separado:

```bash
python -m scripts.evaluate_test --run runs/laboratorio --device cuda
```

Genera R², MAE, MSE y RMSE, compara con persistencia y dibuja valores reales frente a predichos y errores por fecha. También guarda métricas por año y los mayores errores para analizar cuándo falla. Utiliza los pesos y escaladores guardados, sin reentrenar. La evaluación real de esta versión está completada; sus tablas están incluidas en el paquete de la aplicación. Consulta la guía antes de abrir prueba: ese bloque no debe usarse para elegir hiperparámetros.

La aplicación incorpora observaciones mediante CSV manual y predice con los pesos y el escalador guardados. El histórico actualizado se guarda por separado de la copia del experimento.

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

1. Ampliar la preparación con variables externas y disponibilidad documentada, comparando con la versión univariante mediante un protocolo que reconozca que esta prueba ya se ha consultado.
2. Si se amplía la evaluación de incertidumbre, reservar una calibración independiente: el rango actual es empírico y orientativo, sin garantía de cobertura.
3. Actualizar las cinco diapositivas y los enlaces si se amplía el experimento o cambia el despliegue.

No guardar tokens ni otras credenciales en el código o en este repositorio.

### Actualización de la entrega (20/11/2026)

La presentación sigue la plantilla UAX: `output/pptx/Presentacion_Brent_UAX_entrega_v2.pptx` y `output/pdf/01_presentacion_brent.pdf` (portada + cinco diapositivas).

Para reproducir la demostración del 20/11 en la aplicación pública, cargar `data/updates/brent_2026-11-20.csv`, que añade diez observaciones oficiales hasta el 17/11; seleccionar la fecha 20/11/2026. La procedencia y huella están en el JSON adjunto. Los pesos, escaladores y evaluación original no cambian. Resultados: `reports/demo_app_2026-11-20.json`.

El contexto económico de 2026 se documenta con EIA, Short-Term Energy Outlook de noviembre: https://www.eia.gov/outlooks/steo/archives/Nov26.pdf . La actualización de ese texto está incluida en el código local; requiere republicar el Space para aparecer en su versión alojada.

Sigue pendiente incorporar las variables externas al entrenamiento y obtener una nueva evaluación independiente. No se afirma cumplimiento completo.
