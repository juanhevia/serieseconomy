# Aplicación de predicción del Brent

La interfaz está construida en HTML, CSS y JavaScript (`web/`), con un servidor Python local (`oilprice/web.py`). No usa Streamlit ni necesita compilar un frontend. Abrir el HTML directamente no basta para ejecutar las predicciones: debe estar funcionando el servidor Python.

La aplicación utiliza los modelos entrenados en el laboratorio y funciona en CPU. No requiere volver a entrenar ni disponer de GPU. Incluye el histórico del experimento, las tres redes seleccionadas previamente en validación y los resultados de prueba. Los archivos necesarios están en `artifacts/brent_v1`, versionados junto al código; la carpeta completa `runs/laboratorio` se conserva como respaldo local.

## Abrir en el Mac Intel de este proyecto

El entorno `.venv-app` está preparado y separado de `.venv`. Desde la carpeta del proyecto:

```bash
.venv-app/bin/python app.py
```

Para preparar ese entorno desde cero con Python 3.12:

```bash
python3.12 -m venv .venv-app
.venv-app/bin/python -m pip install -r requirements-app-mac-intel.txt
.venv-app/bin/python app.py
```

PyTorch 2.2.2 es la última versión oficial para Mac Intel. El entrenamiento se realizó con PyTorch 2.10 y ROCm; se ha comprobado que los pesos seleccionados cargan y reproducen las predicciones en CPU con una tolerancia de 0,005 USD. La versión de NumPy está fijada para evitar incompatibilidades binarias con PyTorch. No se debe mezclar este entorno con el de preparación de datos.

## Abrir en Linux o en otro equipo con PyTorch instalado

Desde la raíz del repositorio y con el entorno activado:

```bash
python -m pip install -r requirements-app.txt
python app.py
```

En un equipo nuevo debe instalarse también PyTorch compatible con su Python y sistema. En el laboratorio puede reutilizarse la instalación ya comprobada. La aplicación siempre hace la inferencia en CPU.

El navegador muestra la aplicación en `http://127.0.0.1:8502`. El servidor solo escucha en este ordenador. Puede cambiarse el puerto con `python app.py --port 8503`. Para detenerla, pulsar Ctrl+C en la terminal.

## Uso

1. Seleccionar 5, 63 o 252 días lunes-viernes. Los festivos no se excluyen.
2. Revisar la fecha de emisión. Por defecto se propone la última fecha respaldada por los datos y el retraso supuesto, nunca una fecha futura. Si los datos son antiguos se indica expresamente; no se presenta como una predicción de hoy.
3. Leer el precio previsto, su variación frente al último precio disponible, la fecha objetivo y el rango orientativo.
4. Consultar «Resultados reales» para comparar con persistencia y ver precios reales frente a predichos. «Historia del Brent» muestra extremos y contexto económico con fuentes.

El gráfico de predicción muestra el histórico disponible y un punto futuro con su rango. No representa una trayectoria diaria prevista entre ambas fechas.

## Incorporar un CSV sin reentrenar

El archivo debe ser UTF-8, separado por comas, de hasta 5 MB y tener exactamente estas columnas:

```csv
Date,brent
```

Cada fila debe contener una fecha real `AAAA-MM-DD` y un precio numérico en USD/barril con punto decimal. Se admiten fechas de lunes a viernes, sin duplicados ni valores vacíos o infinitos. Se ordenan las fechas y se rechazan fechas futuras. Los precios extremos finitos no se borran automáticamente.

Puede cargarse un histórico completo o solamente observaciones nuevas. Las filas ya presentes con el mismo precio se ignoran; un precio distinto para una fecha ya observada se rechaza para impedir cambios silenciosos de fuente o revisiones. Se pueden completar huecos. La copia original del experimento no se modifica.

Pulsar «Incorporar datos» actualiza el histórico del servidor local (compartido entre las pestañas abiertas) y propone una nueva fecha de emisión. Pulsar «Guardar cambios» lo conserva en `data/inference/history.csv`, que se recupera al volver a abrir la aplicación. También puede descargarse un CSV y cargarlo en otro ordenador. Este histórico personal está excluido de Git: para trasladarlo hay que copiarlo o descargarlo. Si un fichero guardado resulta inválido, se muestra el error y se puede apartar ese fichero para volver a abrir con la copia original.

La ventana contiene 60 entradas. Se aplica el retraso supuesto de publicación de tres días lunes-viernes, con un máximo de cinco días consecutivos de relleno causal. Los huecos largos impiden predecir. Los datos posteriores a la fecha de emisión, y los que todavía no estarían disponibles bajo ese retraso, no se usan como entradas.

## Elección de modelos y rango de incertidumbre

Se conservan RNN a 5 días, LSTM a 63 y RNN a 252, elegidos en validación. No se cambia la selección tras observar prueba. Los tres empeoraron el RMSE de persistencia en prueba, por lo que la interfaz lo explica junto a la predicción.

El rango se construye sumando a la predicción los percentiles 10 y 90 de los residuos `real − predicho` de validación. Es un rango empírico orientativo del 80 % central, no un intervalo con garantía de cobertura. La validación se utilizó también para selección y parada temprana, los errores se solapan temporalmente y la distribución puede cambiar. Se muestra la cobertura observada en prueba, calculada con el rango ya fijado. No se recalibra usando prueba ni al subir un CSV. Los límites no se recortan artificialmente a cero.

Para una evaluación rigurosa de intervalos sería necesario reservar un bloque de calibración independiente dentro del desarrollo, antes de consultar prueba. Esta versión documenta su limitación; no afirma haber completado esa ampliación.

## Reproducibilidad

`scripts/export_app.py` exporta los tres modelos seleccionados, sus reglas de preparación, las referencias de validación/prueba y los cuantiles de los residuos. Verifica los archivos de entrenamiento y no sobrescribe un paquete existente:

```bash
python -m scripts.export_app --run runs/laboratorio --output artifacts/brent_otra_version
```

La aplicación comprueba las huellas de los archivos del paquete y de la copia fuente. Para usar otra versión debe cambiarse explícitamente la ruta del paquete en `oilprice/web.py`. La exportación no entrena. Las predicciones se descargan con fecha de emisión, fecha objetivo, modelo, último precio disponible y rango.

Las pruebas `test_inference.py` comprueban causalidad, huecos, validación de CSV, persistencia de datos, cobertura y equivalencia con predicciones del laboratorio. `test_app.py` comprueba el servidor HTTP, los tres horizontes, la carga de CSV, el rechazo de revisiones y la recuperación del histórico guardado. Las pruebas utilizan un histórico temporal y no entrenan los modelos:

```bash
.venv-app/bin/python -m unittest discover -s tests -p 'test_inference.py' -v
.venv-app/bin/python -m unittest discover -s tests -p 'test_app.py' -v
```
