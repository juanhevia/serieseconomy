# Preparación para entrenar RNN, LSTM y GRU

Esta versión prepara **Brent univariante**: cada red recibirá 60 precios históricos como entrada y predecirá un único precio al final de un horizonte de 5, 63 o 252 días. Se conserva el dataset multivariante de la exploración para una ampliación posterior; todavía no se utiliza como entrada del modelo.

## Reproducir en tu ordenador o en el laboratorio

Desde la raíz del repositorio, con Python 3.12 o superior y las dependencias instaladas:

```bash
poetry install
poetry run python -m scripts.prepare_data
poetry run python -m scripts.evaluate_baseline
poetry run python -m unittest discover -s tests -v
```

Estos comandos funcionan sin conexión: la copia de FRED sin rellenar está incluida en `data/raw/`. Las ventanas se generan en `data/processed/brent_univariate_v1/`, excluido de Git. Los informes se guardan en `reports/`.

Si falta la copia sin rellenar:

```bash
poetry run python scripts/download_raw_brent.py
```

La descarga sí necesita Internet. Si la copia ya existe y coincide con su huella y periodo, el comando la reutiliza. No sobrescribe una copia existente. Para otro periodo, crear otra configuración con un nuevo `experiment`, `as_of` y `raw_path`, y pasarla mediante `--config` a descarga y preparación. Pasar el nuevo directorio a `evaluate_baseline` con `--prepared`.

`scripts/download_data.py` sigue siendo el script original de la asignatura para el CSV multivariante. La nueva preparación **no usa ese CSV rellenado**, y ninguno de los comandos anteriores lo modifica.

## Fechas y calendario

La configuración está en `configs/preparation.json`. Se mantienen los cortes propuestos en la exploración:

| Bloque | Inicio incluido | Fin excluido |
|---|---|---|
| Entrenamiento | 01/01/2000 | 18/10/2018 |
| Validación | 18/10/2018 | 27/10/2022 |
| Prueba reservada | 27/10/2022 | 06/11/2026 |

El corte de la copia es el 05/11/2026, para mantener el periodo de la exploración. Los porcentajes exactos cambian al excluir objetivos faltantes y aplicar ventanas, horizontes y retrasos.

Se usa un calendario de **lunes a viernes que incluye festivos**. De esta forma el horizonte de 5 pasos siempre avanza cinco días de lunes a viernes, sin comprimir el tiempo al eliminar festivos. Si no hay un precio observado en la fecha objetivo, se descarta esa muestra. No se interpreta el horizonte como número de sesiones efectivas de un mercado.

## Huecos y precios extremos

- El precio objetivo es siempre una observación de FRED. No se rellena.
- Las entradas se rellenan hacia delante como máximo cinco filas consecutivas, usando únicamente el precio anterior.
- No se rellenan huecos iniciales hacia atrás, no se interpolan con datos futuros y no se convierten nulos en ceros.
- Si una ventana sigue teniendo algún nulo, se descarta completa.
- Los precios extremos se conservan. No se recortan con umbrales calculados sobre todo el histórico.
- `calendar_audit.csv` permite revisar el precio observado, la entrada disponible, su fecha de origen y si procede de relleno.

## Disponibilidad de la información

La fecha de una observación no equivale necesariamente a su fecha de publicación. FRED publica por separado la fecha del dato y la actualización de la serie y advierte que los datos pueden revisarse. Fuente: [Brent de EIA vía FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU).

Para este experimento se **supone** que cada precio se conoce tres días lunes-viernes después de su fecha de observación. El parámetro `publication_lag_bdays` permite cambiar el supuesto. No se ha comprobado que ese retraso cubra todas las publicaciones históricas.

En un origen `t`, la última entrada puede proceder como máximo de `t - 3 días lunes-viernes`. Si hubo un hueco, puede ser anterior. La referencia ingenua utiliza ese mismo último precio disponible, para comparar en igualdad de información.

Además, una etiqueta de entrenamiento debe estar disponible antes de que empiece validación. Una etiqueta de validación debe estar disponible antes de prueba. Así se excluyen también los objetivos que, aunque pertenecieran al bloque por su fecha, se conocerían después de su cierre.

**Límite:** se usa la versión histórica descargada actualmente, no las versiones disponibles en cada fecha pasada. El retraso fijo no elimina el posible efecto de revisiones posteriores. Es una evaluación retrospectiva bajo un supuesto explícito; para simular decisiones en tiempo real habría que disponer de publicaciones y versiones históricas verificadas.

## Escalado y validación temporal

Se calcula `(precio - media) / desviación estándar`, usando la desviación poblacional. La media y la desviación solo se ajustan con observaciones de entrenamiento disponibles antes del siguiente bloque, excluyendo nulos y sin duplicar precios por relleno. El mismo escalador transforma `X` e `y`. No se recortan valores extremos ni valores de validación fuera del rango de entrenamiento.

Hay tres folds con entrenamiento creciente, dentro del bloque de entrenamiento principal:

| Fold | Entrenamiento desde | Validación desde | Validación hasta, excluido |
|---|---|---|---|
| 1 | 01/01/2000 | 02/01/2012 | 01/01/2014 |
| 2 | 01/01/2000 | 01/01/2014 | 01/01/2016 |
| 3 | 01/01/2000 | 01/01/2016 | 18/10/2018 |

Cada fold ajusta su propio escalador. Los orígenes finales de cada bloque se excluyen cuando el objetivo, más su retraso supuesto, cruza el corte. Las ventanas de validación pueden usar historia anterior al bloque: esa historia ya estaría disponible al predecir.

Estos folds sirven para elegir hiperparámetros. La validación principal permite comparar las configuraciones seleccionadas. La prueba permanece reservada para la evaluación final. No se han entrenado redes todavía.

## Archivos que utilizarán las redes

`manifest.json` registra configuración, huellas, escaladores y número de muestras. Los archivos `.npz` contienen:

| Clave | Contenido |
|---|---|
| `x` | Entradas escaladas `float32`, forma `(N, 60, 1)` |
| `y` | Objetivo escalado `float32`, forma `(N, 1)` |
| `y_usd` | Precio objetivo original, forma `(N,)` |
| `persistence_usd` | Último precio disponible en el origen, predicción ingenua |
| `origin_date` | Fecha en la que se emite la predicción |
| `target_date` | Fecha exacta del precio futuro objetivo |
| `last_observation_date` | Fecha del último precio conocido en el origen |
| `input_imputed_fraction` | Fracción de la ventana que procede de relleno |

Ejemplo desde la raíz del proyecto:

```python
import json
from pathlib import Path
import numpy as np

folder = Path("data/processed/brent_univariate_v1")
manifest = json.loads((folder / "manifest.json").read_text())
with np.load(folder / "main_train_h5.npz", allow_pickle=False) as data:
    x_train, y_train = data["x"], data["y"]
scaler = manifest["scalers"]["main"]
# Una vez entrenada la red: predicciones_usd = predicciones_escaladas * scaler["scale"] + scaler["mean"]
print(x_train.shape, y_train.shape)
```

`main_train_h5.npz` corresponde al entrenamiento principal para 5 días. Los nombres `fold_01_train_h5.npz` y `fold_01_validation_h5.npz` corresponden al primer fold. Cada grupo tiene su escalador en el manifiesto.

Las tres redes y la referencia ingenua deben utilizar los mismos archivos para cada horizonte. Los orígenes disponibles difieren entre horizontes; no se deben comparar sus errores como si la muestra fuera idéntica.

## Comprobaciones

Las pruebas comprueban que modificar precios futuros no altera ni las ventanas de entrenamiento ni su escalador, que una entrada no usa observaciones posteriores a su disponibilidad supuesta, que los objetivos no son imputados, que los horizontes respetan el calendario y que el relleno se detiene tras el límite configurado.

El evaluador de persistencia solo abre archivos cuyo bloque es `validation`, incluidos los folds. No abre ni evalúa los archivos `test`. Sus métricas se expresan en unidades originales: MAE y RMSE en USD/barril, MSE en (USD/barril)².
