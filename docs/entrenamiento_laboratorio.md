# Entrenar las redes en el laboratorio

El programa está preparado para ejecutar **RNN, LSTM y GRU** con los datos del Brent. El entrenamiento real queda para el laboratorio. Las pruebas del código usan únicamente datos artificiales.

## 1. Obtener el proyecto

En una terminal del ordenador del laboratorio:

```bash
git clone https://github.com/juanhevia/serieseconomy.git
cd serieseconomy
```

Si ya tienes una copia, entra en ella y ejecuta `git pull --ff-only`. Todos los comandos siguientes se ejecutan desde esta carpeta, donde está `pyproject.toml`.

## 2. Preparar Python y PyTorch

Usa **Python 3.12**. Si el profesor proporciona un entorno con GPU configurada, utiliza sus indicaciones para activar ese entorno e instalar las dependencias. No hace falta reinstalar el controlador de la GPU desde este proyecto.

Para crear un entorno propio en Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install poetry
poetry install
python -m pip install -r requirements-training.txt
```

En Windows PowerShell, los dos primeros comandos son:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Después se usan los mismos comandos de instalación con `python`, `pip` y `poetry`.

**La instalación de PyTorch debe ser compatible con la GPU y el controlador del laboratorio.** El archivo `requirements-training.txt` solicita PyTorch 2.6 o superior de la serie 2.x; no fija una variante de CUDA porque todavía no conocemos el equipo. Para NVIDIA, comprueba la instalación con el paso 3. Si no detecta CUDA, usa el comando correspondiente al equipo en el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/), o la instalación indicada por el profesor. No continúes con el entrenamiento hasta que la comprobación termine correctamente.

PyTorch se instala aparte del archivo `poetry.lock` para poder elegir la variante de GPU. El programa registra las versiones efectivas de Python, NumPy, PyTorch, CUDA y cuDNN. `poetry install` conserva esta instalación adicional; no uses `poetry sync` después de añadir PyTorch, porque podría eliminarlo del entorno.

## 3. Comprobar la GPU sin entrenar con datos reales

```bash
python -m scripts.train_models --device cuda --check
```

Debe mostrar `"device": "cuda:0"`, el nombre de la GPU y un mensaje de comprobación completada. Solo ejecuta operaciones de las tres redes con tensores artificiales. Si falla, el entrenamiento no cambia automáticamente a CPU.

## 4. Generar las ventanas y revisar el plan

```bash
python -m scripts.prepare_data
python -m scripts.train_models --device cuda --dry-run
```

Las ventanas se regeneran desde la copia de FRED incluida en el repositorio; no necesitan Internet. El segundo comando comprueba los archivos, sus huellas, dimensiones y escaladores, sin ajustar modelos ni abrir prueba.

Con la configuración actual mostrará **63 ajustes** como máximo:

- 3 arquitecturas × 3 horizontes × 2 candidatos × 3 folds = 54 ajustes para elegir hiperparámetros.
- 3 arquitecturas × 3 horizontes = 9 ajustes sobre el entrenamiento principal, con el candidato seleccionado para cada combinación.

Cada ajuste tiene hasta 40 épocas, con parada temprana. No se ha medido todavía el tiempo en la GPU del laboratorio; no se presupone que termine en una clase.

## 5. Iniciar el entrenamiento

```bash
python -m scripts.train_models --device cuda --output runs/laboratorio
```

La terminal mostrará la red, horizonte, candidato, fold, época y error de validación escalado. Por ejemplo, `lstm_h63_c01_fold_02` identifica LSTM, horizonte de 63 días, segundo candidato y segundo fold.

No ejecutes dos procesos simultáneos sobre la misma carpeta de salida. Para pausar, pulsa **Ctrl+C**. Para continuar, vuelve a ejecutar exactamente el mismo comando.

## 6. Si se interrumpe

El programa guarda cada ajuste completado por separado. Al volver a ejecutar:

- Verifica que los datos, el código, la configuración y el entorno coinciden con la ejecución original.
- Reutiliza las tareas terminadas, tras comprobar sus archivos.
- Repite desde la semilla inicial la tarea que estuviera a medias. **No continúa desde la época exacta interrumpida.**

Si cambias de GPU, versión de PyTorch, datos o configuración, utiliza una carpeta nueva, por ejemplo `--output runs/laboratorio_v2`. Así no se mezclan experimentos ni tiempos obtenidos en distintos entornos.

**Antes de salir del laboratorio, copia toda la carpeta `runs/laboratorio` a tu almacenamiento personal**, incluso si el entrenamiento no ha terminado. Está excluida de Git y no se sube automáticamente a GitHub. Guarda también la versión del código utilizada. Para reanudar en otro equipo hará falta un entorno coincidente; en caso contrario, inicia otro experimento.

## 7. Leer los resultados

Al terminar aparecerán en `runs/laboratorio/`:

| Archivo o carpeta | Contenido |
|---|---|
| `comparison_validation.md` | Tabla legible con MAE, MSE, RMSE, mejora frente a persistencia, parámetros y tiempos |
| `comparison_validation.csv` | La misma comparación para analizar o graficar |
| `search_results.csv` | Resultados de los candidatos en los folds temporales |
| `selection_validation.json` | Mejor red provisional por horizonte y si supera a persistencia |
| `signature.json` | Configuración, versiones, equipo y huellas del experimento |
| `completed.json` | Confirmación de que terminó el conjunto de ajustes |
| `jobs/` | Pesos, metadatos, historial y predicciones de cada ajuste |

Cada tarea contiene `best.pt` (pesos de la mejor época), `metadata.json` (arquitectura, escalador y preparación), `history.csv`, `validation_predictions.npz` y `result.json`. Estos archivos se guardan también durante una ejecución todavía incompleta.

Una mejora de RMSE positiva significa que la red mejora la referencia ingenua; una negativa significa que empeora. Si ninguna red mejora persistencia, el informe lo refleja. No se inventa una ventaja de las redes.

## Cómo se realiza la comparación

- Las tres redes usan una única capa recurrente y una salida lineal, sin bidireccionalidad.
- Reciben los mismos datos, ventanas, horizontes y escaladores. No se mezclan filas de entrenamiento, validación o prueba.
- Los lotes de entrenamiento se recorren en orden, con `shuffle=False`.
- Cada red prueba los mismos candidatos: 16 o 32 unidades ocultas y tasa de aprendizaje 0,001, usando Adam, lotes de 128 y recorte de gradiente a norma 1.
- Cada fold utiliza su escalador ya ajustado solo con entrenamiento. Se inicia una red nueva para cada ajuste.
- Se seleccionan los hiperparámetros con el RMSE conjunto de los folds, ponderando por su número de muestras y calculado en USD/barril.
- La parada temprana usa la validación del fold correspondiente. En el ajuste principal usa la validación principal. Se guardan los pesos con el menor MSE de validación.
- Se fija la semilla 42 y se solicitan operaciones deterministas. No se garantiza igualdad numérica entre hardware o versiones diferentes.
- El mismo presupuesto máximo de épocas no implica idéntico número de parámetros ni duración. Ambos se registran.

La configuración está en `configs/training.json`. Si la cambias, conserva la modificación y utiliza una carpeta de resultados nueva. Para estudiar estabilidad se podrán repetir los experimentos con otras semillas; esta primera configuración utiliza una sola.

## Qué queda fuera de esta ejecución

**El programa no abre ni evalúa los archivos de prueba.** La validación se usa para elegir hiperparámetros, época y red, por lo que sus errores son cifras de desarrollo y selección. La evaluación independiente en prueba se hará después de fijar el procedimiento.

Esta versión sigue siendo univariante y conserva los supuestos de la [preparación](preparacion_datos.md), incluido el retraso de publicación supuesto de tres días lunes-viernes. No utiliza aún variables externas, no calcula intervalos de incertidumbre y no crea la aplicación final.

## Pruebas del código

Con PyTorch instalado:

```bash
python -m unittest discover -s tests -v
```

Las pruebas de entrenamiento usan muestras artificiales pequeñas. Comprueban las tres arquitecturas, los pesos guardados, la parada temprana, la selección, la reanudación, el rechazo de archivos alterados y la ausencia de acceso al bloque de prueba. Sin PyTorch, esas pruebas opcionales se omiten y siguen disponibles las de preparación.

En el Mac Intel se comprueba la lógica en un entorno temporal con PyTorch 2.2.2 y NumPy 1.26.4, separado del entorno de preparación. PyTorch dejó de publicar paquetes oficiales para Mac Intel a partir de 2.3; véase el [anuncio oficial](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690). No instales esa combinación antigua encima del entorno de preparación. La ejecución CUDA se verificará en el laboratorio con el paso 3.
