"""Exporta pesos existentes para Hugging Face Models y una demo HTML sin servidor."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import torch
from oilprice.web import Application

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/huggingface'
SPACE=OUT/'space';MODEL=OUT/'model'
SPACE.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
app=Application(history_path=OUT/'unused.csv')
bundle={'history':app.state()['history'],'models':{},'source_sha256':app.bundle['models']['5']['metadata']['raw_sha256'],'evaluation':{str(h):app.evaluation(h) for h in [5,63,252]},'metrics_csv':(app.folder/'metrics_test.csv').read_text()}
for h,m in app.bundle['models'].items():
    weights=torch.load(app.folder/m['weights'],map_location='cpu',weights_only=True)
    bundle['models'][h]={**m,'tensors':{k:v.tolist() for k,v in weights.items()}}
(SPACE/'bundle.json').write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False))
for name in ['app.js','styles.css','favicon.svg','browser_engine.mjs','static_bootstrap.mjs']:
    shutil.copy2(ROOT/'web'/name,SPACE/name)
html=(ROOT/'web/index.html').read_text().replace('<script src="/app.js" defer></script>','<script type="module" src="./static_bootstrap.mjs"></script>').replace('href="/styles.css"','href="./styles.css"').replace('href="/favicon.svg"','href="./favicon.svg"').replace('href="/"','href="./"').replace('Inferencia local','Inferencia en navegador').replace('Aplicación local','Demo pública').replace('Histórico conservado · sin cambios pendientes','Histórico conservado · sin cambios pendientes')
html=html.replace('USD por barril · Demo pública','USD por barril · Demo pública').replace('Los precios existentes no se sustituyen.','Los CSV se procesan en este navegador y no se envían a un servidor. Guardar conserva los datos en este navegador. Los precios existentes no se sustituyen.')
(SPACE/'index.html').write_text(html)
(SPACE/'README.md').write_text('''---
title: Brent Series Economy
emoji: 📈
colorFrom: green
colorTo: purple
sdk: static
app_file: index.html
---
# Brent · Series Economy

Aplicación académica de Juan Hevia Losa. HTML/CSS/JavaScript con los pesos de RNN (5 días), LSTM (63) y RNN (252) entrenados en PyTorch. Inferencia en el navegador; no se reentrena. Los CSV permanecen en el navegador. Guardar utiliza almacenamiento local del navegador; puede descargarse el histórico como CSV.

Código y ejecución original en Python: https://github.com/juanhevia/serieseconomy

Modelos: https://huggingface.co/juanhevia/brent-rnn-models

Los modelos elegidos en validación no superaron a persistencia en prueba. Rangos empíricos orientativos sin garantía de cobertura. Modelo univariante; variables externas pendientes. Copia histórica hasta el 03/11/2026: cargar observaciones recientes para actualizar. No representa asesoramiento de inversión.
''')
rows=list(csv.DictReader((app.folder/'metrics_validation.csv').open()))
for r in rows:
    if not r['job_id']:continue
    source=ROOT/'runs/laboratorio/jobs'/r['job_id'];stem=r['architecture'].lower()+'_h'+r['horizon_bdays']
    shutil.copy2(source/'best.pt',MODEL/(stem+'.pt'));shutil.copy2(source/'metadata.json',MODEL/(stem+'.json'))
for name in ['metrics_test.csv','metrics_validation.csv','manifest.json']:
    shutil.copy2(app.folder/name,MODEL/name)
shutil.copy2(ROOT/'oilprice/models.py',MODEL/'models.py')
(MODEL/'README.md').write_text('''---
language:
- es
library_name: pytorch
tags:
- time-series
- rnn
- lstm
- gru
- oil-price
- brent
---
# Brent: RNN, LSTM y GRU

Autor: Juan Hevia Losa. Caso práctico individual de aprendizaje automático.

Nueve redes entrenadas: RNN, LSTM y GRU a 5, 63 y 252 días lunes-viernes. Cada archivo `arquitectura_hN.pt` contiene un `state_dict` de PyTorch y su JSON asociado conserva arquitectura, escalador, preparación y metadatos. `models.py` permite reconstruir cada red. No son modelos Transformers ni requieren la API de inferencia de Hugging Face.

## Reproducción

```python
import json
import torch
from models import load_regressor
metadata = json.load(open("rnn_h5.json"))
model = load_regressor("rnn_h5.pt", metadata, "cpu")
# x: tensor [lote, 60, 1], preparado y escalado según metadata.
# with torch.inference_mode(): scaled_prediction = model(x)
```

Para ejecutar la inferencia completa, clonar https://github.com/juanhevia/serieseconomy y seguir su README. Ese repositorio incluye el histórico de referencia y el paquete verificable de la aplicación. Este repositorio de modelos conserva también GRU y las redes no seleccionadas para permitir la comparación. La selección de la app fue RNN a 5 días, LSTM a 63 y RNN a 252, fijada antes de consultar prueba.

## Datos y entrenamiento

Brent spot, EIA vía FRED (DCOILBRENTEU), copia de 2000 a noviembre de 2026. 6.501 precios observados. Versión univariante. Ventana de 60 entradas, retraso de publicación supuesto de tres días, relleno causal de hasta cinco días y escalado solo con entrenamiento. Tres folds temporales, dos tamaños ocultos (16/32), hasta 40 épocas y una semilla (42). 54 ajustes de búsqueda y 9 principales. PyTorch 2.10.0 con ROCm 7.1 en el laboratorio. Inferencia CPU verificada también con PyTorch 2.2.2.

## Evaluación y límites

`metrics_test.csv` y `metrics_validation.csv` incluyen las tres redes y persistencia para cada horizonte. Los modelos seleccionados empeoran el RMSE de persistencia en prueba un 3,33 %, 2,04 % y 2,42 %. GRU presenta menor RMSE entre redes en prueba, pero no se reelige usando ese bloque. A 252 días las predicciones son casi constantes. El efecto de cambios abruptos no se anticipa adecuadamente.

Los rangos de la app se basan en los percentiles 10/90 de residuos de validación, bloque también usado para selección y parada temprana: no garantizan cobertura futura. No hay prueba de rentabilidad ni significación estadística. Las variables externas exigidas por el enunciado siguen pendientes. La copia revisada de FRED no reconstruye la información histórica en tiempo real.

Aplicación: https://huggingface.co/spaces/juanhevia/brent-observatorio

## Integridad

`sha256.json` registra huellas de los archivos exportados. Los pesos se cargan con `weights_only=True`. `manifest.json` es el manifiesto original del paquete de la aplicación, cuyos archivos completos permanecen en GitHub; sus nombres h5.pt/h63.pt/h252.pt corresponden aquí a rnn_h5.pt/lstm_h63.pt/rnn_h252.pt.
''')
(MODEL/'sha256.json').write_text(json.dumps({p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(MODEL.iterdir()) if p.is_file() and p.name!='sha256.json'},indent=2))
print('Exportación preparada:',OUT)
