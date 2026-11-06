# Preparación de datos: Brent univariante

Copia de FRED: `data/raw/brent_2000-01-01_2026-11-05.csv`. SHA-256: `ae1d1210a122d4ed018040d38a2d48cd508b5b93b9ed4e7d416334bd645255f3`.

Calendario de 7,004 filas lunes-viernes, del 2000-01-03 al 2026-11-05.
Precios observados: 6,501. Fechas sin precio observado: 503.

Ventanas de 60 filas y objetivos a [5, 63, 252] días lunes-viernes. No se cuentan sesiones de mercado.
Se rellenan solo entradas con el último valor, como máximo 5 filas; jamás se rellenan objetivos ni hacia atrás.
Se asume un retraso de publicación de 3 días lunes-viernes. Es una hipótesis configurable, no un retraso histórico verificado.
Los objetivos de cada bloque deben estar observados y disponibles, según ese supuesto, antes del inicio del siguiente bloque.

## Muestras listas para las redes

| Grupo | Bloque | Horizonte | Muestras | Primer origen | Último origen | Último objetivo |
|---|---|---:|---:|---|---|---|
| main | train | 5 | 4207 | 2000-03-30 | 2018-10-05 | 2018-10-12 |
| main | train | 63 | 4159 | 2000-03-30 | 2018-07-17 | 2018-10-12 |
| main | train | 252 | 4042 | 2000-03-30 | 2017-10-25 | 2018-10-12 |
| main | validation | 5 | 1013 | 2018-10-18 | 2022-10-14 | 2022-10-21 |
| main | validation | 63 | 960 | 2018-10-18 | 2022-07-26 | 2022-10-21 |
| main | validation | 252 | 773 | 2018-10-18 | 2021-11-03 | 2022-10-21 |
| main | test | 5 | 1010 | 2022-10-27 | 2026-10-26 | 2026-11-02 |
| main | test | 63 | 955 | 2022-10-27 | 2026-08-05 | 2026-11-02 |
| main | test | 252 | 772 | 2022-10-27 | 2025-11-13 | 2026-11-02 |
| fold_01 | train | 5 | 2484 | 2000-03-30 | 2011-12-16 | 2011-12-23 |
| fold_01 | train | 63 | 2436 | 2000-03-30 | 2011-09-27 | 2011-12-23 |
| fold_01 | train | 252 | 2319 | 2000-03-30 | 2011-01-05 | 2011-12-23 |
| fold_01 | validation | 5 | 494 | 2012-01-02 | 2013-12-19 | 2013-12-26 |
| fold_01 | validation | 63 | 438 | 2012-01-02 | 2013-09-30 | 2013-12-26 |
| fold_01 | validation | 252 | 257 | 2012-01-02 | 2013-01-08 | 2013-12-26 |
| fold_02 | train | 5 | 2985 | 2000-03-30 | 2013-12-19 | 2013-12-26 |
| fold_02 | train | 63 | 2937 | 2000-03-30 | 2013-09-30 | 2013-12-26 |
| fold_02 | train | 252 | 2820 | 2000-03-30 | 2013-01-08 | 2013-12-26 |
| fold_02 | validation | 5 | 501 | 2014-01-01 | 2015-12-21 | 2015-12-28 |
| fold_02 | validation | 63 | 444 | 2014-01-01 | 2015-09-30 | 2015-12-28 |
| fold_02 | validation | 252 | 260 | 2014-01-01 | 2015-01-08 | 2015-12-28 |
| fold_03 | train | 5 | 3493 | 2000-03-30 | 2015-12-21 | 2015-12-28 |
| fold_03 | train | 63 | 3445 | 2000-03-30 | 2015-09-30 | 2015-12-28 |
| fold_03 | train | 252 | 3328 | 2000-03-30 | 2015-01-08 | 2015-12-28 |
| fold_03 | validation | 5 | 707 | 2016-01-01 | 2018-10-05 | 2018-10-12 |
| fold_03 | validation | 63 | 651 | 2016-01-01 | 2018-07-17 | 2018-10-12 |
| fold_03 | validation | 252 | 463 | 2016-01-01 | 2017-10-25 | 2018-10-12 |

## Escalado

Se usa (precio - media) / desviación estándar poblacional. Cada fold ajusta su propio escalador con precios originales disponibles antes de validación; no usa precios de validación o prueba.
El mismo escalador transforma las entradas y el objetivo. Para recuperar USD/barril: predicción * scale + mean. No se recortan valores fuera del rango de entrenamiento.

## Límites del experimento

Esta fase prepara únicamente Brent; las otras variables del CSV exploratorio no entran todavía en las redes.
La copia de FRED contiene el histórico revisado al descargarla. El retraso fijo no sustituye a las fechas reales de publicación ni a los datos disponibles en cada momento.
Los resultados serán una comparación retrospectiva bajo este supuesto, no un backtest de inversión en tiempo real.
El bloque test se prepara, pero no se calcula ninguna métrica de prueba en esta fase.

Fuente: [EIA vía FRED, DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU).
