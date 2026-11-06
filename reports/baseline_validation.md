# Referencia ingenua: persistencia

Predicción = último precio disponible en la fecha de origen, usando exactamente la misma información que recibirán las redes.
Los objetivos son precios realmente observados. MAE y RMSE están en USD/barril; MSE en (USD/barril)².
Supuesto de disponibilidad: 3 días lunes-viernes de retraso fijo. No equivale a una reconstrucción histórica de publicaciones.
Huella de la copia fuente: `ae1d1210a122d4ed018040d38a2d48cd508b5b93b9ed4e7d416334bd645255f3`.

## Validación principal

| Horizonte (días lunes-viernes) | Muestras | MAE | RMSE | MSE |
|---:|---:|---:|---:|---:|
| 5 | 1013 | 3.862 | 5.477 | 30.000 |
| 63 | 960 | 12.286 | 16.740 | 280.212 |
| 252 | 773 | 27.473 | 31.076 | 965.722 |

## Validación walk-forward dentro de entrenamiento

| Fold | Horizonte | Muestras | MAE | RMSE | MSE |
|---|---:|---:|---:|---:|---:|
| fold_01 | 5 | 494 | 3.179 | 3.903 | 15.233 |
| fold_01 | 63 | 438 | 8.721 | 11.385 | 129.615 |
| fold_01 | 252 | 257 | 5.785 | 8.021 | 64.341 |
| fold_02 | 5 | 501 | 2.900 | 3.732 | 13.929 |
| fold_02 | 63 | 444 | 12.159 | 16.197 | 262.332 |
| fold_02 | 252 | 260 | 45.841 | 46.995 | 2208.565 |
| fold_03 | 5 | 707 | 2.455 | 3.056 | 9.337 |
| fold_03 | 63 | 651 | 6.204 | 7.546 | 56.936 |
| fold_03 | 252 | 463 | 14.974 | 17.224 | 296.667 |

## Interpretación

Esta tabla fija la referencia que deben superar las redes en las mismas fechas de cada horizonte. Menor MAE o RMSE significa menor error.
Los horizontes tienen distintos orígenes válidos, por lo que sus cifras no son una comparación con muestra idéntica entre horizontes.
Dentro de cada horizonte se mantendrán exactamente los mismos orígenes y objetivos para los cuatro modelos.
Los errores de horizontes solapados no son independientes. No se calculan intervalos de confianza suponiendo independencia.
No se han entrenado redes ni evaluado el bloque de prueba.
