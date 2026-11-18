# Modelos de la aplicación

Paquete exportado de la ejecución real `runs/laboratorio` con `python -m scripts.export_app`. Contiene los pesos de RNN a 5 días, LSTM a 63 y RNN a 252, elegidos en validación. No se ha cambiado la selección usando prueba.

`manifest.json` conserva la configuración y el entorno del entrenamiento, los escaladores, las huellas de los archivos y la selección original. Los modelos se cargan como pesos con `weights_only=True`. La aplicación verifica los archivos antes de utilizarlos.

`metrics_validation.csv` y `metrics_test.csv` contienen los resultados de las tres redes y persistencia. `predictions_test.csv` permite reproducir los gráficos de prueba. No son predicciones nuevas ni un archivo de entrada para actualizar precios.

Los rangos utilizan los percentiles 10 y 90 de los errores reales menos predichos en validación. Ese bloque también se usó para seleccionar y parar el entrenamiento: son rangos orientativos sin garantía de cobertura. Su cobertura en prueba fue 75,0 %, 77,3 % y 90,4 % para los tres horizontes, respectivamente. La prueba solo se usa para medir esa cobertura; no ajusta los límites.

La copia fuente sin rellenar permanece en `data/raw/`. Los datos añadidos por el usuario se guardan aparte. Para más detalles, consultar [la guía de la aplicación](../../docs/aplicacion.md).
