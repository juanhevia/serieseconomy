"""Aplicación local: python -m streamlit run app.py."""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from oilprice.inference import (forecast, history_csv, latest_origin, load_bundle,
                                merge_history, read_update, save_history, today)

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "artifacts/brent_v1"
HISTORY = Path(os.environ.get("OILPRICE_HISTORY_PATH", str(ROOT / "data/inference/history.csv")))

st.set_page_config(page_title="Brent · Predicción y evidencia", page_icon="📈", layout="wide")
st.title("Brent · Predicción y evidencia")
st.caption("RNN, LSTM y GRU · Precios en dólares por barril · Modelos entrenados en el laboratorio")

try:
    bundle, base = load_bundle(BUNDLE, ROOT)
    if "history" not in st.session_state:
        saved = base
        if HISTORY.exists():
            saved, _ = merge_history(base, read_update(HISTORY.read_bytes()))
        st.session_state.history = saved
except (OSError, ValueError, KeyError) as exc:
    st.error(f"No se pueden cargar los datos o modelos: {exc}")
    st.info("Consulta docs/aplicacion.md para restaurar el paquete o revisar el histórico guardado.")
    st.stop()

with st.sidebar:
    st.header("Actualizar los datos")
    st.write("Añade observaciones del Brent con un CSV. Los modelos no se vuelven a entrenar.")
    uploaded = st.file_uploader("CSV con columnas Date,brent", type=["csv"], max_upload_size=5)
    st.caption("Fechas AAAA-MM-DD, precios con punto decimal. Puedes incluir el histórico completo o solo datos nuevos.")
    if st.button("Incorporar CSV", disabled=uploaded is None):
        try:
            updated, count = merge_history(st.session_state.history, read_update(uploaded.getvalue()))
            st.session_state.history = updated
            st.session_state.origin = latest_origin(updated, bundle["models"]["5"]["metadata"]["preparation_config"]["publication_lag_bdays"]).date()
            st.success(f"{count} observaciones incorporadas. Guarda el histórico para conservarlas al cerrar.")
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Guardar histórico actualizado"):
        try:
            save_history(st.session_state.history, HISTORY)
            st.success("Histórico guardado en este ordenador.")
        except OSError as exc:
            st.error(f"No se pudo guardar: {exc}")
    st.download_button("Descargar histórico CSV", history_csv(st.session_state.history), "brent_actualizado.csv", "text/csv")
    st.download_button("Descargar plantilla vacía", "Date,brent\n", "plantilla_brent.csv", "text/csv")
    st.caption("La copia original del experimento se conserva. No se sustituyen precios ya registrados.")

history = st.session_state.history
metadata = bundle["models"]["5"]["metadata"]
lag = metadata["preparation_config"]["publication_lag_bdays"]
default_origin = latest_origin(history, lag)
if "origin" not in st.session_state:
    st.session_state.origin = default_origin.date()
prediction_tab, evidence_tab, history_tab, method_tab = st.tabs(["Predicción", "Resultados reales", "Historia del Brent", "Cómo interpretarlo"])

with prediction_tab:
    controls = st.columns([1, 1, 2])
    horizon = controls[0].selectbox("Horizonte", [5, 63, 252], format_func=lambda h: {5: "Corto · 5 días", 63: "Medio · 63 días", 252: "Largo · 252 días"}[h])
    origin = controls[1].date_input("Fecha de la predicción", value=None,
                                    min_value=pd.Timestamp("2000-01-01").date(), max_value=today().date(), key="origin")
    controls[2].caption("Horizontes de lunes a viernes, incluidos festivos. Se supone un retraso de publicación de tres días de ese calendario.")
    result = None
    try:
        result = forecast(history, origin, horizon, bundle, BUNDLE)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
    if result:
        if pd.Timestamp(origin) < pd.offsets.BDay().rollback(today()):
            st.info(f"Predicción fechada el {origin:%d/%m/%Y}, no a fecha de hoy. Incorpora observaciones recientes para actualizarla.")
        st.subheader(f"{result['architecture']} · objetivo {pd.Timestamp(result['target_date']):%d/%m/%Y}")
        a, b, c = st.columns(3)
        a.metric("Precio previsto", f"{result['predicted_usd']:.2f} USD")
        b.metric("Último precio disponible", f"{result['last_available_usd']:.2f} USD")
        percent = "no calculable" if result["change_pct"] is None else f"{result['change_pct']:+.1f} %"
        c.metric("Variación prevista", f"{result['change_usd']:+.2f} USD", percent, delta_color="off")
        st.write(f"**Rango orientativo:** {result['range_low_usd']:.2f}–{result['range_high_usd']:.2f} USD/barril.")
        st.caption(f"Basado en el 80 % central de los errores de validación. En prueba cubrió el {100*result['range_test_coverage']:.1f} % de los valores reales; no garantiza una cobertura futura del 80 %.")
        cutoff = pd.Timestamp(result["last_observation_date"])
        known = history.loc[:cutoff].dropna().tail(260).rename_axis("Fecha").reset_index(name="Precio")
        point = pd.DataFrame({"Fecha": [pd.Timestamp(result["target_date"])], "Precio": [result["predicted_usd"]],
                              "Inferior": [result["range_low_usd"]], "Superior": [result["range_high_usd"]]})
        line = alt.Chart(known).mark_line(color="#24768b").encode(x=alt.X("Fecha:T", title="Fecha"), y=alt.Y("Precio:Q", title="USD/barril", scale=alt.Scale(zero=False)), tooltip=["Fecha:T", alt.Tooltip("Precio:Q", format=".2f")])
        interval = alt.Chart(point).mark_rule(color="#d88032", strokeWidth=4).encode(x="Fecha:T", y="Inferior:Q", y2="Superior:Q", tooltip=["Inferior:Q", "Superior:Q"])
        dot = alt.Chart(point).mark_point(color="#d88032", filled=True, size=100).encode(x="Fecha:T", y="Precio:Q", tooltip=["Fecha:T", alt.Tooltip("Precio:Q", format=".2f")])
        st.altair_chart((line + interval + dot).properties(height=340), width="stretch")
        st.caption(f"Azul: últimas 260 observaciones disponibles. Naranja: una predicción y su rango, no una trayectoria. Última observación utilizada: {cutoff:%d/%m/%Y}.")
        direction = "por encima" if result["change_usd"] > 0 else "por debajo" if result["change_usd"] < 0 else "al mismo nivel"
        st.write(f"La red sitúa el precio {direction} del último dato disponible. Es una estimación para una fecha concreta: no significa que el precio siga una subida o bajada continua hasta entonces.")
        st.warning("El modelo elegido para este horizonte no superó a mantener el último precio conocido en la prueba final. Su predicción debe leerse junto con esa referencia.")
        if horizon == 252:
            st.write("A largo plazo, las redes produjeron valores casi constantes y no anticiparon los picos de la serie. La fiabilidad para detectar giros es limitada.")
        st.download_button("Descargar esta predicción", pd.DataFrame([result]).to_csv(index=False), f"prediccion_brent_{horizon}.csv", "text/csv")
        with st.expander("Datos utilizados"):
            st.write(f"Ventana: 60 días lunes-viernes. Porcentaje de entradas rellenadas causalmente: {100*result['input_imputed_fraction']:.1f} %. Pesos y escalador fijos.")

with evidence_tab:
    st.subheader("Qué ocurrió en datos no usados para ajustar las redes")
    st.write("La selección se hizo en validación: RNN a 5 días, LSTM a 63 y RNN a 252. Se conserva esa selección aunque GRU obtuviera el menor RMSE en prueba.")
    metrics = pd.read_csv(BUNDLE / "metrics_test.csv")
    shown = metrics.loc[metrics.horizon_bdays == horizon].rename(columns={"architecture": "Modelo", "n": "Muestras", "r2": "R²", "mae_usd": "MAE (USD)", "mse_usd2": "MSE (USD²)", "rmse_usd": "RMSE (USD)", "improvement_rmse_pct": "Mejora RMSE (%)"}).drop(columns="horizon_bdays")
    st.caption(f"Resultados para el horizonte seleccionado: {horizon} días. PERSISTENCE mantiene el último precio disponible.")
    st.dataframe(shown.round(3), hide_index=True, width="stretch")
    predictions = pd.read_csv(BUNDLE / "predictions_test.csv", parse_dates=["target_date"])
    selected = bundle["models"][str(horizon)]["architecture"]
    subset = predictions[(predictions.horizon_bdays == horizon) & (predictions.architecture == selected)]
    chart_data = subset.set_index("target_date")[["actual_usd", "predicted_usd"]].rename(columns={"actual_usd": "Real", "predicted_usd": selected})
    st.line_chart(chart_data, color=["#24768b", "#d88032"], height=300)
    st.caption("Predicciones con orígenes sucesivos; cada fecha usa el histórico que estaría disponible entonces.")
    st.write("Los mayores errores se concentran en movimientos bruscos. A 63 días, la ventaja global de GRU depende del tramo de 2026; a 252 días las redes apenas varían su predicción.")
    st.download_button("Descargar comparación completa", (BUNDLE / "metrics_test.csv").read_bytes(), "comparacion_prueba.csv", "text/csv")

with history_tab:
    st.subheader("Picos, caídas y contexto económico")
    st.caption("Contexto descriptivo del histórico cargado. Estas explicaciones no son variables de entrada de la red.")
    observed = history.dropna()
    events = []
    for label, start, end, mode in [("2008 · máximo", "2008-01-01", "2008-07-31", "max"),
                                   ("2008 · caída", "2008-08-01", "2008-12-31", "min"),
                                   ("2014–16", "2014-01-01", "2016-12-31", "min"),
                                   ("2020", "2020-01-01", "2020-12-31", "min"),
                                   ("2022", "2022-01-01", "2022-12-31", "max"),
                                   ("Reciente", "2026-01-01", "2026-12-31", "max")]:
        sample = observed.loc[start:end]
        if not sample.empty:
            date = sample.idxmax() if mode == "max" else sample.idxmin()
            events.append({"Fecha": date, "Precio": sample.loc[date], "Episodio": label})
    # El muestreo solo aligera el gráfico; los puntos de los extremos conservan el dato diario.
    historical = observed.iloc[::3].rename_axis("Fecha").reset_index(name="Precio")
    marks = pd.DataFrame(events)
    curve = alt.Chart(historical).mark_line(color="#24768b").encode(x="Fecha:T", y=alt.Y("Precio:Q", title="USD/barril", scale=alt.Scale(zero=False)))
    dots = alt.Chart(marks).mark_point(color="#d88032", size=80, filled=True).encode(x="Fecha:T", y="Precio:Q", tooltip=["Episodio", "Fecha:T", "Precio:Q"])
    labels = dots.mark_text(dy=-13, fontSize=11, color="#d88032").encode(text="Episodio:N")
    st.altair_chart((curve + dots + labels).properties(height=340), width="stretch")
    st.markdown("""
- **2008:** el máximo fue seguido por una fuerte caída en el contexto de la crisis financiera y la contracción de la demanda. [EIA: precios y consumo](https://www.eia.gov/finance/performanceprofiles/overview.php).
- **2014–2016:** la oferta superior a la demanda y la acumulación de inventarios presionaron los precios a la baja. [EIA: balance de 2015](https://www.eia.gov/TODAYINENERGY/detail.php?id=24432).
- **2020:** las restricciones asociadas a la pandemia redujeron la demanda de petróleo. El contrato de futuros WTI llegó a cotizar en negativo; no debe confundirse con el Brent de este gráfico. [EIA: mercado de 2020](https://www.eia.gov/tODAyinenergy/detail.php?id=46336).
- **2022:** la invasión de Ucrania y los bajos inventarios contribuyeron a las subidas; después pesaron las preocupaciones por la demanda. [EIA: balance de 2022](https://www.eia.gov/todayinenergy/detail.php?id=55079).
- **Tramo reciente:** los picos y cambios rápidos de 2026 coinciden con grandes errores de predicción en esta copia. No se atribuyen aquí a un acontecimiento concreto sin un análisis adicional de fuentes.
""")

with method_tab:
    st.subheader("Una herramienta para estudiar la predicción del Brent")
    st.markdown("""
- **Actualización manual:** el CSV añade precios o completa huecos. Guardar el histórico permite recuperarlo al volver a abrir la aplicación. No se entrena ni se recalcula el escalador.
- **Disponibilidad:** se aplica el mismo retraso supuesto de tres días lunes-viernes y un máximo de cinco días de relleno hacia delante. Un hueco largo impide predecir.
- **Rango orientativo:** se suman a la predicción los percentiles 10 y 90 de los errores reales menos predichos en validación. Ese bloque también se usó para elegir los modelos y la época, por lo que el rango puede ser optimista. La cobertura observada en prueba es descriptiva; no se ajustó el rango con prueba.
- **Fiabilidad:** MAE expresa el error absoluto medio; RMSE da más peso a errores grandes. R² puede ser negativo y no es un porcentaje de aciertos. Una pequeña mejora no demuestra significación estadística.
- **Limitaciones:** modelo univariante, una semilla y disponibilidad histórica aproximada. Cambios de oferta, demanda o acontecimientos no presentes en el histórico pueden invalidar la estimación. No incorpora noticias ni variables externas.
""")
    st.caption("La aplicación ejecuta los modelos en CPU. No necesita la GPU del laboratorio ni conexión a Internet para predecir con los datos guardados.")
