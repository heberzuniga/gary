
import io
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from typing import Tuple, Dict, List

# ML / Stats
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Visualization
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pronóstico de Precios de Soya - ML", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        st.warning("Formato no soportado. Sube CSV o XLSX.")
        return pd.DataFrame()
    return df

def infer_datetime_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
        # attempt parse
        try:
            pd.to_datetime(df[col])
            return col
        except Exception:
            continue
    return ""

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    d["quarter"] = d[date_col].dt.quarter
    d["dayofyear"] = d[date_col].dt.dayofyear
    d["weekofyear"] = d[date_col].dt.isocalendar().week.astype(int)
    d["is_month_start"] = d[date_col].dt.is_month_start.astype(int)
    d["is_month_end"] = d[date_col].dt.is_month_end.astype(int)
    return d

def add_lags_rolls(df: pd.DataFrame, target: str, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    d = df.copy()
    for L in lags:
        d[f"{target}_lag{L}"] = d[target].shift(L)
    for W in rolls:
        d[f"{target}_rollmean{W}"] = d[target].rolling(W).mean()
        d[f"{target}_rollstd{W}"] = d[target].rolling(W).std()
    d = d.dropna()
    return d

def regression_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

def scores_to_df(scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, m in scores.items():
        rows.append({
            "Modelo": name,
            "MAE": m["mae"],
            "RMSE": m["rmse"],
            "R2": m["r2"]
        })
    return pd.DataFrame(rows).sort_values(by="RMSE")

def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def narrative_from_scores(best_name: str, metrics: Dict[str, float], horizon: int) -> str:
    return (
        f"El mejor modelo seleccionado fue **{best_name}**. "
        f"Al evaluar sobre el conjunto de prueba, obtuvo **MAE={metrics['mae']:.3f}**, "
        f"**RMSE={metrics['rmse']:.3f}** y **R²={metrics['r2']:.3f}**. "
        f"Para un horizonte de pronóstico de {horizon} períodos, estos resultados sugieren "
        f"que el error absoluto típico es de aproximadamente {metrics['mae']:.2f} unidades "
        f"del precio y que el desvío cuadrático medio (RMSE) representa el error esperado "
        f"alrededor de la predicción. Un R² de {metrics['r2']:.2%} indica la fracción de la "
        f"variabilidad explicada por el modelo."
    )

def make_forecast_frame(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DataFrame:
    future_idx = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    return pd.DataFrame({"__future_index__": future_idx})

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Menú")
page = st.sidebar.radio(
    "Navegación",
    [
        "1) Proyecto",
        "2) Cargar Datos",
        "3) Exploración (EDA)",
        "4) Ingeniería de Características",
        "5) Selección / Entrenamiento de Modelos",
        "6) Pronóstico y Escenarios",
        "7) Visualizaciones",
        "8) Reporte y Exportación",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Basado en tu investigación: 'Pronósticos de Precios de Soya con Machine Learning'.")

# Session State
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()
if "features" not in st.session_state:
    st.session_state["features"] = []
if "target" not in st.session_state:
    st.session_state["target"] = ""
if "date_col" not in st.session_state:
    st.session_state["date_col"] = ""
if "freq" not in st.session_state:
    st.session_state["freq"] = "MS"  # default monthly start
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "X_columns" not in st.session_state:
    st.session_state["X_columns"] = []

# -----------------------------
# 1) Proyecto
# -----------------------------
if page.startswith("1"):
    st.title("📈 Pronóstico de Precios de Soya con Machine Learning")
    st.markdown("""
**Objetivo:** Construir una aplicación interactiva para **cargar datos**, **explorarlos**, **ingenierizar características**,
**entrenar distintos modelos**, **compararlos** y **generar pronósticos** con reportes explicativos y gráficos profesionales.

**Sugerencia de flujo (según el PDF):**
1. **Cargar Datos:** Subir CSV/XLSX con precios de soya (y variables explicativas si las hubiera).
2. **EDA:** Validar estructura, ausentes, outliers, correlaciones.
3. **Ingeniería:** Crear retardos (_lags_) y ventanas móviles, además de **variables temporales** (año, mes, etc.).
4. **Modelos:** Comparar modelos (Lineal, Ridge, Lasso, Random Forest, Gradient Boosting) y **SARIMAX** para series temporales.
5. **Pronóstico:** Elegir horizonte, frecuencia, escenarios (choques de variables) y visualizar bandas.
6. **Reporte:** Generar un texto automático con interpretación y exportar resultados.

> En cada paso, puedes elegir **tipos de gráficos** (línea, área, barras, dispersión, boxplot, heatmap) para facilitar tu análisis.
""")

# -----------------------------
# 2) Cargar Datos
# -----------------------------
elif page.startswith("2"):
    st.title("📥 Cargar Datos")
    uploaded = st.file_uploader("Sube un archivo CSV o XLSX", type=["csv", "xlsx"])
    if uploaded:
        df = load_data(uploaded)
        if not df.empty:
            st.session_state["data"] = df.copy()
            st.success("Datos cargados correctamente.")
            st.write("Vista previa:", df.head())

            # Selección de columnas clave
            date_col_guess = infer_datetime_column(df)
            date_col = st.selectbox("Columna de fecha", options=[""] + list(df.columns), index=(list(df.columns).index(date_col_guess) + 1) if date_col_guess in df.columns else 0)
            target = st.selectbox("Variable objetivo (precio soya)", options=[""] + [c for c in df.columns if c != date_col])
            features = st.multiselect("Variables explicativas (opcionales)", options=[c for c in df.columns if c not in [date_col, target]])

            freq = st.selectbox("Frecuencia temporal", options=["D", "W", "MS", "M", "Q", "YS"], index=2, help="D: diaria, W: semanal, MS/M: mensual, Q: trimestral, YS: anual")
            if st.button("Guardar configuraciones"):
                st.session_state["date_col"] = date_col
                st.session_state["target"] = target
                st.session_state["features"] = features
                st.session_state["freq"] = freq
                st.success("Configuración guardada.")

# -----------------------------
# 3) EDA
# -----------------------------
elif page.startswith("3"):
    st.title("🔎 Exploración de Datos (EDA)")
    df = st.session_state["data"]
    if df.empty:
        st.info("Sube datos en la sección 2.")
    else:
        st.write("Dimensiones:", df.shape)
        st.write("Tipos de datos:", df.dtypes)
        with st.expander("Estadísticos descriptivos"):
            st.write(df.describe(include="all"))

        chart_type = st.selectbox("Tipo de gráfico", ["Línea", "Área", "Barras", "Dispersión", "Boxplot", "Heatmap (correlación)"])
        date_col = st.session_state.get("date_col", "")
        target = st.session_state.get("target", "")
        if date_col and (date_col in df.columns):
            try:
                df_plot = df.copy()
                df_plot[date_col] = pd.to_datetime(df_plot[date_col])
                df_plot = df_plot.sort_values(date_col)
                if chart_type == "Línea":
                    fig = px.line(df_plot, x=date_col, y=target if target else df_plot.columns[1:])
                elif chart_type == "Área":
                    fig = px.area(df_plot, x=date_col, y=target if target else df_plot.columns[1:])
                elif chart_type == "Barras":
                    fig = px.bar(df_plot, x=date_col, y=target if target else df_plot.columns[1:])
                elif chart_type == "Dispersión":
                    ycol = st.selectbox("Eje Y", options=[c for c in df_plot.columns if c != date_col])
                    fig = px.scatter(df_plot, x=date_col, y=ycol)
                elif chart_type == "Boxplot":
                    ycol = st.selectbox("Variable", options=[c for c in df_plot.columns if c != date_col])
                    fig = px.box(df_plot, x=date_col, y=ycol)
                else:
                    corr = df.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(corr, color_continuous_scale="RdBu_r", origin="lower")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo graficar: {e}")
        else:
            st.info("Define la columna de fecha y objetivo en 'Cargar Datos'.")

# -----------------------------
# 4) Ingeniería
# -----------------------------
elif page.startswith("4"):
    st.title("🧪 Ingeniería de Características")
    df = st.session_state["data"]
    date_col = st.session_state.get("date_col", "")
    target = st.session_state.get("target", "")
    if df.empty or not date_col or not target:
        st.info("Sube datos y define fecha/objetivo en secciones anteriores.")
    else:
        lags = st.multiselect("Lags (retardos)", options=[1,2,3,6,9,12,18,24], default=[1,2,3,6,12])
        rolls = st.multiselect("Ventanas móviles", options=[3,6,12,24], default=[3,6,12])

        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col])
        df2 = df2.sort_values(date_col)
        df2 = add_time_features(df2, date_col)
        df2 = add_lags_rolls(df2, target, lags, rolls)

        st.write("Muestra con nuevas columnas:", df2.head())
        st.session_state["engineered"] = df2

# -----------------------------
# 5) Selección / Entrenamiento
# -----------------------------
elif page.startswith("5"):
    st.title("🤖 Selección y Entrenamiento de Modelos")
    df = st.session_state.get("engineered", pd.DataFrame())
    date_col = st.session_state.get("date_col", "")
    target = st.session_state.get("target", "")
    if df.empty:
        st.info("Primero realiza ingeniería de características.")
    else:
        # Definir features
        candidate_features = [c for c in df.columns if c not in [date_col, target]]
        chosen = st.multiselect("Selecciona predictores", options=candidate_features, default=[c for c in candidate_features if "lag" in c or "roll" in c or c in ["year","month","quarter","weekofyear"]])
        test_size = st.slider("Proporción de test", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)

        X = df[chosen].copy()
        y = df[target].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        models = regression_models(random_state=random_state)
        results = {}
        preds = {}

        for name, model in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=False) if isinstance(model, (Lasso, Ridge)) else "passthrough"),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            results[name] = evaluate_regression(y_test, y_pred)
            preds[name] = y_pred

        res_df = scores_to_df(results)
        st.subheader("Comparación de Modelos (conjunto de prueba)")
        st.dataframe(res_df, use_container_width=True)

        # Gráfico comparativo de predicciones
        best_name = res_df.iloc[0]["Modelo"]
        st.session_state["trained_model"] = models[best_name].fit(X_train, y_train)
        st.session_state["X_columns"] = list(X.columns)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test.values, name="Real"))
        for name in results.keys():
            fig.add_trace(go.Scatter(y=preds[name], name=f"Pred {name}"))
        fig.update_layout(title="Comparación de Predicciones en Test")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(narrative_from_scores(best_name, results[best_name], horizon=12))

        st.info("Alternativa de serie temporal: SARIMAX")
        p = st.number_input("p (AR)", value=1, step=1)
        d = st.number_input("d (dif)", value=1, step=1)
        q = st.number_input("q (MA)", value=1, step=1)
        seasonal = st.checkbox("Estacional", value=True)
        P = st.number_input("P", value=1, step=1)
        D = st.number_input("D", value=1, step=1)
        Q = st.number_input("Q", value=1, step=1)
        s = st.number_input("s (periodicidad, ej. 12 mensual)", value=12, step=1)

        if st.button("Entrenar SARIMAX"):
            try:
                series = df.set_index(pd.to_datetime(df[date_col]))[target].asfreq(st.session_state["freq"])
                series = series.fillna(method="ffill")
                order = (int(p), int(d), int(q))
                seasonal_order = (int(P), int(D), int(Q), int(s)) if seasonal else (0,0,0,0)
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                sarima = model.fit(disp=False)
                st.session_state["sarimax"] = sarima
                st.success("SARIMAX entrenado.")
                st.write(sarima.summary().as_text())
            except Exception as e:
                st.error(f"No se pudo entrenar SARIMAX: {e}")

# -----------------------------
# 6) Pronóstico y Escenarios
# -----------------------------
elif page.startswith("6"):
    st.title("🔮 Pronóstico y Simulación de Escenarios")
    df = st.session_state.get("engineered", pd.DataFrame())
    date_col = st.session_state.get("date_col", "")
    target = st.session_state.get("target", "")
    freq = st.session_state.get("freq", "MS")

    if df.empty or not st.session_state.get("trained_model"):
        st.info("Necesitas entrenar un modelo en la sección 5.")
    else:
        horizon = st.slider("Horizonte de pronóstico (períodos)", 1, 36, 12, 1)
        last_date = pd.to_datetime(df[date_col]).max()
        future = [f for f in st.session_state["X_columns"] if f in df.columns]
        base = df[future].iloc[[-1]].copy()

        # Construir marco futuro con supuestos sencillos (mantener últimos valores)
        future_df = pd.concat([base]*horizon, ignore_index=True)
        # Permitir ajustes de escenario
        st.subheader("Ajustes de escenario")
        for col in future:
            if col.endswith("rollmean12") or col.endswith("rollmean6"):
                v = st.number_input(f"Δ {col} (%)", value=0.0, step=0.5)
                future_df[col] = future_df[col] * (1 + v/100.0)

        y_pred = st.session_state["trained_model"].predict(future_df)
        pred_idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]
        pred_frame = pd.DataFrame({date_col: pred_idx, "forecast": y_pred})

        fig = px.line(pred_frame, x=date_col, y="forecast", title="Pronóstico (Modelo ML)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pred_frame, use_container_width=True)

        if "sarimax" in st.session_state:
            try:
                sarima = st.session_state["sarimax"]
                steps = horizon
                sar_pred = sarima.get_forecast(steps=steps)
                sar_ci = sar_pred.conf_int()
                sar_mean = sar_pred.predicted_mean.reset_index()
                sar_mean.columns = [date_col, "forecast_sarimax"]
                sar_ci = sar_ci.reset_index(drop=True)
                sar_plot = sar_mean.copy()
                sar_plot["lower"] = sar_ci.iloc[:,0].values
                sar_plot["upper"] = sar_ci.iloc[:,1].values

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sar_plot[date_col], y=sar_plot["forecast_sarimax"], name="SARIMAX"))
                fig2.add_trace(go.Scatter(x=sar_plot[date_col], y=sar_plot["upper"], name="Upper", fill=None, mode="lines"))
                fig2.add_trace(go.Scatter(x=sar_plot[date_col], y=sar_plot["lower"], name="Lower", fill='tonexty', mode="lines"))
                fig2.update_layout(title="Pronóstico SARIMAX con bandas de confianza")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(sar_plot, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo calcular el pronóstico SARIMAX: {e}")

# -----------------------------
# 7) Visualizaciones
# -----------------------------
elif page.startswith("7"):
    st.title("📊 Visualizaciones flexibles")
    df = st.session_state["data"]
    if df.empty:
        st.info("Sube datos en la sección 2.")
    else:
        kind = st.selectbox("Tipo de gráfico", ["Línea","Área","Barras","Dispersión","Boxplot","Heatmap (correlación)"])
        xcol = st.selectbox("Eje X", options=list(df.columns))
        ycols = st.multiselect("Ejes Y", options=[c for c in df.columns if c != xcol], default=[c for c in df.columns if df[c].dtype!=object][:1])
        if ycols:
            try:
                if kind == "Línea":
                    fig = px.line(df, x=xcol, y=ycols)
                elif kind == "Área":
                    fig = px.area(df, x=xcol, y=ycols)
                elif kind == "Barras":
                    fig = px.bar(df, x=xcol, y=ycols, barmode="group")
                elif kind == "Dispersión":
                    fig = px.scatter(df, x=xcol, y=ycols[0])
                elif kind == "Boxplot":
                    fig = px.box(df, x=xcol, y=ycols[0])
                else:
                    corr = df.select_dtypes(include=[np.number]).corr()
                    fig = px.imshow(corr, color_continuous_scale="RdBu_r", origin="lower")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo graficar: {e}")
        else:
            st.info("Selecciona al menos una variable Y.")

# -----------------------------
# 8) Reporte y Exportación
# -----------------------------
elif page.startswith("8"):
    st.title("📝 Reporte y Exportación")
    best = st.session_state.get("trained_model", None)
    if best is None:
        st.info("Entrena un modelo primero.")
    else:
        st.markdown("Genera un **reporte textual** con interpretación automática de resultados.")
        comentarios = st.text_area("Comentarios adicionales (opcional)", "")
        report = f"""
# Reporte de Pronóstico de Precios de Soya

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Resumen Ejecutivo
Este informe resume el proceso de modelado y pronóstico aplicado a la serie de precios de soya.
Se utilizaron transformaciones temporales (retardos y ventanas móviles), y se compararon múltiples modelos.

## Configuración de Datos
- Columna de fecha: **{st.session_state.get('date_col','(no definida)')}**
- Variable objetivo: **{st.session_state.get('target','(no definida)')}**
- Frecuencia: **{st.session_state.get('freq','MS')}**
- Predictores usados: **{', '.join(st.session_state.get('X_columns', [])) or '(no definidos)'}**

## Resultados Principales
El mejor modelo seleccionado fue el entrenado en la sección 5. Se generaron pronósticos en la sección 6.
{comentarios}

## Recomendaciones
- Revisar la estabilidad de los hiperparámetros con validación temporal (TimeSeriesSplit).
- Evaluar la sensibilidad del pronóstico ante escenarios de shocks (ej. variaciones porcentuales en medias móviles).
- Actualizar el modelo periódicamente a medida que ingresan nuevos datos.
"""
        st.download_button("Descargar reporte (.md)", report, file_name="reporte_soya.md")

        st.markdown("---")
        st.subheader("Exportar artefactos")
        st.write("Puedes exportar las columnas de ingeniería y/o los pronósticos generados en la sección 6.")
        if "engineered" in st.session_state:
            csv_bytes = st.session_state["engineered"].to_csv(index=False).encode("utf-8")
            st.download_button("Descargar datos ingenierizados (CSV)", csv_bytes, file_name="datos_ingenierizados.csv")

st.caption("© 2025 – SolverTic SRL | App ML para Pronóstico de Precios de Soya")
