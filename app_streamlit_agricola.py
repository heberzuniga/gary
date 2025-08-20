
# app_streamlit_agricola.py
# Streamlit front-end elegante para forecasting univariado
# Permite subir CSV, elegir columnas, probar varios modelos (si están instalados),
# graficar y descargar resultados. Diseñada para estudiantes/docentes.

import io
import math
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Imports opcionales (el app se adapta si no están)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ----------------------------
# Utilidades
# ----------------------------
def _plot_series(df, date_col, y_col, yhat_col=None, title="Serie temporal"):
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df[y_col], name="Observado", mode="lines"))
        if yhat_col and yhat_col in df:
            fig.add_trace(go.Scatter(x=df[date_col], y=df[yhat_col], name="Pronóstico", mode="lines"))
        fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title=y_col, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(df[date_col], df[y_col], label="Observado")
        if yhat_col and yhat_col in df:
            plt.plot(df[date_col], df[yhat_col], label="Pronóstico")
        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel(y_col)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def train_test_split_time(df, test_size:int):
    if test_size <= 0:
        return df.copy(), pd.DataFrame(columns=df.columns)
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    val = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0
    return float(val)

# ----------------------------
# Modelos
# ----------------------------
@dataclass
class ForecastResult:
    model_name: str
    horizon: int
    metrics: dict
    fitted: Optional[object]
    df_pred: pd.DataFrame  # columnas: ds, y, yhat, split ("train"/"test"/"forecast")

def baseline_last_value(train, test, date_col, y_col, horizon):
    # Pronóstico ingenuo: último valor
    last_val = float(train[y_col].iloc[-1])
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = np.nan
    if len(test) > 0:
        full.loc[full["split"]=="test", "yhat"] = last_val
    # horizonte futuro
    if horizon > 0:
        freq = pd.infer_freq(train[date_col]) or "D"
        future_dates = pd.date_range(train[date_col].iloc[-1], periods=horizon+1, freq=freq)[1:]
        df_forecast = pd.DataFrame({date_col: future_dates, "yhat": last_val, "split": "forecast"})
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {
            "MAE": mae(test[y_col], [last_val]*len(test)),
            "RMSE": rmse(test[y_col], [last_val]*len(test)),
            "MAPE%": mape(test[y_col], [last_val]*len(test)),
        }
    return ForecastResult("baseline_last", horizon, metrics, None, full)

def moving_average(train, test, date_col, y_col, horizon, window=7):
    yhat_test = []
    hist = train[y_col].tolist()
    for _ in range(len(test)):
        val = float(np.mean(hist[-window:])) if len(hist) >= 1 else np.nan
        yhat_test.append(val)
        hist.append(test[y_col].iloc[len(yhat_test)-1])
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = np.nan
    if len(test) > 0:
        full.loc[full["split"]=="test", "yhat"] = yhat_test
    # futuro
    if horizon > 0:
        hist2 = train[y_col].tolist() + test[y_col].tolist()
        fut = []
        for _ in range(horizon):
            val = float(np.mean(hist2[-window:])) if len(hist2) >= 1 else np.nan
            fut.append(val)
            hist2.append(val)  # recursivo
        freq = pd.infer_freq(full[date_col]) or "D"
        future_dates = pd.date_range(full[date_col].iloc[-1], periods=horizon+1, freq=freq)[1:]
        df_forecast = pd.DataFrame({date_col: future_dates, "yhat": fut, "split": "forecast"})
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {"MAE": mae(test[y_col], yhat_test),
                   "RMSE": rmse(test[y_col], yhat_test),
                   "MAPE%": mape(test[y_col], yhat_test)}
    return ForecastResult(f"moving_average(w={window})", horizon, metrics, None, full)

def model_prophet(train, test, date_col, y_col, horizon, seasonality_mode="additive"):
    if not HAS_PROPHET:
        st.warning("Prophet no está instalado. Usa `pip install prophet` para habilitarlo.")
        return baseline_last_value(train, test, date_col, y_col, horizon)
    df_train = train.rename(columns={date_col: "ds", y_col: "y"})[["ds","y"]]
    m = Prophet(seasonality_mode=seasonality_mode)
    m.fit(df_train)
    # predicción en train+test
    df_all = pd.concat([train, test])
    future_all = df_all.rename(columns={date_col: "ds"})[["ds"]]
    fc_all = m.predict(future_all)
    yhat_all = fc_all["yhat"].values
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = yhat_all[:len(full)]
    # futuro
    if horizon > 0:
        future = m.make_future_dataframe(periods=horizon, include_history=False)
        fc = m.predict(future)
        df_forecast = pd.DataFrame({
            date_col: fc["ds"],
            "yhat": fc["yhat"],
            "split": "forecast"
        })
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {"MAE": mae(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "RMSE": rmse(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "MAPE%": mape(test[y_col], full.loc[full["split"]=="test","yhat"])}
    return ForecastResult("prophet", horizon, metrics, m, full)

def model_sarima(train, test, date_col, y_col, horizon, order=(1,1,1), seasonal_order=(0,0,0,0)):
    if not HAS_STATSMODELS:
        st.warning("statsmodels no está instalado. Usa `pip install statsmodels` para habilitar SARIMA.")
        return baseline_last_value(train, test, date_col, y_col, horizon)
    y_train = train.set_index(date_col)[y_col].astype(float)
    res = sm.tsa.statespace.SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    # pred en train+test
    pred_all = res.get_prediction(start=0, end=len(train)+len(test)-1)
    yhat_all = pred_all.predicted_mean.values
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = yhat_all[:len(full)]
    # futuro
    if horizon > 0:
        fcast = res.get_forecast(steps=horizon).predicted_mean.reset_index(drop=True)
        freq = pd.infer_freq(full[date_col]) or "D"
        future_dates = pd.date_range(full[date_col].iloc[-1], periods=horizon+1, freq=freq)[1:]
        df_forecast = pd.DataFrame({date_col: future_dates, "yhat": fcast, "split": "forecast"})
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {"MAE": mae(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "RMSE": rmse(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "MAPE%": mape(test[y_col], full.loc[full["split"]=="test","yhat"])}
    return ForecastResult(f"SARIMA{order}x{seasonal_order}", horizon, metrics, res, full)

def model_svr(train, test, date_col, y_col, horizon, C=10.0, epsilon=0.1, kernel="rbf"):
    if not HAS_SKLEARN:
        st.warning("scikit-learn no está instalado. Usa `pip install scikit-learn` para habilitar SVR.")
        return baseline_last_value(train, test, date_col, y_col, horizon)
    # index temporal -> feature t (1..n)
    n_train = len(train)
    n_test = len(test)
    X_train = np.arange(n_train).reshape(-1,1)
    y_train = train[y_col].values.astype(float)
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=C, epsilon=epsilon, kernel=kernel))])
    pipe.fit(X_train, y_train)
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = np.nan
    X_all = np.arange(n_train + n_test).reshape(-1,1)
    yhat_all = pipe.predict(X_all)
    full.loc[:, "yhat"] = yhat_all[:len(full)]
    # futuro
    if horizon > 0:
        X_future = np.arange(n_train + n_test, n_train + n_test + horizon).reshape(-1,1)
        fut = pipe.predict(X_future)
        freq = pd.infer_freq(full[date_col]) or "D"
        future_dates = pd.date_range(full[date_col].iloc[-1], periods=horizon+1, freq=freq)[1:]
        df_forecast = pd.DataFrame({date_col: future_dates, "yhat": fut, "split":"forecast"})
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {"MAE": mae(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "RMSE": rmse(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "MAPE%": mape(test[y_col], full.loc[full["split"]=="test","yhat"])}
    return ForecastResult("SVR", horizon, metrics, pipe, full)

def model_xgb(train, test, date_col, y_col, horizon, max_depth=4, n_estimators=300, learning_rate=0.05):
    if not HAS_XGBOOST:
        st.warning("XGBoost no está instalado. Usa `pip install xgboost` para habilitarlo.")
        return baseline_last_value(train, test, date_col, y_col, horizon)
    n_train = len(train)
    n_test = len(test)
    X_train = np.arange(n_train).reshape(-1,1)
    y_train = train[y_col].values.astype(float)
    model = XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    full = pd.concat([train.assign(split="train"), test.assign(split="test")])
    full["yhat"] = np.nan
    X_all = np.arange(n_train + n_test).reshape(-1,1)
    yhat_all = model.predict(X_all)
    full.loc[:, "yhat"] = yhat_all[:len(full)]
    # futuro
    if horizon > 0:
        X_future = np.arange(n_train + n_test, n_train + n_test + horizon).reshape(-1,1)
        fut = model.predict(X_future)
        freq = pd.infer_freq(full[date_col]) or "D"
        future_dates = pd.date_range(full[date_col].iloc[-1], periods=horizon+1, freq=freq)[1:]
        df_forecast = pd.DataFrame({date_col: future_dates, "yhat": fut, "split":"forecast"})
        full = pd.concat([full, df_forecast], ignore_index=True)
    metrics = {}
    if len(test) > 0:
        metrics = {"MAE": mae(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "RMSE": rmse(test[y_col], full.loc[full["split"]=="test","yhat"]),
                   "MAPE%": mape(test[y_col], full.loc[full["split"]=="test","yhat"])}
    return ForecastResult("XGBoost(1D)", horizon, metrics, model, full)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Agrícola – Forecasting Web", page_icon="🌾", layout="wide")

st.title("🌾 Agrícola – Forecasting con IA (Streamlit)")
st.caption("Sube tu CSV, elige columnas y genera pronósticos con distintos modelos.")

with st.sidebar:
    st.header("1) Datos")
    up = st.file_uploader("Sube un CSV", type=["csv"])
    sep = st.selectbox("Separador", [",",";","\\t","|"], index=0)
    decimal = st.selectbox("Decimal", [".", ","], index=0)

    st.header("2) Columnas")
    date_col = st.text_input("Columna de fecha", value="ds")
    y_col = st.text_input("Columna objetivo (y)", value="y")
    parse_dates = st.checkbox("Parsear fechas automáticamente", value=True)

    st.header("3) Split & Horizonte")
    test_size = st.number_input("Tamaño de test (últimas N filas)", min_value=0, value=30, step=1)
    horizon = st.number_input("Horizonte futuro (pasos)", min_value=0, value=30, step=1)

    st.header("4) Modelo")
    modelos = ["Baseline (último valor)", "Moving Average", "Prophet", "SARIMA", "SVR", "XGBoost"]
    # deshabilitar según disponibilidad
    help_txt = []
    if not HAS_PROPHET: help_txt.append("Prophet no disponible")
    if not HAS_STATSMODELS: help_txt.append("SARIMA no disponible")
    if not HAS_SKLEARN: help_txt.append("SVR no disponible")
    if not HAS_XGBOOST: help_txt.append("XGBoost no disponible")
    if help_txt:
        st.caption(" / ".join(help_txt))
    model_name = st.selectbox("Selecciona el modelo", modelos, index=2 if HAS_PROPHET else 0)

    st.header("5) Parámetros")
    ma_window = st.slider("Moving Average window", min_value=2, max_value=60, value=7) if model_name=="Moving Average" else None
    sarima_order_p = st.number_input("SARIMA p", 0, 5, 1) if model_name=="SARIMA" else None
    sarima_order_d = st.number_input("SARIMA d", 0, 2, 1) if model_name=="SARIMA" else None
    sarima_order_q = st.number_input("SARIMA q", 0, 5, 1) if model_name=="SARIMA" else None
    sarima_S = st.number_input("SARIMA período estacional (S)", 0, 52, 0) if model_name=="SARIMA" else None
    sarima_P = st.number_input("SARIMA P", 0, 5, 0) if model_name=="SARIMA" else None
    sarima_D = st.number_input("SARIMA D", 0, 2, 0) if model_name=="SARIMA" else None
    sarima_Q = st.number_input("SARIMA Q", 0, 5, 0) if model_name=="SARIMA" else None

    svr_C = st.number_input("SVR C", 0.1, 1000.0, 10.0) if model_name=="SVR" else None
    svr_eps = st.number_input("SVR epsilon", 0.001, 1.0, 0.1) if model_name=="SVR" else None
    svr_kernel = st.selectbox("SVR kernel", ["rbf","linear","poly"]) if model_name=="SVR" else None

    xgb_depth = st.number_input("XGB max_depth", 1, 12, 4) if model_name=="XGBoost" else None
    xgb_n = st.number_input("XGB n_estimators", 50, 1500, 300, step=50) if model_name=="XGBoost" else None
    xgb_lr = st.number_input("XGB learning_rate", 0.001, 0.5, 0.05) if model_name=="XGBoost" else None

    run_btn = st.button("🚀 Ejecutar")

# ----------------------------
# Carga de datos
# ----------------------------
if up is not None:
    try:
        sep_effective = {"\\t":"\t"}.get(sep, sep)
        df = pd.read_csv(up, sep=sep_effective, decimal=decimal)
    except Exception as e:
        st.error(f"No pude leer el CSV: {e}")
        st.stop()

    st.success(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    st.dataframe(df.head(50))

    if date_col not in df.columns or y_col not in df.columns:
        st.warning("Revisa los nombres de tus columnas en la barra lateral.")
        st.stop()

    # Parseo de fecha y orden
    if parse_dates:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception as e:
            st.error(f"No pude parsear las fechas: {e}")
            st.stop()

    df = df.dropna(subset=[date_col, y_col]).copy()
    df = df.sort_values(by=[date_col])
    df = df[[date_col, y_col]]
    df.reset_index(drop=True, inplace=True)

    # ----------------------------
    # Entrenamiento / Pronóstico
    # ----------------------------
    if run_btn:
        train, test = train_test_split_time(df, int(test_size))

        if model_name == "Baseline (último valor)":
            res = baseline_last_value(train, test, date_col, y_col, int(horizon))
        elif model_name == "Moving Average":
            res = moving_average(train, test, date_col, y_col, int(horizon), window=int(ma_window))
        elif model_name == "Prophet":
            res = model_prophet(train, test, date_col, y_col, int(horizon))
        elif model_name == "SARIMA":
            order = (int(sarima_order_p), int(sarima_order_d), int(sarima_order_q))
            seas = (int(sarima_P), int(sarima_D), int(sarima_Q), int(sarima_S))
            res = model_sarima(train, test, date_col, y_col, int(horizon), order=order, seasonal_order=seas)
        elif model_name == "SVR":
            res = model_svr(train, test, date_col, y_col, int(horizon), C=float(svr_C), epsilon=float(svr_eps), kernel=svr_kernel)
        elif model_name == "XGBoost":
            res = model_xgb(train, test, date_col, y_col, int(horizon), max_depth=int(xgb_depth), n_estimators=int(xgb_n), learning_rate=float(xgb_lr))
        else:
            res = baseline_last_value(train, test, date_col, y_col, int(horizon))

        st.subheader("📈 Serie con Pronóstico")
        show = res.df_pred.copy()
        show.rename(columns={date_col: "Fecha"}, inplace=True)
        _plot_series(show.rename(columns={"Fecha": date_col}), date_col, y_col, "yhat", title=f"{res.model_name} – y vs yhat")

        # Métricas
        st.subheader("📊 Métricas (sobre TEST)")
        if res.metrics:
            st.json({k: round(v, 4) if isinstance(v, (int,float)) else v for k,v in res.metrics.items()})
        else:
            st.info("Sin conjunto de test o no se pudieron calcular métricas.")

        # Descarga
        st.subheader("⬇️ Descargas")
        # DataFrame con yhat
        csv_buf = io.StringIO()
        res.df_pred.to_csv(csv_buf, index=False)
        st.download_button("Descargar CSV con y yhat", csv_buf.getvalue().encode("utf-8"), file_name="forecast_results.csv", mime="text/csv")

        # Guardar modelo si existe
        try:
            import joblib, tempfile
            if res.fitted is not None:
                with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                    joblib.dump(res.fitted, tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("Descargar modelo entrenado (.pkl)", f.read(), file_name=f"model_{res.model_name}.pkl")
        except Exception:
            st.caption("joblib no disponible o modelo no serializable.")

else:
    st.info("👈 Sube un CSV desde la barra lateral para comenzar.")
