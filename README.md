# Pronóstico de Precios de Soya con ML (Streamlit)

Esta app reproduce el flujo de tu investigación (carga de datos → EDA → ingeniería → modelos → pronóstico → reporte).
Permite subir CSV/XLSX, elegir algoritmos y visualizar resultados con gráficos configurables.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud
1. Sube esta carpeta a un repositorio en GitHub.
2. Ve a share.streamlit.io, conéctalo a tu repo, y selecciona `app.py` como archivo principal.
3. Asegúrate de incluir `requirements.txt`.

## Datos
- Sube un CSV/XLSX con al menos una columna de **fecha** y la columna objetivo (precio de soya).
- En ingeniería de características puedes crear **lags** y **ventanas móviles** y variables de calendario.

## Modelos
- Modelos de regresión: Linear, Ridge, Lasso, RandomForest, GradientBoosting.
- Alternativa de series temporales: **SARIMAX** (statsmodels).

## Reporte
- Genera un reporte `.md` con narrativa automática de resultados y recomendaciones.
