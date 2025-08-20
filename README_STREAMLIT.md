
# Agrícola – Streamlit Web

App web lista para subir tu CSV y generar pronósticos con distintos modelos (Prophet, SARIMA, SVR, XGBoost, Moving Average, Baseline).

## Cómo ejecutar
```bash
pip install -r requirements_streamlit.txt
streamlit run app_streamlit_agricola.py
```

## CSV esperado
- **Columna de fecha** (por defecto `ds`), puede cambiarse desde la barra lateral.
- **Columna objetivo** (por defecto `y`).

## Características
- Split temporal por "últimas N filas".
- Horizonte futuro configurable.
- Gráficas con Plotly (o Matplotlib si no está disponible).
- Descarga de resultados (CSV) y del modelo (.pkl) cuando aplica.
- Manejo elegante de dependencias opcionales: si un modelo no está instalado, la app lo indica y sugiere `pip install`.

## Nota
Esta app es **independiente del notebook** original y sirve como front-end genérico de forecasting para tus datos. Si deseas enlazarla con tu pipeline existente, puedo integrar llamadas a tus funciones específicas.
