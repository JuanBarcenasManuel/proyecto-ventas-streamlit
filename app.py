import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción de Ventas: Pollo", layout="wide")

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Archivo 'model.pkl' no encontrado. Asegúrate de subirlo a tu repo.")
        return None

model = load_model()

# --- FUNCIONES DE PROCESAMIENTO CORREGIDAS ---
def create_features_row(date):
    # Convertimos la fecha de Streamlit a un objeto de Pandas para acceder a sus atributos
    date_pd = pd.Timestamp(date)
    
    return pd.DataFrame({
        'día': [date_pd.day],
        'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month],
        'trimestre': [date_pd.quarter],
        'año': [date_pd.year],
        'díadelaño': [date_pd.dayofyear]
    })

# --- INTERFAZ DE USUARIO ---
st.title("🍗 Dashboard de Proyección de Ventas: Pollo")
st.markdown("Este modelo utiliza **XGBoost** para predecir las ventas diarias.")

# Sidebar para inputs manuales
st.sidebar.header("Predicción Individual")
fecha_sel = st.sidebar.date_input("Selecciona una fecha", datetime(2025, 11, 1))

# Inputs para los Lags
st.sidebar.subheader("Datos Históricos Recientes")
lag1 = st.sidebar.number_input("Ventas ayer (Lag 1)", value=15000)
lag7 = st.sidebar.number_input("Ventas hace 1 semana (Lag 7)", value=14500)
roll7 = st.sidebar.number_input("Promedio última semana", value=14800)

# --- COLUMNAS PARA ORGANIZAR EL DASHBOARD ---
col1, col2 = st.columns([1, 2])

with col1:
    if st.sidebar.button("Predecir"):
        if model is not None:
            # Crear el vector de características
            features_df = create_features_row(fecha_sel)
            features_df['Ventas_Netas_lag1'] = lag1
            features_df['Ventas_Netas_lag7'] = lag7
            features_df['Ventas_Netas_lag14'] = lag7 * 0.9 
            features_df['Ventas_Netas_lag30'] = lag7 * 1.1
            features_df['Ventas_Netas_rolling7'] = roll7
            features_df['Ventas_Netas_rolling30'] = roll7 * 0.95
            
            # Orden de columnas (Asegúrate que coincida con tu entrenamiento)
            order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
                    'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 'Ventas_Netas_lag30', 
                    'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
            
            prediction = model.predict(features_df[order])
            
            st.metric(label=f"Venta Predicha para {fecha_sel}", value=f"${prediction[0]:,.2f}")
        else:
            st.warning("Carga un modelo válido para predecir.")

with col2:
    st.subheader("Serie de Tiempo: Histórico y Tendencia")
    # Generamos datos de ejemplo (Reemplaza esto cargando tu CSV real)
    fechas_hist = pd.date_range(end=datetime.now(), periods=30)
    datos_hist = np.random.randint(13000, 16000, size=30)
    
    df_grafico = pd.DataFrame({'Fecha': fechas_hist, 'Ventas': datos_hist})
    
    # Crear gráfico con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_grafico['Fecha'], y=df_grafico['Ventas'], mode='lines+markers', name='Ventas Históricas'))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
    
    st.plotly_chart(fig, use_container_width=True)
