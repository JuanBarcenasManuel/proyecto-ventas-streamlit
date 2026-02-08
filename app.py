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
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- FUNCIONES DE PROCESAMIENTO (Iguales a tu script) ---
def create_features_row(date):
    return pd.DataFrame({
        'día': [date.day],
        'díadelasemana': [date.dayofweek],
        'mes': [date.month],
        'trimestre': [date.quarter],
        'año': [date.year],
        'díadelaño': [date.dayofyear]
    })

# --- INTERFAZ DE USUARIO ---
st.title("🍗 Dashboard de Proyección de Ventas: Pollo")
st.markdown("Este modelo utiliza **XGBoost** para predecir las ventas diarias.")

# Sidebar para inputs manuales
st.sidebar.header("Predicción Individual")
fecha_sel = st.sidebar.date_input("Selecciona una fecha", datetime(2025, 11, 1))

# Inputs para los Lags (en una app real, esto vendría de tu última base de datos)
st.sidebar.subheader("Datos Históricos Recientes")
lag1 = st.sidebar.number_input("Ventas ayer (Lag 1)", value=15000)
lag7 = st.sidebar.number_input("Ventas hace 1 semana (Lag 7)", value=14500)
roll7 = st.sidebar.number_input("Promedio última semana", value=14800)

if st.sidebar.button("Predecir"):
    # Crear el vector de características para el modelo
    features_df = create_features_row(fecha_sel)
    features_df['Ventas_Netas_lag1'] = lag1
    features_df['Ventas_Netas_lag7'] = lag7
    features_df['Ventas_Netas_lag14'] = lag7 * 0.9 # Simplificación para el ejemplo
    features_df['Ventas_Netas_lag30'] = lag7 * 1.1
    features_df['Ventas_Netas_rolling7'] = roll7
    features_df['Ventas_Netas_rolling30'] = roll7 * 0.95
    
    # El orden de las columnas debe ser el mismo que en tu entrenamiento
    order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
            'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 'Ventas_Netas_lag30', 
            'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
    
    prediction = model.predict(features_df[order])
    
    st.metric(label=f"Venta Predicha para {fecha_sel}", value=f"${prediction[0]:,.2f}")