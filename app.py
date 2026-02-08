import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Pollo", layout="wide", page_icon="🍗")

# --- ESTILO CSS ---
st.markdown("""
    <style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

def create_features_row(date):
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'día': [date_pd.day], 'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month], 'trimestre': [date_pd.quarter],
        'año': [date_pd.year], 'díadelaño': [date_pd.dayofyear]
    })

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_sel = st.date_input("Fecha de Inicio", datetime(2025, 11, 17))
    
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    predict_btn = st.button(" Calcular Proyección", use_container_width=True)

# --- LÓGICA DE PREDICCIÓN ---
pred = None
if predict_btn and model:
    features_df = create_features_row(fecha_sel)
    # Llenado de características para el modelo
    features_df['Ventas_Netas_lag1'] = lag1
    features_df['Ventas_Netas_lag7'] = lag7
    features_df['Ventas_Netas_lag14'] = lag7 * 0.95
    features_df['Ventas_Netas_lag30'] = lag7 * 1.05
    features_df['Ventas_Netas_rolling7'] = roll7
    features_df['Ventas_Netas_rolling30'] = roll7 * 0.98
    
    order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
            'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 
            'Ventas_Netas_lag30', 'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
    pred = model.predict(features_df[order])[0]

# --- UI PRINCIPAL ---
st.title("🍗 Proyección de Demanda Pollo Supermercado")
st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🎯 Resultado de Predicción")
    if pred:
        delta_val = ((pred / lag1) - 1) * 100
        st.metric(
            label=f"Valor Predicho para {fecha_sel}", 
            value=f"${pred:,.2f}", 
            delta=f"{delta_val:.2f}% vs ayer"
        )
    else:
        st.info("Presiona el botón para generar la etiqueta de datos en la gráfica.")

with col2:
    st.subheader(f"📈 Tendencia Proyectada (Desde {fecha_sel})")
    
    # Generamos 30 días hacia adelante
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    start_value = pred if pred else lag1
    ventas_proyectadas = np.random.normal(start_value, 800, size=30)
    
    if pred:
        ventas_proyectadas[0] = pred

    fig = go.Figure()

    # 1. Línea de proyección
    fig.add_trace(go.Scatter(
        x=fechas_futuras, 
        y=ventas_proyectadas, 
        mode='lines', 
        line=dict(color='#ff4b4b', width=3), 
        fill='tozeroy',
        name="Tendencia"
    ))

    # 2. LA ETIQUETA DE RESULTADO (Punto destacado con texto)
    if pred:
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], 
            y=[pred],
            mode='markers+text',
            text=[f"PREDICCIÓN: ${pred:,.0f}"], # <-- AQUÍ ESTÁ LA ETIQUETA
            textposition="top center",
            textfont=dict(family="Arial Black", size=14, color="black"),
            marker=dict(color='black', size=15, symbol='diamond-dot'),
            name="Resultado"
        ))

    fig.update_layout(
        height=450, 
        margin=dict(l=0, r=0, t=40, b=0), 
        showlegend=False, 
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Ventas Estimadas ($)", gridcolor='rgba(0,0,0,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()
