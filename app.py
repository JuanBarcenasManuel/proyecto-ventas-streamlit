import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Pollo", layout="wide", page_icon="🍗")

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
    fecha_sel = st.date_input("📅 Fecha de Inicio", datetime(2025, 11, 17))
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    predict_btn = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- LÓGICA DE PREDICCIÓN ---
pred = None
if predict_btn and model:
    features_df = create_features_row(fecha_sel)
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
st.title("🍗 Proyección de Demanda Pollo")

col1, col2 = st.columns([1, 2])

with col1:
    if pred is not None:
        st.metric(label="Resultado Predicho", value=f"${pred:,.2f}")
    else:
        st.info("Presiona el botón para calcular")

with col2:
    # Generamos los datos para la gráfica
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    # Si no hay pred, usamos lag1 para la simulación
    base = pred if pred is not None else lag1
    ventas_proyectadas = np.random.normal(base, 500, size=30)
    if pred is not None: ventas_proyectadas[0] = pred

    fig = go.Figure()

    # Serie de tiempo
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=ventas_proyectadas,
        mode='lines+markers',
        line=dict(color='#ff4b4b'),
        name="Proyección"
    ))

    # Si hay predicción, agregamos el diamante y la ANOTACIÓN FORZADA
    if pred is not None:
        # 1. El diamante
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], y=[pred],
            mode='markers',
            marker=dict(color='black', size=15, symbol='diamond'),
            showlegend=False
        ))

        # 2. LA ANOTACIÓN (El texto que no falla)
        fig.add_annotation(
            x=fechas_futuras[0],
            y=pred,
            text=f"<b>VALOR: ${pred:,.0f}</b>",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40, # Distancia hacia arriba
            bgcolor="black",
            font=dict(color="white", size=14),
            bordercolor="black",
            borderwidth=2,
            borderpad=4,
            opacity=0.9
        )

    fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
