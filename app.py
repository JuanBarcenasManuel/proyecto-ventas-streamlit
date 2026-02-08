import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
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
    fecha_sel = st.date_input("📅 Fecha de Inicio", datetime(2025, 11, 11))
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    predict_btn = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- LÓGICA DE PREDICCIÓN ---
pred = None
# IMPORTANTE: Guardamos el estado de la predicción para que no se borre al refrescar
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
        st.metric(label=f"Predicción para {fecha_sel}", value=f"${pred:,.2f}")
    else:
        st.info("Haz clic en 'Calcular Proyección' para ver el valor en el gráfico.")

with col2:
    # Generar datos
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    # Simulación de tendencia
    base = pred if pred is not None else lag1
    ventas_proy = np.random.normal(base, 500, size=30)
    if pred is not None: ventas_proy[0] = pred

    fig = go.Figure()

    # Gráfico de línea con puntos
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=ventas_proy,
        mode='lines+markers',
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=6),
        name="Tendencia"
    ))

    # ETIQUETA FORZADA (Solo si hay predicción)
    if pred is not None:
        # Añadimos el punto diamante negro
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], y=[pred],
            mode='markers',
            marker=dict(color='black', size=14, symbol='diamond'),
            showlegend=False
        ))

        # Añadimos la anotación con flecha y caja de texto
        fig.add_annotation(
            x=fechas_futuras[0],
            y=pred,
            text=f"VALOR PREDICHO:<br><b>${pred:,.0f}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40, # Mover flecha a la derecha
            ay=-50, # Mover flecha hacia arriba
            bgcolor="rgba(0,0,0,0.8)",
            font=dict(color="white", size=14),
            bordercolor="black",
            borderwidth=2,
            borderpad=6,
            align="center"
        )

    # Ajustamos márgenes para que la etiqueta no se corte
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Ventas ($)",
            gridcolor='rgba(0,0,0,0.1)',
            range=[min(ventas_proy)*0.9, max(ventas_proy)*1.2] # Damos espacio arriba para la etiqueta
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
