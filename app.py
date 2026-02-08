import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción Pollo", layout="wide", page_icon="🍗")

# --- 2. CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

# Inicializamos el estado de la predicción si no existe
if 'mi_prediccion' not in st.session_state:
    st.session_state.mi_prediccion = None

def create_features_row(date):
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'día': [date_pd.day], 'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month], 'trimestre': [date_pd.quarter],
        'año': [date_pd.year], 'díadelaño': [date_pd.dayofyear]
    })

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_sel = st.date_input("📅 Fecha de Inicio", datetime(2025, 11, 11))
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    
    if st.button("🚀 Calcular Proyección", use_container_width=True):
        if model:
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
            
            # GUARDAMOS EN EL SESSION STATE
            resultado = model.predict(features_df[order])[0]
            st.session_state.mi_prediccion = resultado
        else:
            st.error("No se encontró el archivo model.pkl")

# --- 4. UI PRINCIPAL ---
st.title("🍗 Dashboard de Proyección: Pollo")

col1, col2 = st.columns([1, 2])

# Recuperamos el valor del estado persistente
pred_final = st.session_state.mi_prediccion

with col1:
    st.subheader("🎯 Resultado")
    if pred_final is not None:
        st.metric(label=f"Predicción {fecha_sel}", value=f"${pred_final:,.2f}")
    else:
        st.info("Presiona el botón para calcular.")

with col2:
    st.subheader("📈 Gráfico de Tendencia")
    
    # Datos de la serie
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    base = pred_final if pred_final is not None else lag1
    
    # Generamos una serie que siempre empiece en el valor de la predicción
    ventas_proy = np.random.normal(base, 600, size=30)
    if pred_final is not None:
        ventas_proy[0] = pred_final

    fig = go.Figure()

    # Línea de tendencia
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=ventas_proy,
        mode='lines+markers',
        line=dict(color='#ff4b4b', width=3),
        name="Proyección"
    ))

    # SOLO DIBUJAMOS LA ETIQUETA SI YA SE CALCULÓ
    if pred_final is not None:
        # El diamante negro
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], y=[pred_final],
            mode='markers',
            marker=dict(color='black', size=15, symbol='diamond'),
            showlegend=False
        ))

        # LA ANOTACIÓN (Burbuja de texto)
        fig.add_annotation(
            x=fechas_futuras[0],
            y=pred_final,
            text=f"<b>VALOR PREDICHO:<br>${pred_final:,.0f}</b>",
            showarrow=True,
            arrowhead=2,
            ax=50, ay=-50,
            bgcolor="black",
            font=dict(color="white", size=14),
            borderpad=6
        )

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(range=[min(ventas_proy)*0.8, max(ventas_proy)*1.3]) # Espacio para la etiqueta
    )
    
    st.plotly_chart(fig, use_container_width=True)
