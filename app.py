import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Ventas Pollo", layout="wide", page_icon="🍗")

# --- 2. CARGAR MODELO (CON RUTA ABSOLUTA) ---
@st.cache_resource
def load_model():
    # Buscamos la carpeta donde está este archivo app.py
    ruta_base = os.path.dirname(__file__)
    # Unimos la carpeta con el nombre del archivo
    ruta_modelo = os.path.join(ruta_base, 'model.pkl')
    
    try:
        with open(ruta_modelo, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Si falla, intentamos la carga simple por si acaso
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None
    except Exception as e:
        return None

model = load_model()

# Inicializar memoria de la predicción
if 'pred_valor' not in st.session_state:
    st.session_state.pred_valor = None

def create_features_row(date):
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'día': [date_pd.day], 'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month], 'trimestre': [date_pd.quarter],
        'año': [date_pd.year], 'díadelaño': [date_pd.dayofyear]
    })

# --- 3. SIDEBAR (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_sel = st.date_input("📅 Fecha de Inicio", datetime(2025, 11, 11))
    
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    
    if st.button("🚀 Calcular Proyección", use_container_width=True):
        if model is not None:
            features_df = create_features_row(fecha_sel)
            # Preparar datos para el modelo
            features_df['Ventas_Netas_lag1'] = lag1
            features_df['Ventas_Netas_lag7'] = lag7
            features_df['Ventas_Netas_lag14'] = lag7 * 0.95
            features_df['Ventas_Netas_lag30'] = lag7 * 1.05
            features_df['Ventas_Netas_rolling7'] = roll7
            features_df['Ventas_Netas_rolling30'] = roll7 * 0.98
            
            order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
                    'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 
                    'Ventas_Netas_lag30', 'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
            
            # Realizar y guardar predicción
            res = model.predict(features_df[order])[0]
            st.session_state.pred_valor = res
        else:
            st.error("⚠️ Error: El archivo 'model.pkl' no se pudo cargar. Revisa que esté en la raíz de tu GitHub.")

# --- 4. CUERPO PRINCIPAL ---
st.title("🍗 Dashboard de Proyección: Pollo")

if model is None:
    st.warning("⚠️ El modelo no está cargado. Asegúrate de que 'model.pkl' esté en tu repositorio de GitHub.")

col1, col2 = st.columns([1, 2], gap="large")

pred_actual = st.session_state.pred_valor

with col1:
    st.subheader("🎯 Resultado")
    if pred_actual is not None:
        st.metric(label=f"Predicción para {fecha_sel}", value=f"${pred_actual:,.2f}")
    else:
        st.info("Configura los datos y presiona el botón.")

with col2:
    st.subheader("📈 Gráfico de Tendencia")
    
    # Datos para graficar
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    inicio_y = pred_actual if pred_actual is not None else lag1
    ventas_y = np.random.normal(inicio_y, 600, size=30)
    if pred_actual is not None:
        ventas_y[0] = pred_actual

    fig = go.Figure()

    # Línea principal
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=ventas_y,
        mode='lines+markers',
        line=dict(color='#ff4b4b', width=3),
        name="Proyección"
    ))

    # ETIQUETA FORZADA
    if pred_actual is not None:
        # Diamante
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], y=[pred_actual],
            mode='markers',
            marker=dict(color='black', size=15, symbol='diamond'),
            showlegend=False
        ))

        # Cuadro de texto (Anotación)
        fig.add_annotation(
            x=fechas_futuras[0],
            y=pred_actual,
            text=f"<b>VALOR PREDICHO:<br>${pred_actual:,.0f}</b>",
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
        yaxis=dict(range=[min(ventas_y)*0.8, max(ventas_y)*1.3])
    )
    
    st.plotly_chart(fig, use_container_width=True)
