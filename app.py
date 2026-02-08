import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción Ventas Pollo", layout="wide", page_icon="🍗")

# Estilo personalizado con CSS para que se vea más pro
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

model = load_model()

def create_features_row(date):
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'día': [date_pd.day], 'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month], 'trimestre': [date_pd.quarter],
        'año': [date_pd.year], 'díadelaño': [date_pd.dayofyear]
    })

# --- ENCABEZADO ---
st.title("🍗 Proyección de Demanda: Avícola")
st.info("Utiliza inteligencia artificial (XGBoost) para optimizar el inventario basado en ventas históricas.")

# --- SIDEBAR MEJORADO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046769.png", width=100) # Un icono de comida
    st.header("Configuración")
    fecha_sel = st.date_input("📅 Fecha a Predecir", datetime.now())
    
    st.divider()
    st.subheader("Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000, step=500)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500, step=500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800, step=500)
    
    predict_btn = st.button("🚀 Calcular Predicción", use_container_width=True)

# --- CUERPO PRINCIPAL ---
col_stats, col_chart = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("Resultado")
    if predict_btn and model:
        features_df = create_features_row(fecha_sel)
        # Añadir lags
        for col, val in zip(['Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_rolling7'], [lag1, lag7, roll7]):
            features_df[col] = val
        
        # Mantenemos las que faltan por defecto o cálculo simple
        features_df['Ventas_Netas_lag14'] = lag7 * 0.95
        features_df['Ventas_Netas_lag30'] = lag7 * 1.05
        features_df['Ventas_Netas_rolling30'] = roll7 * 0.98

        order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
                'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 
                'Ventas_Netas_lag30', 'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
        
        pred = model.predict(features_df[order])[0]
        
        # Mostrar métricas bonitas
        st.metric(label="Venta Estimada", value=f"${pred:,.2f}", delta=f"{((pred/lag1)-1)*100:.1f}% vs ayer")
        
        with st.expander("Ver detalles técnicos"):
            st.write("Características enviadas al modelo:")
            st.dataframe(features_df[order].T)
    else:
        st.write("Configura los valores y haz clic en predecir.")

with col_chart:
    st.subheader("📈 Análisis de Tendencias")
    
    # Datos simulados más realistas (reemplazar por CSV)
    fechas = pd.date_range(end=datetime.now(), periods=30)
    ventas = np.random.normal(15000, 1500, size=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fechas, y=ventas,
        mode='lines+markers',
        name='Historial',
        line=dict(color='#ff4b4b', width=3),
        fill='tozeroy' # Área rellena para que se vea más moderno
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TABLA DE DATOS ---
st.divider()
st.subheader("📋 Datos Recientes")
df_tabla = pd.DataFrame({'Fecha': fechas, 'Ventas ($)': ventas}).sort_values(by='Fecha', ascending=False)
st.dataframe(df_tabla.head(7), use_container_width=True)
