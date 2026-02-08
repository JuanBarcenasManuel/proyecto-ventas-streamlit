import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Ventas Pollo", layout="wide", page_icon="🍗")

# Estilo CSS para mejorar la estética de las tarjetas de métricas
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
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
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# --- PROCESAMIENTO DE FECHAS ---
def create_features_row(date):
    # pd.Timestamp soluciona el error 'AttributeError: datetime.date object has no attribute dayofweek'
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'día': [date_pd.day], 
        'díadelasemana': [date_pd.dayofweek],
        'mes': [date_pd.month], 
        'trimestre': [date_pd.quarter],
        'año': [date_pd.year], 
        'díadelaño': [date_pd.dayofyear]
    })

# --- ENCABEZADO ---
st.title("🍗 Proyección de Demanda: Avícola")
st.markdown("---")

# --- SIDEBAR (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_sel = st.date_input("📅 Fecha a Predecir", datetime.now())
    
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    
    predict_btn = st.button("🚀 Calcular Predicción", use_container_width=True)

# --- LÓGICA DE PREDICCIÓN ---
pred = None
if predict_btn:
    if model:
        features_df = create_features_row(fecha_sel)
        # Asignación de variables según el entrenamiento del modelo
        features_df['Ventas_Netas_lag1'] = lag1
        features_df['Ventas_Netas_lag7'] = lag7
        features_df['Ventas_Netas_lag14'] = lag7 * 0.95
        features_df['Ventas_Netas_lag30'] = lag7 * 1.05
        features_df['Ventas_Netas_rolling7'] = roll7
        features_df['Ventas_Netas_rolling30'] = roll7 * 0.98

        order = ['día', 'díadelasemana', 'mes', 'trimestre', 'año', 'díadelaño',
                'Ventas_Netas_lag1', 'Ventas_Netas_lag7', 'Ventas_Netas_lag14', 
                'Ventas_Netas_lag30', 'Ventas_Netas_rolling7', 'Ventas_Netas_rolling30']
        
        # Realizar la predicción
        pred = model.predict(features_df[order])[0]
    else:
        st.error("Modelo no disponible.")

# --- CUERPO PRINCIPAL ---
col_stats, col_chart = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("🎯 Resultado")
    if pred is not None:
        # Métrica principal con indicador de cambio (delta)
        delta_val = ((pred / lag1) - 1) * 100
        st.metric(
            label=f"Venta Predicha ({fecha_sel})", 
            value=f"${pred:,.2f}", 
            delta=f"{delta_val:.2f}% vs ayer"
        )
        
        with st.expander("🔍 Ver variables del modelo"):
            st.dataframe(features_df[order].T, column_config={"0": "Valor"})
    else:
        st.info("Ajusta los parámetros en el panel izquierdo y presiona 'Calcular Predicción'.")

with col_chart:
    st.subheader("📈 Análisis de Tendencias")
    
    # Datos simulados (Reemplaza con pd.read_csv('tu_archivo.csv') para datos reales)
    fechas_hist = pd.date_range(end=datetime.now(), periods=30)
    ventas_hist = np.random.normal(15000, 1200, size=30)
    
    fig = go.Figure()

    # Gráfica de área para el histórico
    fig.add_trace(go.Scatter(
        x=fechas_hist, y=ventas_hist,
        mode='lines',
        name='Histórico',
        line=dict(color='#ff4b4b', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 75, 75, 0.2)'
    ))

    # ETIQUETA DE DATOS: Si hay predicción, agregar punto destacado con texto
    if pred is not None:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(fecha_sel)], 
            y=[pred],
            mode='markers+text',
            name='Predicción Actual',
            text=[f"Predicción: ${pred:,.0f}"],
            textposition="top center",
            marker=dict(color='black', size=12, symbol='diamond'),
            textfont=dict(size=14, color="black", family="Arial Black")
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=False,
        yaxis=dict(gridcolor='LightGray', title="Ventas ($)"),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TABLA INFERIOR ---
st.divider()
st.subheader("📋 Resumen de datos recientes")
df_resumen = pd.DataFrame({'Fecha': fechas_hist, 'Ventas': ventas_hist}).sort_values(by='Fecha', ascending=False)
st.dataframe(df_resumen.head(5), use_container_width=True)
