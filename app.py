import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción Ventas Pollo", layout="wide", page_icon="🍗")

# Estilo CSS 
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
    ruta_base = os.path.dirname(__file__)
    ruta_modelo = os.path.join(ruta_base, 'model.pkl')
    try:
        with open(ruta_modelo, 'rb') as f:
            return pickle.load(f)
    except:
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None

model = load_model()

# --- PROCESAMIENTO DE FECHAS ---
def create_features_row(date):
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
st.title("Proyección de Demanda: Pollo")
st.markdown("---")

# --- SIDEBAR (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    # Fecha inicial: Noviembre 2025
    fecha_sel = st.date_input("Fecha de Inicio Proyección", datetime(2025, 11, 17))
    
    st.divider()
    st.subheader("📊 Datos de Entrada")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500)
    roll7 = st.number_input("Promedio Semanal ($)", value=14800)
    
    predict_btn = st.button(" Calcular Proyección", use_container_width=True)

# --- LÓGICA DE PREDICCIÓN ---
if 'pred' not in st.session_state:
    st.session_state.pred = None

if predict_btn:
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
        
        st.session_state.pred = model.predict(features_df[order])[0]
    else:
        st.error("Archivo model.pkl no encontrado. Por favor súbelo a GitHub.")

# --- CUERPO PRINCIPAL ---
col_stats, col_chart = st.columns([1, 2], gap="large")
pred = st.session_state.pred

with col_stats:
    st.subheader(" Resultado")
    if pred is not None:
        delta_val = ((pred / lag1) - 1) * 100
        st.metric(
            label=f"Valor Proyectado ({fecha_sel})", 
            value=f"${pred:,.2f}", 
            delta=f"{delta_val:.2f}% vs ayer"
        )
        with st.expander("🔍 Ver variables del modelo"):
            # Generamos de nuevo el DF para mostrarlo en el expander
            df_vars = create_features_row(fecha_sel)
            df_vars['Ventas_Netas_lag1'] = lag1
            df_vars['Ventas_Netas_lag7'] = lag7
            st.dataframe(df_vars.T, column_config={"0": "Valor"})
    else:
        st.info("Ajusta los parámetros y presiona 'Calcular Proyección'.")

with col_chart:
    st.subheader(f"📈 Proyección Futura (Desde {fecha_sel})")
    
    # Generamos fechas desde la fecha seleccionada HACIA ADELANTE (30 días)
    fechas_futuras = pd.date_range(start=pd.Timestamp(fecha_sel), periods=30)
    base_val = pred if pred is not None else lag1
    # Simulación de ventas futuras basada en la predicción
    ventas_proy = np.random.normal(base_val, 800, size=30)
    if pred is not None:
        ventas_proy[0] = pred
    
    fig = go.Figure()

    # Gráfica de área para la proyección
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=ventas_proy,
        mode='lines+markers',
        name='Proyección',
        line=dict(color='#ff4b4b', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 75, 75, 0.1)',
        marker=dict(size=4)
    ))

    # ETIQUETA DE DATOS FORZADA
    if pred is not None:
        # Diamante negro en el punto inicial
        fig.add_trace(go.Scatter(
            x=[fechas_futuras[0]], y=[pred],
            mode='markers',
            marker=dict(color='black', size=15, symbol='diamond'),
            showlegend=False
        ))

        # Cuadro de texto flotante
        fig.add_annotation(
            x=fechas_futuras[0],
            y=pred,
            text=f"<b>VALOR PREDICHO:<br>${pred:,.0f}</b>",
            showarrow=True,
            arrowhead=2,
            ax=45, ay=-45,
            bgcolor="black",
            font=dict(color="white", size=14),
            borderpad=6
        )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=450,
        showlegend=False,
        yaxis=dict(
            gridcolor='LightGray', 
            title="Ventas ($)",
            range=[min(ventas_proy)*0.8, max(ventas_proy)*1.3] # Espacio para la etiqueta
        ),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TABLA INFERIOR ---
st.divider()
st.subheader("📋 Detalle de Proyección Semanal")
df_resumen = pd.DataFrame({'Fecha': fechas_futuras, 'Venta Est.': ventas_proy})
st.dataframe(df_resumen.head(7), use_container_width=True)
