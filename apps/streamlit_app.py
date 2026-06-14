import os
# Forzar al SDK de Gemini a usar la API v1 estable antes de importar cualquier módulo de google
os.environ["API_VERSION"] = "v1"
os.environ["GOOGLE_API_VERSION"] = "v1"

import sys
from pathlib import Path
import warnings

# 1. Silenciar advertencias de discrepancia de versiones de scikit-learn - Evitar carteles en la UI
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Añadir la carpeta raiz del proyecto al path para poder importar'apps.*'
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# Forzar la API v1 estable en las llamadas internas de la SDK
try:
    genai.client.API_VERSION = "v1"
except Exception:
    pass
try:
    genai.client.api_version = "v1"
except Exception:
    pass

# Importar las funciones analiticas reales del repositorio del equipo
from apps.utils import load_model_and_preprocessor, predict_from_inputs

# Configuracion visual de la aplicacion
st.set_page_config(
    page_title="Predicción de Intención de Compra - TIF", 
    page_icon="🛍️",
    layout="wide"
)

# Encabezado institucional (Criterio: Audiencia no tecnica)
st.title("🛍️ Sistema de Inteligencia Artificial: Intención de Compra E-commerce")
st.caption("Tecnicatura Universitaria en Ciencia de Datos e IA Aplicada - UPATECO 2026")
st.markdown("""
Esta interfaz web analiza las métricas de comportamiento y navegación de un cliente en tiempo real.
A través del modelo predictivo **XGBoost Classifier**, el sistema estima la probabilidad de que el usuario concrete su compra.
""")
st.divider()

# Carga de artefactos analiticos con manejo seguro de cache
@st.cache_resource(show_spinner="Cargando el núcleo computacional...")
def load_artifacts():
    model, preprocessor = load_model_and_preprocessor()
    return model, preprocessor

model, preprocessor = load_artifacts()

# Función para construir el medidor visual interactivo (Gauge)
def build_indicator_gauge(prob: float, title: str = "Probabilidad"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': "lightcoral"},
                {'range': [35, 65], 'color': "gold"},
                {'range': [65, 100], 'color': "lightgreen"}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Validacion fisica de los archivos serializados (.pkl)
if model is None:
    st.error("⚠️ Error crítico: El archivo del modelo `modelo_ganador.pkl` no pudo ser hallado en la carpeta `models/`.")
else:
    # Organizacion por pestañas (Negocio vs Criterios Tecnicos)
    tab_simulador, tab_metricas, tab_hibrido = st.tabs(["🔮 Simulador de Sesión", "📊 Evaluación del Modelo", "🤖 Arquitectura Híbrida Conversacional"])

    with tab_simulador:
        st.header("Formulario de Comportamiento del Usuario")
        st.write("Complete los siguientes parámetros lógicos observados en la sesión web:")

        col_inputs1, col_inputs2, col_inputs3 = st.columns(3)

        with col_inputs1:
            st.markdown("### 📈 Páginas Consultadas")
            page_values = st.number_input("Valor de la Página (PageValues)", min_value=0.0, value=0.0, help="Promedio del valor económico de las páginas visitadas antes de finalizar la sesión.")
            product_related = st.number_input("Páginas de Productos", min_value=0, value=12, step=1)
            product_related_duration = st.number_input("Tiempo en Productos (seg)", min_value=0.0, value=350.0)

        with col_inputs2:
            st.markdown("### 🚨 Métricas de Deserción")
            exit_rates = st.slider("Tasa de Salida (Exit Rates)", 0.0, 1.0, 0.02, 0.01)
            bounce_rates = st.slider("Tasa de Rebote (Bounce Rates)", 0.0, 1.0, 0.01, 0.01)
            administrative = st.number_input("Páginas Administrativas", min_value=0, value=2, step=1)

        with col_inputs3:
            st.markdown("### 🗓️ Entorno de la Sesión")
            month = st.selectbox("Mes de Navegación", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=10) # Noviembre por estacionalidad comercial
            visitor_type = st.selectbox("Tipo de Visitante", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
            weekend = st.checkbox("¿Ocurre en Fin de Semana?")

        st.markdown("---")

        if st.button("Analizar Intención de Compra", type="primary"):
            try:
                # Estructuracion completa del diccionario con las 17 llaves nativas del dataset
                inputs_dict = {
                    'Administrative': administrative,
                    'Administrative_Duration': 0.0,
                    'Informational': 0,
                    'Informational_Duration': 0.0,
                    'ProductRelated': product_related,
                    'ProductRelated_Duration': product_related_duration,
                    'BounceRates': bounce_rates,
                    'ExitRates': exit_rates,
                    'PageValues': page_values,
                    'SpecialDay': 0.0,
                    'Month': month,
                    'OperatingSystems': 2,  
                    'Browser': 2,           
                    'Region': 1,            
                    'TrafficType': 1,       
                    'VisitorType': visitor_type,
                    'Weekend': weekend
                }

                # SOLUCIÓN DEL ERROR: Invocación en el orden requerido por utils.py -> (model, preprocessor, inputs_dict)
                prediccion, prob_compra = predict_from_inputs(model, preprocessor, inputs_dict)

                # Guardar en session_state para cruzar datos con la pestaña conversacional (RAG)
                st.session_state['last_prediction'] = {
                    'prediccion': prediccion,
                    'prob_compra': prob_compra,
                    'inputs': inputs_dict
                }

                # Presentacion de resultados con valor explicativo comercial (Exigido en las pautas)
                st.subheader("🎯 Diagnóstico del Algoritmo Predictivo")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if prediccion == 1:
                        st.success("### **Predicción: COMPRA**")
                    else:
                        st.error("### **Predicción: NO COMPRA**")
                    
                    st.metric(label="Confianza / Probabilidad de Conversión", value=f"{prob_compra * 100:.2f}%")

                with col_res2:
                    st.write("**Justificación Conceptual de la Decisión:**")
                    if page_values > 0:
                        st.info(f"💡 **Hallazgo Clave:** El usuario visitó secciones con un `PageValues` de {page_values}. La ingeniería de características demuestra que este es el factor estadístico con mayor peso positivo para concretar transacciones.")
                    else:
                        st.warning("⚠️ **Alerta de Abandono:** El valor económico de la página es cero. Sin interacción con productos de alto interés o carritos activos, la tasa de conversión decrece bruscamente.")
                    
                    if exit_rates > 0.04:
                        st.error(f"📉 **Punto de Fuga:** La tasa de salida observada ({exit_rates * 100}%) supera los límites recomendados, indicando desinterés o fricciones de navegación.")
                    
                    if month == "Nov":
                        st.info("🗓️ **Estacionalidad Promocional:** La sesión se ejecuta dentro de Noviembre, mes históricamente potenciado por campañas anuales de descuento masivo.")

                # Sección Gráfica del Gauge
                st.write("")
                col_gauge1, col_gauge2 = st.columns([1, 1])
                with col_gauge1:
                    st.plotly_chart(build_indicator_gauge(prob_compra, title="Nivel de Intención de Compra"), use_container_width=True)
                with col_gauge2:
                    st.markdown("""
                    **Segmentación de Acciones de Negocio Recomendadas:**
                    * 🔴 **Baja Conversión (0% - 35%):** Navegación casual. No invasiva. Se recomienda retargeting por e-mail a las 24 horas.
                    * 🟡 **Intención Media (35% - 65%):** El usuario evalúa alternativas. Activar un banner interactivo con envío gratis o cupones de descuento por tiempo limitado.
                    * 🟢 **Conversión Inminente (65% - 100%):** Cliente decidido. Optimizar el checkout reduciendo campos de pago para evitar la fricción final.
                    """)

            except Exception as ex:
                st.error(f"Ocurrió un error al procesar la inferencia en el pipeline: {ex}.")

    with tab_metricas:
        st.header("Métricas de Rendimiento y Calidad (XGBoost Classifier)")
        st.write("Estadísticas extraídas del conjunto de prueba independiente para validar el sistema:")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label="Exactitud (Accuracy Global)", value="88.20 %")
        c2.metric(label="Área Bajo la Curva (ROC-AUC)", value="0.9205")
        c3.metric(label="Sensibilidad (Recall - Captura de Compras)", value="72.51 %")
        c4.metric(label="Puntuación F1 (F1-Score)", value="0.6580")
        
        st.markdown("---")
        
        # Carga dinámica del gráfico de importancia si existe en models/
        im_path = root / 'models' / 'feature_importance_xgboost.png'
        if pprocessor_path := root / 'models' / 'feature_importance_xgboost.png':
            if im_path.exists():
                st.subheader("📊 Gráfico de Importancia Estructural de Características")
                st.image(str(im_path), caption="Importancia de atributos calculada nativamente por el core de XGBoost.", use_container_width=True)

    # Configurar barra lateral para la API Key de Gemini de forma segura
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Configuración del Agente IA")
    
    # Intentar obtener la API key de los secretos de Streamlit de forma segura
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_api_key = None

    # Si no está en secretos, habilitar el campo manual en el sidebar
    if not gemini_api_key:
        if 'gemini_api_key' in st.session_state:
            gemini_api_key = st.session_state['gemini_api_key']
        else:
            user_key = st.sidebar.text_input("Ingresa tu Gemini API Key:", type="password", help="Obtenela gratis en Google AI Studio")
            if user_key:
                gemini_api_key = user_key
                st.session_state['gemini_api_key'] = user_key
                st.rerun()

    if gemini_api_key:
        st.sidebar.success("✔️ API Key de Gemini configurada.")
    else:
        st.sidebar.info("💡 Para usar el chat con IA conversacional, ingresa tu API Key.")

    with tab_hibrido:
        st.header("🤖 Ecosistema Híbrido Conversacional RAG")
        st.write("Consulta al Agente Inteligente de E-commerce. Responde en tiempo real basándose en las políticas comerciales y las predicciones.")

        if not gemini_api_key:
            st.warning("⚠️ Se requiere una API Key de Gemini para activar el chat. Por favor, ingresala en el campo correspondiente de la barra lateral.")
            
            # Mostrar la explicación conceptual del bloque 3 si no está la API Key
            st.markdown("""
            ---
            ### 📋 Concepto: Arquitectura Híbrida de IA
            En sintonía con las directivas de la materia, esta pestaña implementa una **Arquitectura RAG (Retrieval-Augmented Generation)**:
            1. **Conexión de Datos:** La interfaz captura los datos no estructurados de la empresa (políticas de descuento y marketing en `data/politicas_comerciales.txt`).
            2. **Luz sobre el Contexto:** El LLM (*Gemini 1.5 Flash*) absorbe el contexto normativo y los datos probabilísticos del modelo **XGBoost** para dar respuestas comerciales precisas y no alucinadas.
            """)
        else:
            import os
            os.environ["API_VERSION"] = "v1"
            try:
                genai.configure(api_key=gemini_api_key)
            except Exception as e:
                st.error(f"Error de configuración de la API de Gemini: {e}")
                st.stop()

            # Intentar cargar base de conocimientos
            politicas_path = root / 'data' / 'politicas_comerciales.txt'
            politicas_content = ""
            if politicas_path.exists():
                try:
                    with open(politicas_path, 'r', encoding='utf-8') as f:
                        politicas_content = f.read()
                except Exception as e:
                    st.warning(f"Error al cargar las políticas comerciales: {e}")
            else:
                st.warning("Archivo de políticas comerciales `data/politicas_comerciales.txt` no encontrado.")

            # Mostrar estado de los datos del simulador cruzados
            if 'last_prediction' in st.session_state:
                lp = st.session_state['last_prediction']
                pred_label = "COMPRA" if lp['prediccion'] == 1 else "NO COMPRA"
                st.success(f"📊 **Contexto del Simulador Activo:** Cargados datos de predicción ({pred_label} con {lp['prob_compra'] * 100:.2f}% de probabilidad). El Agente usará esta información en tus preguntas.")
            else:
                st.info("💡 **Consejo:** Hacé una corrida de análisis en la pestaña *Simulador de Sesión* para que el agente converse sobre el perfil de ese cliente en particular.")

            # Inicializar historial del chat
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Mostrar historial de chat
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input de chat
            if prompt := st.chat_input("Preguntale al agente (ej. '¿Qué campaña aplico al cliente de la simulación?')"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Armar el contexto integrado (RAG + XGBoost)
                context = f"""
                Actúas como un asistente analítico e inteligente especializado en E-commerce y Marketing digital.
                Tu objetivo es asesorar al equipo comercial utilizando la información provista.
                
                --- INICIO BASE DE CONOCIMIENTOS DE LA EMPRESA (RAG) ---
                {politicas_content}
                --- FIN BASE DE CONOCIMIENTOS ---
                
                """

                if 'last_prediction' in st.session_state:
                    lp = st.session_state['last_prediction']
                    pred_label = "COMPRA" if lp['prediccion'] == 1 else "NO COMPRA"
                    context += f"""
                    --- RESULTADOS DE LA ÚLTIMA SIMULACIÓN (XGBOOST) ---
                    - Clasificación del Modelo: El cliente tiene intención de {pred_label}.
                    - Probabilidad de Conversión de Compra: {lp['prob_compra'] * 100:.2f}%
                    - Parámetros detallados de la navegación del cliente:
                      * Valor de la Página (PageValues): {lp['inputs']['PageValues']}
                      * Páginas de Productos visitadas (ProductRelated): {lp['inputs']['ProductRelated']}
                      * Tiempo en Productos en segundos (ProductRelated_Duration): {lp['inputs']['ProductRelated_Duration']}
                      * Tasa de Salida (ExitRates): {lp['inputs']['ExitRates']}
                      * Tasa de Rebote (BounceRates): {lp['inputs']['BounceRates']}
                      * Mes de la Sesión (Month): {lp['inputs']['Month']}
                      * Tipo de Visitante (VisitorType): {lp['inputs']['VisitorType']}
                      * Ocurre en Fin de Semana (Weekend): {lp['inputs']['Weekend']}
                    ----------------------------------------------------
                    """
                else:
                    context += "\n[Nota: Aún no se ha ejecutado ninguna predicción en el simulador para esta sesión.]\n"

                context += """
                INSTRUCCIONES DE RESPUESTA:
                1. Responde con un tono profesional, experto y comercial, muy claro.
                2. Usa la Base de Conocimientos para justificar tus respuestas. Si la consulta se refiere a qué descuento o campaña aplicar, busca en las políticas y asociala al perfil del cliente.
                3. Si hay datos de predicción disponibles, utilízalos para dar un diagnóstico personalizado y citá la probabilidad estimada por el modelo.
                4. Mantén las respuestas fluidas y directas en español.
                
                --- HISTORIAL DE LA CONVERSACIÓN ---
                """

                # Incluir el historial de mensajes anteriores en el contexto
                for msg in st.session_state.messages[:-1]:
                    role_label = "Usuario" if msg["role"] == "user" else "Asistente"
                    context += f"{role_label}: {msg['content']}\n"

                context += f"\nUsuario: {prompt}\nAsistente:"

                try:
                    import requests
                    with st.spinner("Procesando consulta..."):
                        # Llamada directa HTTP a la API v1 de Gemini para evitar bugs de la SDK vieja
                        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
                        headers = {
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "contents": [
                                {
                                    "parts": [
                                        {"text": context}
                                    ]
                                }
                            ]
                        }
                        
                        response = requests.post(url, headers=headers, json=payload)
                        if response.status_code == 200:
                            response_json = response.json()
                            response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                        else:
                            try:
                                error_msg = response.json()['error']['message']
                            except Exception:
                                error_msg = response.text
                            raise Exception(f"HTTP {response.status_code}: {error_msg}")

                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"Error al conectar con Gemini API: {e}")