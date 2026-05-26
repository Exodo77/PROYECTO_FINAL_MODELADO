import sys
from pathlib import Path

# Añadir la carpeta raíz del proyecto al path para poder importar `apps.*`
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from apps.utils import load_model_and_preprocessor, predict_from_inputs, prepare_input_dataframe


st.set_page_config(page_title="Demo: Probabilidad de Compra", layout="wide")


@st.cache_data(show_spinner=False)
def load_artifacts():
    model, preprocessor = load_model_and_preprocessor()
    return model, preprocessor


def build_indicator_gauge(prob: float, title: str = "Probabilidad"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "lightcoral"},
                {'range': [35, 65], 'color': "gold"},
                {'range': [65, 100], 'color': "lightgreen"}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main():
    st.title("Demo de Probabilidad de Compra")
    st.markdown("Ajusta las entradas a la izquierda y usa los botones para predecir, muestrear o inspeccionar categorías.")

    model, preprocessor = load_artifacts()
    if model is None:
        st.warning("No se encontró el modelo en `models/modelo_ganador.pkl`. La predicción no funcionará hasta que exista.")

    # If there are pending sampled values from a previous click, apply them
    pending_map = {
        'pending_Administrative_Duration': 'Administrative_Duration',
        'pending_PageValues': 'PageValues',
        'pending_ProductRelated_Duration': 'ProductRelated_Duration',
        'pending_Browser': 'Browser',
        'pending_TrafficType': 'TrafficType',
        'pending_Month': 'Month',
    }
    for pkey, wkey in pending_map.items():
        if pkey in st.session_state:
            st.session_state[wkey] = st.session_state.pop(pkey)

    def _try_rerun():
        rerun_fn = getattr(st, "rerun", None)
        if callable(rerun_fn):
            rerun_fn()
            return
        experimental_rerun_fn = getattr(st, "experimental_rerun", None)
        if callable(experimental_rerun_fn):
            experimental_rerun_fn()
            return
        # last resort: change query params to trigger a rerun on supported versions
        import time
        try:
            st.experimental_set_query_params(_rerun=int(time.time()))
        except Exception:
            # give up silently; Streamlit will rerun on next user interaction
            pass

    # Sidebar controls
    st.sidebar.header("Controles")
    with st.sidebar.form("inputs_form"):
        st.subheader("Entradas del usuario")
        Administrative_Duration = st.number_input('Duración en páginas administrativas', min_value=0.0, value=0.0, step=1.0, format="%.1f", key="Administrative_Duration")
        PageValues = st.number_input('Valor de Pagina(PageValues)', min_value=0.0, value=1.0, step=0.5, format="%.2f", key="PageValues")
        ProductRelated_Duration = st.number_input('Duración en pagina de producto(segundos)', min_value=0.0, value=50.0, step=1.0, format="%.1f", key="ProductRelated_Duration")
        Browser = st.selectbox('Browser', options=list(range(1, 14)), index=11, key="Browser")
        TrafficType = st.selectbox('Tipo de tráfico', options=list(range(1, 21)), index=0, key="TrafficType")
        Month = st.selectbox('Mes', options=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], index=0, key="Month")
        # Add any other quick inputs here if desired

        # Action buttons inside the form
        predict_btn = st.form_submit_button('Predecir')
        sample_btn = st.form_submit_button('Muestra aleatoria')
        reset_btn = st.form_submit_button('Restablecer valores')

    # handle reset
    if reset_btn:
        _try_rerun()

    # handle sampling
    if sample_btn:
        # sample plausible values (simple heuristic)
        # store them in "pending_..." keys so we can transfer them before widgets are instantiated
        st.session_state["pending_Administrative_Duration"] = float(np.random.uniform(0, 300))
        st.session_state["pending_PageValues"] = round(float(np.random.uniform(0, 10)), 2)
        st.session_state["pending_ProductRelated_Duration"] = float(np.random.uniform(0, 300))
        st.session_state["pending_Browser"] = int(np.random.randint(1, 14))
        st.session_state["pending_TrafficType"] = int(np.random.randint(1, 21))
        st.session_state["pending_Month"] = np.random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        _try_rerun()

    if predict_btn:
        inputs = {
            'Administrative_Duration': st.session_state.get('Administrative_Duration', Administrative_Duration),
            'PageValues': st.session_state.get('PageValues', PageValues),
            'ProductRelated_Duration': st.session_state.get('ProductRelated_Duration', ProductRelated_Duration),
            'Browser': st.session_state.get('Browser', Browser),
            'TrafficType': st.session_state.get('TrafficType', TrafficType),
            'Month': st.session_state.get('Month', Month),
        }

        if model is None:
            st.error('No hay modelo cargado. Coloca `models/modelo_ganador.pkl` y `models/preprocessor.pkl` si quieres resultados reales.')
            st.stop()

        prob, pred = predict_from_inputs(inputs, model, preprocessor)

        # Layout: left = inputs summary, center = gauge
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Resumen de entradas")
            st.write(pd.DataFrame([inputs]).T.rename(columns={0: "Valor"}))
            st.markdown("**Segmento estimado:** " + ("Alta" if prob > 0.65 else ("Media" if prob >= 0.35 else "Baja")))
            st.markdown("**Recomendación:**")
            if prob > 0.65:
                st.info(
                    """Reducir fricciones en checkout y optimizar el flujo de compra.

Activar remarketing agresivo y ofertas personalizadas de upselling.

Priorizar inversión publicitaria y recursos comerciales en este segmento."""
                )
            elif prob >= 0.35:
                st.info(
                    """Implementar campañas de urgencia con descuentos moderados y tiempo limitado.

Reforzar confianza con testimonios, garantías de devolución y prueba social.

Aplicar remarketing selectivo y cupones personalizados para impulsar conversión."""
                )
            else:
                st.info(
                    """Nutrir con contenido educativo y remarketing de bajo costo.

Capturar para newsletter y campañas de reactivación a largo plazo.

Reasignar presupuesto a segmentos medio/alto si no responden tras 2-3 interacciones."""
                )

        with col2:
            st.subheader("Probabilidad (gauge)")
            gauge = build_indicator_gauge(prob, title="Probabilidad de Compra (%)")
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(
                """
                **Leyenda de segmentos**

                - <span style='color:#d9534f; font-weight:600;'>Baja</span>: menor a 35%
                - <span style='color:#f0ad4e; font-weight:600;'>Media</span>: de 35% a 65%
                - <span style='color:#5cb85c; font-weight:600;'>Alta</span>: mayor a 65%
                """,
                unsafe_allow_html=True,
            )

        # show feature importance image if available
        im_path = Path(__file__).resolve().parent.parent / 'models' / 'feature_importance_xgboost.png'
        if im_path.exists():
            st.markdown("---")
            st.image(str(im_path), caption='Importancia de features (XGBoost)', use_container_width=True)


if __name__ == '__main__':
    main()
