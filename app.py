"""Aplicación Streamlit para priorización preventiva mensual en Candelaria, Valle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import classification_features, regression_features

st.set_page_config(page_title="Candelaria Vial", page_icon="🚦", layout="wide")
ARTIFACT_DIR = Path("artifacts")
REG_MODEL = ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib"
CLF_MODEL = ARTIFACT_DIR / "modelo_alerta_alta_siniestralidad.joblib"
METRICS = ARTIFACT_DIR / "metricas.json"


def artifacts_ready() -> bool:
    return REG_MODEL.exists() and CLF_MODEL.exists() and METRICS.exists()


@st.cache_resource(show_spinner=False)
def ensure_artifacts() -> None:
    """Entrena una sola vez si el repositorio aún no trae artefactos."""
    if artifacts_ready():
        return
    from src.train import main

    main()


@st.cache_resource(show_spinner=False)
def load_models():
    return (
        joblib.load(REG_MODEL),
        joblib.load(CLF_MODEL),
        json.loads(METRICS.read_text(encoding="utf-8")),
    )


def recommended_actions(high_risk: bool, expected_count: float) -> list[str]:
    if high_risk:
        return [
            "Priorizar revisión de señalización, iluminación y puntos de conflicto en el corregimiento.",
            "Considerar controles o acciones preventivas durante el mes, sujetos a validación en terreno.",
            "Contrastar la alerta con conocimiento técnico local antes de asignar recursos.",
        ]
    return [
        "Mantener monitoreo mensual y contrastar la estimación con reportes locales.",
        f"La regresión estima {expected_count:.2f} siniestros; una alerta baja no significa ausencia de riesgo.",
        "Validar siempre en terreno antes de intervenir.",
    ]


st.title("🚦 Candelaria Vial")
st.caption(
    "Apoyo a la priorización preventiva para la Secretaría de Tránsito y Transporte de Candelaria, Valle."
)

if not artifacts_ready():
    with st.spinner("Primera ejecución: descargando datos y entrenando los modelos del proyecto..."):
        try:
            ensure_artifacts()
        except Exception as exc:
            st.error(
                "No fue posible preparar los modelos automáticamente. "
                "Si la descarga pública está bloqueada, agrega el CSV como "
                "`data/raw/accidentalidad_candelaria.csv` y vuelve a desplegar."
            )
            st.exception(exc)
            st.stop()

monthly_model, alert_model, metadata = load_models()

st.info(
    f"Fuente: {metadata['registros_incidentes']:,} registros entre "
    f"{metadata['fecha_minima']} y {metadata['fecha_maxima']}. "
    f"El año de prueba temporal es {metadata['anio_prueba']}."
)

st.subheader("Prioriza el próximo mes")
col_a, col_b = st.columns(2)
with col_a:
    corregimiento = st.selectbox("Corregimiento", metadata["corregimientos"])
    target_date = st.date_input("Mes que se quiere priorizar", value=pd.Timestamp("2026-01-01"))
with col_b:
    previous = st.number_input(
        "Siniestros registrados el mes anterior", min_value=0, value=0, step=1
    )
    rolling = st.number_input(
        "Promedio de siniestros de los tres meses previos",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )

target_month = pd.Timestamp(target_date).month
row = pd.DataFrame(
    {
        "corregimiento": [corregimiento],
        "anio": [pd.Timestamp(target_date).year],
        "mes": [target_month],
        "mes_seno": [np.sin(2 * np.pi * target_month / 12)],
        "mes_coseno": [np.cos(2 * np.pi * target_month / 12)],
        "siniestros_lag_1": [previous],
        "promedio_3_meses_previo": [rolling],
    }
)

reg_cat, reg_num = regression_features()
clf_cat, clf_num = classification_features()
expected_count = max(
    0.0, float(monthly_model.predict(row[reg_cat + reg_num])[0])
)
high_probability = float(alert_model.predict_proba(row[clf_cat + clf_num])[:, 1][0])
high_label = high_probability >= 0.5

metric_a, metric_b = st.columns(2)
metric_a.metric("Siniestros estimados", f"{expected_count:.2f}")
metric_b.metric(
    "Alerta de alta siniestralidad",
    "ALTA" if high_label else "NO ALTA",
    help=f"Probabilidad estimada por el clasificador: {high_probability:.1%}",
)

st.caption(
    f"En esta versión, 'alta siniestralidad' significa "
    f"{metadata['umbral_alta_siniestralidad']} o más siniestros en un mes. "
    "El umbral se calcula únicamente con el periodo de entrenamiento."
)

st.subheader("Acciones sugeridas")
for action in recommended_actions(high_label, expected_count):
    st.write(f"- {action}")

with st.expander("Cómo interpretar los resultados"):
    st.write(
        "La regresión estima cuántos siniestros pueden registrarse el próximo mes por corregimiento. "
        "La clasificación responde si ese mes se parece a un escenario de alta siniestralidad. "
        "Ambos modelos trabajan a nivel corregimiento-mes y usan calendario e información de meses previos. "
        "No identifican causas ni predicen el comportamiento de personas específicas."
    )

st.subheader("Transparencia del modelo")
left, right = st.columns(2)
with left:
    st.caption(f"Regresión seleccionada: {metadata['modelo_regresion_seleccionado']}")
    reg_factors = pd.DataFrame(metadata.get("factores_importantes_regresion", []))
    if reg_factors.empty:
        st.write("El modelo seleccionado no expone importancias globales comparables.")
    else:
        st.dataframe(reg_factors, hide_index=True, use_container_width=True)
with right:
    st.caption(f"Clasificación seleccionada: {metadata['modelo_clasificacion_seleccionado']}")
    clf_factors = pd.DataFrame(metadata.get("factores_importantes_clasificacion", []))
    if clf_factors.empty:
        st.write("El modelo seleccionado no expone importancias globales comparables.")
    else:
        st.dataframe(clf_factors, hide_index=True, use_container_width=True)

st.warning(
    "Los registros son administrativos y pueden tener subregistro o cambios de reporte. "
    "La aplicación prioriza revisión humana; no debe usarse para sancionar personas, atribuir culpa "
    "ni sustituir una inspección técnica."
)
