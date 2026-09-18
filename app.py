"""Aplicación Streamlit para priorización preventiva mensual en Candelaria, Valle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import classification_features, regression_features


st.set_page_config(
    page_title="Candelaria Vial",
    page_icon="🚦",
    layout="wide",
)


ARTIFACT_DIR = Path("artifacts")
REG_MODEL = ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib"
CLF_MODEL = ARTIFACT_DIR / "modelo_alerta_alta_siniestralidad.joblib"
METRICS = ARTIFACT_DIR / "metricas.json"


# ============================================================
# CONFIGURACIÓN DEL UMBRAL
# ============================================================

UMBRAL_ALTA_SINIESTRALIDAD = 5


# ============================================================
# FUNCIONES
# ============================================================

def artifacts_ready() -> bool:
    return (
        REG_MODEL.exists()
        and CLF_MODEL.exists()
        and METRICS.exists()
    )


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
        json.loads(
            METRICS.read_text(
                encoding="utf-8"
            )
        ),
    )


def recommended_actions(
    high_risk: bool,
    expected_count: float,
) -> list[str]:

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


# ============================================================
# TÍTULO
# ============================================================

st.title("🚦 Candelaria Vial")

st.caption(
    "Apoyo a la priorización preventiva para la Secretaría de Tránsito y Transporte de Candelaria, Valle."
)


# ============================================================
# PREPARACIÓN DE MODELOS
# ============================================================

if not artifacts_ready():

    with st.spinner(
        "Primera ejecución: descargando datos y entrenando los modelos del proyecto..."
    ):

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


# ============================================================
# INFORMACIÓN DE LOS DATOS
# ============================================================

st.info(
    f"Fuente: {metadata['registros_incidentes']:,} registros entre "
    f"{metadata['fecha_minima']} y {metadata['fecha_maxima']}. "
    f"El año de prueba temporal es {metadata['anio_prueba']}."
)


# ============================================================
# ENTRADAS
# ============================================================

st.subheader("Prioriza el próximo mes")


col_a, col_b = st.columns(2)


# ------------------------------------------------------------
# COLUMNA A
# ------------------------------------------------------------

with col_a:

    corregimiento = st.selectbox(
        "Corregimiento",
        metadata["corregimientos"],
    )

    anio_prediccion = st.number_input(
        "Año que se quiere priorizar",
        min_value=2020,
        max_value=2100,
        value=2026,
        step=1,
    )

    meses = {
        1: "Enero",
        2: "Febrero",
        3: "Marzo",
        4: "Abril",
        5: "Mayo",
        6: "Junio",
        7: "Julio",
        8: "Agosto",
        9: "Septiembre",
        10: "Octubre",
        11: "Noviembre",
        12: "Diciembre",
    }

    mes_nombre = st.selectbox(
        "Mes que se quiere priorizar",
        list(meses.values()),
    )

    target_month = next(
        numero
        for numero, nombre in meses.items()
        if nombre == mes_nombre
    )


# ------------------------------------------------------------
# COLUMNA B
# ------------------------------------------------------------

with col_b:

    previous = st.number_input(
        "Siniestros registrados el mes anterior",
        min_value=0,
        value=0,
        step=1,
    )

    rolling = st.number_input(
        "Promedio de siniestros de los tres meses previos",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )


# ============================================================
# CREACIÓN DE LA FILA PARA EL MODELO
# ============================================================

row = pd.DataFrame(
    {
        "corregimiento": [corregimiento],
        "anio": [anio_prediccion],
        "mes": [target_month],
        "mes_seno": [
            np.sin(
                2 * np.pi * target_month / 12
            )
        ],
        "mes_coseno": [
            np.cos(
                2 * np.pi * target_month / 12
            )
        ],
        "siniestros_lag_1": [previous],
        "promedio_3_meses_previo": [rolling],
    }
)


# ============================================================
# CARACTERÍSTICAS DE LOS MODELOS
# ============================================================

reg_cat, reg_num = regression_features()

clf_cat, clf_num = classification_features()


# ============================================================
# PREDICCIÓN DE SINIESTROS
# ============================================================

expected_count = max(
    0.0,
    float(
        monthly_model.predict(
            row[reg_cat + reg_num]
        )[0]
    ),
)


# ============================================================
# PROBABILIDAD DEL CLASIFICADOR
# ============================================================

high_probability = float(
    alert_model.predict_proba(
        row[clf_cat + clf_num]
    )[:, 1][0]
)


# ============================================================
# ALERTA DE ALTA SINIESTRALIDAD
# ============================================================
#
# Umbral:
#
# Menos de 5  -> NO ALTA
# 5 o más      -> ALTA
#
# ============================================================

high_label = (
    expected_count >= UMBRAL_ALTA_SINIESTRALIDAD
)


# ============================================================
# RESULTADOS
# ============================================================

metric_a, metric_b = st.columns(2)


metric_a.metric(
    "Siniestros estimados",
    f"{expected_count:.2f}",
)


metric_b.metric(
    "Alerta de alta siniestralidad",
    "ALTA" if high_label else "NO ALTA",
    help=(
        f"Estimación mensual: {expected_count:.2f} siniestros. "
        f"Umbral de alta siniestralidad: "
        f"{UMBRAL_ALTA_SINIESTRALIDAD} siniestros."
    ),
)


# ============================================================
# EXPLICACIÓN DEL UMBRAL
# ============================================================

st.caption(
    f"Para esta aplicación, se considera alta siniestralidad "
    f"cuando la estimación mensual es igual o superior a "
    f"{UMBRAL_ALTA_SINIESTRALIDAD} siniestros."
)


# ============================================================
# ACCIONES SUGERIDAS
# ============================================================

st.subheader("Acciones sugeridas")


for action in recommended_actions(
    high_label,
    expected_count,
):

    st.write(
        f"- {action}"
    )


# ============================================================
# INTERPRETACIÓN
# ============================================================

with st.expander(
    "Cómo interpretar los resultados"
):

    st.write(
        "La regresión estima cuántos siniestros pueden registrarse "
        "en el mes seleccionado por corregimiento. "
        "La alerta de alta siniestralidad se activa cuando la "
        f"estimación alcanza o supera {UMBRAL_ALTA_SINIESTRALIDAD} "
        "siniestros. "
        "El modelo trabaja a nivel corregimiento-mes y utiliza "
        "información del calendario y de meses previos. "
        "No identifica causas ni predice el comportamiento de "
        "personas específicas."
    )


# ============================================================
# TRANSPARENCIA DEL MODELO
# ============================================================

st.subheader(
    "Transparencia del modelo"
)


left, right = st.columns(2)


# ------------------------------------------------------------
# MODELO DE REGRESIÓN
# ------------------------------------------------------------

with left:

    st.caption(
        f"Regresión seleccionada: "
        f"{metadata['modelo_regresion_seleccionado']}"
    )

    reg_factors = pd.DataFrame(
        metadata.get(
            "factores_importantes_regresion",
            [],
        )
    )

    if reg_factors.empty:

        st.write(
            "El modelo seleccionado no expone "
            "importancias globales comparables."
        )

    else:

        st.dataframe(
            reg_factors,
            hide_index=True,
            use_container_width=True,
        )


# ------------------------------------------------------------
# MODELO DE CLASIFICACIÓN
# ------------------------------------------------------------

with right:

    st.caption(
        f"Clasificación seleccionada: "
        f"{metadata['modelo_clasificacion_seleccionado']}"
    )

    clf_factors = pd.DataFrame(
        metadata.get(
            "factores_importantes_clasificacion",
            [],
        )
    )

    if clf_factors.empty:

        st.write(
            "El modelo seleccionado no expone "
            "importancias globales comparables."
        )

    else:

        st.dataframe(
            clf_factors,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# ADVERTENCIA
# ============================================================

st.warning(
    "Los registros son administrativos y pueden tener subregistro "
    "o cambios de reporte. La aplicación prioriza revisión humana; "
    "no debe usarse para sancionar personas, atribuir culpa ni "
    "sustituir una inspección técnica."
)
