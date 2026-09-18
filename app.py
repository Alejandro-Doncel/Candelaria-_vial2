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

# Versión esperada de los artefactos. Esto evita que un despliegue antiguo
# conserve un metricas.json/modelo entrenado con el corte 0.02.
EXPECTED_ARTIFACT_VERSION = "camino_b_final_038_v2"
EXPECTED_HIGH_COUNT_THRESHOLD = 5
EXPECTED_DECISION_THRESHOLD = 0.38


def artifacts_ready() -> bool:
    if not (REG_MODEL.exists() and CLF_MODEL.exists() and METRICS.exists()):
        return False

    try:
        metadata = json.loads(METRICS.read_text(encoding="utf-8"))
        return (
            metadata.get("artifact_version") == EXPECTED_ARTIFACT_VERSION
            and int(metadata.get("umbral_alta_siniestralidad", -1))
            == EXPECTED_HIGH_COUNT_THRESHOLD
            and abs(
                float(metadata.get("umbral_decision_clasificacion", -1.0))
                - EXPECTED_DECISION_THRESHOLD
            ) < 1e-12
            and metadata.get("modelo_clasificacion_seleccionado") == "Random Forest"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


@st.cache_resource(show_spinner=False)
def ensure_artifacts(version_key: str) -> None:
    """Entrena una sola vez si el repositorio aún no trae artefactos."""
    if artifacts_ready():
        return
    from src.train import main

    main()
    if not artifacts_ready():
        raise RuntimeError("Los artefactos generados no corresponden a la versión final 0.38.")


@st.cache_resource(show_spinner=False)
def load_models(version_key: str):
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
            ensure_artifacts(EXPECTED_ARTIFACT_VERSION)
        except Exception as exc:
            st.error(
                "No fue posible preparar los modelos automáticamente. "
                "Si la descarga pública está bloqueada, agrega el CSV como "
                "`data/raw/accidentalidad_candelaria.csv` y vuelve a desplegar."
            )
            st.exception(exc)
            st.stop()

monthly_model, alert_model, metadata = load_models(EXPECTED_ARTIFACT_VERSION)

HIGH_COUNT_THRESHOLD = int(metadata.get("umbral_alta_siniestralidad", 5))
DECISION_THRESHOLD = float(metadata.get("umbral_decision_clasificacion", 0.38))

st.info(
    f"Fuente: {metadata['registros_incidentes']:,} registros entre "
    f"{metadata['fecha_minima']} y {metadata['fecha_maxima']}. "
    f"Validación temporal: {metadata.get('anio_validacion', 'año previo')}; "
    f"prueba final: {metadata['anio_prueba']}."
)

st.subheader("Prioriza un mes")
col_a, col_b = st.columns(2)

# La fecha se pide como dos entradas separadas. No se usa un calendario de día/mes/año.
with col_a:
    corregimiento = st.selectbox("Corregimiento", metadata["corregimientos"])

    anio_prediccion = st.number_input(
        "Año que se quiere priorizar",
        min_value=2020,
        max_value=2100,
        value=max(2026, int(metadata.get("anio_prueba", 2025)) + 1),
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
    mes_numero = st.selectbox(
        "Mes que se quiere priorizar",
        options=list(meses.keys()),
        format_func=lambda x: meses[x],
    )

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

row = pd.DataFrame(
    {
        "corregimiento": [corregimiento],
        "anio": [int(anio_prediccion)],
        "mes": [int(mes_numero)],
        "mes_seno": [np.sin(2 * np.pi * mes_numero / 12)],
        "mes_coseno": [np.cos(2 * np.pi * mes_numero / 12)],
        "siniestros_lag_1": [previous],
        "promedio_3_meses_previo": [rolling],
    }
)

reg_cat, reg_num = regression_features()
clf_cat, clf_num = classification_features()

expected_count = max(
    0.0,
    float(monthly_model.predict(row[reg_cat + reg_num])[0]),
)

high_probability = float(
    alert_model.predict_proba(row[clf_cat + clf_num])[:, 1][0]
)

# IMPORTANTE: la alerta se decide con el CLASIFICADOR, no con la regresión.
high_label = high_probability >= DECISION_THRESHOLD

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Siniestros estimados", f"{expected_count:.2f}")
metric_b.metric(
    "Alerta de alta siniestralidad",
    "ALTA" if high_label else "NO ALTA",
)
metric_c.metric(
    "Puntaje estimado de ALTA",
    f"{high_probability:.1%}",
)

st.caption(
    f"La clase real **ALTA** se define como **{HIGH_COUNT_THRESHOLD} o más siniestros** "
    f"en un corregimiento-mes. La alerta del clasificador se activa cuando su "
    f"puntaje alcanza **{DECISION_THRESHOLD:.0%}**. Son dos umbrales distintos: "
    "uno define la etiqueta histórica y el otro convierte el puntaje del modelo en una alerta."
)

st.warning(
    "El puntaje mostrado es la salida estimada por el modelo y no debe interpretarse "
    "como una probabilidad perfectamente calibrada ni como certeza de que ocurrirán "
    f"{HIGH_COUNT_THRESHOLD} o más siniestros."
)

st.subheader("Acciones sugeridas")
for action in recommended_actions(high_label, expected_count):
    st.write(f"- {action}")

with st.expander("Cómo interpretar los resultados"):
    st.write(
        "La regresión y la clasificación son dos problemas distintos. La regresión estima "
        "un conteo esperado, que puede ser decimal, mientras que el Random Forest de "
        "clasificación asigna un puntaje a la posibilidad de entrar en la categoría ALTA. "
        "Por eso ambas salidas no tienen que coincidir exactamente."
    )
    st.write(
        f"El umbral de decisión {DECISION_THRESHOLD:.0%} fue fijado usando la validación "
        "temporal previa al año de prueba y no se reajusta con el test final."
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
    st.caption(f"Clasificación operativa: {metadata['modelo_clasificacion_seleccionado']}")
    clf_factors = pd.DataFrame(metadata.get("factores_importantes_clasificacion", []))
    if clf_factors.empty:
        st.write("El modelo seleccionado no expone importancias globales comparables.")
    else:
        st.dataframe(clf_factors, hide_index=True, use_container_width=True)

st.warning(
    "Los registros son administrativos y pueden tener subregistro o cambios de reporte. "
    "La clasificación final presenta una limitación real por la escasez de meses ALTA: "
    "el sistema sirve para apoyar priorización y revisión humana, no para sancionar, "
    "atribuir culpa ni sustituir una inspección técnica."
)
