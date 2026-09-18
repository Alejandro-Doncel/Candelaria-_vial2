# Candelaria Vial: priorización preventiva con datos abiertos

Proyecto de Técnicas de Aprendizaje de Máquina — Pontificia Universidad Javeriana.

## Problema

La Secretaría de Tránsito y Transporte de Candelaria (Valle) debe decidir cada mes dónde concentrar controles, señalización, inspecciones y educación vial. El proyecto trabaja a nivel **corregimiento-mes** y responde dos preguntas:

1. **Regresión:** ¿cuántos siniestros se esperan el próximo mes por corregimiento?
2. **Clasificación:** ¿el próximo mes tendrá alta siniestralidad un corregimiento?

El EDA original se conserva: gravedad, hora, día, vías, corregimientos, cobertura temporal, estadística descriptiva y correlaciones siguen utilizándose para entender la fuente y sus limitaciones.

## Fuente de datos

Conjunto `Accidentalidad Vial Municipio de Candelaria, Valle`, publicado en Datos Abiertos Colombia.

- Página: <https://www.datos.gov.co/Transporte/Accidentalidad-Vial-Municipio-de-Candelaria-Valle-/7wbf-88zm>
- API CSV: <https://www.datos.gov.co/resource/7wbf-88zm.csv?$limit=100000>
- Identificador: `7wbf-88zm`

`src/data.py` usa primero `data/raw/accidentalidad_candelaria.csv` si existe; si no, intenta descargar el CSV oficial.

## Definición final de alta siniestralidad

La etiqueta de clasificación quedó fijada así:

> **ALTA = 5 o más siniestros en un corregimiento-mes.**

Este umbral define la clase real. No se baja para balancear artificialmente las clases ni para mejorar métricas.

## Modelos

### Problema 1 — regresión

Se comparan Regresión lineal, Árbol de decisión y Random Forest. El año previo al test se usa para seleccionar/optimizar sin mirar el año final de prueba.

### Problema 2 — clasificación

Se comparan Regresión logística, Árbol de decisión y Random Forest. El clasificador operativo de la aplicación es **Random Forest**.

Parámetros congelados del RF operativo:

- `n_estimators=350`
- `min_samples_leaf=2`
- `max_features=0.5`
- `class_weight="balanced_subsample"`
- `random_state=42`

## Partición temporal y umbral de decisión

La lógica final separa los años cronológicamente:

- años anteriores a 2024: ajuste previo;
- **2024: validación temporal** para revisar el comportamiento y fijar el corte de alerta;
- 2021–2024: reentrenamiento final;
- **2025: test final**, sin reajustar el umbral después de ver sus resultados.

En la validación 2024, los cortes `0.37` y `0.38` lograron el mismo resultado (4 TP, 0 FP, 0 FN). Se conservó el mayor de los empatados:

> **Umbral de decisión del Random Forest = 0.38.**

Este `0.38` NO significa “5 siniestros”. Son conceptos distintos:

- `>= 5 siniestros` define qué observaciones históricas son ALTA;
- `score RF >= 0.38` activa la alerta del clasificador.

## Resultado final y limitación real

Con el umbral `0.38` congelado, el test 2025 produjo:

- TN = 429
- FP = 1
- FN = 2
- TP = 0
- Accuracy ≈ 0.9931
- Precision = 0
- Recall = 0
- F1 = 0
- AUC-ROC ≈ 0.9419
- AUC-PR ≈ 0.2696

Los dos eventos ALTA reales del test recibieron scores aproximados de `0.0604` y `0.3459`. Esto muestra que el modelo no reconoció ambos episodios con la misma intensidad.

Como análisis de sensibilidad se observó que bajar drásticamente el corte hasta `0.02` permitía recuperar los dos positivos, pero generaba alrededor de **71 falsas alertas**. Ese corte **no se adopta**. Modificar el umbral después de mirar 2025 sería ajustar el proceso al test y, además, el costo operativo de decenas de falsas alarmas no está validado por la Secretaría.

La conclusión final no es que los dos eventos sean datos atípicos. La limitación es que la clase ALTA es muy poco frecuente y las variables disponibles no contienen necesariamente todos los factores que explican un pico de siniestralidad. El clasificador puede ordenar riesgo de manera útil (AUC alto) y, aun así, no disponer de una frontera binaria estable que detecte todos los positivos sin disparar demasiadas falsas alertas.

## Aplicación Streamlit

La app pide **año y mes por separado**; no usa un calendario de fecha completa. También solicita corregimiento, siniestros del mes anterior y promedio de los tres meses previos.

Muestra simultáneamente:

- siniestros estimados por la regresión;
- alerta `ALTA` / `NO ALTA` obtenida del **clasificador Random Forest**;
- puntaje estimado de ALTA;
- umbral preventivo de decisión (`0.38`);
- validación de versión de artefactos para impedir que un despliegue reutilice modelos/`metricas.json` antiguos con corte `0.02`;
- definición histórica de ALTA (`>= 5 siniestros`).

La etiqueta de la app **no** se calcula comparando la predicción de regresión contra 5.

## Estructura

```text
app.py
requirements.txt
README.md
PASOS_STREAMLIT.txt
src/
  data.py
  features.py
  train.py
notebooks/
  Proyecto_Candelaria_Vial.ipynb
docs/
  guion_presentacion.md
scripts/
  generate_notebook.py
```

## Ejecutar el entrenamiento

```bash
pip install -r requirements.txt
python -m src.train
```

Se crean:

```text
artifacts/
  modelo_priorizacion_mensual.joblib
  modelo_alerta_alta_siniestralidad.joblib
  metricas.json
```

## Uso de IA documentado

El notebook incluye interacciones de IA en formato **Prompt → crítica/respuesta → acción**, incluyendo las decisiones que cambiaron el proyecto:

- reformulación del objetivo de clasificación;
- mantenimiento de `ALTA >= 5` pese al desbalance;
- separación entre umbral de la etiqueta y umbral de decisión;
- elección de `0.38` usando validación 2024;
- rechazo del corte `0.02` pese a recuperar los positivos, por la explosión de falsos positivos;
- decisión de no seguir reajustando después de observar 2025;
- interpretación de la limitación como problema de rareza de la clase y falta de variables explicativas, no como “datos atípicos”.

## Limitaciones y uso responsable

Los datos son registros administrativos y pueden presentar subregistro, cambios de cobertura o cambios de reporte. Los modelos no demuestran causalidad, no predicen comportamientos individuales y no deben utilizarse para sancionar personas. La salida es un insumo para priorización y revisión humana.
