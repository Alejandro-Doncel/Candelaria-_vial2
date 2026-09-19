# Guion breve — Candelaria Vial

## Problema y stakeholder

La Secretaría de Tránsito y Transporte de Candelaria debe priorizar recursos preventivos cada mes. Se propone una herramienta agregada a nivel corregimiento-mes, no un sistema de sanción ni de predicción individual.

## Dos problemas de aprendizaje

**Regresión:** estimar cuántos siniestros se esperan el próximo mes por corregimiento.

**Clasificación:** estimar si el próximo mes tendrá alta siniestralidad un corregimiento.

La definición final es explícita: **ALTA = 5 o más siniestros en un corregimiento-mes**. No se cambia esa definición para mejorar artificialmente el balance de clases.

## EDA

Se mantiene el análisis original de cobertura por año, gravedad, meses, corregimientos, hora/día, tendencia central, dispersión, asimetría y correlaciones. El EDA sirve para entender calidad, sesgos y cambios de registro; no se interpreta causalmente.

## Modelos

Regresión: Regresión lineal, Árbol de decisión y Random Forest.  
Clasificación: Regresión logística, Árbol de decisión y Random Forest.

El Random Forest es el clasificador operativo de la app. Los baselines se conservan solo como referencia. Ridge y DummyClassifier no forman parte de la comparación.

## Validación temporal y umbral

La metodología evita usar 2025 para escoger el corte:

1. Se entrena el clasificador con los años anteriores a 2024.
2. Se usa 2024 como validación temporal.
3. Se evalúan cortes entre 0.01 y 0.50 y se selecciona el que maximiza F1; si hay empate, se prioriza mayor Precision y luego el corte más alto.
4. Los cortes 0.11 y 0.12 empataron en F1 y Precision; se tomó **0.12** por ser el mayor de los empatados.
5. El modelo se reentrena con 2021–2024.
6. El corte 0.12 queda congelado y 2025 se utiliza como test final.

Con 0.12 en validación 2024 se obtuvieron TN = 390, FP = 38, FN = 1 y TP = 3.

El `0.12` es un corte del score del modelo; no debe confundirse con `5 siniestros`, que es la definición de la clase real.

## Resultado final de clasificación

Con 0.12 en 2025:

- TN = 404
- FP = 26
- FN = 1
- TP = 1
- Accuracy = 93.75%
- Precision ≈ 3.70%
- Recall = 50%
- F1 ≈ 0.069
- AUC-ROC ≈ 0.942
- AUC-PR ≈ 0.270

La Accuracy sigue siendo insuficiente por sí sola debido al fuerte desbalance: solo hubo dos positivos reales en el test.

Los dos positivos obtuvieron scores aproximados de 0.060 y 0.346. Con el corte 0.12, uno quedó por debajo y el otro por encima.

Como prueba de sensibilidad, un corte de 0.02 recupera los dos positivos, pero genera 71 falsos positivos. No se adopta: sería una cantidad muy alta de falsas alertas y no existe una valoración operativa de la Secretaría que justifique ese costo. Además, no se debe escoger un nuevo corte después de mirar el test.

## Conclusión de la limitación

El falso negativo del test no se considera un “dato atípico”. Es un evento real que el modelo no logró identificar con el corte fijado. La clase ALTA es extremadamente poco frecuente, por lo que hay pocos patrones positivos para aprender. Además, la base no contiene factores externos como flujo vehicular, lluvia, obras, festividades o cambios de infraestructura que podrían explicar algunos picos.

La conclusión es que el modelo muestra capacidad de ordenamiento de riesgo, pero **la frontera binaria es inestable**: intentar capturar todos los casos puede producir demasiadas falsas alertas. Se documenta esto como una limitación real del problema y de los datos disponibles, en vez de seguir reajustando hasta obtener métricas favorables.

## Aplicación

La app solicita **año y mes en campos separados**, además de corregimiento, siniestros del mes anterior y promedio de los tres meses previos.

Muestra:

- conteo esperado de siniestros;
- alerta ALTA/NO ALTA del clasificador Random Forest;
- score estimado de ALTA;
- corte de decisión 0.12;
- definición de ALTA como >= 5 siniestros.

La clasificación no se obtiene aplicando `>=5` a la predicción de regresión.

## Ética y límites

Los datos pueden tener subregistro y cambios administrativos. Un corregimiento con más registros no necesariamente tiene mayor riesgo causal. La herramienta no debe usarse para sancionar, atribuir culpa ni estigmatizar territorios. La salida orienta revisión humana y trabajo de campo.
