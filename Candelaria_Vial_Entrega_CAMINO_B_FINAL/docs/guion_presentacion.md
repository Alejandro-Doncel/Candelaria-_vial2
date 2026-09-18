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
3. En 2024 los umbrales 0.37 y 0.38 dieron 4 TP, 0 FP y 0 FN; se tomó **0.38** por ser el mayor umbral empatado.
4. El modelo se reentrena con 2021–2024.
5. El corte 0.38 queda congelado y 2025 se utiliza como test final.

El `0.38` es un corte del score del modelo; no debe confundirse con `5 siniestros`, que es la definición de la clase real.

## Resultado final de clasificación

Con 0.38 en 2025:

- TN = 429
- FP = 1
- FN = 2
- TP = 0
- Accuracy ≈ 99.31%
- Precision = 0
- Recall = 0
- F1 = 0
- AUC-ROC ≈ 0.942
- AUC-PR ≈ 0.270

La Accuracy es engañosa por el fuerte desbalance: solo hubo dos positivos reales en el test.

Los dos positivos obtuvieron scores aproximados de 0.060 y 0.346. Por eso no basta con bajar un poco el umbral para capturar ambos.

Como prueba de sensibilidad, un corte cercano a 0.02 recuperaba los dos positivos, pero generaba alrededor de 71 falsos positivos. No se adopta: sería una cantidad muy alta de falsas alertas y no existe una valoración operativa de la Secretaría que justifique ese costo. Además, no se debe escoger un nuevo corte después de mirar el test.

## Conclusión de la limitación

Los dos falsos negativos no se consideran “datos atípicos”. Son eventos reales que el modelo no logró identificar con el corte fijado. La clase ALTA es extremadamente poco frecuente, por lo que hay pocos patrones positivos para aprender. Además, la base no contiene factores externos como flujo vehicular, lluvia, obras, festividades o cambios de infraestructura que podrían explicar algunos picos.

La conclusión es que el modelo muestra capacidad de ordenamiento de riesgo, pero **la frontera binaria es inestable**: intentar capturar todos los casos puede producir demasiadas falsas alertas. Se documenta esto como una limitación real del problema y de los datos disponibles, en vez de seguir reajustando hasta obtener métricas favorables.

## Aplicación

La app solicita **año y mes en campos separados**, además de corregimiento, siniestros del mes anterior y promedio de los tres meses previos.

Muestra:

- conteo esperado de siniestros;
- alerta ALTA/NO ALTA del clasificador Random Forest;
- score estimado de ALTA;
- corte de decisión 0.38;
- definición de ALTA como >= 5 siniestros.

La clasificación no se obtiene aplicando `>=5` a la predicción de regresión.

## Ética y límites

Los datos pueden tener subregistro y cambios administrativos. Un corregimiento con más registros no necesariamente tiene mayor riesgo causal. La herramienta no debe usarse para sancionar, atribuir culpa ni estigmatizar territorios. La salida orienta revisión humana y trabajo de campo.
