# Guion breve — Candelaria Vial

## Problema y stakeholder

La Secretaría de Tránsito y Transporte de Candelaria debe priorizar recursos preventivos cada mes. Se propone una herramienta agregada a nivel corregimiento-mes, no un sistema de sanción ni de predicción individual.

## Dos problemas de aprendizaje

**Regresión:** estimar cuántos siniestros se esperan el próximo mes por corregimiento.

**Clasificación:** estimar si el próximo mes tendrá alta siniestralidad un corregimiento.

La definición de alta siniestralidad se calcula solo con el entrenamiento: es un conteo superior al percentil 75 de los meses con actividad registrada. El umbral exacto se reporta al ejecutar el proyecto.

## EDA

Se mantiene el análisis original de cobertura por año, gravedad, meses, corregimientos, hora/día, tendencia central, dispersión, asimetría y correlaciones. El EDA sirve para entender calidad, sesgos y cambios de registro; no se interpreta causalmente.

## Modelos

Regresión: Regresión lineal, Árbol de decisión y Random Forest.  
Clasificación: Regresión logística, Árbol de decisión y Random Forest.

Los baselines se conservan solo como referencia. Ridge y DummyClassifier no forman parte de la comparación de algoritmos.

## Evaluación

La prueba es temporal: el último año no se usa para seleccionar hiperparámetros. La optimización usa el año inmediatamente anterior como validación temporal interna.

Regresión: MAE, RMSE y R².  
Clasificación: Accuracy, Precision, Recall, F1, AUC-ROC y AUC-PR, además de matriz de confusión y curvas ROC/PR en el cuaderno.

## Aplicación

La app muestra simultáneamente la cantidad esperada de siniestros y la alerta de alta siniestralidad. El resultado orienta revisión humana y debe contrastarse con conocimiento local y trabajo de campo.

## Ética y límites

Los datos pueden tener subregistro y cambios administrativos. Un corregimiento con más registros no necesariamente tiene mayor riesgo causal. La herramienta no debe usarse para sancionar, atribuir culpa ni estigmatizar territorios.
