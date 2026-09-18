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

`src/data.py` funciona así:

- si existe `data/raw/accidentalidad_candelaria.csv`, usa ese CSV;
- si no existe, intenta descargarlo automáticamente desde la URL oficial.

Por eso **no es necesario subir una carpeta `data/` vacía a GitHub**.

## Modelos

### Problema 1 — regresión

Se comparan los tres algoritmos vistos en el curso:

- Regresión lineal
- Árbol de decisión
- Random Forest

Como referencia se calcula un baseline de promedio histórico. Ridge fue retirado.

### Problema 2 — clasificación

Se comparan:

- Regresión logística
- Árbol de decisión
- Random Forest

Como referencia se calcula una clase mayoritaria sin usar `DummyClassifier`.

La etiqueta `alta_siniestralidad` se construye **solo con el periodo de entrenamiento**. Se calcula el percentil 75 de los meses que tuvieron al menos un siniestro y se define como alta siniestralidad un conteo estrictamente superior a ese valor. Así se evita que la abundancia de meses con cero convierta automáticamente un solo siniestro en una alerta alta.

Las variables predictoras de ambos problemas son información disponible antes del mes objetivo: corregimiento, calendario, siniestros del mes anterior y promedio de los tres meses previos.

## Partición y optimización

El último año disponible se reserva para prueba temporal. El año inmediatamente anterior funciona como validación interna para `RandomizedSearchCV`; el año de prueba no participa en la selección de hiperparámetros.

## Estructura

```text
app.py
requirements.txt
README.md
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

`artifacts/` y `data/` se crean automáticamente cuando hacen falta; GitHub no necesita carpetas vacías.

## Ejecutar el entrenamiento manualmente

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
python -m src.train
```

Esto crea:

```text
artifacts/
  modelo_priorizacion_mensual.joblib
  modelo_alerta_alta_siniestralidad.joblib
  metricas.json
```

## Desplegar directamente en Streamlit Community Cloud

El repositorio está preparado para que **no sea obligatorio entrenar manualmente antes del deploy**.

1. Subir el contenido de esta carpeta a un repositorio de GitHub.
2. Crear una app en Streamlit Community Cloud usando `app.py` como archivo principal.
3. En la primera ejecución, si no existen `artifacts/`, la app descarga el CSV y ejecuta `src.train` automáticamente.
4. Después carga los dos modelos y muestra la predicción de regresión y la alerta de clasificación.

Si el servidor no pudiera acceder a Datos Abiertos, la única acción adicional es descargar el CSV y añadirlo al repositorio con esta ruta exacta:

```text
data/raw/accidentalidad_candelaria.csv
```

No es necesario modificar código.

## Uso de la aplicación

La app solicita:

- corregimiento;
- mes a priorizar;
- siniestros del mes anterior;
- promedio de siniestros de los tres meses previos.

Entrega:

- cantidad estimada de siniestros;
- alerta `ALTA` / `NO ALTA`;
- probabilidad estimada;
- recomendaciones de revisión preventiva y advertencias de uso.

## Limitaciones

Los datos son registros administrativos y pueden presentar subregistro, cambios de cobertura o cambios en mecanismos de reporte. Los modelos no demuestran causalidad, no predicen comportamientos individuales y no deben utilizarse para sancionar personas. La salida es un insumo para priorización y revisión humana.
