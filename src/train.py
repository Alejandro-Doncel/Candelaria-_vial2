"""Entrenamiento, validación temporal y persistencia de los dos problemas."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data import DATASET_ID, SOURCE_PAGE, load_raw_data
from src.features import (
    HIGH_SINIESTRALIDAD_THRESHOLD,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    add_classification_target,
    classification_features,
    high_siniestralidad_threshold,
    prepare_incidents,
    prepare_monthly_counts,
    regression_features,
    time_split,
)

ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42
ARTIFACT_VERSION = "camino_b_final_012_v3"

# Umbral de decisión del clasificador. Se eligió usando exclusivamente la
# validación temporal 2024 y después quedó congelado para evaluar 2025.
CLASSIFICATION_DECISION_THRESHOLD = 0.12

# Umbral muy bajo que se probó únicamente como análisis de sensibilidad.
# NO se usa en la app ni para seleccionar el modelo final.
SENSITIVITY_THRESHOLD = 0.02

# Parámetros del Random Forest operativo congelados después de la validación.
RF_CLASSIFIER_PARAMS = {
    "n_estimators": 350,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def make_pipeline(categorical: list[str], numeric: list[str], estimator: object) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(categorical, numeric)), ("model", estimator)]
    )


def regression_models() -> dict[str, object]:
    return {
        "Regresión lineal": LinearRegression(),
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def classification_models() -> dict[str, object]:
    """Tres algoritmos del curso antes de la optimización temporal."""
    return {
        "Regresión logística": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def classification_search_spaces() -> dict[str, dict[str, object]]:
    return {
        "Regresión logística": {
            "model__C": np.logspace(-2, 1, 15),
        },
        "Árbol de decisión": {
            "model__max_depth": [3, 4, 5, 6, 8, 10],
            "model__min_samples_leaf": [2, 4, 6, 8, 12, 16],
        },
        "Random Forest": {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [6, 8, 12, 16, None],
            "model__min_samples_leaf": [2, 4, 6, 8],
            "model__max_features": [0.5, 0.8, 1.0],
        },
    }


def regression_scores(actual: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(actual, prediction))
    return {
        "MSE": mse,
        "RMSE": float(mse ** 0.5),
        "MAE": float(mean_absolute_error(actual, prediction)),
        "R2": float(r2_score(actual, prediction)),
    }


def classification_scores(
    actual: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
    auc_roc = float(roc_auc_score(actual, probability)) if actual.nunique() > 1 else float("nan")
    auc_pr = float(average_precision_score(actual, probability)) if actual.nunique() > 1 else float("nan")
    return {
        "Accuracy": float(accuracy_score(actual, prediction)),
        "Precision": float(precision_score(actual, prediction, zero_division=0)),
        "Recall": float(recall_score(actual, prediction, zero_division=0)),
        "F1": float(f1_score(actual, prediction, zero_division=0)),
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "umbral_decision": float(threshold),
    }


def threshold_table(
    actual: pd.Series,
    probability: np.ndarray,
    start: float = 0.01,
    stop: float = 0.50,
    step: float = 0.01,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in np.arange(start, stop + 1e-9, step):
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
        precision = float(precision_score(actual, prediction, zero_division=0))
        recall = float(recall_score(actual, prediction, zero_division=0))
        rows.append(
            {
                "Umbral": round(float(threshold), 2),
                "Accuracy": float(accuracy_score(actual, prediction)),
                "Precision": precision,
                "Recall": recall,
                "F1": float(f1_score(actual, prediction, zero_division=0)),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "TN": int(tn),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["F1", "Precision", "Umbral"],
        ascending=[False, False, False],
    )


def feature_importance(model: Pipeline, limit: int = 8) -> list[dict[str, float | str]]:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return []
    names = model.named_steps["prep"].get_feature_names_out()
    values = estimator.feature_importances_
    order = np.argsort(values)[::-1][:limit]
    return [
        {"variable": str(names[index]), "importancia": float(values[index])}
        for index in order
    ]


def tune_regression_with_temporal_validation(
    train_before_validation: pd.DataFrame,
    validation: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[str, dict[str, object], pd.DataFrame]:
    """Selecciona regresión usando solo el año de validación, nunca el test."""
    features = categorical + numeric
    models = regression_models()
    rows: list[dict[str, float | str]] = []
    fitted_specs: dict[str, object] = {}

    # Regresión lineal no tiene hiperparámetros relevantes en esta comparación.
    linear = make_pipeline(categorical, numeric, deepcopy(models["Regresión lineal"]))
    linear.fit(train_before_validation[features], train_before_validation[TARGET_REGRESSION])
    pred = linear.predict(validation[features])
    rows.append({"Modelo": "Regresión lineal", **regression_scores(validation[TARGET_REGRESSION], pred)})
    fitted_specs["Regresión lineal"] = deepcopy(models["Regresión lineal"])

    spaces = {
        "Árbol de decisión": {
            "model__max_depth": [3, 4, 5, 6, 8, 10, None],
            "model__min_samples_leaf": [2, 4, 6, 8, 12, 16],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [5, 8, 10, 12, None],
            "model__min_samples_leaf": [2, 4, 6, 8],
            "model__max_features": [0.5, 0.8, 1.0],
        },
    }

    for name in ["Árbol de decisión", "Random Forest"]:
        best_mae = np.inf
        best_estimator = None
        for params in ParameterSampler(spaces[name], n_iter=10, random_state=RANDOM_STATE):
            candidate = make_pipeline(categorical, numeric, deepcopy(models[name]))
            candidate.set_params(**params)
            candidate.fit(train_before_validation[features], train_before_validation[TARGET_REGRESSION])
            val_pred = candidate.predict(validation[features])
            mae = mean_absolute_error(validation[TARGET_REGRESSION], val_pred)
            if mae < best_mae:
                best_mae = float(mae)
                best_estimator = deepcopy(candidate.named_steps["model"])
        if best_estimator is None:
            raise RuntimeError(f"No fue posible optimizar {name}.")
        candidate = make_pipeline(categorical, numeric, deepcopy(best_estimator))
        candidate.fit(train_before_validation[features], train_before_validation[TARGET_REGRESSION])
        val_pred = candidate.predict(validation[features])
        rows.append({"Modelo": f"{name} (optimizado)", **regression_scores(validation[TARGET_REGRESSION], val_pred)})
        fitted_specs[f"{name} (optimizado)"] = deepcopy(best_estimator)

    validation_results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    selected_name = str(validation_results.iloc[0]["Modelo"])
    return selected_name, fitted_specs, validation_results



def tune_classification_with_temporal_validation(
    train_before_validation: pd.DataFrame,
    validation: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[
    dict[str, Pipeline],
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Optimiza los tres clasificadores usando solo el año de validación."""
    features = categorical + numeric
    models = classification_models()
    spaces = classification_search_spaces()

    validation_models: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    best_params: dict[str, dict[str, object]] = {}
    validation_rows: list[dict[str, float | int | str]] = []
    optimization_rows: list[dict[str, float | str]] = []

    for name, estimator in models.items():
        sampled = list(
            ParameterSampler(
                spaces[name],
                n_iter=10,
                random_state=RANDOM_STATE,
            )
        )
        best_auc = -np.inf
        best_pipe = None
        best_probability = None
        selected_params = None

        for params in sampled:
            candidate = make_pipeline(categorical, numeric, deepcopy(estimator))
            candidate.set_params(**params)
            candidate.fit(
                train_before_validation[features],
                train_before_validation[TARGET_CLASSIFICATION],
            )
            probability = candidate.predict_proba(validation[features])[:, 1]
            auc = roc_auc_score(validation[TARGET_CLASSIFICATION], probability)

            if auc > best_auc:
                best_auc = float(auc)
                best_pipe = candidate
                best_probability = probability
                selected_params = params

        if best_pipe is None or best_probability is None or selected_params is None:
            raise RuntimeError(f"No fue posible optimizar {name}.")

        validation_models[name] = best_pipe
        validation_probabilities[name] = best_probability
        best_params[name] = selected_params

        validation_rows.append(
            {
                "Modelo": name,
                **classification_scores(
                    validation[TARGET_CLASSIFICATION],
                    best_probability,
                    0.50,
                ),
            }
        )
        optimization_rows.append(
            {
                "Modelo": name,
                "Método": "ParameterSampler, 10 combinaciones",
                "Criterio": "ROC-AUC en validación temporal",
                "Mejor ROC-AUC": best_auc,
                "Hiperparámetros finales": str(selected_params),
            }
        )

    return (
        validation_models,
        validation_probabilities,
        best_params,
        pd.DataFrame(validation_rows),
        pd.DataFrame(optimization_rows),
    )


def main() -> dict:
    raw = load_raw_data()
    incidents = prepare_incidents(raw)
    monthly = prepare_monthly_counts(incidents)

    monthly_train, monthly_test = time_split(monthly)
    test_year = int(monthly_test["anio"].min())
    validation_year = test_year - 1

    # Etiqueta final: ALTA = 5 o más siniestros en un corregimiento-mes.
    high_threshold, p75_active_train = high_siniestralidad_threshold(monthly_train)
    monthly = add_classification_target(monthly, high_threshold)
    monthly_train = add_classification_target(monthly_train, high_threshold)
    monthly_test = add_classification_target(monthly_test, high_threshold)

    train_before_validation = monthly.loc[monthly["anio"] < validation_year].copy()
    validation = monthly.loc[monthly["anio"] == validation_year].copy()
    if train_before_validation.empty or validation.empty:
        raise ValueError("No hay suficiente historia para usar el año previo como validación temporal.")

    reg_categorical, reg_numeric = regression_features()
    clf_categorical, clf_numeric = classification_features()
    reg_features = reg_categorical + reg_numeric
    clf_features = clf_categorical + clf_numeric

    # ------------------------------------------------------------------
    # REGRESIÓN: selección por 2024 (o año previo al test) y test final.
    # ------------------------------------------------------------------
    selected_reg_name, reg_specs, reg_validation_results = tune_regression_with_temporal_validation(
        train_before_validation,
        validation,
        reg_categorical,
        reg_numeric,
    )
    selected_reg_estimator = reg_specs[selected_reg_name]
    final_reg = make_pipeline(reg_categorical, reg_numeric, deepcopy(selected_reg_estimator))
    final_reg.fit(monthly_train[reg_features], monthly_train[TARGET_REGRESSION])
    reg_test_pred = final_reg.predict(monthly_test[reg_features])
    reg_test_result = {
        "Modelo": selected_reg_name,
        **regression_scores(monthly_test[TARGET_REGRESSION], reg_test_pred),
    }

    baseline_reg_pred = np.repeat(monthly_train[TARGET_REGRESSION].mean(), len(monthly_test))
    baseline_reg_result = {
        "Modelo": "Baseline: promedio histórico",
        **regression_scores(monthly_test[TARGET_REGRESSION], baseline_reg_pred),
    }

    # ------------------------------------------------------------------
    # CLASIFICACIÓN: optimización temporal + RF operativo.
    # ------------------------------------------------------------------
    (
        validation_models,
        validation_probabilities,
        clf_best_params,
        classification_validation_results,
        classification_optimization_summary,
    ) = tune_classification_with_temporal_validation(
        train_before_validation,
        validation,
        clf_categorical,
        clf_numeric,
    )

    rf_val_probability = validation_probabilities["Random Forest"]
    thresholds_validation = threshold_table(
        validation[TARGET_CLASSIFICATION],
        rf_val_probability,
    )
    selected_threshold = float(thresholds_validation.iloc[0]["Umbral"])

    if abs(selected_threshold - CLASSIFICATION_DECISION_THRESHOLD) > 1e-12:
        raise RuntimeError(
            "El conjunto público cambió o la validación ya no reproduce el corte 0.12 "
            f"documentado en el informe (corte obtenido: {selected_threshold:.2f})."
        )

    frozen_validation_result = classification_scores(
        validation[TARGET_CLASSIFICATION],
        rf_val_probability,
        CLASSIFICATION_DECISION_THRESHOLD,
    )

    final_rf = make_pipeline(
        clf_categorical,
        clf_numeric,
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    final_rf.set_params(**clf_best_params["Random Forest"])
    final_rf.fit(monthly_train[clf_features], monthly_train[TARGET_CLASSIFICATION])
    rf_test_probability = final_rf.predict_proba(monthly_test[clf_features])[:, 1]

    final_classification_result = {
        "Modelo": "Random Forest",
        **classification_scores(
            monthly_test[TARGET_CLASSIFICATION],
            rf_test_probability,
            CLASSIFICATION_DECISION_THRESHOLD,
        ),
    }

    # Baseline de mayoría: referencia para mostrar por qué Accuracy es engañosa.
    prevalence = float(monthly_train[TARGET_CLASSIFICATION].mean())
    baseline_probability = np.repeat(prevalence, len(monthly_test))
    baseline_classification_result = {
        "Modelo": "Baseline: clase mayoritaria",
        **classification_scores(monthly_test[TARGET_CLASSIFICATION], baseline_probability, 0.50),
    }

    # Sensibilidad post-hoc: se documenta el costo de bajar mucho el umbral.
    # NO se usa para reoptimizar después de observar 2025.
    sensitivity_result = {
        "Modelo": "Random Forest",
        **classification_scores(
            monthly_test[TARGET_CLASSIFICATION],
            rf_test_probability,
            SENSITIVITY_THRESHOLD,
        ),
    }

    # Scores de los positivos reales del test para documentar la inestabilidad.
    positive_cases = monthly_test.loc[
        monthly_test[TARGET_CLASSIFICATION] == 1,
        ["corregimiento", "periodo", TARGET_REGRESSION, TARGET_CLASSIFICATION],
    ].copy()
    positive_cases["probabilidad_alta"] = rf_test_probability[
        monthly_test[TARGET_CLASSIFICATION].to_numpy() == 1
    ]
    positive_cases["periodo"] = positive_cases["periodo"].astype(str)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_reg, ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib")
    joblib.dump(final_rf, ARTIFACT_DIR / "modelo_alerta_alta_siniestralidad.joblib")

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "dataset_id": DATASET_ID,
        "source": SOURCE_PAGE,
        "registros_incidentes": int(len(incidents)),
        "registros_panel_mensual": int(len(monthly)),
        "fecha_minima": str(incidents["fecha"].min().date()),
        "fecha_maxima": str(incidents["fecha"].max().date()),
        "anio_validacion": validation_year,
        "anio_prueba": test_year,
        "p75_meses_activos_entrenamiento_solo_diagnostico": p75_active_train,
        "umbral_alta_siniestralidad": int(HIGH_SINIESTRALIDAD_THRESHOLD),
        "definicion_alta_siniestralidad": "ALTA si el corregimiento-mes registra 5 o más siniestros.",
        "umbral_decision_clasificacion": float(CLASSIFICATION_DECISION_THRESHOLD),
        "interpretacion_umbral_decision": (
            "Corte del puntaje/probabilidad estimada por el clasificador para activar la alerta. "
            "No debe confundirse con los 5 siniestros que definen la clase real ALTA."
        ),
        "modelo_regresion_seleccionado": selected_reg_name,
        "modelo_clasificacion_seleccionado": "Random Forest",
        "parametros_random_forest_clasificacion": {
            key.replace("model__", ""): value
            for key, value in clf_best_params["Random Forest"].items()
        },
        "resultados_regresion_validacion": reg_validation_results.round(6).to_dict(orient="records"),
        "resultado_regresion_test": reg_test_result,
        "baseline_regresion_test": baseline_reg_result,
        "resultados_clasificacion_validacion_umbral_050": classification_validation_results.round(6).to_dict(orient="records"),
        "optimizacion_clasificacion": classification_optimization_summary.to_dict(orient="records"),
        "resultado_rf_validacion_umbral_012": frozen_validation_result,
        "tabla_umbrales_rf_validacion_top20": thresholds_validation.head(20).round(6).to_dict(orient="records"),
        "resultado_clasificacion_test": final_classification_result,
        "baseline_clasificacion_test": baseline_classification_result,
        "sensibilidad_test_umbral_002_no_operativo": sensitivity_result,
        "casos_positivos_test": positive_cases.to_dict(orient="records"),
        "factores_importantes_regresion": feature_importance(final_reg),
        "factores_importantes_clasificacion": feature_importance(final_rf),
        "corregimientos": sorted(monthly["corregimiento"].unique().tolist()),
        "limitacion_clasificacion": (
            "La clase ALTA es muy poco frecuente. Con el umbral 0.12 seleccionado en validación, "
            "el test final detectó uno de los dos casos ALTA y generó falsas alertas. Reducir todavía más "
            "el corte aumenta la sensibilidad, pero también las falsas alertas. El resultado se reporta "
            "como una limitación real y no se reajusta mirando el año de prueba."
        ),
    }

    (ARTIFACT_DIR / "metricas.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\n=== Candelaria Vial — entrenamiento final ===")
    print(f"ALTA siniestralidad: >= {HIGH_SINIESTRALIDAD_THRESHOLD} siniestros/mes")
    print(f"Validación temporal: {validation_year}; prueba final: {test_year}")
    print(f"Umbral de decisión RF congelado: {CLASSIFICATION_DECISION_THRESHOLD:.2f}")
    print("\nRegresión — validación:")
    print(reg_validation_results.round(4).to_string(index=False))
    print("\nRegresión — test:")
    print(pd.DataFrame([baseline_reg_result, reg_test_result]).round(4).to_string(index=False))
    print("\nClasificación RF — validación con 0.12:")
    print(pd.DataFrame([frozen_validation_result]).round(4).to_string(index=False))
    print("\nClasificación RF — test con 0.12:")
    print(pd.DataFrame([final_classification_result]).round(4).to_string(index=False))
    print("\nAnálisis de sensibilidad NO operativo — test con 0.02:")
    print(pd.DataFrame([sensitivity_result]).round(4).to_string(index=False))
    print("\nCasos positivos reales del test y score del RF:")
    print(positive_cases.to_string(index=False))
    print("\nArtefactos guardados en artifacts/.")
    return metadata


if __name__ == "__main__":
    main()
