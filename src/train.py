"""Entrenamiento, comparación y persistencia de los dos problemas del proyecto."""

from __future__ import annotations

import json
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
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data import DATASET_ID, SOURCE_PAGE, load_raw_data
from src.features import (
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


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def regression_models() -> dict[str, object]:
    """Los tres algoritmos vistos en clase para el problema de regresión."""
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
    """Los tres algoritmos vistos en clase para la alerta de alta siniestralidad."""
    return {
        "Regresión logística": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=4,
            max_features=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def regression_scores(actual: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
    }


def classification_scores(actual: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    if actual.nunique() < 2:
        auc_roc = float("nan")
    else:
        auc_roc = float(roc_auc_score(actual, probability))
    return {
        "Accuracy": float(accuracy_score(actual, prediction)),
        "Precision": float(precision_score(actual, prediction, zero_division=0)),
        "Recall": float(recall_score(actual, prediction, zero_division=0)),
        "F1": float(f1_score(actual, prediction, zero_division=0)),
        "AUC_ROC": auc_roc,
        "AUC_PR": float(average_precision_score(actual, probability)),
    }


def make_pipeline(categorical: list[str], numeric: list[str], estimator: object) -> Pipeline:
    return Pipeline(
        [("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)]
    )


def fit_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    target: str,
    models: dict[str, object],
    problem: str,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    features = categorical + numeric
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        model = make_pipeline(categorical, numeric, estimator)
        model.fit(train[features], train[target])
        if problem == "regression":
            scores = regression_scores(test[target], model.predict(test[features]))
        else:
            scores = classification_scores(test[target], model.predict_proba(test[features])[:, 1])
        rows.append({"Modelo": name, **scores})
        fitted[name] = model
    metric = "MAE" if problem == "regression" else "AUC_ROC"
    return fitted, pd.DataFrame(rows).sort_values(metric, ascending=problem == "regression")


def tune_model(
    train_tune: pd.DataFrame,
    val_tune: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    target: str,
    estimator: object,
    params: dict,
    scoring: str,
) -> Pipeline:
    """RandomizedSearchCV con validación temporal explícita."""
    features = categorical + numeric
    combined = pd.concat([train_tune, val_tune], ignore_index=True)
    split = PredefinedSplit(
        test_fold=np.array([-1] * len(train_tune) + [0] * len(val_tune))
    )
    search = RandomizedSearchCV(
        make_pipeline(categorical, numeric, estimator),
        param_distributions=params,
        n_iter=10,
        scoring=scoring,
        cv=split,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(combined[features], combined[target])
    return search.best_estimator_


def feature_importance(model: Pipeline, limit: int = 5) -> list[dict[str, float | str]]:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return []
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = estimator.feature_importances_
    order = np.argsort(values)[::-1][:limit]
    return [
        {"variable": str(names[index]), "importancia": float(values[index])}
        for index in order
    ]


def main() -> dict:
    raw = load_raw_data()
    incidents = prepare_incidents(raw)
    monthly = prepare_monthly_counts(incidents)

    monthly_train, monthly_test = time_split(monthly)
    high_threshold, p75_active_train = high_siniestralidad_threshold(monthly_train)
    monthly_train = add_classification_target(monthly_train, high_threshold)
    monthly_test = add_classification_target(monthly_test, high_threshold)
    monthly_labeled = add_classification_target(monthly, high_threshold)

    reg_categorical, reg_numeric = regression_features()
    clf_categorical, clf_numeric = classification_features()

    # Baseline simple para interpretar si ML realmente aporta.
    baseline_mean = float(monthly_train[TARGET_REGRESSION].mean())
    baseline_reg_pred = np.repeat(baseline_mean, len(monthly_test))
    baseline_reg = {"Modelo": "Baseline: promedio histórico", **regression_scores(
        monthly_test[TARGET_REGRESSION], baseline_reg_pred
    )}

    prevalence = float(monthly_train[TARGET_CLASSIFICATION].mean())
    baseline_clf_prob = np.repeat(prevalence, len(monthly_test))
    baseline_clf = {"Modelo": "Baseline: clase mayoritaria", **classification_scores(
        monthly_test[TARGET_CLASSIFICATION], baseline_clf_prob
    )}

    fitted_reg, base_reg_results = fit_models(
        monthly_train,
        monthly_test,
        reg_categorical,
        reg_numeric,
        TARGET_REGRESSION,
        regression_models(),
        "regression",
    )
    fitted_clf, base_clf_results = fit_models(
        monthly_train,
        monthly_test,
        clf_categorical,
        clf_numeric,
        TARGET_CLASSIFICATION,
        classification_models(),
        "classification",
    )

    test_year = int(monthly["anio"].max())
    validation_year = test_year - 1
    tune_train = monthly_labeled[monthly_labeled["anio"] < validation_year].copy()
    tune_val = monthly_labeled[monthly_labeled["anio"] == validation_year].copy()

    if tune_train.empty or tune_val.empty:
        raise ValueError("No hay suficiente historia para la validación temporal interna.")

    reg_spaces = {
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
    clf_spaces = {
        "Regresión logística": {"model__C": np.logspace(-2, 1, 15)},
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

    tuned_reg: dict[str, Pipeline] = {}
    for name, space in reg_spaces.items():
        tuned_reg[name] = tune_model(
            tune_train,
            tune_val,
            reg_categorical,
            reg_numeric,
            TARGET_REGRESSION,
            regression_models()[name],
            space,
            "neg_mean_absolute_error",
        )

    tuned_clf: dict[str, Pipeline] = {}
    for name, space in clf_spaces.items():
        tuned_clf[name] = tune_model(
            tune_train,
            tune_val,
            clf_categorical,
            clf_numeric,
            TARGET_CLASSIFICATION,
            classification_models()[name],
            space,
            "roc_auc",
        )

    # Reentrenamiento con todo el periodo de entrenamiento y evaluación final en test.
    reg_features = reg_categorical + reg_numeric
    clf_features = clf_categorical + clf_numeric
    tuned_reg_rows = []
    for name, model in tuned_reg.items():
        model.fit(monthly_train[reg_features], monthly_train[TARGET_REGRESSION])
        key = name + " (optimizado)"
        fitted_reg[key] = model
        tuned_reg_rows.append(
            {"Modelo": key, **regression_scores(
                monthly_test[TARGET_REGRESSION], model.predict(monthly_test[reg_features])
            )}
        )

    tuned_clf_rows = []
    for name, model in tuned_clf.items():
        model.fit(monthly_train[clf_features], monthly_train[TARGET_CLASSIFICATION])
        key = name + " (optimizado)"
        fitted_clf[key] = model
        tuned_clf_rows.append(
            {"Modelo": key, **classification_scores(
                monthly_test[TARGET_CLASSIFICATION], model.predict_proba(monthly_test[clf_features])[:, 1]
            )}
        )

    regression_results = pd.concat(
        [pd.DataFrame([baseline_reg]), base_reg_results, pd.DataFrame(tuned_reg_rows)],
        ignore_index=True,
    ).sort_values("MAE")
    classification_results = pd.concat(
        [pd.DataFrame([baseline_clf]), base_clf_results, pd.DataFrame(tuned_clf_rows)],
        ignore_index=True,
    ).sort_values("AUC_ROC", ascending=False, na_position="last")

    # Selección solo entre los algoritmos de ML; el baseline es referencia.
    candidate_reg = regression_results[~regression_results["Modelo"].str.startswith("Baseline")]
    candidate_clf = classification_results[~classification_results["Modelo"].str.startswith("Baseline")]
    best_regression_name = str(candidate_reg.iloc[0]["Modelo"])
    best_classification_name = str(candidate_clf.iloc[0]["Modelo"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        fitted_reg[best_regression_name],
        ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib",
    )
    joblib.dump(
        fitted_clf[best_classification_name],
        ARTIFACT_DIR / "modelo_alerta_alta_siniestralidad.joblib",
    )

    metadata = {
        "dataset_id": DATASET_ID,
        "source": SOURCE_PAGE,
        "registros_incidentes": int(len(incidents)),
        "registros_panel_mensual": int(len(monthly)),
        "fecha_minima": str(incidents["fecha"].min().date()),
        "fecha_maxima": str(incidents["fecha"].max().date()),
        "anio_entrenamiento_maximo": int(monthly_train["anio"].max()),
        "anio_prueba": int(monthly_test["anio"].min()),
        "p75_meses_activos_entrenamiento": p75_active_train,
        "umbral_alta_siniestralidad": int(high_threshold),
        "definicion_alta_siniestralidad": (
            "Conteo mensual igual o superior al primer entero estrictamente mayor que el P75 "
            "de los meses con al menos un siniestro en el periodo de entrenamiento."
        ),
        "modelo_regresion_seleccionado": best_regression_name,
        "modelo_clasificacion_seleccionado": best_classification_name,
        "factores_importantes_regresion": feature_importance(fitted_reg[best_regression_name]),
        "factores_importantes_clasificacion": feature_importance(fitted_clf[best_classification_name]),
        "resultados_regresion": regression_results.round(4).to_dict(orient="records"),
        "resultados_clasificacion": classification_results.round(4).to_dict(orient="records"),
        "corregimientos": sorted(monthly["corregimiento"].unique().tolist()),
    }
    (ARTIFACT_DIR / "metricas.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\nProblema 1 — regresión: siniestros esperados el próximo mes")
    print(regression_results.round(4).to_string(index=False))
    print("\nProblema 2 — clasificación: ¿habrá alta siniestralidad el próximo mes?")
    print(f"Umbral de alta siniestralidad: >= {high_threshold} siniestros/mes")
    print(classification_results.round(4).to_string(index=False))
    print("\nArtefactos guardados en artifacts/.")
    return metadata


if __name__ == "__main__":
    main()
