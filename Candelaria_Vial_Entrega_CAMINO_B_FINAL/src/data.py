"""Descarga y validación del conjunto público de siniestralidad vial."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "7wbf-88zm"
DATASET_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.csv?$limit=100000"
SOURCE_PAGE = (
    "https://www.datos.gov.co/Transporte/"
    "Accidentalidad-Vial-Municipio-de-Candelaria-Valle-/7wbf-88zm"
)
DEFAULT_CACHE = Path("data/raw/accidentalidad_candelaria.csv")

REQUIRED_COLUMNS = {
    "clase_de_accidente",
    "gravedad",
    "vias",
    "corregimiento",
    "d_a_semana",
    "fecha_de_ocurrecia",
    "hora_ocurrencia",
}


def download_raw_data(cache_path: str | Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Descarga el CSV oficial y conserva una copia local."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(DATASET_URL, timeout=90)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "No fue posible descargar el CSV desde Datos Abiertos. "
            "Como alternativa, descarga manualmente el archivo y guárdalo como "
            f"{cache_path.as_posix()}."
        ) from exc

    cache_path.write_bytes(response.content)
    data = pd.read_csv(cache_path, low_memory=False)
    validate_schema(data)
    return data


def load_raw_data(
    cache_path: str | Path = DEFAULT_CACHE, refresh: bool = False
) -> pd.DataFrame:
    """Usa el CSV local si existe; de lo contrario lo descarga automáticamente."""
    cache_path = Path(cache_path)
    if refresh or not cache_path.exists():
        return download_raw_data(cache_path)

    data = pd.read_csv(cache_path, low_memory=False)
    validate_schema(data)
    return data


def validate_schema(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "El conjunto descargado cambió. Faltan columnas requeridas: "
            f"{sorted(missing)}"
        )
