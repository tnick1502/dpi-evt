"""Add Russian display columns to metadata and keep the original identifiers."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from dpi_evt.constants import (
    GENERATOR_FAMILY_DISPLAY_NAMES,
    LOAD_DISPLAY_NAMES,
    RESPONSE_TYPE_DISPLAY_NAMES,
    SOIL_DISPLAY_NAMES,
)

__all__ = ["localize_series", "localize_meta_frame"]


def localize_series(values: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """Map categorical identifiers to Russian display labels.

    Unmapped values are retained.

    :param values: series of category identifiers
    :param mapping: identifier-to-label mapping
    :return: series with display labels
    """
    return values.map(mapping).fillna(values)


def localize_meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add localized metadata columns without changing the original identifiers.

    soil_type, load_mode and generator_family receive companion columns ending in _ru.
    Grouping and stratification continue to use the original columns.

    :param df: dataset metadata
    :return: a copy with localized display columns
    """
    localized = df.copy()
    if "soil_type" in localized.columns:
        localized["soil_type_ru"] = localize_series(localized["soil_type"], SOIL_DISPLAY_NAMES)
    if "load_mode" in localized.columns:
        localized["load_mode_ru"] = localize_series(localized["load_mode"], LOAD_DISPLAY_NAMES)
    if "generator_family" in localized.columns:
        localized["generator_family_ru"] = localize_series(
            localized["generator_family"], GENERATOR_FAMILY_DISPLAY_NAMES
        )
    if "response_type" in localized.columns:
        localized["response_type_ru"] = localize_series(
            localized["response_type"], RESPONSE_TYPE_DISPLAY_NAMES
        )
    return localized
