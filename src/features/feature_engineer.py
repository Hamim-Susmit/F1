from __future__ import annotations

from typing import Any, Dict

import sqlalchemy as sa

from f1_data_ingest import extract_race_features


class FeatureEngineer:
    """Lightweight wrapper around the feature extraction pipeline."""

    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine

    def build_features(self, race_id: int, driver_id: int) -> Dict[str, Any]:
        with self.engine.begin() as conn:
            return extract_race_features(conn, race_id=race_id, driver_id=driver_id)
