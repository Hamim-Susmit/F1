import os

import sqlalchemy as sa
from fastapi.testclient import TestClient

from f1_data_ingest import drivers, metadata, teams
from f1_prediction_api import app


def setup_test_db(tmp_path) -> str:
    db_path = tmp_path / "test.db"
    database_url = f"sqlite:///{db_path}"
    engine = sa.create_engine(database_url, future=True)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            drivers.insert(),
            [
                {"driver_id": 1, "driver_code": "HAM", "full_name": "Lewis Hamilton"},
                {"driver_id": 2, "driver_code": "VER", "full_name": "Max Verstappen"},
            ],
        )
        conn.execute(
            teams.insert(),
            [
                {"team_id": 1, "team_name": "Mercedes"},
                {"team_id": 2, "team_name": "Red Bull"},
            ],
        )
    return database_url


def test_root_and_health() -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_driver_and_team_endpoints(tmp_path, monkeypatch) -> None:
    database_url = setup_test_db(tmp_path)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("API_KEY", "test-key")

    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    drivers_resp = client.get("/drivers", headers=headers)
    teams_resp = client.get("/teams", headers=headers)

    assert drivers_resp.status_code == 200
    assert teams_resp.status_code == 200
    assert len(drivers_resp.json()["drivers"]) == 2
    assert len(teams_resp.json()["teams"]) == 2
