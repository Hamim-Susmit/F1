import sqlalchemy as sa

from f1_data_ingest import metadata, extract_race_features


def test_extract_race_features_empty(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    engine = sa.create_engine(f"sqlite:///{db_path}", future=True)
    metadata.create_all(engine)
    with engine.begin() as conn:
        features = extract_race_features(conn, race_id=1, driver_id=44)
    assert features == {}
