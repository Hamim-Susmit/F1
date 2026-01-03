import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import sqlalchemy as sa
from fastf1 import Cache, get_event_schedule, get_session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from tqdm import tqdm


LOG = logging.getLogger("f1_ingest")


metadata = sa.MetaData()

races = sa.Table(
    "races",
    metadata,
    sa.Column("race_id", sa.Integer, primary_key=True),
    sa.Column("season", sa.Integer, nullable=False),
    sa.Column("round", sa.Integer, nullable=False),
    sa.Column("race_name", sa.String, nullable=False),
    sa.Column("circuit_name", sa.String, nullable=False),
    sa.Column("race_date", sa.Date, nullable=False),
    sa.Column("winning_time", sa.Float),
    sa.UniqueConstraint("season", "round", name="uq_races_season_round"),
)

drivers = sa.Table(
    "drivers",
    metadata,
    sa.Column("driver_id", sa.Integer, primary_key=True),
    sa.Column("driver_code", sa.String, nullable=False, unique=True),
    sa.Column("full_name", sa.String, nullable=False),
    sa.Column("nationality", sa.String),
)

teams = sa.Table(
    "teams",
    metadata,
    sa.Column("team_id", sa.Integer, primary_key=True),
    sa.Column("team_name", sa.String, nullable=False, unique=True),
    sa.Column("engine_manufacturer", sa.String),
)

race_results = sa.Table(
    "race_results",
    metadata,
    sa.Column("result_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.team_id"), nullable=False),
    sa.Column("grid_position", sa.Integer),
    sa.Column("finishing_position", sa.Integer),
    sa.Column("points", sa.Float),
    sa.Column("status", sa.String),
    sa.Column("race_time", sa.Float),
    sa.Column("fastest_lap", sa.Boolean, default=False),
)

qualifying_results = sa.Table(
    "qualifying_results",
    metadata,
    sa.Column("qual_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("q1_time", sa.Float),
    sa.Column("q2_time", sa.Float),
    sa.Column("q3_time", sa.Float),
    sa.Column("grid_position", sa.Integer),
)

lap_times = sa.Table(
    "lap_times",
    metadata,
    sa.Column("lap_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("lap_number", sa.Integer, nullable=False),
    sa.Column("lap_time", sa.Float),
    sa.Column("tire_compound", sa.String),
)

pit_stops = sa.Table(
    "pit_stops",
    metadata,
    sa.Column("stop_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("stop_number", sa.Integer, nullable=False),
    sa.Column("lap", sa.Integer, nullable=False),
    sa.Column("duration", sa.Float),
    sa.Column("tire_in", sa.String),
    sa.Column("tire_out", sa.String),
)

race_weather = sa.Table(
    "race_weather",
    metadata,
    sa.Column("weather_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("session_type", sa.String, nullable=False),
    sa.Column("air_temp", sa.Float),
    sa.Column("track_temp", sa.Float),
    sa.Column("rain_probability", sa.Float),
    sa.Column("rain_amount", sa.Float),
    sa.Column("humidity", sa.Float),
    sa.Column("wind_speed", sa.Float),
    sa.Column("wind_direction", sa.Float),
    sa.Column("conditions", sa.String),
    sa.Column("wet_race_probability", sa.Float),
    sa.Column("temp_deviation", sa.Float),
    sa.Column("weather_change_probability", sa.Float),
    sa.Column("is_estimated", sa.Boolean, default=False),
    sa.Column("source", sa.String),
    sa.Column("observed_at", sa.DateTime(timezone=True)),
)

race_weather_forecasts = sa.Table(
    "race_weather_forecasts",
    metadata,
    sa.Column("forecast_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("session_type", sa.String, nullable=False),
    sa.Column("forecast_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("air_temp", sa.Float),
    sa.Column("track_temp", sa.Float),
    sa.Column("rain_probability", sa.Float),
    sa.Column("rain_amount", sa.Float),
    sa.Column("humidity", sa.Float),
    sa.Column("wind_speed", sa.Float),
    sa.Column("wind_direction", sa.Float),
    sa.Column("conditions", sa.String),
    sa.Column("source", sa.String),
)

circuits = sa.Table(
    "circuits",
    metadata,
    sa.Column("circuit_id", sa.Integer, primary_key=True),
    sa.Column("circuit_name", sa.String, nullable=False, unique=True),
    sa.Column("country", sa.String),
    sa.Column("circuit_length", sa.Float),
    sa.Column("number_of_corners", sa.Integer),
    sa.Column("longest_straight", sa.Float),
    sa.Column("circuit_type", sa.String),
    sa.Column("altitude", sa.Integer),
    sa.Column("lap_record", sa.Float),
    sa.Column("lap_record_holder", sa.String),
    sa.Column("lap_record_year", sa.Integer),
)

circuit_characteristics = sa.Table(
    "circuit_characteristics",
    metadata,
    sa.Column("char_id", sa.Integer, primary_key=True),
    sa.Column("circuit_id", sa.Integer, sa.ForeignKey("circuits.circuit_id"), nullable=False),
    sa.Column("high_speed_corners", sa.Integer),
    sa.Column("medium_speed_corners", sa.Integer),
    sa.Column("low_speed_corners", sa.Integer),
    sa.Column("overtaking_difficulty", sa.Integer),
    sa.Column("drs_zones", sa.Integer),
    sa.Column("pit_lane_time_loss", sa.Float),
    sa.Column("track_evolution_factor", sa.Float),
    sa.Column("safety_car_probability", sa.Float),
    sa.Column("average_pit_stops", sa.Float),
    sa.Column("typical_strategy", sa.String),
    sa.Column("overtaking_frequency", sa.Float),
    sa.Column("grid_finish_correlation", sa.Float),
    sa.Column("qualifying_importance", sa.Float),
    sa.Column("tire_degradation_severity", sa.Float),
    sa.Column("strategy_variation", sa.Float),
    sa.UniqueConstraint("circuit_id", name="circuit_characteristics_circuit_id_key"),
)

circuit_history = sa.Table(
    "circuit_history",
    metadata,
    sa.Column("history_id", sa.Integer, primary_key=True),
    sa.Column("circuit_id", sa.Integer, sa.ForeignKey("circuits.circuit_id"), nullable=False),
    sa.Column("season", sa.Integer, nullable=False),
    sa.Column("pole_win_rate", sa.Float),
    sa.Column("average_winning_margin", sa.Float),
    sa.Column("safety_car_frequency", sa.Float),
    sa.Column("red_flag_count", sa.Integer),
    sa.Column("weather_variability", sa.Float),
    sa.UniqueConstraint("circuit_id", "season", name="uq_circuit_history"),
)

driver_features = sa.Table(
    "driver_features",
    metadata,
    sa.Column("feature_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("last_3_races_avg_position", sa.Float),
    sa.Column("last_5_races_avg_position", sa.Float),
    sa.Column("last_3_races_points_scored", sa.Float),
    sa.Column("recent_dnf_rate", sa.Float),
    sa.Column("recent_qualifying_avg_position", sa.Float),
    sa.Column("positions_gained_last_3_races", sa.Float),
    sa.Column("avg_finish_position_at_circuit", sa.Float),
    sa.Column("best_result_at_circuit", sa.Float),
    sa.Column("worst_result_at_circuit", sa.Float),
    sa.Column("qualifying_to_race_conversion_at_circuit", sa.Float),
    sa.Column("avg_positions_gained_lap1_at_circuit", sa.Float),
    sa.Column("crash_rate_at_circuit", sa.Float),
    sa.Column("qualifying_pace_percentile", sa.Float),
    sa.Column("race_pace_percentile", sa.Float),
    sa.Column("overtaking_success_rate", sa.Float),
    sa.Column("defending_success_rate", sa.Float),
    sa.Column("wet_weather_performance_multiplier", sa.Float),
    sa.Column("first_lap_position_change_avg", sa.Float),
    sa.Column("consistency_score", sa.Float),
    sa.Column("tire_management_score", sa.Float),
    sa.Column("total_f1_races", sa.Integer),
    sa.Column("races_at_this_circuit", sa.Integer),
    sa.Column("years_with_current_team", sa.Integer),
    sa.Column("championship_experience", sa.Integer),
    sa.Column("podium_count", sa.Integer),
    sa.Column("win_count", sa.Integer),
    sa.Column("qualifying_gap_to_teammate", sa.Float),
    sa.Column("race_gap_to_teammate", sa.Float),
    sa.Column("head_to_head_qualifying", sa.Float),
    sa.Column("head_to_head_race", sa.Float),
    sa.Column("championship_position", sa.Integer),
    sa.Column("points_from_leader", sa.Float),
    sa.Column("contract_year_flag", sa.Boolean),
    sa.Column("recent_incidents_count", sa.Integer),
    sa.Column("consecutive_race_finishes", sa.Integer),
    sa.UniqueConstraint("race_id", "driver_id", name="uq_driver_features"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastF1 data ingester for PostgreSQL.")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--cache-dir", default=os.environ.get("FASTF1_CACHE", ".fastf1_cache"))
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--weather-api-key", default=os.environ.get("OPENWEATHER_API_KEY"))
    parser.add_argument("--weather-base-url", default=os.environ.get("WEATHER_BASE_URL", "https://api.openweathermap.org"))
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_engine(database_url: Optional[str]) -> Engine:
    if not database_url:
        raise ValueError("DATABASE_URL is required, e.g. postgresql+psycopg2://user:pass@host/db")
    return sa.create_engine(database_url, future=True)


def enable_cache(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    Cache.enable_cache(cache_dir)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)


def to_seconds(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_session_with_retry(year: int, round_number: int, session_type: str, retries: int = 3):
    last_exc = None
    for attempt in range(retries):
        try:
            session = get_session(year, round_number, session_type)
            session.load()
            return session
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOG.warning(
                "Failed to load session %s %s %s (attempt %s/%s): %s",
                year,
                round_number,
                session_type,
                attempt + 1,
                retries,
                exc,
            )
            time.sleep(2 + attempt)
    raise last_exc


def get_schedule(year: int) -> pd.DataFrame:
    schedule = get_event_schedule(year, include_testing=False)
    return schedule[schedule["EventFormat"] == "race"]


def upsert_rows(conn, table: sa.Table, rows: List[Dict[str, Any]], constraint: Optional[str] = None) -> None:
    if not rows:
        return
    stmt = pg_insert(table).values(rows)
    if constraint:
        stmt = stmt.on_conflict_do_nothing(constraint=constraint)
    else:
        stmt = stmt.on_conflict_do_nothing()
    conn.execute(stmt)


def ensure_drivers(conn, driver_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    upsert_rows(conn, drivers, driver_rows, constraint="drivers_driver_code_key")
    result = conn.execute(sa.select(drivers.c.driver_id, drivers.c.driver_code))
    return {row.driver_code: row.driver_id for row in result}


def ensure_teams(conn, team_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    upsert_rows(conn, teams, team_rows, constraint="teams_team_name_key")
    result = conn.execute(sa.select(teams.c.team_id, teams.c.team_name))
    return {row.team_name: row.team_id for row in result}


def ensure_race(
    conn,
    season: int,
    round_number: int,
    race_name: str,
    circuit_name: str,
    race_date: date,
    winning_time: Optional[float],
) -> int:
    stmt = pg_insert(races).values(
        season=season,
        round=round_number,
        race_name=race_name,
        circuit_name=circuit_name,
        race_date=race_date,
        winning_time=winning_time,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_races_season_round",
        set_={
            "race_name": race_name,
            "circuit_name": circuit_name,
            "race_date": race_date,
            "winning_time": winning_time,
        },
    ).returning(races.c.race_id)
    return conn.execute(stmt).scalar_one()


def ensure_circuit(
    conn,
    circuit_name: str,
    country: Optional[str],
    circuit_type: Optional[str],
) -> int:
    stmt = pg_insert(circuits).values(
        circuit_name=circuit_name,
        country=country,
        circuit_type=circuit_type,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="circuits_circuit_name_key",
        set_={
            "country": country,
            "circuit_type": circuit_type,
        },
    ).returning(circuits.c.circuit_id)
    return conn.execute(stmt).scalar_one()


def build_driver_rows(results_df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "driver_code": row["Abbreviation"],
            "full_name": row["FullName"],
            "nationality": row.get("Nationality"),
        }
        for _, row in results_df.iterrows()
    ]


def build_team_rows(results_df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "team_name": row["TeamName"],
            "engine_manufacturer": row.get("Engine"),
        }
        for _, row in results_df.iterrows()
    ]


def build_race_results(
    results_df: pd.DataFrame,
    race_id: int,
    driver_map: Dict[str, int],
    team_map: Dict[str, int],
    fastest_driver: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in results_df.iterrows():
        driver_code = row["Abbreviation"]
        team_name = row["TeamName"]
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_map.get(driver_code),
                "team_id": team_map.get(team_name),
                "grid_position": int(row["GridPosition"]) if pd.notna(row["GridPosition"]) else None,
                "finishing_position": int(row["Position"]) if pd.notna(row["Position"]) else None,
                "points": float(row["Points"]) if pd.notna(row["Points"]) else None,
                "status": row.get("Status"),
                "race_time": to_seconds(row.get("Time")),
                "fastest_lap": driver_code == fastest_driver,
            }
        )
    return rows


def build_qualifying_results(
    qualifying_df: pd.DataFrame,
    race_id: int,
    driver_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in qualifying_df.iterrows():
        driver_code = row["Abbreviation"]
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_map.get(driver_code),
                "q1_time": to_seconds(row.get("Q1")),
                "q2_time": to_seconds(row.get("Q2")),
                "q3_time": to_seconds(row.get("Q3")),
                "grid_position": int(row["Position"]) if pd.notna(row.get("Position")) else None,
            }
        )
    return rows


def build_lap_times(
    laps: pd.DataFrame,
    race_id: int,
    driver_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in laps.iterrows():
        driver_code = row.get("Driver")
        lap_number = row.get("LapNumber")
        if pd.isna(driver_code) or pd.isna(lap_number):
            continue
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_map.get(driver_code),
                "lap_number": int(lap_number),
                "lap_time": to_seconds(row.get("LapTime")),
                "tire_compound": row.get("Compound"),
            }
        )
    return rows


def build_pit_stops(
    pit_laps: pd.DataFrame,
    full_laps: pd.DataFrame,
    race_id: int,
    driver_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full_laps = full_laps.copy()
    full_laps["LapNumber"] = full_laps["LapNumber"].astype("Int64")
    for _, row in pit_laps.iterrows():
        driver_code = row.get("Driver")
        lap_number = row.get("LapNumber")
        stop_number = row.get("PitStopNumber")
        if pd.isna(driver_code) or pd.isna(lap_number) or pd.isna(stop_number):
            continue
        lap_number = int(lap_number)
        stop_number = int(stop_number)
        pit_in = row.get("PitInTime")
        pit_out = row.get("PitOutTime")
        duration = None
        if pd.notna(pit_in) and pd.notna(pit_out):
            duration = to_seconds(pit_out - pit_in)
        prev_compound = None
        next_compound = None
        driver_laps = full_laps[full_laps["Driver"] == driver_code]
        prev_lap = driver_laps[driver_laps["LapNumber"] == lap_number - 1]
        next_lap = driver_laps[driver_laps["LapNumber"] == lap_number + 1]
        if not prev_lap.empty:
            prev_compound = prev_lap.iloc[0].get("Compound")
        if not next_lap.empty:
            next_compound = next_lap.iloc[0].get("Compound")
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_map.get(driver_code),
                "stop_number": stop_number,
                "lap": lap_number,
                "duration": duration,
                "tire_in": next_compound,
                "tire_out": prev_compound,
            }
        )
    return rows


def chunked(iterable: List[Dict[str, Any]], size: int = 1000) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def build_data_quality_report(conn: sa.Connection) -> Dict[str, Any]:
    def count(table: sa.Table) -> int:
        return conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()

    def null_count(table: sa.Table, column: sa.Column) -> int:
        stmt = sa.select(sa.func.count()).select_from(table).where(column.is_(None))
        return conn.execute(stmt).scalar_one()

    report = {
        "row_counts": {
            "races": count(races),
            "drivers": count(drivers),
            "teams": count(teams),
            "race_results": count(race_results),
            "qualifying_results": count(qualifying_results),
            "lap_times": count(lap_times),
            "pit_stops": count(pit_stops),
            "race_weather": count(race_weather),
            "race_weather_forecasts": count(race_weather_forecasts),
            "circuits": count(circuits),
            "circuit_characteristics": count(circuit_characteristics),
            "circuit_history": count(circuit_history),
            "driver_features": count(driver_features),
        },
        "missing_values": {
            "race_results_race_time": null_count(race_results, race_results.c.race_time),
            "qualifying_q1_time": null_count(qualifying_results, qualifying_results.c.q1_time),
            "lap_times_lap_time": null_count(lap_times, lap_times.c.lap_time),
            "pit_stops_duration": null_count(pit_stops, pit_stops.c.duration),
            "race_weather_air_temp": null_count(race_weather, race_weather.c.air_temp),
            "circuits_circuit_length": null_count(circuits, circuits.c.circuit_length),
            "driver_features_last_3_avg": null_count(
                driver_features, driver_features.c.last_3_races_avg_position
            ),
        },
    }
    return report


def classify_conditions(rain_amount: Optional[float], rain_probability: Optional[float]) -> str:
    if rain_amount and rain_amount >= 0.5:
        return "wet"
    if rain_probability and rain_probability >= 50:
        return "mixed"
    return "dry"


def get_session_times(event: pd.Series) -> List[Tuple[str, datetime]]:
    sessions = []
    for index in range(1, 6):
        name = event.get(f"Session{index}")
        session_date = event.get(f"Session{index}Date")
        if pd.isna(name) or pd.isna(session_date):
            continue
        if hasattr(session_date, "to_pydatetime"):
            session_dt = session_date.to_pydatetime()
        else:
            session_dt = session_date
        if session_dt.tzinfo is None:
            session_dt = session_dt.replace(tzinfo=timezone.utc)
        sessions.append((str(name), session_dt))
    return sessions


def fetch_openweather(
    api_key: str,
    base_url: str,
    lat: float,
    lon: float,
    timestamp: datetime,
) -> Optional[Dict[str, Any]]:
    unix_time = int(timestamp.timestamp())
    url = f"{base_url}/data/3.0/onecall/timemachine"
    params = {"lat": lat, "lon": lon, "dt": unix_time, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        LOG.warning("Weather API request failed: %s", exc)
        return None


def fetch_openweather_forecast(
    api_key: str,
    base_url: str,
    lat: float,
    lon: float,
) -> Optional[Dict[str, Any]]:
    url = f"{base_url}/data/3.0/onecall"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "exclude": "minutely,alerts"}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        LOG.warning("Weather forecast request failed: %s", exc)
        return None


def circuit_average_conditions(conn: sa.Connection, circuit_name: str, month: int) -> Dict[str, Optional[float]]:
    stmt = (
        sa.select(
            sa.func.avg(race_weather.c.air_temp).label("air_temp"),
            sa.func.avg(race_weather.c.track_temp).label("track_temp"),
            sa.func.avg(race_weather.c.rain_probability).label("rain_probability"),
            sa.func.avg(race_weather.c.rain_amount).label("rain_amount"),
            sa.func.avg(race_weather.c.humidity).label("humidity"),
            sa.func.avg(race_weather.c.wind_speed).label("wind_speed"),
        )
        .select_from(race_weather.join(races, race_weather.c.race_id == races.c.race_id))
        .where(
            races.c.circuit_name == circuit_name,
            sa.extract("month", races.c.race_date) == month,
            race_weather.c.is_estimated.is_(False),
        )
    )
    row = conn.execute(stmt).mappings().first()
    if not row:
        return {}
    return dict(row)


def build_weather_rows(
    conn: sa.Connection,
    race_id: int,
    circuit_name: str,
    event_date: date,
    lat: Optional[float],
    lon: Optional[float],
    sessions: List[Tuple[str, datetime]],
    api_key: Optional[str],
    base_url: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    month = event_date.month
    averages = circuit_average_conditions(conn, circuit_name, month)
    for session_type, session_dt in sessions:
        weather_payload = None
        if api_key and lat is not None and lon is not None:
            weather_payload = fetch_openweather(api_key, base_url, lat, lon, session_dt)
        data = None
        if weather_payload and "data" in weather_payload and weather_payload["data"]:
            data = weather_payload["data"][0]
        is_estimated = data is None
        air_temp = None
        track_temp = None
        humidity = None
        wind_speed = None
        wind_deg = None
        rain_amount = None
        rain_probability = None
        if data:
            air_temp = data.get("temp")
            track_temp = data.get("feels_like")
            humidity = data.get("humidity")
            wind_speed = data.get("wind_speed")
            wind_deg = data.get("wind_deg")
            rain_amount = data.get("rain", {}).get("1h") if isinstance(data.get("rain"), dict) else data.get("rain")
            rain_probability = data.get("pop")
        if is_estimated and averages:
            air_temp = averages.get("air_temp")
            track_temp = averages.get("track_temp")
            rain_probability = averages.get("rain_probability")
            rain_amount = averages.get("rain_amount")
            humidity = averages.get("humidity")
            wind_speed = averages.get("wind_speed")
        conditions = classify_conditions(rain_amount, rain_probability)
        avg_temp = averages.get("air_temp") if averages else None
        temp_deviation = None
        if air_temp is not None and avg_temp is not None:
            temp_deviation = air_temp - avg_temp
        wet_prob = None
        if rain_probability is not None:
            wet_prob = min(1.0, max(0.0, float(rain_probability)))
            if wet_prob > 1:
                wet_prob = wet_prob / 100
        weather_change_probability = None
        if rain_probability is not None and avg_temp is not None and air_temp is not None:
            weather_change_probability = min(1.0, abs(air_temp - avg_temp) / 15.0)
        rows.append(
            {
                "race_id": race_id,
                "session_type": session_type,
                "air_temp": air_temp,
                "track_temp": track_temp,
                "rain_probability": rain_probability,
                "rain_amount": rain_amount,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_deg,
                "conditions": conditions,
                "wet_race_probability": wet_prob,
                "temp_deviation": temp_deviation,
                "weather_change_probability": weather_change_probability,
                "is_estimated": is_estimated,
                "source": "openweather" if not is_estimated else "estimated",
                "observed_at": session_dt,
            }
        )
    return rows


def build_weather_forecast_rows(
    race_id: int,
    sessions: List[Tuple[str, datetime]],
    lat: Optional[float],
    lon: Optional[float],
    api_key: Optional[str],
    base_url: str,
) -> List[Dict[str, Any]]:
    if not api_key or lat is None or lon is None:
        return []
    forecast_payload = fetch_openweather_forecast(api_key, base_url, lat, lon)
    if not forecast_payload or "hourly" not in forecast_payload:
        return []
    hourly = forecast_payload["hourly"]
    rows: List[Dict[str, Any]] = []
    for session_type, session_dt in sessions:
        target_ts = int(session_dt.timestamp())
        closest = min(hourly, key=lambda h: abs(h.get("dt", target_ts) - target_ts))
        rain_amount = None
        if "rain" in closest:
            rain_amount = closest["rain"].get("1h") if isinstance(closest["rain"], dict) else closest["rain"]
        rows.append(
            {
                "race_id": race_id,
                "session_type": session_type,
                "forecast_time": datetime.now(timezone.utc),
                "air_temp": closest.get("temp"),
                "track_temp": closest.get("feels_like"),
                "rain_probability": closest.get("pop"),
                "rain_amount": rain_amount,
                "humidity": closest.get("humidity"),
                "wind_speed": closest.get("wind_speed"),
                "wind_direction": closest.get("wind_deg"),
                "conditions": classify_conditions(rain_amount, closest.get("pop")),
                "source": "openweather_forecast",
            }
        )
    return rows


def ingest_season(engine: Engine, season: int, sleep_time: float, weather_api_key: Optional[str], base_url: str) -> None:
    schedule = get_schedule(season)
    with engine.begin() as conn:
        for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"Season {season}"):
            round_number = int(event["RoundNumber"])
            race_name = event["EventName"]
            circuit_name = event["Location"]
            race_date = event["EventDate"].date() if pd.notna(event["EventDate"]) else None
            if not race_date:
                LOG.warning("Skipping round %s due to missing race date", round_number)
                continue
            try:
                race_session = load_session_with_retry(season, round_number, "R")
                qualifying_session = load_session_with_retry(season, round_number, "Q")
            except Exception as exc:  # noqa: BLE001
                LOG.error("Skipping round %s due to load failure: %s", round_number, exc)
                continue
            race_results_df = race_session.results
            if race_results_df is None or race_results_df.empty:
                LOG.warning("No race results for %s round %s", season, round_number)
                continue
            winning_time = None
            winner_row = race_results_df[race_results_df["Position"] == 1]
            if not winner_row.empty:
                winning_time = to_seconds(winner_row.iloc[0].get("Time"))
            race_id = ensure_race(
                conn,
                season,
                round_number,
                race_name,
                circuit_name,
                race_date,
                winning_time,
            )
            ensure_circuit(
                conn,
                circuit_name,
                event.get("Country"),
                event.get("CircuitType"),
            )
            lat = event.get("Latitude")
            lon = event.get("Longitude")
            sessions = get_session_times(event)
            weather_rows = build_weather_rows(
                conn,
                race_id,
                circuit_name,
                race_date,
                lat,
                lon,
                sessions,
                weather_api_key,
                base_url,
            )
            upsert_rows(conn, race_weather, weather_rows)
            if season >= 2026:
                forecast_rows = build_weather_forecast_rows(
                    race_id,
                    sessions,
                    lat,
                    lon,
                    weather_api_key,
                    base_url,
                )
                upsert_rows(conn, race_weather_forecasts, forecast_rows)
            driver_rows = build_driver_rows(race_results_df)
            team_rows = build_team_rows(race_results_df)
            driver_map = ensure_drivers(conn, driver_rows)
            team_map = ensure_teams(conn, team_rows)
            fastest_lap = None
            if race_session.laps is not None and not race_session.laps.empty:
                fastest_lap_data = race_session.laps.pick_fastest()
                if fastest_lap_data is not None:
                    fastest_lap = fastest_lap_data.get("Driver")
            results_rows = build_race_results(race_results_df, race_id, driver_map, team_map, fastest_lap)
            upsert_rows(conn, race_results, results_rows)
            qualifying_df = qualifying_session.results
            if qualifying_df is not None and not qualifying_df.empty:
                qualifying_rows = build_qualifying_results(qualifying_df, race_id, driver_map)
                upsert_rows(conn, qualifying_results, qualifying_rows)
            if race_session.laps is not None and not race_session.laps.empty:
                laps_df = race_session.laps
                lap_rows = build_lap_times(laps_df, race_id, driver_map)
                for chunk in chunked(lap_rows, size=5000):
                    upsert_rows(conn, lap_times, chunk)
                pit_laps = laps_df.pick_pitstops()
                if pit_laps is not None and not pit_laps.empty:
                    pit_rows = build_pit_stops(pit_laps, laps_df, race_id, driver_map)
                    for chunk in chunked(pit_rows, size=2000):
                        upsert_rows(conn, pit_stops, chunk)
            time.sleep(sleep_time)


def build_circuit_metrics(
    conn: sa.Connection,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    circuits_df = pd.read_sql(
        sa.select(circuits.c.circuit_id, circuits.c.circuit_name),
        conn,
    )
    races_df = pd.read_sql(
        sa.select(
            races.c.race_id,
            races.c.circuit_name,
            races.c.season,
            races.c.race_date,
        ),
        conn,
    )
    results_df = pd.read_sql(
        sa.select(
            race_results.c.race_id,
            race_results.c.driver_id,
            race_results.c.grid_position,
            race_results.c.finishing_position,
            race_results.c.race_time,
            race_results.c.status,
        ),
        conn,
    )
    qualifying_df = pd.read_sql(
        sa.select(
            qualifying_results.c.race_id,
            qualifying_results.c.driver_id,
            qualifying_results.c.grid_position.label("qualifying_position"),
        ),
        conn,
    )
    laps_df = pd.read_sql(
        sa.select(
            lap_times.c.race_id,
            lap_times.c.driver_id,
            lap_times.c.lap_number,
            lap_times.c.lap_time,
        ),
        conn,
    )
    pits_df = pd.read_sql(
        sa.select(
            pit_stops.c.race_id,
            pit_stops.c.driver_id,
            pit_stops.c.stop_number,
        ),
        conn,
    )
    weather_df = pd.read_sql(
        sa.select(
            race_weather.c.race_id,
            race_weather.c.air_temp,
            race_weather.c.rain_probability,
        ),
        conn,
    )
    drivers_df = pd.read_sql(
        sa.select(
            drivers.c.driver_id,
            drivers.c.full_name,
        ),
        conn,
    )

    results_df = results_df.merge(races_df, on="race_id", how="left")
    qualifying_df = qualifying_df.merge(races_df, on="race_id", how="left")
    laps_df = laps_df.merge(races_df, on="race_id", how="left")
    pits_df = pits_df.merge(races_df, on="race_id", how="left")
    weather_df = weather_df.merge(races_df, on="race_id", how="left")
    laps_with_driver = laps_df.merge(drivers_df, on="driver_id", how="left")

    return circuits_df, {
        "races": races_df,
        "results": results_df,
        "qualifying": qualifying_df,
        "laps": laps_df,
        "pits": pits_df,
        "weather": weather_df,
        "laps_with_driver": laps_with_driver,
    }, races_df


def calculate_circuit_characteristics(
    circuit_name: str,
    data: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[float]]:
    results_df = data["results"][data["results"]["circuit_name"] == circuit_name]
    qualifying_df = data["qualifying"][data["qualifying"]["circuit_name"] == circuit_name]
    laps_df = data["laps"][data["laps"]["circuit_name"] == circuit_name]
    pits_df = data["pits"][data["pits"]["circuit_name"] == circuit_name]

    overtaking_frequency = None
    grid_finish_correlation = None
    if not results_df.empty:
        valid = results_df.dropna(subset=["grid_position", "finishing_position"])
        if not valid.empty:
            overtaking_frequency = (valid["finishing_position"] - valid["grid_position"]).abs().mean()
            grid_finish_correlation = valid["grid_position"].corr(valid["finishing_position"])

    qualifying_importance = None
    if not results_df.empty and not qualifying_df.empty:
        combined = results_df.merge(
            qualifying_df,
            on=["race_id", "driver_id"],
            how="inner",
        )
        combined = combined.dropna(subset=["qualifying_position", "finishing_position"])
        if not combined.empty:
            qualifying_importance = combined["qualifying_position"].corr(combined["finishing_position"])

    tire_degradation_severity = None
    if not laps_df.empty:
        lap_valid = laps_df.dropna(subset=["lap_time", "lap_number"])
        if not lap_valid.empty:
            lap_valid = lap_valid.sort_values(["race_id", "driver_id", "lap_number"])
            first_last = lap_valid.groupby(["race_id", "driver_id"])["lap_time"].agg(["first", "last"])
            if not first_last.empty:
                tire_degradation_severity = (first_last["last"] - first_last["first"]).mean()

    track_evolution_factor = None
    if not laps_df.empty:
        lap_valid = laps_df.dropna(subset=["lap_time", "lap_number"])
        if not lap_valid.empty:
            evolution_scores = []
            for race_id, race_laps in lap_valid.groupby("race_id"):
                max_lap = race_laps["lap_number"].max()
                first_slice = race_laps[race_laps["lap_number"] <= 5]
                last_slice = race_laps[race_laps["lap_number"] >= max_lap - 4]
                if first_slice.empty or last_slice.empty:
                    continue
                first_mean = first_slice["lap_time"].mean()
                last_mean = last_slice["lap_time"].mean()
                if first_mean and first_mean > 0:
                    evolution = (first_mean - last_mean) / first_mean
                    evolution_scores.append(max(0.0, min(1.0, evolution)))
            if evolution_scores:
                track_evolution_factor = float(pd.Series(evolution_scores).mean())

    safety_car_probability = None
    if not results_df.empty:
        safety_car_probability = (
            results_df.groupby("race_id")["status"]
            .apply(lambda s: s.fillna("").str.contains("Safety Car|SC|VSC", case=False).any())
            .mean()
        )

    average_pit_stops = None
    strategy_variation = None
    if not pits_df.empty:
        pit_counts = pits_df.groupby(["race_id", "driver_id"])["stop_number"].max().reset_index(name="pit_stops")
        if not pit_counts.empty:
            average_pit_stops = pit_counts["pit_stops"].mean()
            variation_by_race = pit_counts.groupby("race_id")["pit_stops"].nunique()
            strategy_variation = (variation_by_race > 1).mean()

    typical_strategy = None
    if average_pit_stops is not None:
        if average_pit_stops <= 1.5:
            typical_strategy = "1-stop"
        elif average_pit_stops <= 2.5:
            typical_strategy = "2-stop"
        else:
            typical_strategy = "variable"

    overtaking_difficulty = None
    if overtaking_frequency is not None:
        scaled = 10 - min(9, int(round(overtaking_frequency)))
        overtaking_difficulty = max(1, scaled)

    return {
        "overtaking_frequency": overtaking_frequency,
        "grid_finish_correlation": grid_finish_correlation,
        "qualifying_importance": qualifying_importance,
        "tire_degradation_severity": tire_degradation_severity,
        "strategy_variation": strategy_variation,
        "average_pit_stops": average_pit_stops,
        "typical_strategy": typical_strategy,
        "overtaking_difficulty": overtaking_difficulty,
        "track_evolution_factor": track_evolution_factor,
        "safety_car_probability": safety_car_probability,
    }


def calculate_circuit_history(
    circuit_name: str,
    season: int,
    data: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[float]]:
    results_df = data["results"][
        (data["results"]["circuit_name"] == circuit_name) & (data["results"]["season"] == season)
    ]
    races_df = data["races"][
        (data["races"]["circuit_name"] == circuit_name) & (data["races"]["season"] == season)
    ]
    weather_df = data["weather"][
        (data["weather"]["circuit_name"] == circuit_name) & (data["weather"]["season"] == season)
    ]

    pole_win_rate = None
    average_winning_margin = None
    safety_car_frequency = None
    red_flag_count = None
    weather_variability = None

    if not results_df.empty:
        winners = results_df[results_df["finishing_position"] == 1]
        if not winners.empty:
            pole_win_rate = (winners["grid_position"] == 1).mean()
        race_times = results_df.pivot_table(
            index="race_id",
            columns="finishing_position",
            values="race_time",
            aggfunc="first",
        )
        if 1 in race_times and 2 in race_times:
            margins = race_times[2] - race_times[1]
            average_winning_margin = margins.dropna().mean()
        safety_car_frequency = (
            results_df.groupby("race_id")["status"]
            .apply(lambda s: s.fillna("").str.contains("Safety Car|SC|VSC", case=False).any())
            .mean()
        )
        red_flag_count = (
            results_df.groupby("race_id")["status"]
            .apply(lambda s: s.fillna("").str.contains("Red Flag", case=False).any())
            .sum()
        )

    if not weather_df.empty:
        variability = []
        if weather_df["air_temp"].notna().any():
            variability.append(weather_df["air_temp"].std())
        if weather_df["rain_probability"].notna().any():
            variability.append(weather_df["rain_probability"].std())
        if variability:
            weather_variability = float(pd.Series(variability).mean())

    return {
        "pole_win_rate": pole_win_rate,
        "average_winning_margin": average_winning_margin,
        "safety_car_frequency": safety_car_frequency,
        "red_flag_count": red_flag_count,
        "weather_variability": weather_variability,
    }


def update_circuit_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        circuits_df, data, races_df = build_circuit_metrics(conn)

        lap_records = None
        if not data["laps_with_driver"].empty:
            lap_records = (
                data["laps_with_driver"]
                .dropna(subset=["lap_time"])
                .sort_values("lap_time")
                .groupby("circuit_name")
                .first()
                .reset_index()
            )

        for _, circuit in circuits_df.iterrows():
            circuit_id = circuit["circuit_id"]
            circuit_name = circuit["circuit_name"]
            metrics = calculate_circuit_characteristics(circuit_name, data)
            lap_record = None
            lap_record_holder = None
            lap_record_year = None
            if lap_records is not None and not lap_records.empty:
                record_row = lap_records[lap_records["circuit_name"] == circuit_name]
                if not record_row.empty:
                    record_row = record_row.iloc[0]
                    lap_record = record_row.get("lap_time")
                    lap_record_holder = record_row.get("full_name")
                    lap_record_year = record_row.get("season")

            circuit_stmt = pg_insert(circuits).values(
                circuit_id=circuit_id,
                circuit_name=circuit_name,
                lap_record=lap_record,
                lap_record_holder=lap_record_holder,
                lap_record_year=lap_record_year,
            )
            circuit_stmt = circuit_stmt.on_conflict_do_update(
                constraint="circuits_circuit_name_key",
                set_={
                    "lap_record": lap_record,
                    "lap_record_holder": lap_record_holder,
                    "lap_record_year": lap_record_year,
                },
            )
            conn.execute(circuit_stmt)

            characteristics_stmt = pg_insert(circuit_characteristics).values(
                circuit_id=circuit_id,
                high_speed_corners=None,
                medium_speed_corners=None,
                low_speed_corners=None,
                drs_zones=None,
                pit_lane_time_loss=None,
                **metrics,
            )
            characteristics_stmt = characteristics_stmt.on_conflict_do_update(
                constraint="circuit_characteristics_circuit_id_key",
                set_=metrics,
            )
            conn.execute(characteristics_stmt)

            seasons = races_df[races_df["circuit_name"] == circuit_name]["season"].unique()
            for season in seasons:
                history = calculate_circuit_history(circuit_name, season, data)
                history_stmt = pg_insert(circuit_history).values(
                    circuit_id=circuit_id,
                    season=int(season),
                    **history,
                )
                history_stmt = history_stmt.on_conflict_do_update(
                    constraint="uq_circuit_history",
                    set_=history,
                )
                conn.execute(history_stmt)


def weighted_average(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def exponential_decay_average(values: List[float], decay: float = 0.7) -> Optional[float]:
    if not values:
        return None
    weights = [decay**i for i in range(len(values))]
    return weighted_average(values, weights)


def calculate_driver_features(
    conn: sa.Connection,
    driver_id: int,
    target_race_id: int,
    lookback_period: int,
) -> Dict[str, Any]:
    races_df = pd.read_sql(
        sa.select(races.c.race_id, races.c.season, races.c.round, races.c.circuit_name),
        conn,
    )
    races_df = races_df.sort_values(["season", "round"]).reset_index(drop=True)
    races_df["order_index"] = range(len(races_df))
    results_df = pd.read_sql(
        sa.select(
            race_results.c.race_id,
            race_results.c.driver_id,
            race_results.c.team_id,
            race_results.c.grid_position,
            race_results.c.finishing_position,
            race_results.c.points,
            race_results.c.status,
        ),
        conn,
    )
    qualifying_df = pd.read_sql(
        sa.select(
            qualifying_results.c.race_id,
            qualifying_results.c.driver_id,
            qualifying_results.c.grid_position.label("qualifying_position"),
        ),
        conn,
    )
    lap_times_df = pd.read_sql(
        sa.select(
            lap_times.c.race_id,
            lap_times.c.driver_id,
            lap_times.c.lap_number,
            lap_times.c.lap_time,
        ),
        conn,
    )
    weather_df = pd.read_sql(
        sa.select(
            race_weather.c.race_id,
            race_weather.c.conditions,
            race_weather.c.session_type,
        ),
        conn,
    )
    target_race = races_df[races_df["race_id"] == target_race_id]
    if target_race.empty:
        return {}
    target_race_row = target_race.iloc[0]
    target_circuit = target_race_row["circuit_name"]
    target_season = target_race_row["season"]
    target_order = target_race_row["order_index"]

    results_df = results_df.merge(races_df, on="race_id", how="left")
    qualifying_df = qualifying_df.merge(races_df, on="race_id", how="left")
    lap_times_df = lap_times_df.merge(races_df, on="race_id", how="left")
    weather_df = weather_df.merge(races_df, on="race_id", how="left")

    driver_results = results_df[results_df["driver_id"] == driver_id].sort_values(
        ["season", "round"]
    )
    driver_results = driver_results[driver_results["order_index"] <= target_order]
    recent_results = driver_results.tail(lookback_period)

    last_three = recent_results.tail(3)["finishing_position"].dropna().tolist()
    last_three_avg = weighted_average(last_three[::-1], [0.5, 0.3, 0.2][: len(last_three)])

    last_five = recent_results.tail(5)["finishing_position"].dropna().tolist()
    last_five_avg = exponential_decay_average(last_five[::-1])

    last_three_points = recent_results.tail(3)["points"].dropna().sum()

    last_ten = recent_results.tail(10)
    dnf_mask = last_ten["status"].fillna("").str.contains("DNF|DNS|DSQ|Ret", case=False)
    recent_dnf_rate = dnf_mask.mean() if not last_ten.empty else None

    recent_qualifying = qualifying_df[
        (qualifying_df["driver_id"] == driver_id) & (qualifying_df["order_index"] <= target_order)
    ].sort_values(["season", "round"])
    recent_qualifying_avg = recent_qualifying.tail(10)["qualifying_position"].dropna().mean()

    positions_gained = None
    if not recent_results.empty:
        gains = (recent_results["grid_position"] - recent_results["finishing_position"]).dropna()
        positions_gained = gains.tail(3).mean()

    circuit_results = results_df[
        (results_df["driver_id"] == driver_id)
        & (results_df["circuit_name"] == target_circuit)
        & (results_df["season"] >= target_season - 5)
        & (results_df["season"] <= target_season)
    ]
    avg_finish_circuit = circuit_results["finishing_position"].dropna().mean()
    best_result = circuit_results["finishing_position"].dropna().min()
    worst_result = circuit_results["finishing_position"].dropna().max()

    qualifying_conversion = None
    circuit_qual = qualifying_df[
        (qualifying_df["driver_id"] == driver_id) & (qualifying_df["circuit_name"] == target_circuit)
    ]
    if not circuit_results.empty and not circuit_qual.empty:
        merged = circuit_results.merge(circuit_qual, on=["race_id", "driver_id"], how="inner")
        if not merged.empty:
            qualifying_conversion = (
                merged["qualifying_position"].dropna().mean() - merged["finishing_position"].dropna().mean()
            )

    crash_rate = None
    if not circuit_results.empty:
        crash_rate = (
            circuit_results["status"].fillna("").str.contains("Accident|Crash|Collision", case=False).mean()
        )

    qualifying_pace_percentile = None
    if not recent_qualifying.empty:
        recent_qualifying["rank"] = recent_qualifying.groupby("race_id")["qualifying_position"].rank(
            method="average"
        )
        recent_qualifying["field_size"] = recent_qualifying.groupby("race_id")["qualifying_position"].transform(
            "count"
        )
        recent_qualifying["percentile"] = 1 - (
            (recent_qualifying["rank"] - 1) / (recent_qualifying["field_size"] - 1)
        )
        qualifying_pace_percentile = recent_qualifying.tail(lookback_period)["percentile"].mean()

    race_pace_percentile = None
    if not recent_results.empty:
        recent_results = recent_results.copy()
        recent_results["rank"] = recent_results.groupby("race_id")["finishing_position"].rank(
            method="average"
        )
        recent_results["field_size"] = recent_results.groupby("race_id")["finishing_position"].transform("count")
        recent_results["percentile"] = 1 - (
            (recent_results["rank"] - 1) / (recent_results["field_size"] - 1)
        )
        race_pace_percentile = recent_results["percentile"].mean()

    overtaking_success_rate = None
    if not recent_results.empty:
        opportunities = recent_results["grid_position"].dropna() - 1
        overtakes = (recent_results["grid_position"] - recent_results["finishing_position"]).clip(lower=0)
        if not opportunities.empty and opportunities.sum() > 0:
            overtaking_success_rate = (overtakes.sum() / opportunities.sum()).item()

    wet_multiplier = None
    if not weather_df.empty:
        wet_races = weather_df[
            (weather_df["conditions"] == "wet") & (weather_df["session_type"].str.contains("Race", case=False))
        ]["race_id"].unique()
        dry_races = weather_df[
            (weather_df["conditions"] == "dry") & (weather_df["session_type"].str.contains("Race", case=False))
        ]["race_id"].unique()
        wet_results = driver_results[driver_results["race_id"].isin(wet_races)]
        dry_results = driver_results[driver_results["race_id"].isin(dry_races)]
        if not wet_results.empty and not dry_results.empty:
            wet_avg = wet_results["finishing_position"].dropna().mean()
            dry_avg = dry_results["finishing_position"].dropna().mean()
            if dry_avg and dry_avg > 0:
                wet_multiplier = wet_avg / dry_avg

    consistency_score = None
    lap_subset = lap_times_df[
        (lap_times_df["driver_id"] == driver_id) & (lap_times_df["order_index"] <= target_order)
    ]
    lap_subset = lap_subset.dropna(subset=["lap_time"])
    if not lap_subset.empty:
        std_dev = lap_subset["lap_time"].std()
        if std_dev and std_dev > 0:
            consistency_score = 1 / std_dev

    tire_management_score = None
    if not lap_subset.empty:
        degradation_scores = []
        for _, race_group in lap_subset.groupby("race_id"):
            race_group = race_group.sort_values("lap_number")
            if race_group.empty:
                continue
            first = race_group["lap_time"].iloc[0]
            last = race_group["lap_time"].iloc[-1]
            if first and first > 0:
                degradation_scores.append((last - first) / first)
        if degradation_scores:
            tire_management_score = float(pd.Series(degradation_scores).mean())

    total_f1_races = int(driver_results["race_id"].nunique())
    races_at_circuit = int(circuit_results["race_id"].nunique())
    podium_count = int((driver_results["finishing_position"] <= 3).sum())
    win_count = int((driver_results["finishing_position"] == 1).sum())

    years_with_current_team = None
    if not driver_results.empty:
        latest_team = driver_results.iloc[-1]["team_id"]
        years_with_current_team = driver_results[driver_results["team_id"] == latest_team]["season"].nunique()

    teammate_qual_gap = None
    teammate_race_gap = None
    head_to_head_qualifying = None
    head_to_head_race = None
    if not driver_results.empty:
        target_team = driver_results.iloc[-1]["team_id"]
        team_results = results_df[
            (results_df["team_id"] == target_team)
            & (results_df["order_index"] <= target_order)
        ]
        teammate_results = team_results[team_results["driver_id"] != driver_id]
        if not teammate_results.empty:
            merged = driver_results.merge(
                teammate_results,
                on="race_id",
                suffixes=("_driver", "_mate"),
            )
            if not merged.empty:
                race_gap = merged["finishing_position_driver"] - merged["finishing_position_mate"]
                teammate_race_gap = race_gap.mean()
                head_to_head_race = (race_gap < 0).mean()
            qual_merged = qualifying_df[qualifying_df["driver_id"] == driver_id].merge(
                qualifying_df[qualifying_df["driver_id"].isin(teammate_results["driver_id"])],
                on="race_id",
                suffixes=("_driver", "_mate"),
            )
            if not qual_merged.empty:
                qual_gap = qual_merged["qualifying_position_driver"] - qual_merged["qualifying_position_mate"]
                teammate_qual_gap = qual_gap.mean()
                head_to_head_qualifying = (qual_gap < 0).mean()

    recent_incidents = driver_results.tail(5)
    recent_incidents_count = int(
        recent_incidents["status"]
        .fillna("")
        .str.contains("Accident|Crash|Collision|DNF", case=False)
        .sum()
    )

    consecutive_finishes = 0
    for _, row in driver_results[::-1].iterrows():
        status = str(row.get("status") or "")
        if "Finished" in status or row.get("finishing_position"):
            consecutive_finishes += 1
        else:
            break

    return {
        "last_3_races_avg_position": last_three_avg,
        "last_5_races_avg_position": last_five_avg,
        "last_3_races_points_scored": last_three_points,
        "recent_dnf_rate": recent_dnf_rate,
        "recent_qualifying_avg_position": recent_qualifying_avg,
        "positions_gained_last_3_races": positions_gained,
        "avg_finish_position_at_circuit": avg_finish_circuit,
        "best_result_at_circuit": best_result,
        "worst_result_at_circuit": worst_result,
        "qualifying_to_race_conversion_at_circuit": qualifying_conversion,
        "avg_positions_gained_lap1_at_circuit": None,
        "crash_rate_at_circuit": crash_rate,
        "qualifying_pace_percentile": qualifying_pace_percentile,
        "race_pace_percentile": race_pace_percentile,
        "overtaking_success_rate": overtaking_success_rate,
        "defending_success_rate": None,
        "wet_weather_performance_multiplier": wet_multiplier,
        "first_lap_position_change_avg": None,
        "consistency_score": consistency_score,
        "tire_management_score": tire_management_score,
        "total_f1_races": total_f1_races,
        "races_at_this_circuit": races_at_circuit,
        "years_with_current_team": years_with_current_team,
        "championship_experience": None,
        "podium_count": podium_count,
        "win_count": win_count,
        "qualifying_gap_to_teammate": teammate_qual_gap,
        "race_gap_to_teammate": teammate_race_gap,
        "head_to_head_qualifying": head_to_head_qualifying,
        "head_to_head_race": head_to_head_race,
        "championship_position": None,
        "points_from_leader": None,
        "contract_year_flag": None,
        "recent_incidents_count": recent_incidents_count,
        "consecutive_race_finishes": consecutive_finishes,
    }


def update_driver_features(engine: Engine, lookback_period: int = 10) -> None:
    with engine.begin() as conn:
        races_df = pd.read_sql(
            sa.select(races.c.race_id, races.c.season, races.c.round).order_by(races.c.season, races.c.round),
            conn,
        )
        results_df = pd.read_sql(
            sa.select(race_results.c.race_id, race_results.c.driver_id).distinct(),
            conn,
        )
        for race_id in races_df["race_id"].tolist():
            drivers_in_race = results_df[results_df["race_id"] == race_id]["driver_id"].tolist()
            for driver_id in drivers_in_race:
                feature_values = calculate_driver_features(conn, driver_id, race_id, lookback_period)
                if not feature_values:
                    continue
                stmt = pg_insert(driver_features).values(
                    race_id=race_id,
                    driver_id=driver_id,
                    **feature_values,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_driver_features",
                    set_=feature_values,
                )
                conn.execute(stmt)


def main() -> None:
    setup_logging()
    args = parse_args()
    enable_cache(args.cache_dir)
    engine = get_engine(args.database_url)
    create_schema(engine)
    for season in range(args.start_season, args.end_season + 1):
        LOG.info("Starting ingestion for season %s", season)
        ingest_season(engine, season, args.sleep, args.weather_api_key, args.weather_base_url)
    update_circuit_tables(engine)
    update_driver_features(engine)
    with engine.begin() as conn:
        report = build_data_quality_report(conn)
    report_path = "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOG.info("Data quality report written to %s", report_path)


if __name__ == "__main__":
    main()
