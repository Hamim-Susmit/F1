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

predictions = sa.Table(
    "predictions",
    metadata,
    sa.Column("prediction_id", sa.Integer, primary_key=True),
    sa.Column("race_id", sa.Integer, sa.ForeignKey("races.race_id"), nullable=False),
    sa.Column("driver_id", sa.Integer, sa.ForeignKey("drivers.driver_id"), nullable=False),
    sa.Column("model_version", sa.String, nullable=False),
    sa.Column("predicted_position", sa.Integer),
    sa.Column("predicted_time", sa.Float),
    sa.Column("dnf_probability", sa.Float),
    sa.Column("podium_probability", sa.Float),
    sa.Column("confidence", sa.Float),
    sa.Column("generated_at", sa.DateTime(timezone=True)),
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


LOG = logging.getLogger("f1_ingest")


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
            "team_features": count(team_features),
            "situational_features": count(situational_features),
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
            "team_features_momentum": null_count(
                team_features, team_features.c.momentum_score
            ),
            "situational_features_grid_position": null_count(
                situational_features, situational_features.c.grid_position
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
    pit_df = pd.read_sql(
        sa.select(
            pit_stops.c.race_id,
            pit_stops.c.driver_id,
            pit_stops.c.stop_number,
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

    results_df = results_df.merge(
        races_df[["race_id", "order_index"]], on="race_id", how="left"
    )
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


def calculate_team_features(
    conn: sa.Connection,
    team_id: int,
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
    pit_df = pd.read_sql(
        sa.select(
            pit_stops.c.race_id,
            pit_stops.c.driver_id,
            pit_stops.c.duration,
        ),
        conn,
    )
    circuits_df = pd.read_sql(
        sa.select(circuits.c.circuit_name, circuits.c.circuit_type),
        conn,
    )

    target_race = races_df[races_df["race_id"] == target_race_id]
    if target_race.empty:
        return {}
    target_order = target_race.iloc[0]["order_index"]
    target_circuit = target_race.iloc[0]["circuit_name"]

    results_df = results_df.merge(races_df, on="race_id", how="left")
    qualifying_df = qualifying_df.merge(races_df, on="race_id", how="left")
    pit_df = pit_df.merge(races_df, on="race_id", how="left")
    circuits_df = circuits_df.set_index("circuit_name")

    team_results = results_df[(results_df["team_id"] == team_id) & (results_df["order_index"] <= target_order)]
    recent_results = team_results.sort_values(["season", "round"]).tail(lookback_period)

    points_by_race = recent_results.groupby("race_id")["points"].sum()
    performance_trend = None
    if len(points_by_race) >= 2:
        x = pd.Series(range(len(points_by_race)), dtype=float)
        y = points_by_race.reset_index(drop=True).astype(float)
        performance_trend = float(pd.Series(y).corr(pd.Series(x)) or 0) * y.std()

    points_last3 = points_by_race.tail(3).mean() if len(points_by_race) >= 3 else None
    points_prev3 = points_by_race.tail(6).head(3).mean() if len(points_by_race) >= 6 else None
    points_delta = None
    if points_last3 is not None and points_prev3 is not None:
        points_delta = points_last3 - points_prev3

    constructor_change = None
    if len(points_by_race) >= 5:
        recent_rank = points_by_race.rank(ascending=False).iloc[-1]
        past_rank = points_by_race.rank(ascending=False).iloc[-5]
        constructor_change = float(past_rank - recent_rank)

    upgrades_impact = None
    if len(points_by_race) >= 2:
        upgrades_impact = points_by_race.diff().mean()

    dnf_rate = None
    mechanical_dnf = None
    if not recent_results.empty:
        dnf_mask = recent_results["status"].fillna("").str.contains("DNF|DNS|DSQ|Ret", case=False)
        dnf_rate = dnf_mask.mean()
        mechanical_mask = recent_results["status"].fillna("").str.contains(
            "Engine|Gearbox|Hydraulic|Mechanical|Power|ERS", case=False
        )
        mechanical_dnf = mechanical_mask.mean()

    races_both_finished = None
    if not recent_results.empty:
        finish_by_race = (
            recent_results.groupby("race_id")["status"]
            .apply(lambda s: (~s.fillna("").str.contains("DNF|DNS|DSQ|Ret", case=False)).all())
        )
        races_both_finished = finish_by_race.mean()

    engine_penalty_frequency = (
        recent_results["status"].fillna("").str.contains("Penalty", case=False).mean()
        if not recent_results.empty
        else None
    )

    qualifying_balance = None
    if not recent_results.empty:
        team_qual = qualifying_df[
            (qualifying_df["race_id"].isin(recent_results["race_id"])) &
            (qualifying_df["driver_id"].isin(recent_results["driver_id"]))
        ]
        if not team_qual.empty:
            qual_avg = team_qual["qualifying_position"].mean()
            race_avg = recent_results["finishing_position"].mean()
            if qual_avg is not None and race_avg is not None:
                qualifying_balance = qual_avg - race_avg

    straight_line_speed = None
    high_speed_corner = None
    low_speed_corner = None
    if not recent_results.empty:
        avg_finish = recent_results["finishing_position"].mean()
        if avg_finish is not None:
            rank_score = max(1, 11 - int(round(avg_finish)))
            straight_line_speed = rank_score
            high_speed_corner = rank_score
            low_speed_corner = rank_score

    overall_downforce = None
    if high_speed_corner is not None and low_speed_corner is not None:
        overall_downforce = "high" if high_speed_corner >= low_speed_corner else "medium"

    tire_wear = None
    if not recent_results.empty:
        tire_wear = "neutral"

    circuit_type = circuits_df.loc[target_circuit]["circuit_type"] if target_circuit in circuits_df.index else None
    team_perf_circuit_type = None
    if circuit_type:
        circuit_races = results_df[
            (results_df["team_id"] == team_id) &
            (results_df["order_index"] <= target_order)
        ].merge(circuits_df, left_on="circuit_name", right_index=True, how="left")
        subset = circuit_races[circuit_races["circuit_type"] == circuit_type]
        if not subset.empty:
            team_perf_circuit_type = subset["points"].mean()

    historical_avg_points = None
    best_finish = None
    circuit_history = results_df[
        (results_df["team_id"] == team_id)
        & (results_df["circuit_name"] == target_circuit)
        & (results_df["season"] >= target_race.iloc[0]["season"] - 5)
    ]
    if not circuit_history.empty:
        historical_avg_points = circuit_history["points"].mean()
        best_finish = circuit_history["finishing_position"].min()

    match_score = None
    if circuit_type:
        if overall_downforce == "high" and circuit_type == "Permanent":
            match_score = 0.8
        elif overall_downforce == "high" and circuit_type == "Street":
            match_score = 0.6
        elif overall_downforce == "medium":
            match_score = 0.5

    pit_recent = pit_df[
        (pit_df["race_id"].isin(recent_results["race_id"])) &
        (pit_df["driver_id"].isin(recent_results["driver_id"]))
    ]
    avg_pit_duration = pit_recent["duration"].mean() if not pit_recent.empty else None
    pit_consistency = pit_recent["duration"].std() if not pit_recent.empty else None

    strategy_success = None
    if avg_pit_duration is not None:
        strategy_success = 1 - min(1.0, avg_pit_duration / 5.0)

    team_orders = None
    if not recent_results.empty:
        team_orders = (recent_results["finishing_position"].diff().abs() < 1).mean()

    reliability_score = None
    if dnf_rate is not None:
        reliability_score = 1 - dnf_rate

    momentum_score = None
    if points_last3 is not None and performance_trend is not None and reliability_score is not None:
        recent_performance = points_last3 / 25 if points_last3 else 0
        development_rate = performance_trend / 10 if performance_trend else 0
        driver_perf_avg = recent_results["finishing_position"].mean() if not recent_results.empty else 0
        driver_perf_avg = 1 - (driver_perf_avg / 20) if driver_perf_avg else 0
        team_morale_proxy = races_both_finished if races_both_finished is not None else 0
        momentum_score = (
            recent_performance * 0.3
            + development_rate * 0.25
            + reliability_score * 0.2
            + driver_perf_avg * 0.15
            + team_morale_proxy * 0.1
        )

    return {
        "performance_trend_last_5_races": performance_trend,
        "points_per_race_last_3_vs_previous_3": points_delta,
        "constructor_position_change": constructor_change,
        "upgrades_impact_score": upgrades_impact,
        "dnf_rate_last_10_races": dnf_rate,
        "mechanical_dnf_rate": mechanical_dnf,
        "races_both_cars_finished": races_both_finished,
        "engine_penalty_frequency": engine_penalty_frequency,
        "straight_line_speed_ranking": straight_line_speed,
        "high_speed_corner_ranking": high_speed_corner,
        "low_speed_corner_ranking": low_speed_corner,
        "overall_downforce_level": overall_downforce,
        "tire_wear_rate": tire_wear,
        "qualifying_vs_race_pace_balance": qualifying_balance,
        "team_performance_at_circuit_type": team_perf_circuit_type,
        "historical_avg_points_at_circuit": historical_avg_points,
        "best_finish_at_circuit": best_finish,
        "car_characteristics_circuit_match_score": match_score,
        "avg_pit_stop_duration": avg_pit_duration,
        "pit_stop_consistency": pit_consistency,
        "strategy_success_rate": strategy_success,
        "team_orders_likelihood": team_orders,
        "momentum_score": momentum_score,
    }


def update_team_features(engine: Engine, lookback_period: int = 10) -> None:
    with engine.begin() as conn:
        races_df = pd.read_sql(
            sa.select(races.c.race_id, races.c.season, races.c.round).order_by(races.c.season, races.c.round),
            conn,
        )
        team_ids = pd.read_sql(sa.select(teams.c.team_id), conn)["team_id"].tolist()
        for race_id in races_df["race_id"].tolist():
            for team_id in team_ids:
                feature_values = calculate_team_features(conn, team_id, race_id, lookback_period)
                if not feature_values:
                    continue
                stmt = pg_insert(team_features).values(
                    race_id=race_id,
                    team_id=team_id,
                    **feature_values,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_team_features",
                    set_=feature_values,
                )
                conn.execute(stmt)


def extract_race_features(conn: sa.Connection, race_id: int, driver_id: int) -> Dict[str, Any]:
    race_row = conn.execute(
        sa.select(races.c.race_id, races.c.season, races.c.round, races.c.circuit_name).where(
            races.c.race_id == race_id
        )
    ).mappings().first()
    if not race_row:
        return {}

    season = race_row["season"]
    round_number = race_row["round"]
    race_scope = sa.or_(
        races.c.season < season,
        sa.and_(races.c.season == season, races.c.round <= round_number),
    )
    races_df = pd.read_sql(
        sa.select(races.c.race_id, races.c.season, races.c.round, races.c.circuit_name)
        .where(race_scope)
        .order_by(races.c.season, races.c.round),
        conn,
    )
    races_df = races_df.reset_index(drop=True)
    races_df["order_index"] = range(len(races_df))
    race_ids = races_df["race_id"].tolist()
    if not race_ids:
        return {}
    results_df = pd.read_sql(
        sa.select(
            race_results.c.race_id,
            race_results.c.driver_id,
            race_results.c.team_id,
            race_results.c.grid_position,
            race_results.c.finishing_position,
            race_results.c.points,
            race_results.c.status,
            races.c.season,
            races.c.round,
            races.c.circuit_name,
        )
        .select_from(race_results.join(races, race_results.c.race_id == races.c.race_id))
        .where(race_scope),
        conn,
    )
    qualifying_df = pd.read_sql(
        sa.select(
            qualifying_results.c.race_id,
            qualifying_results.c.driver_id,
            qualifying_results.c.grid_position.label("qualifying_position"),
        ).where(qualifying_results.c.race_id == race_id),
        conn,
    )
    laps_df = pd.read_sql(
        sa.select(
            lap_times.c.race_id,
            lap_times.c.driver_id,
            lap_times.c.lap_number,
            lap_times.c.lap_time,
            lap_times.c.tire_compound,
        ).where(lap_times.c.race_id == race_id),
        conn,
    )
    pit_df = pd.read_sql(
        sa.select(
            pit_stops.c.race_id,
            pit_stops.c.driver_id,
            pit_stops.c.stop_number,
            pit_stops.c.duration,
        ).where(pit_stops.c.race_id == race_id),
        conn,
    )
    weather_df = pd.read_sql(
        sa.select(
            race_weather.c.race_id,
            race_weather.c.conditions,
            race_weather.c.wet_race_probability,
            race_weather.c.weather_change_probability,
        ).where(race_weather.c.race_id == race_id),
        conn,
    )
    circuits_df = pd.read_sql(
        sa.select(circuits.c.circuit_name, circuits.c.circuit_type),
        conn,
    ).set_index("circuit_name")
    circuit_char_df = pd.read_sql(
        sa.select(
            circuits.c.circuit_id,
            circuits.c.circuit_name,
            circuit_characteristics.c.safety_car_probability,
            circuit_characteristics.c.overtaking_difficulty,
            circuit_characteristics.c.tire_degradation_severity,
            circuit_characteristics.c.track_evolution_factor,
        ).select_from(
            circuits.join(circuit_characteristics, circuits.c.circuit_id == circuit_characteristics.c.circuit_id, isouter=True)
        ),
        conn,
    )
    circuit_history_df = pd.read_sql(
        sa.select(
            circuits.c.circuit_name,
            circuit_history.c.season,
            circuit_history.c.red_flag_count,
        ).select_from(
            circuits.join(circuit_history, circuits.c.circuit_id == circuit_history.c.circuit_id, isouter=True)
        ),
        conn,
    )
    team_features_df = pd.read_sql(
        sa.select(
            team_features.c.race_id,
            team_features.c.team_id,
            team_features.c.momentum_score,
            team_features.c.performance_trend_last_5_races,
            team_features.c.straight_line_speed_ranking,
            team_features.c.overall_downforce_level,
            team_features.c.strategy_success_rate,
            team_features.c.historical_avg_points_at_circuit,
            team_features.c.dnf_rate_last_10_races,
        ).where(team_features.c.race_id == race_id),
        conn,
    )
    driver_features_df = pd.read_sql(
        sa.select(
            driver_features.c.race_id,
            driver_features.c.driver_id,
            driver_features.c.recent_dnf_rate,
            driver_features.c.wet_weather_performance_multiplier,
            driver_features.c.avg_finish_position_at_circuit,
            driver_features.c.last_3_races_avg_position,
            driver_features.c.overtaking_success_rate,
            driver_features.c.tire_management_score,
            driver_features.c.qualifying_pace_percentile,
            driver_features.c.race_pace_percentile,
        ).where(driver_features.c.race_id == race_id),
        conn,
    )

    order_index = races_df.index[races_df["race_id"] == race_id][0]
    circuit_name = race_row["circuit_name"]

    results_df = results_df.merge(races_df, on="race_id", how="left")
    qualifying_df = qualifying_df.merge(races_df, on="race_id", how="left")
    laps_df = laps_df.merge(races_df, on="race_id", how="left")

    driver_result = results_df[(results_df["race_id"] == race_id) & (results_df["driver_id"] == driver_id)]
    if driver_result.empty:
        return {}
    driver_result = driver_result.iloc[0]
    team_id = driver_result["team_id"]

    grid_position = driver_result.get("grid_position")
    grid_side = None
    if pd.notna(grid_position):
        grid_side = "clean" if int(grid_position) % 2 == 1 else "dirty"

    starting_tire = None
    driver_lap1 = laps_df[
        (laps_df["race_id"] == race_id)
        & (laps_df["driver_id"] == driver_id)
        & (laps_df["lap_number"] == 1)
    ]
    if not driver_lap1.empty:
        starting_tire = driver_lap1.iloc[0].get("tire_compound")

    positions_historically_gained = None
    if pd.notna(grid_position):
        historical = results_df[
            (results_df["circuit_name"] == circuit_name)
            & (results_df["grid_position"] == grid_position)
        ]
        if not historical.empty:
            positions_historically_gained = (
                historical["grid_position"] - historical["finishing_position"]
            ).dropna().mean()

    expected_lap1_position = None
    lap1_times = laps_df[(laps_df["race_id"] == race_id) & (laps_df["lap_number"] == 1)].dropna(subset=["lap_time"])
    if not lap1_times.empty:
        lap1_times = lap1_times.sort_values("lap_time")
        lap1_times["lap1_position"] = range(1, len(lap1_times) + 1)
        driver_lap1_row = lap1_times[lap1_times["driver_id"] == driver_id]
        if not driver_lap1_row.empty:
            expected_lap1_position = driver_lap1_row.iloc[0]["lap1_position"]

    optimal_tire_strategy = None
    pit_counts = pit_df[pit_df["race_id"] == race_id]
    if not pit_counts.empty:
        pit_count = pit_counts[pit_counts["driver_id"] == driver_id]["stop_number"].max()
        if pd.notna(pit_count):
            if pit_count <= 1:
                optimal_tire_strategy = "1-stop"
            elif pit_count == 2:
                optimal_tire_strategy = "2-stop"
            else:
                optimal_tire_strategy = "3-stop"

    tire_compound_advantage = None
    if starting_tire:
        compound_counts = laps_df[(laps_df["race_id"] == race_id) & (laps_df["lap_number"] == 1)]["tire_compound"]
        if not compound_counts.empty:
            total = compound_counts.count()
            same = (compound_counts == starting_tire).sum()
            tire_compound_advantage = 1 - (same / total) if total else None

    pit_window_overlap_count = None
    if pd.notna(grid_position):
        pit_window_overlap_count = results_df[
            (results_df["race_id"] == race_id)
            & (results_df["grid_position"] >= grid_position - 2)
            & (results_df["grid_position"] <= grid_position + 2)
        ].shape[0]

    undercut_opportunity = None
    overcut_opportunity = None
    strategic_position_value = None
    if pd.notna(grid_position) and pd.notna(driver_result.get("finishing_position")):
        position_gain = driver_result["grid_position"] - driver_result["finishing_position"]
        undercut_opportunity = max(0.0, position_gain / 5.0)
        overcut_opportunity = max(0.0, -position_gain / 5.0)
        strategic_position_value = position_gain

    championship_position = None
    points_gap_next = None
    points_gap_prev = None
    mathematical_possible = None
    must_win = None
    remaining_races = 0
    standings = (
        results_df[results_df["order_index"] <= order_index]
        .groupby("driver_id")["points"].sum()
        .sort_values(ascending=False)
    )
    if driver_id in standings.index:
        championship_position = int(standings.index.get_loc(driver_id) + 1)
        driver_points = standings.loc[driver_id]
        if championship_position > 1:
            points_gap_prev = standings.iloc[championship_position - 2] - driver_points
        if championship_position < len(standings):
            points_gap_next = driver_points - standings.iloc[championship_position]
        remaining_races = results_df["race_id"].nunique() - (order_index + 1)
        max_points_possible = remaining_races * 26
        leader_points = standings.iloc[0]
        mathematical_possible = (leader_points - driver_points) <= max_points_possible
        must_win = (
            (leader_points - driver_points) > max(0, (remaining_races - 1) * 26)
            if remaining_races > 0
            else False
        )

    team_orders_risk = None
    if team_id:
        team_points = (
            results_df[(results_df["team_id"] == team_id) & (results_df["order_index"] <= order_index)]
            .groupby("driver_id")["points"]
            .sum()
        )
        if len(team_points) == 2:
            diff = abs(team_points.iloc[0] - team_points.iloc[1])
            team_orders_risk = max(0.0, 1 - diff / 50)

    circuit_char_row = circuit_char_df[circuit_char_df["circuit_name"] == circuit_name]
    safety_car_prob = circuit_char_row["safety_car_probability"].iloc[0] if not circuit_char_row.empty else None
    circuit_history_row = circuit_history_df[
        (circuit_history_df["circuit_name"] == circuit_name) & (circuit_history_df["season"] == season)
    ]
    red_flag_prob = None
    if not circuit_history_row.empty:
        red_flag_prob = min(1.0, circuit_history_row["red_flag_count"].iloc[0] / 5.0)

    driver_feature_row = driver_features_df[
        (driver_features_df["race_id"] == race_id) & (driver_features_df["driver_id"] == driver_id)
    ]
    expected_dnf_rate = driver_feature_row["recent_dnf_rate"].iloc[0] if not driver_feature_row.empty else None

    weather_race = weather_df[weather_df["race_id"] == race_id]
    wet_race_prob = None
    weather_change_prob = None
    if not weather_race.empty:
        wet_race_prob = weather_race["wet_race_probability"].dropna().mean()
        weather_change_prob = weather_race["weather_change_probability"].dropna().mean()

    driver_wet_adv = None
    if not driver_feature_row.empty and wet_race_prob is not None:
        multiplier = driver_feature_row["wet_weather_performance_multiplier"].iloc[0]
        if multiplier is not None:
            driver_wet_adv = multiplier * wet_race_prob

    team_wet_setup = None
    team_feature_row = team_features_df[
        (team_features_df["race_id"] == race_id) & (team_features_df["team_id"] == team_id)
    ]
    if not team_feature_row.empty and wet_race_prob is not None:
        team_wet_setup = (team_feature_row["momentum_score"].iloc[0] or 0) * wet_race_prob

    grid_position_std_dev = None
    expected_race_pace_gaps = None
    competitive_balance = None
    race_qual = qualifying_df[qualifying_df["race_id"] == race_id]
    if not race_qual.empty:
        grid_position_std_dev = race_qual["qualifying_position"].std()
        expected_race_pace_gaps = grid_position_std_dev
        if grid_position_std_dev and grid_position_std_dev > 0:
            competitive_balance = 1 / grid_position_std_dev

    active_aero = None
    overtake_efficiency = None
    power_unit_risk = None
    regulation_adaptation = None
    if season >= 2026 and not team_feature_row.empty:
        active_aero = team_feature_row["performance_trend_last_5_races"].iloc[0]
        overtake_efficiency = team_feature_row["momentum_score"].iloc[0]
        power_unit_risk = (team_feature_row["dnf_rate_last_10_races"].iloc[0] or 0)
        regulation_adaptation = team_feature_row["momentum_score"].iloc[0]

    driver_circuit_affinity = None
    car_circuit_compatibility = None
    weather_adjusted_pace = None
    wet_advantage_index = None
    risk_adjusted_strategy = None
    intra_team_competition_factor = None
    team_orders_adjustment = None
    motivational_boost = None
    expected_qualifying_position = None
    expected_race_pace = None
    win_probability_raw = None

    avg_finish_at_circuit = (
        driver_feature_row["avg_finish_position_at_circuit"].iloc[0]
        if not driver_feature_row.empty
        else None
    )
    recent_form = (
        driver_feature_row["last_3_races_avg_position"].iloc[0]
        if not driver_feature_row.empty
        else None
    )
    overtaking_rate = (
        driver_feature_row["overtaking_success_rate"].iloc[0]
        if not driver_feature_row.empty
        else None
    )
    tire_management = (
        driver_feature_row["tire_management_score"].iloc[0]
        if not driver_feature_row.empty
        else None
    )
    qualifying_skill = (
        driver_feature_row["qualifying_pace_percentile"].iloc[0]
        if not driver_feature_row.empty
        else None
    )
    race_skill = (
        driver_feature_row["race_pace_percentile"].iloc[0]
        if not driver_feature_row.empty
        else None
    )

    circuit_characteristics_row = circuit_char_df[circuit_char_df["circuit_name"] == circuit_name]
    overtaking_difficulty = None
    tire_deg_severity = None
    if not circuit_characteristics_row.empty:
        overtaking_difficulty = circuit_characteristics_row["overtaking_difficulty"].iloc[0]
        tire_deg_severity = circuit_characteristics_row["tire_degradation_severity"].iloc[0]

    driver_style_match = None
    if overtaking_rate is not None:
        driver_style_match = overtaking_rate
        if overtaking_difficulty is not None:
            driver_style_match = overtaking_rate * (1 - min(1.0, overtaking_difficulty / 10))
    if tire_deg_severity is not None and tire_management is not None:
        tire_component = min(1.0, abs(tire_management) / (abs(tire_deg_severity) + 1))
        driver_style_match = ((driver_style_match or 0) + tire_component) / 2
    if wet_race_prob is not None and not driver_feature_row.empty:
        wet_multiplier = driver_feature_row["wet_weather_performance_multiplier"].iloc[0]
        if wet_multiplier is not None:
            driver_style_match = ((driver_style_match or 0) + wet_multiplier * wet_race_prob) / 2

    historical_perf_score = None
    if avg_finish_at_circuit is not None and avg_finish_at_circuit > 0:
        historical_perf_score = 1 / avg_finish_at_circuit
    recent_form_score = None
    if recent_form is not None and recent_form > 0:
        recent_form_score = 1 / recent_form
    if historical_perf_score is not None and recent_form_score is not None and driver_style_match is not None:
        driver_circuit_affinity = (
            historical_perf_score * 0.4
            + driver_style_match * 0.3
            + recent_form_score * 0.3
        )

    if not team_feature_row.empty:
        downforce_match = 0.5
        circuit_type = circuits_df.loc[circuit_name]["circuit_type"] if circuit_name in circuits_df.index else None
        if circuit_type and team_feature_row["overall_downforce_level"].iloc[0]:
            if circuit_type.lower() == "street" and team_feature_row["overall_downforce_level"].iloc[0] == "high":
                downforce_match = 0.8
            elif circuit_type.lower() == "permanent":
                downforce_match = 0.7
        power_advantage = team_feature_row["straight_line_speed_ranking"].iloc[0]
        if power_advantage is not None:
            power_advantage = power_advantage / 10
        tire_management_team = team_feature_row["strategy_success_rate"].iloc[0]
        historical_perf = team_feature_row["historical_avg_points_at_circuit"].iloc[0]
        if historical_perf is not None:
            historical_perf = min(1.0, historical_perf / 25)
        car_circuit_compatibility = (
            downforce_match * 0.35
            + (power_advantage or 0) * 0.25
            + (tire_management_team or 0) * 0.2
            + (historical_perf or 0) * 0.2
        )

    if race_skill is not None and wet_race_prob is not None:
        wet_multiplier = driver_feature_row["wet_weather_performance_multiplier"].iloc[0] if not driver_feature_row.empty else 1
        wet_pace = race_skill * (wet_multiplier or 1)
        weather_adjusted_pace = race_skill * (1 - wet_race_prob) + wet_pace * wet_race_prob
        field_avg_wet = driver_features_df[driver_features_df["race_id"] == race_id][
            "wet_weather_performance_multiplier"
        ].mean()
        if wet_multiplier is not None and field_avg_wet is not None:
            wet_advantage_index = (wet_multiplier - field_avg_wet) * wet_race_prob

    championship_pressure = None
    if points_gap_prev is not None and points_gap_next is not None and remaining_races >= 0:
        pressure_gap = (points_gap_prev + points_gap_next) / max(1, remaining_races)
        championship_pressure = min(1.0, max(0.0, pressure_gap / 50))
    if optimal_tire_strategy:
        optimal_value = {"1-stop": 1, "2-stop": 2, "3-stop": 3}.get(optimal_tire_strategy, 2)
        aggressive_value = min(3, optimal_value + 1)
        if championship_pressure is not None:
            risk_adjusted_strategy = optimal_value * (1 - championship_pressure) + aggressive_value * championship_pressure

    intra_team_competition_factor = team_orders_risk
    if team_orders_risk is not None:
        team_orders_adjustment = max(0.0, 1 - team_orders_risk)

    if not driver_feature_row.empty:
        motivational_boost = (driver_feature_row["last_3_races_avg_position"].iloc[0] or 0)
        if motivational_boost:
            motivational_boost = min(1.0, 1 / motivational_boost)

    if qualifying_skill is not None:
        car_perf = team_feature_row["momentum_score"].iloc[0] if not team_feature_row.empty else 0
        circuit_match = car_circuit_compatibility or 0
        weather_factor = 1 - (wet_race_prob or 0)
        recent_form_factor = (1 / recent_form) if recent_form else 0
        expected_qualifying_position = (
            (1 - qualifying_skill) * 10
            + (1 - car_perf) * 5
            - circuit_match * 2
            + (1 - weather_factor) * 2
            - recent_form_factor * 2
        )

    if expected_qualifying_position is not None and race_skill is not None:
        expected_race_pace = (
            expected_qualifying_position * 0.4
            + (1 - race_skill) * 10 * 0.4
            + (1 - (tire_management or 0)) * 2 * 0.2
        )

    if expected_race_pace is not None and grid_position is not None:
        reliability = 1 - (expected_dnf_rate or 0)
        strategic_value = strategic_position_value or 0
        win_probability_raw = max(
            0.0,
            min(
                1.0,
                (1 / (expected_race_pace + 1)) * 0.5
                + (1 / (grid_position + 1)) * 0.2
                + (safety_car_prob or 0) * 0.1
                + reliability * 0.1
                + (strategic_value / 10) * 0.1,
            ),
        )

    first_lap_incident_prob = None
    if safety_car_prob is not None or red_flag_prob is not None:
        base = [p for p in [safety_car_prob, red_flag_prob] if p is not None]
        if base:
            first_lap_incident_prob = min(1.0, float(pd.Series(base).mean()))

    return {
        "grid_position": grid_position,
        "starting_tire_compound": starting_tire,
        "grid_side": grid_side,
        "positions_historically_gained_from_this_grid": positions_historically_gained,
        "expected_lap1_position": expected_lap1_position,
        "optimal_tire_strategy": optimal_tire_strategy,
        "tire_compound_advantage": tire_compound_advantage,
        "pit_window_overlap_count": pit_window_overlap_count,
        "undercut_opportunity_score": undercut_opportunity,
        "overcut_opportunity_score": overcut_opportunity,
        "strategic_position_value": strategic_position_value,
        "championship_position": championship_position,
        "points_gap_to_next": points_gap_next,
        "points_gap_to_previous": points_gap_prev,
        "mathematical_championship_possible": mathematical_possible,
        "must_win_race_flag": must_win,
        "team_orders_risk_score": team_orders_risk,
        "first_lap_incident_probability": first_lap_incident_prob,
        "safety_car_probability": safety_car_prob,
        "red_flag_probability": red_flag_prob,
        "expected_dnf_rate": expected_dnf_rate,
        "wet_race_probability": wet_race_prob,
        "weather_change_during_race_probability": weather_change_prob,
        "driver_wet_weather_advantage": driver_wet_adv,
        "team_wet_setup_effectiveness": team_wet_setup,
        "grid_position_std_dev": grid_position_std_dev,
        "expected_race_pace_gaps": expected_race_pace_gaps,
        "competitive_balance_index": competitive_balance,
        "active_aero_adaptation_score": active_aero,
        "overtake_mode_efficiency": overtake_efficiency,
        "new_power_unit_reliability_risk": power_unit_risk,
        "regulation_adaptation_success": regulation_adaptation,
        "driver_circuit_affinity": driver_circuit_affinity,
        "car_circuit_compatibility": car_circuit_compatibility,
        "weather_adjusted_pace": weather_adjusted_pace,
        "wet_advantage_index": wet_advantage_index,
        "risk_adjusted_strategy": risk_adjusted_strategy,
        "intra_team_competition_factor": intra_team_competition_factor,
        "team_orders_adjustment": team_orders_adjustment,
        "motivational_boost": motivational_boost,
        "expected_qualifying_position": expected_qualifying_position,
        "expected_race_pace": expected_race_pace,
        "win_probability_raw": win_probability_raw,
    }


def update_situational_features(engine: Engine) -> None:
    with engine.begin() as conn:
        race_driver_pairs = pd.read_sql(
            sa.select(race_results.c.race_id, race_results.c.driver_id).distinct(),
            conn,
        )
        for _, row in race_driver_pairs.iterrows():
            features = extract_race_features(
                conn,
                int(row["race_id"]),
                int(row["driver_id"]),
                lookback_races=50,
            )
            if not features:
                continue
            stmt = pg_insert(situational_features).values(
                race_id=int(row["race_id"]),
                driver_id=int(row["driver_id"]),
                **features,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_situational_features",
                set_=features,
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
    update_team_features(engine)
    update_situational_features(engine)
    with engine.begin() as conn:
        report = build_data_quality_report(conn)
    report_path = "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOG.info("Data quality report written to %s", report_path)


if __name__ == "__main__":
    main()
