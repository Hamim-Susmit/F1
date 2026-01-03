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
        },
        "missing_values": {
            "race_results_race_time": null_count(race_results, race_results.c.race_time),
            "qualifying_q1_time": null_count(qualifying_results, qualifying_results.c.q1_time),
            "lap_times_lap_time": null_count(lap_times, lap_times.c.lap_time),
            "pit_stops_duration": null_count(pit_stops, pit_stops.c.duration),
            "race_weather_air_temp": null_count(race_weather, race_weather.c.air_temp),
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


def main() -> None:
    setup_logging()
    args = parse_args()
    enable_cache(args.cache_dir)
    engine = get_engine(args.database_url)
    create_schema(engine)
    for season in range(args.start_season, args.end_season + 1):
        LOG.info("Starting ingestion for season %s", season)
        ingest_season(engine, season, args.sleep, args.weather_api_key, args.weather_base_url)
    with engine.begin() as conn:
        report = build_data_quality_report(conn)
    report_path = "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOG.info("Data quality report written to %s", report_path)


if __name__ == "__main__":
    main()
