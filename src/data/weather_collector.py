from __future__ import annotations

from typing import Any, Dict

import requests


def fetch_weather(
    base_url: str,
    api_key: str,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    """Fetch current weather from OpenWeather-compatible API."""
    response = requests.get(
        f"{base_url}/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
