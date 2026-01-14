import os
import requests
import datetime
from functools import lru_cache

BASE_URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst"

def _base_datetime():
    now = datetime.datetime.now()
    minute = 0 if now.minute < 30 else 30
    base_time = f"{now.hour:02d}{minute:02d}"
    base_date = now.strftime("%Y%m%d")
    return base_date, base_time

@lru_cache(maxsize=1)
def _cached_weather(nx: int, ny: int, cache_key: str):
    service_key = os.environ.get("SERVICE_KEY")
    if not service_key:
        raise RuntimeError("SERVICE_KEY is not set")

    base_date, base_time = _base_datetime()

    params = {
        "authKey": service_key,
        "pageNo": 1,
        "numOfRows": 100,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    res = requests.get(BASE_URL, params=params, timeout=5)
    res.raise_for_status()

    items = res.json()["response"]["body"]["items"]["item"]

    temp = None
    pty = "SUNNY"

    for it in items:
        if it["category"] == "T1H":
            temp = float(it["obsrValue"])
        elif it["category"] == "PTY" and it["obsrValue"] != "0":
            pty = "RAIN"

    if temp is None:
        raise RuntimeError("Temperature data not found")

    return {"temp": temp, "pty": pty}

def get_current_weather(lat: float, lon: float):
    nx, ny = 55, 127  # TODO: latlon_to_grid 적용 가능
    now = datetime.datetime.now()
    cache_key = now.strftime("%Y%m%d%H") + ("0" if now.minute < 30 else "5")
    return _cached_weather(nx, ny, cache_key)
