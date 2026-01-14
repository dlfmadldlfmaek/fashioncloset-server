# services/weather.py
import datetime
import requests
from functools import lru_cache

# ⚠️ import 시 네트워크 호출 절대 없음

BASE_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/"
    "VilageFcstInfoService_2.0/getUltraSrtNcst"
)


def _base_datetime():
    """
    기상청 규칙에 맞는 base_date / base_time 계산
    """
    now = datetime.datetime.now()
    minute = 0 if now.minute < 30 else 30
    base_time = f"{now.hour:02d}{minute:02d}"
    base_date = now.strftime("%Y%m%d")
    return base_date, base_time


@lru_cache(maxsize=32)
def _fetch_weather(nx: int, ny: int, cache_key: str):
    """
    ⚠️ 실제 외부 API 호출
    이 함수는 요청 시점에만 호출됨
    """
    import os

    service_key = os.getenv("SERVICE_KEY")
    if not service_key:
        raise RuntimeError("SERVICE_KEY not set")

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
    wind = 0.0
    pty = "SUNNY"

    for it in items:
        if it["category"] == "T1H":
            temp = float(it["obsrValue"])
        elif it["category"] == "WSD":
            wind = float(it["obsrValue"])
        elif it["category"] == "PTY" and it["obsrValue"] != "0":
            pty = "RAIN"

    if temp is None:
        raise RuntimeError("Temperature not found")

    return {
        "temp": round(temp, 1),
        "feelsLike": round(temp, 1),  # 단순화 (체감온도 공식은 이후)
        "wind": round(wind, 1),
        "pty": pty,
    }


def get_current_weather(lat: float, lon: float):
    """
    🔥 외부에서 호출하는 유일한 함수
    어떤 상황에서도 Exception을 밖으로 던지지 않는다.
    """
    try:
        # 🔥 grid 계산은 여기서 import (안전)
        from services.grid import latlon_to_grid

        nx, ny = latlon_to_grid(lat, lon)

        now = datetime.datetime.now()
        cache_key = now.strftime("%Y%m%d%H") + (
            "0" if now.minute < 30 else "5"
        )

        return _fetch_weather(nx, ny, cache_key)

    except Exception:
        # 실패해도 서버는 살아야 함
        return {
            "temp": 0.0,
            "feelsLike": 0.0,
            "wind": 0.0,
            "pty": "SUNNY",
        }
