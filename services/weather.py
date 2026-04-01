# services/weather.py
from __future__ import annotations

import datetime as dt
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import requests
from zoneinfo import ZoneInfo

BASE_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/"
    "VilageFcstInfoService_2.0/getUltraSrtNcst"
)

_KST = ZoneInfo("Asia/Seoul")

_TOTAL_BUDGET_SEC = 2.5
_REQ_TIMEOUT = (1.2, 1.2)

# transient 에러에 대해서만 짧게 재시도
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _now_kst() -> dt.datetime:
    return dt.datetime.now(tz=_KST)


def _to_30min_slot(t: dt.datetime) -> Tuple[str, str]:
    minute = 30 if t.minute >= 30 else 0
    base_time = f"{t.hour:02d}{minute:02d}"
    base_date = t.strftime("%Y%m%d")
    return base_date, base_time


def _candidate_base_times(now: dt.datetime, tries: int = 3) -> List[Tuple[str, str]]:
    """
    발표 지연을 고려해 now-10min 기준으로 30분씩 뒤로.
    """
    out: List[Tuple[str, str]] = []
    t = now - dt.timedelta(minutes=10)

    for _ in range(max(1, tries)):
        pair = _to_30min_slot(t)
        if pair not in out:
            out.append(pair)
        t -= dt.timedelta(minutes=30)

    return out


def _parse_kma_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload["response"]["body"]["items"]["item"]

    temp: float | None = None
    wind: float = 0.0
    pty_code: str = "0"

    for it in items:
        cat = str(it.get("category", "")).strip()
        val = it.get("obsrValue")
        if cat == "T1H":
            try:
                temp = float(val)
            except Exception:
                temp = None
        elif cat == "WSD":
            try:
                wind = float(val)
            except Exception:
                wind = 0.0
        elif cat == "PTY":
            pty_code = str(val).strip()

    if temp is None:
        raise RuntimeError("Temperature(T1H) not found from KMA response")

    if pty_code in ("1", "2", "4"):
        pty = "RAIN"
    elif pty_code == "3":
        pty = "SNOW"
    else:
        pty = "SUNNY"

    return {
        "temp": round(temp, 1),
        "feelsLike": round(temp, 1),
        "wind": round(wind, 1),
        "pty": pty,
    }


def _get_service_key() -> str:
    """
    PowerShell/Secret 업로드 시 CRLF가 섞이면 authKey=...%0D%0A 로 나가 401이 발생함.
    """
    key = (os.getenv("SERVICE_KEY") or "").strip()
    if not key:
        raise RuntimeError("SERVICE_KEY not set")

    # strip 후에도 공백이 남아있으면(중간 공백/개행 등) 키 자체가 오염된 것
    if any(ch.isspace() for ch in key):
        raise RuntimeError("SERVICE_KEY contains whitespace; fix Secret/Env value")

    return key


def _is_retryable_http(err: requests.HTTPError) -> bool:
    resp = err.response
    if resp is None:
        return True
    return int(resp.status_code) in _RETRYABLE_STATUS


def _request_kma(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    1회 요청에 대해 아주 짧게 1번만 재시도(총 2회).
    401/403은 즉시 실패(키 문제).
    """
    last_err: Exception | None = None

    for attempt in range(2):
        try:
            res = requests.get(BASE_URL, params=params, timeout=_REQ_TIMEOUT)
            res.raise_for_status()
            return res.json()

        except requests.HTTPError as e:
            status = int(e.response.status_code) if e.response is not None else -1
            # 인증 문제는 재시도해도 해결 안 됨
            if status in (401, 403):
                raise RuntimeError(
                    f"KMA unauthorized (status={status}). "
                    "SERVICE_KEY(개행/공백 포함 여부)와 권한을 확인하세요."
                ) from e

            if attempt == 0 and _is_retryable_http(e):
                last_err = e
                time.sleep(0.15)
                continue
            raise

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == 0:
                last_err = e
                time.sleep(0.15)
                continue
            raise

        except Exception as e:
            last_err = e
            raise

    raise RuntimeError(f"KMA request failed: {last_err}")


@lru_cache(maxsize=256)
def _fetch_weather(nx: int, ny: int, base_date: str, base_time: str) -> Dict[str, Any]:
    params = {
        "authKey": _get_service_key(),
        "pageNo": 1,
        "numOfRows": 100,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    payload = _request_kma(params)
    return _parse_kma_payload(payload)


def get_current_weather(lat: float, lon: float) -> Dict[str, Any]:
    from services.geo import latlon_to_grid

    nx, ny = latlon_to_grid(lat, lon)

    start = time.monotonic()
    now = _now_kst()
    candidates = _candidate_base_times(now, tries=3)

    last_err: Exception | None = None

    for base_date, base_time in candidates:
        if (time.monotonic() - start) > _TOTAL_BUDGET_SEC:
            break

        try:
            return _fetch_weather(nx, ny, base_date, base_time)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"KMA weather fetch failed (budget={_TOTAL_BUDGET_SEC}s, tried={len(candidates)}): {last_err}"
    )
