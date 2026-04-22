# main.py
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import anyio
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from services.rate_limit import limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

StyleAnchors = Dict[str, Any]


def _md5(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    return hashlib.md5(p.read_bytes()).hexdigest()


def _ensure_routers(app: FastAPI) -> None:
    if not hasattr(app.state, "routers") or not isinstance(getattr(app.state, "routers", None), dict):
        app.state.routers = {
            "recommend": {"enabled": False, "error": None},
            "like": {"enabled": False, "error": None},
            "tryon": {"enabled": False, "error": None},
            "quota": {"enabled": False, "error": None},
            "retention": {"enabled": False, "error": None},
            "ad": {"enabled": False, "error": None},
            "classify": {"enabled": False, "error": None},
        }


async def _load_style_anchors(app: FastAPI, *, force_reload: bool = False) -> None:
    try:
        from services.style_encoder import ANCHOR_JSON_PATH, load_style_anchors

        logger.info("STYLE_ANCHOR_DIR=%s", os.getenv("STYLE_ANCHOR_DIR"))
        logger.info("STYLE_ANCHOR_JSON=%s", os.getenv("STYLE_ANCHOR_JSON"))
        logger.info("ANCHOR_JSON_PATH=%s exists=%s", ANCHOR_JSON_PATH, ANCHOR_JSON_PATH.exists())
        if ANCHOR_JSON_PATH.exists():
            logger.info("anchors.json md5=%s", _md5(str(ANCHOR_JSON_PATH)))

        anchors: StyleAnchors = await anyio.to_thread.run_sync(
            lambda: load_style_anchors(force_reload=bool(force_reload))
        )

        app.state.style_anchors = anchors
        app.state.style_anchors_ready = True
        app.state.style_anchors_error = None
        logger.info("✅ style anchors loaded: %s", list(anchors.keys()))
    except Exception as e:
        app.state.style_anchors = {}
        app.state.style_anchors_ready = False
        app.state.style_anchors_error = repr(e)
        logger.exception("❌ style anchors failed -> fallback empty: %s", e)


def _include_router_safe(app: FastAPI, key: str, importer) -> None:
    _ensure_routers(app)
    try:
        router = importer()
        app.include_router(router)
        app.state.routers[key]["enabled"] = True
        app.state.routers[key]["error"] = None
        logger.info("✅ router included: %s", key)
    except Exception as e:
        app.state.routers[key]["enabled"] = False
        app.state.routers[key]["error"] = repr(e)
        logger.exception("❌ router include failed: %s -> %s", key, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 startup begin")

    _ensure_routers(app)
    app.state.style_anchors = {}
    app.state.style_anchors_ready = False
    app.state.style_anchors_error = None
    app.state.style_anchors_task: Optional[asyncio.Task[None]] = None

    app.state.style_anchors_task = asyncio.create_task(_load_style_anchors(app, force_reload=False))
    yield

    try:
        task = getattr(app.state, "style_anchors_task", None)
        if task and not task.done():
            task.cancel()
    except Exception:
        pass

    logger.info("👋 shutdown done")


app = FastAPI(lifespan=lifespan)

# --- Rate Limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})

_include_router_safe(app, "recommend", lambda: __import__("api.recommend", fromlist=["router"]).router)
_include_router_safe(app, "like", lambda: __import__("api.like", fromlist=["router"]).router)
_include_router_safe(app, "tryon", lambda: __import__("api.tryon", fromlist=["router"]).router)
_include_router_safe(app, "quota", lambda: __import__("api.quota", fromlist=["router"]).router)
_include_router_safe(app, "retention", lambda: __import__("api.retention", fromlist=["router"]).router)
_include_router_safe(app, "ad", lambda: __import__("api.ad", fromlist=["router"]).router)
_include_router_safe(app, "classify", lambda: __import__("api.classify", fromlist=["router"]).router)


@app.get("/")
def health():
    return {
        "status": "ok",
        "style_anchors_ready": bool(getattr(app.state, "style_anchors_ready", False)),
        "loaded_styles": list(getattr(app.state, "style_anchors", {}).keys()),
    }


@app.post("/admin/reload-style-anchors")
async def reload_style_anchors(
    force: bool = Query(True),
    x_admin_key: str = Header(default=""),
):
    expected = os.getenv("ADMIN_KEY", "")
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    await _load_style_anchors(app, force_reload=bool(force))
    return {
        "ok": True,
        "ready": bool(getattr(app.state, "style_anchors_ready", False)),
        "error": getattr(app.state, "style_anchors_error", None),
        "loaded_styles": list(getattr(app.state, "style_anchors", {}).keys()),
    }
