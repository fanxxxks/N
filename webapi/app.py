"""FastAPI entry point for the AlphaGPT web dashboard.

Dev:
    uvicorn webapi.app:app --host 127.0.0.1 --port 8000 --reload
    (frontend: `npm run dev` in webui/, Vite proxies /api to :8000)

Prod:
    npm run build --prefix webui
    uvicorn webapi.app:app --host 0.0.0.0 --port 8000
    (webui/dist is served at /)
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import service

# Windows' mimetypes module resolves `.js` from the registry and frequently
# returns text/plain, which makes browsers refuse ES module scripts and leaves
# the dashboard as a black page. Pin the correct MIME types explicitly.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "webui" / "dist"

app = FastAPI(title="AlphaGPT Web API", version="1.0.0")

# The API can start/stop/reset the simulation, so it is intentionally bound
# to local origins only (dev Vite proxy + the single-origin production build).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/overview")
def overview() -> dict:
    backtest = service.get_backtest()
    strategy = service.get_strategy()
    sim = service.get_sim_state()
    status = service.get_data_status()
    return {
        "backtest": backtest,
        "strategy": strategy,
        "sim": sim,
        "status": status,
    }


@app.get("/api/backtest")
def backtest() -> dict:
    return service.get_backtest()


@app.get("/api/backtest/positions")
def backtest_positions(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    return service.get_backtest_positions(offset, limit)


@app.get("/api/strategy")
def strategy() -> dict:
    return service.get_strategy()


@app.get("/api/sim")
def sim_state() -> dict:
    return service.get_sim_state()


@app.get("/api/sim/days")
def sim_days() -> dict:
    return service.get_sim_days()


@app.get("/api/sim/day/{date}")
def sim_day(date: str) -> dict:
    if not (date.isdigit() and len(date) == 8):
        raise HTTPException(status_code=400, detail="date must be YYYYMMDD")
    return service.get_sim_day(date)


@app.post("/api/sim/stop")
def sim_stop() -> JSONResponse:
    result = service.sim_stop_run()
    return JSONResponse(result, status_code=200 if result.get("ok") else 500)


@app.get("/api/sim/status")
def sim_run_status() -> dict:
    return service.sim_status()


@app.post("/api/sim/start")
def sim_run_start(req: service.SimStartRequest) -> JSONResponse:
    result = service.sim_start(req)
    code = 409 if result.get("conflict") else (200 if result.get("ok") else 400)
    return JSONResponse(result, status_code=code)


@app.post("/api/sim/reset")
def sim_run_reset() -> JSONResponse:
    result = service.sim_reset_run()
    code = 409 if result.get("conflict") else (200 if result.get("ok") else 500)
    return JSONResponse(result, status_code=code)


@app.get("/api/sim/config")
def sim_config_get() -> dict:
    return service.get_sim_config()


@app.put("/api/sim/config")
def sim_config_update(patch: service.SimConfigPatch) -> JSONResponse:
    result = service.write_sim_config(patch)
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)


@app.get("/api/data-status")
def data_status() -> dict:
    return service.get_data_status()


@app.get("/api/logs")
def logs() -> list[dict]:
    return service.list_logs()


@app.get("/api/logs/{name}")
def log_content(name: str, tail: int = Query(1000, ge=1, le=20000)) -> dict:
    return service.read_log(name, tail)


# Single-server production mode: serve the built frontend when it exists.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="webui")
