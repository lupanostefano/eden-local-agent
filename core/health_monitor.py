"""
health_monitor.py — Eden subsystem watchdog
============================================

APScheduler BackgroundScheduler che ogni 5 minuti verifica la salute dei
sottosistemi critici e loggail risultato via log_utils. Risolve la classe
"bug che nessuno vede perche' catturato in try/except": se inner_stream si
blocca, o Ollama non risponde, l'UI `/logs` lo mostra entro 5 minuti.

**Nome file**: `watchdog.py` era gia' usato per il supervisor server →
scelto `health_monitor.py` per evitare collisione.

Check implementati (ognuno → log_watchdog con status HEALTHY/DEGRADED/DOWN):


2. **SIC-1** (inner_stream) — scheduler running + ultimo pensiero entro
   60 minuti (tolleranza su INNER_STREAM_MAX_MINUTES=45).

3. **SIC-2** (emotional_elaboration) — scheduler running + last_elaboration_at
   entro 4 ore (tolleranza su ELABORATION_MAX_HOURS=3).

4. **OLLAMA** — ping locale, verifica che gemma2:9b risponda. Se il
   generator e' down, Eden non puo' funzionare.

5. **DISK** — `data/logs/` e `data/sessions/` scrivibili.

Tutto fail-silente: un check che solleva eccezione viene loggato come
DEGRADED ma non ferma gli altri.

Kill switch: `_ATTIVO = False` → scheduler non parte. Per-runtime: `stop()`.
"""

from __future__ import annotations

import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from core import log_utils


_ATTIVO = True

# Intervallo check periodico (secondi). 300s = 5 minuti.
CHECK_INTERVAL_SEC = 300

# Soglie tolleranza — leggermente piu' permissive dei target nominali.
INNER_STALE_MIN     = 60   # SIC-1 target max 45 min → tolleranza +15
EMOELAB_STALE_HOURS = 4    # SIC-2 target max 3h → tolleranza +1h
OLLAMA_TIMEOUT_SEC  = 8

# Percorsi progetto — via eden_paths per coerenza post-refactor.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
import sys as _sys
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import eden_paths as _ep
_LOGS_DIR     = _ep.LOGS_DIR
_SESSIONS_DIR = _ep.SESSIONS_DIR

# Endpoint Ollama locale.
_OLLAMA_URL = "http://127.0.0.1:11434/api/tags"

# Stato interno scheduler.
_scheduler: Optional[BackgroundScheduler] = None

# Iniezioni da agent.py al boot.
_mem_carica_fn: Optional[Callable[[], dict]] = None
_inner_stream_module: Optional[Any] = None
_emotional_elaboration_module: Optional[Any] = None

# Ultimo snapshot (per endpoint /api/logs/health, evita ricomputo).
_last_snapshot: dict = {
    "ts": None,
    "checks": {},
}


# ============================================================
# API pubblica
# ============================================================

def start(mem_carica_fn: Callable[[], dict],
          inner_stream_module: Any = None,
          emotional_elaboration_module: Any = None) -> None:
    """
    Avvia lo scheduler. Chiamato da agent.py dopo che tutti i moduli sono
    caricati.
    """
    global _scheduler, _mem_carica_fn, _inner_stream_module, _emotional_elaboration_module

    if not _ATTIVO:
        log_utils.log("INFO", "HEALTH", "Watchdog disabilitato (_ATTIVO=False)")
        return

    _mem_carica_fn = mem_carica_fn
    _inner_stream_module = inner_stream_module
    _emotional_elaboration_module = emotional_elaboration_module

    if _scheduler is not None and _scheduler.running:
        log_utils.log("WARNING", "HEALTH", "start() chiamato ma scheduler gia' attivo")
        return

    _scheduler = BackgroundScheduler(
        daemon=True,
        job_defaults={"coalesce": True, "max_instances": 1},
    )
    _scheduler.add_job(_run_checks, "interval", seconds=CHECK_INTERVAL_SEC,
                       id="health_check", next_run_time=datetime.now() + timedelta(seconds=30))
    _scheduler.start()
    log_utils.log_boot("HEALTH", True, f"watchdog ogni {CHECK_INTERVAL_SEC}s")


def stop() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log_utils.log("INFO", "HEALTH", "Watchdog fermato")


def get_snapshot() -> dict:
    """Ritorna ultimo risultato dei check (per /api/logs/health)."""
    return dict(_last_snapshot)


def run_now() -> dict:
    """Forza un run immediato. Utile per test / debug."""
    _run_checks()
    return get_snapshot()


# ============================================================
# Logica check
# ============================================================

def _run_checks() -> None:
    """Esegue tutti i check e aggiorna snapshot."""
    results = {}
    try:
        results["inner_stream"] = _check_inner_stream()
    except Exception as e:
        results["inner_stream"] = {"status": "DEGRADED", "detail": f"check eccezione: {e}"}
        log_utils.log_exception("HEALTH", "check inner_stream fallito", e)
    try:
        results["emotional_elaboration"] = _check_emotional_elaboration()
    except Exception as e:
        results["emotional_elaboration"] = {"status": "DEGRADED", "detail": f"check eccezione: {e}"}
        log_utils.log_exception("HEALTH", "check emotional_elaboration fallito", e)
    try:
        results["ollama"] = _check_ollama()
    except Exception as e:
        results["ollama"] = {"status": "DEGRADED", "detail": f"check eccezione: {e}"}
        log_utils.log_exception("HEALTH", "check ollama fallito", e)
    try:
        results["disk"] = _check_disk()
    except Exception as e:
        results["disk"] = {"status": "DEGRADED", "detail": f"check eccezione: {e}"}
        log_utils.log_exception("HEALTH", "check disk fallito", e)

    # Log ogni risultato via log_watchdog (mapping automatico → severity).
    for name, res in results.items():
        log_utils.log_watchdog(name.upper(), res.get("status", "DEGRADED"),
                                res.get("detail", ""))

    _last_snapshot["ts"] = datetime.now().isoformat(timespec="seconds")
    _last_snapshot["checks"] = results


def _check_inner_stream() -> dict:
    """SIC-1: scheduler running + ultimo pensiero entro INNER_STALE_MIN."""
    if _inner_stream_module is None:
        return {"status": "DEGRADED", "detail": "modulo non iniettato"}
    st = _inner_stream_module.status()
    if not st.get("running"):
        return {"status": "DOWN", "detail": "scheduler fermo"}
    # Timestamp ultimo pensiero: da memory.json mem["inner_stream"][-1]["ts"].
    last_ts = _get_last_inner_stream_ts()
    if last_ts is None:
        # Primo giro: tolleriamo assenza per ~INNER_STALE_MIN.
        return {"status": "HEALTHY", "detail": "scheduler attivo (nessun pensiero ancora)"}
    age_min = (time.time() - last_ts) / 60.0
    if age_min > INNER_STALE_MIN:
        return {"status": "DEGRADED",
                "detail": f"ultimo pensiero {age_min:.0f}min fa (soglia {INNER_STALE_MIN})"}
    return {"status": "HEALTHY", "detail": f"ultimo pensiero {age_min:.0f}min fa"}


def _check_emotional_elaboration() -> dict:
    """SIC-2: scheduler running + last_elaboration_at entro EMOELAB_STALE_HOURS."""
    if _emotional_elaboration_module is None:
        return {"status": "DEGRADED", "detail": "modulo non iniettato"}
    st = _emotional_elaboration_module.status()
    if not st.get("running"):
        return {"status": "DOWN", "detail": "scheduler fermo"}
    last_ts = _get_last_elaboration_ts()
    if last_ts is None:
        return {"status": "HEALTHY",
                "detail": "scheduler attivo (nessuna elaborazione ancora)"}
    age_h = (time.time() - last_ts) / 3600.0
    if age_h > EMOELAB_STALE_HOURS:
        return {"status": "DEGRADED",
                "detail": f"ultima elaborazione {age_h:.1f}h fa (soglia {EMOELAB_STALE_HOURS}h)"}
    return {"status": "HEALTHY", "detail": f"ultima elaborazione {age_h:.1f}h fa"}


def _check_ollama() -> dict:
    """Ping endpoint Ollama locale."""
    try:
        req = urllib.request.Request(_OLLAMA_URL, method="GET")
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
            if resp.status == 200:
                return {"status": "HEALTHY", "detail": f"ping OK ({resp.status})"}
            return {"status": "DEGRADED", "detail": f"status {resp.status}"}
    except urllib.error.URLError as e:
        return {"status": "DOWN", "detail": f"unreachable: {e.reason}"}
    except Exception as e:
        return {"status": "DEGRADED", "detail": f"errore: {e}"}


def _check_disk() -> dict:
    """Verifica che dir critiche siano scrivibili."""
    problems = []
    for d in (_LOGS_DIR, _SESSIONS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".health_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as e:
            problems.append(f"{d.name}: {e}")
    if problems:
        return {"status": "DOWN", "detail": "; ".join(problems)}
    return {"status": "HEALTHY", "detail": "dir scrivibili"}


# ============================================================
# Helpers
# ============================================================

def _get_exchange_count() -> int:
    if _mem_carica_fn is None:
        return 0
    try:
        mem = _mem_carica_fn()
        return int(mem.get("exchange_count", 0))
    except Exception:
        return 0


def _get_last_inner_stream_ts() -> Optional[float]:
    if _mem_carica_fn is None:
        return None
    try:
        mem = _mem_carica_fn()
        stream = mem.get("inner_stream", [])
        if not stream:
            return None
        # Campo corretto: "timestamp" (v1.0+). "ts" era un bug silenzioso.
        last   = stream[-1]
        ts_str = last.get("timestamp") or last.get("ts")
        if not ts_str:
            return None
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return None


def _get_last_elaboration_ts() -> Optional[float]:
    if _mem_carica_fn is None:
        return None
    try:
        mem = _mem_carica_fn()
        ts_str = mem.get("last_elaboration_at")
        if not ts_str:
            return None
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return None
