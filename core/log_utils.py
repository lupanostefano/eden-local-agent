"""
log_utils.py — Eden logging utilities
======================================

Wrapper centralizzato attorno al LogBuffer di agent.py. Risolve tre problemi:

1. **Silent failures invisibili**: try/except: pass nei moduli research falliva
   senza traccia utente (es. bug Observer 2026-04-21 con session_ts=None che
   uccideva silenziosamente la pipeline Phase A per 59 scambi).

2. **Accoppiamento circolare**: i moduli research non possono importare agent
   (agent importa research). Qui iniettiamo il buffer una volta all'avvio,
   tutti i moduli chiamano la stessa API.

3. **Fail-safe by design**: se il buffer non e' stato iniettato (es. test
   standalone), le chiamate sono no-op silenziose — mai sollevano eccezioni
   dentro handler di eccezione.

Uso tipico nei moduli:

    import log_utils

    try:
        do_critical_work()
    except Exception as e:
        log_utils.log_exception("OBSERVER", "osserva() fallita", e)
        # comportamento fail-silent invariato: Eden continua a girare

Livelli severity (convenzione uniforme):
    DEBUG    — diagnostica verbosa, normalmente esclusa dall'UI
    INFO     — operazione normale registrata per audit
    WARNING  — degrado ma recuperabile (es. fallback attivato)
    ERROR    — operazione fallita, impatto locale, Eden continua
    CRITICAL — rischia di corrompere dati-ricerca o silenziare strumenti
               pre-registrati. Richiede attenzione immediata.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional


# Buffer iniettato da agent.py all'avvio. None finche' non viene settato.
_buffer: Any = None

# Severity valide — convenzione centralizzata.
SEVERITIES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def set_buffer(buffer: Any) -> None:
    """
    Inietta il LogBuffer (tipicamente da agent.py al boot).
    Chiamato UNA VOLTA. Chiamate successive sovrascrivono (utile per test).
    """
    global _buffer
    _buffer = buffer


def has_buffer() -> bool:
    """True se il buffer e' stato iniettato. Utile per skip in test."""
    return _buffer is not None


def log(severity: str, module: str, message: str) -> None:
    """
    Append di un log generico. severity non valida → normalizzata a INFO.
    No-op silenzioso se buffer non iniettato.
    """
    if _buffer is None:
        return
    sev = (severity or "INFO").upper()
    if sev not in SEVERITIES:
        sev = "INFO"
    try:
        _buffer.append(sev, (module or "UNKNOWN").upper(), str(message))
    except Exception:
        # Ultimate failsafe: mai sollevare dentro codice di logging.
        pass


def log_exception(module: str, message: str, exc: BaseException,
                  severity: str = "ERROR") -> None:
    """
    Log di un'eccezione con summary breve del traceback.
    Il traceback completo NON va in UI (sporca) — solo ultima frame + tipo.
    """
    try:
        tb_summary = _format_exception_summary(exc)
        full_msg = f"{message} — {tb_summary}"
    except Exception:
        full_msg = f"{message} — <exc repr unavailable>"
    log(severity, module, full_msg)


def log_boot(module: str, ok: bool, detail: str = "") -> None:
    """
    Log di boot sanity-check. ok=True → INFO "OK". ok=False → CRITICAL "DOWN".
    Il modulo emette esattamente una riga al boot → copre il bug classe
    "modulo importato male, nessuno lo sa finche' non serve".
    """
    if ok:
        msg = f"OK{' — ' + detail if detail else ''}"
        log("INFO", module, msg)
    else:
        msg = f"DOWN — {detail or 'motivo non specificato'}"
        log("CRITICAL", module, msg)


def log_watchdog(module: str, status: str, detail: str = "") -> None:
    """
    Log del watchdog di salute. status in {'HEALTHY','DEGRADED','DOWN'}.
    Mappa automaticamente su severity UI:
      HEALTHY  → DEBUG (silenzioso, ma visibile filtrando)
      DEGRADED → WARNING
      DOWN     → CRITICAL
    """
    st = (status or "").upper()
    if st == "HEALTHY":
        sev = "DEBUG"
    elif st == "DEGRADED":
        sev = "WARNING"
    elif st == "DOWN":
        sev = "CRITICAL"
    else:
        sev = "INFO"
    msg = f"[{st}] {detail}" if detail else f"[{st}]"
    log(sev, module, msg)


# ============================================================
# Helpers interni
# ============================================================

def _format_exception_summary(exc: BaseException) -> str:
    """
    Estrae ultima frame + tipo eccezione in formato compatto per UI.
    Esempio output: 'AttributeError@observer.py:125 _session_file_path —
    NoneType has no attribute replace'
    """
    try:
        tb = getattr(exc, "__traceback__", None)
        last_frame = None
        while tb is not None:
            last_frame = tb.tb_frame
            tb = tb.tb_next
        if last_frame is not None:
            fname = last_frame.f_code.co_filename.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            lineno = last_frame.f_lineno
            funcname = last_frame.f_code.co_name
            return f"{type(exc).__name__}@{fname}:{lineno} {funcname} — {exc!s}"
    except Exception:
        pass
    return f"{type(exc).__name__}: {exc!s}"
