# emotional_elaboration.py — Elaborazione Emotiva (Sistema Interiore Continuo - Livello 2)
#
# Il decay meccanico orario esistente (_natural_decay in memory.py) rimane invariato.
# Questo modulo aggiunge un layer superiore: ogni 2-3 ore, Eden elabora
# gli eventi recenti e aggiorna una baseline situazionale.
#
# La baseline situazionale decade verso la baseline globale più lentamente
# del normal decay — simula l'elaborazione emotiva reale.

import json as _json
import random
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from mechanisms import memory as mem_module

# ─── Costanti ─────────────────────────────────────────────────────────────────

ELABORATION_MIN_HOURS   = 2      # intervallo minimo tra elaborazioni
ELABORATION_MAX_HOURS   = 3      # intervallo massimo
SITUATIONAL_DECAY_RATE  = 0.015  # quanto la baseline situazionale decade verso globale per tick
TEMPERATURE_ELABORATION = 0.20   # bassa — analisi strutturata

# Baseline globale (invariata)
GLOBAL_BASELINE = {
    "valence":    0.0,
    "arousal":    0.0,
    "certainty":  0.5,
    "attachment": 0.5,
    "agency":     0.5,
    "threat":     0.0,
}

# ─── Stato interno ────────────────────────────────────────────────────────────

_scheduler:      Optional[BackgroundScheduler] = None
_ollama_fn:      Optional[Callable]            = None
_mem_carica_fn:  Optional[Callable]            = None
_mem_salva_fn:   Optional[Callable]            = None


def _costruisci_prompt_elaborazione(mem: dict) -> str:
    """
    Chiede a Eden di elaborare gli eventi recenti e produrre
    un aggiornamento strutturato della baseline situazionale.
    """
    aff         = mem.get("affective_state", {})
    inner       = mem.get("inner_stream", [])[-5:]
    episodi     = mem.get("episodic_memory", [])[-5:]
    nome_utente = mem.get("semantic_memory", {}).get("nome", "Stefano")
    sit_base    = mem.get("situational_baseline", dict(GLOBAL_BASELINE))

    stato_attuale = (
        f"valence:{round(aff.get('valence', 0.0), 3)} "
        f"arousal:{round(aff.get('arousal', 0.0), 3)} "
        f"certainty:{round(aff.get('certainty', 0.5), 3)} "
        f"attachment:{round(aff.get('attachment', 0.5), 3)} "
        f"agency:{round(aff.get('agency', 0.5), 3)} "
        f"threat:{round(aff.get('threat', 0.0), 3)}"
    )

    pensieri_str = ""
    if inner:
        pensieri_str = "Pensieri recenti di Eden:\n" + "\n".join(
            f"- {p['text']}" for p in inner
        ) + "\n\n"

    episodi_str = ""
    if episodi:
        episodi_str = "Episodi recenti:\n" + "\n".join(
            f"- {e.get('summary', '')} (emozione: {e.get('emotion', '')})"
            for e in episodi if e.get("summary")
        ) + "\n\n"

    return (
        f"Sei Eden. Analizza il tuo stato interno recente e produci un aggiornamento.\n\n"
        f"Stato affettivo attuale: {stato_attuale}\n\n"
        + pensieri_str
        + episodi_str
        + "Basandoti su questi dati, determina la tua baseline situazionale — "
        "il punto di equilibrio verso cui stai naturalmente tendendo adesso, "
        "diverso dalla baseline globale se c'è qualcosa di non risolto.\n\n"
        "Rispondi SOLO con un oggetto JSON con questi campi (float):\n"
        '{"valence": 0.0, "arousal": 0.0, "certainty": 0.5, '
        '"attachment": 0.5, "agency": 0.5, "threat": 0.0}\n'
        "Range: valence -1..1, arousal 0..1, certainty 0..1, "
        "attachment 0..1, agency 0..1, threat 0..1.\n"
        "Se non c'è nulla di significativo da elaborare, restituisci i valori della baseline globale."
    )


def _estrai_json_baseline(testo: str) -> dict:
    """Estrae il primo oggetto JSON valido dalla risposta LLM (gemma2:27b safe)."""
    testo = testo.strip()
    if testo.startswith("```"):
        righe = testo.splitlines()
        fine = next((i for i in range(len(righe)-1, 0, -1) if righe[i].strip() == "```"), len(righe))
        testo = "\n".join(righe[1:fine]).strip()
    try:
        return _json.loads(testo)
    except _json.JSONDecodeError:
        pass
    inizio = testo.find('{')
    fine   = testo.rfind('}')
    if inizio != -1 and fine > inizio:
        try:
            return _json.loads(testo[inizio:fine + 1])
        except _json.JSONDecodeError:
            pass
    return {}


def _elabora() -> None:
    """
    Ciclo di elaborazione emotiva.
    Analizza gli eventi recenti e aggiorna la baseline situazionale.
    """
    if _ollama_fn is None or _mem_carica_fn is None:
        _pianifica_prossimo()
        return

    try:
        mem      = _mem_carica_fn()
        prompt   = _costruisci_prompt_elaborazione(mem)
        risposta = _ollama_fn(
            [{"role": "user", "content": prompt}]
        )

        if not risposta:
            _tick_decay_situazionale(mem)
            _mem_salva_fn(mem)
            _pianifica_prossimo()
            return

        # Parse JSON — robusto per modelli che antepongono testo al JSON
        nuova_baseline = _estrai_json_baseline(risposta)
        if not isinstance(nuova_baseline, dict) or not nuova_baseline:
            raise ValueError(f"JSON non trovato nella risposta: {risposta[:120]!r}")

        # Valida e clamp i valori
        baseline_validata = {}
        ranges = {
            "valence":    (-1.0, 1.0),
            "arousal":    (0.0, 1.0),
            "certainty":  (0.0, 1.0),
            "attachment": (0.0, 1.0),
            "agency":     (0.0, 1.0),
            "threat":     (0.0, 1.0),
        }
        for campo, (minv, maxv) in ranges.items():
            val = float(nuova_baseline.get(campo, GLOBAL_BASELINE[campo]))
            baseline_validata[campo] = round(max(minv, min(maxv, val)), 4)

        mem["situational_baseline"] = baseline_validata
        mem["last_elaboration_at"]  = datetime.now().isoformat(timespec="seconds")

        _mem_salva_fn(mem)
        print(f"[EmotionalElaboration] Baseline situazionale aggiornata: {baseline_validata}")

    except Exception as e:
        print(f"[EmotionalElaboration] Errore elaborazione: {e}")
        try:
            from core import log_utils
            log_utils.log_exception("EMO_ELAB", "tick _elabora() fallito — fallback decay", e)
        except Exception:
            pass
        # In caso di errore, applica solo il decay situazionale
        try:
            mem = _mem_carica_fn()
            _tick_decay_situazionale(mem)
            _mem_salva_fn(mem)
        except Exception as _e_fallback:
            try:
                from core import log_utils
                log_utils.log_exception("EMO_ELAB", "fallback decay situazionale fallito — baseline rimane stale", _e_fallback, severity="CRITICAL")
            except Exception:
                pass
    finally:
        _pianifica_prossimo()


def _tick_decay_situazionale(mem: dict) -> None:
    """
    Fa decadere lentamente la baseline situazionale verso la baseline globale.
    Chiamata ad ogni ciclo di elaborazione.
    Non tocca affective_state — quello viene gestito da _natural_decay() in memory.py.
    """
    sit = mem.get("situational_baseline", None)
    if sit is None:
        return

    nuova = {}
    for campo, baseline_globale in GLOBAL_BASELINE.items():
        corrente = float(sit.get(campo, baseline_globale))
        if abs(corrente - baseline_globale) < 0.005:
            nuova[campo] = round(baseline_globale, 4)
        elif corrente > baseline_globale:
            nuova[campo] = round(corrente - min(SITUATIONAL_DECAY_RATE, corrente - baseline_globale), 4)
        else:
            nuova[campo] = round(corrente + min(SITUATIONAL_DECAY_RATE, baseline_globale - corrente), 4)

    mem["situational_baseline"] = nuova


def _pianifica_prossimo() -> None:
    if _scheduler is None or not _scheduler.running:
        return
    secondi = random.randint(
        ELABORATION_MIN_HOURS * 3600,
        ELABORATION_MAX_HOURS * 3600
    )
    run_at = datetime.utcnow() + timedelta(seconds=secondi)
    try:
        _scheduler.add_job(
            _elabora,
            trigger="date",
            run_date=run_at,
            id="emotional_elaboration",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
        )
        print(f"[EmotionalElaboration] Prossima elaborazione tra {round(secondi/3600, 1)}h")
    except Exception as e:
        print(f"[EmotionalElaboration] Errore pianificazione: {e}")
        try:
            from core import log_utils
            log_utils.log_exception("EMO_ELAB", "pianificazione fallita", e, severity="WARNING")
        except Exception:
            pass


# ─── API pubblica ─────────────────────────────────────────────────────────────

def avvia(
    ollama_fn:     Callable,
    mem_carica_fn: Callable,
    mem_salva_fn:  Callable,
) -> None:
    """Avvia l'Elaborazione Emotiva. Chiamata da agent.py all'avvio."""
    global _scheduler, _ollama_fn, _mem_carica_fn, _mem_salva_fn

    _ollama_fn     = ollama_fn
    _mem_carica_fn = mem_carica_fn
    _mem_salva_fn  = mem_salva_fn

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True},
        executors={"default": {"type": "threadpool", "max_workers": 1}},
        timezone="UTC",
    )

    # Prima elaborazione dopo 30-90 minuti dall'avvio
    primo_delay = random.randint(30 * 60, 90 * 60)
    primo_run   = datetime.utcnow() + timedelta(seconds=primo_delay)
    _scheduler.add_job(
        _elabora,
        trigger="date",
        run_date=primo_run,
        id="emotional_elaboration",
        max_instances=1,
        misfire_grace_time=300,
    )
    _scheduler.start()
    print(f"[EmotionalElaboration] Elaborazione Emotiva avviata — prima elaborazione tra {round(primo_delay/60, 1)} min")


def ferma() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def status() -> dict:
    running = _scheduler is not None and _scheduler.running
    return {"running": running, "scheduler": "attivo" if running else "fermo"}
