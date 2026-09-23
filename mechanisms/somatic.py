"""somatic.py — Grounding somatico via telemetria GPU (Breakpoint B-Graph, 2026-04-30).

Ispirazione Reddit: hardware metrics → "somatic entropy" → routing cognitivo.
Razionale neuroscientifico: Damasio (1999) Somatic Marker Hypothesis — il sé
emerge dall'integrazione continua di segnali corporei. In un agente locale,
gli unici "segnali corporei" sono GPU temp, VRAM utilization, e load del modello.

Mappatura:
    gpu_temp_norm        = (temp_C - 30) / 60         # 30C=0, 90C=1
    vram_util_norm       = vram_used / vram_total      # 0..1
    cpu_load_norm        = (CPU 1-min) / cpu_count    # 0..1+ (può saturare)
    somatic_entropy      = mean(temp_norm, vram_norm) # 0..1 — overall body stress
    somatic_arousal      = vram_util_norm              # carico cognitivo
    somatic_threat       = max(0, temp_norm - 0.7)*3  # >70%temp → threat alta

Effetto su Layer 1: somatic_entropy alta → bias verso block_proactive,
risposte più brevi (analogo a stato di stress fisico).

Le metriche somatiche sono AUTONOME (non dipendono da rating umano), forniscono
un secondo canale di verità che non può essere falsato da bug rule-based.

Versionamento: _SOMATIC_VERSION stampato su ogni snapshot.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

_SOMATIC_VERSION = "v1.3"  # 2026-04-30 sera — + persistenza history su disco + soglie wiring Layer 1
_ATTIVO = True
_TEMP_FLOOR = 30.0      # C
_TEMP_CEIL = 90.0       # C
_VRAM_THRESHOLD_HIGH = 0.85   # >85% VRAM → arousal=high

# Thermal persistence (v1.2 → v1.3) — analogo al cortisolo: non il picco acuto
# conta, ma la durata dell'esposizione allo stress (McEwen 1998, allostatic load).
# Buffer circolare degli ultimi 6 snapshot (ogni 5 min = 30 min finestra).
# EWMA α=0.40: peso recente 40%, poi 24%, 14.4%, ... decrescente verso passato.
# v1.3: history persistita in mem["homeostatic_state"]["somatic_history"] tramite
# load/save espliciti — sopravvive ai restart Eden.
_SNAPSHOT_HISTORY: list[dict] = []
_MAX_HISTORY = 6
_EWMA_ALPHA   = 0.40

# Soglie wiring → Layer 1 (v1.3): l'allostatic_load oltre queste soglie modula
# effettivamente la generazione (block_proactive, temperature_override).
# Calibrate con margine: 0.70 = stress sostenuto chiaro, 0.85 = stress acuto.
_ALLOSTATIC_BLOCK_PROACTIVE = 0.70
_ALLOSTATIC_TEMP_REDUCE     = 0.85
_SOMATIC_THREAT_BLOCK       = 0.50  # threat istantaneo: GPU > ~85°C


import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")


def _compute_persistence(values: list, alpha: float = _EWMA_ALPHA) -> float:
    """EWMA [oldest→newest]: carica accumulata nel tempo, non solo picco corrente.

    Neuroscientifico: McEwen (1998) allostatic load — l'organismo risponde alla
    durata cumulativa dello stress, non al singolo picco. Un'ora a 70°C conta
    più di un picco di 5 min a 85°C.
    α=0.40 → il campione più recente pesa 40%, il precedente 24%, poi 14.4%...
    """
    if not values:
        return 0.0
    ewma = float(values[0])
    for v in values[1:]:
        ewma = alpha * float(v) + (1.0 - alpha) * ewma
    return round(ewma, 4)


def _circadian_arousal(hour: int) -> float:
    """Modello circadiano semplificato dell'arousal fisiologico in funzione dell'ora.

    Basato su: Kleitman (1963) basic rest-activity cycle + Monk et al. (1997).
    Due picchi: mattino (09h) e pomeriggio (15h). Minimo: 04h (sonno profondo).
    Output [0, 1] — additivo al somatic_arousal GPU, non sostitutivo.
    """
    # Tabella oraria: ora → arousal circadiano
    _TABLE = [
        0.10, 0.05, 0.05, 0.05, 0.05, 0.15,  # 00-05
        0.35, 0.55, 0.75, 0.90, 0.85, 0.80,  # 06-11
        0.70, 0.65, 0.60, 0.75, 0.80, 0.75,  # 12-17
        0.65, 0.55, 0.45, 0.35, 0.20, 0.12,  # 18-23
    ]
    return _TABLE[max(0, min(23, int(hour)))]


def _circadian_phase(hour: int) -> str:
    """Fase del ciclo circadiano in linguaggio narrativo."""
    if  5 <= hour < 12: return "mattino"
    if 12 <= hour < 17: return "pomeriggio"
    if 17 <= hour < 22: return "sera"
    return "notte"


def _read_nvml() -> Optional[dict]:
    """Legge tutte le GPU e ritorna la più carica (quella che Eden sta usando).

    Setup dual-GPU (29/04): generatore principale = 3090 Ti, secondaria = 5060 Ti.
    Su single-GPU, ritorna semplicemente l'unica disponibile.
    Override via env EDEN_SOMATIC_GPU_INDEX (forza quella GPU).
    """
    try:
        import pynvml as nv
        nv.nvmlInit()
        n = nv.nvmlDeviceGetCount()
        if n == 0:
            return None
        force_idx = os.environ.get("EDEN_SOMATIC_GPU_INDEX")
        if force_idx is not None:
            try:
                indices = [int(force_idx)]
            except ValueError:
                indices = list(range(n))
        else:
            indices = list(range(n))

        best = None
        for i in indices:
            h = nv.nvmlDeviceGetHandleByIndex(i)
            temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            meminfo = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            try:
                name = nv.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
            except Exception:
                name = f"GPU{i}"
            vram_util = meminfo.used / max(meminfo.total, 1)
            cand = {
                "gpu_index": i,
                "gpu_name": name,
                "gpu_temp_c": int(temp),
                "vram_used_gb": round(meminfo.used / (1024**3), 2),
                "vram_total_gb": round(meminfo.total / (1024**3), 2),
                "vram_util": round(vram_util, 4),
                "gpu_util": int(util.gpu),
                "mem_util": int(util.memory),
            }
            # Preferisci la GPU con VRAM più caricata (= dove gira Ollama)
            if best is None or cand["vram_util"] > best["vram_util"]:
                best = cand
        return best
    except Exception:
        return None


def _read_cpu_load() -> Optional[float]:
    """1-min load average normalizzata sul cpu_count. None se non disponibile."""
    try:
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()[0]
            cnt = os.cpu_count() or 1
            return round(la / cnt, 3)
    except Exception:
        pass
    # Fallback: psutil se disponibile
    try:
        import psutil
        return round(psutil.cpu_percent(interval=0.1) / 100.0, 3)
    except Exception:
        return None


def _load_history(mem: Optional[dict]) -> None:
    """Carica history persistita da memory in _SNAPSHOT_HISTORY (idempotente).

    v1.3: senza questa load, ogni restart Eden azzerava la EWMA e l'allostatic
    load ripartiva da 0 — incoerente con l'analogia McEwen (lo stress cumulativo
    NON si resetta solo perché il sistema riavvia il processo).
    """
    global _SNAPSHOT_HISTORY
    if _SNAPSHOT_HISTORY:
        return  # già caricata in questa sessione
    if not isinstance(mem, dict):
        return
    state = mem.get("homeostatic_state", {})
    persisted = state.get("somatic_history")
    if isinstance(persisted, list) and persisted:
        # Conserva solo gli ultimi _MAX_HISTORY (in caso di file vecchi più lunghi)
        _SNAPSHOT_HISTORY = list(persisted[-_MAX_HISTORY:])


def _save_history(mem: Optional[dict]) -> None:
    """Persiste _SNAPSHOT_HISTORY in mem['homeostatic_state']['somatic_history'].

    Non chiama salva_memoria — il caller (agent.py / scheduler) è responsabile
    della persistenza atomica. Qui solo aggiornamento in-memory del dict.
    """
    if not isinstance(mem, dict):
        return
    state = mem.setdefault("homeostatic_state", {})
    state["somatic_history"] = list(_SNAPSHOT_HISTORY)


def snapshot(mem: Optional[dict] = None) -> dict:
    """Ritorna stato somatico corrente con thermal/vram persistence (v1.3).

    Fail-safe: chiavi None se non leggibile.
    Aggiorna _SNAPSHOT_HISTORY (buffer EWMA) ad ogni chiamata.

    v1.3: se `mem` è fornito, carica history persistita al primo uso e la
    salva in `mem` dopo l'aggiornamento del buffer. Senza `mem` (chiamata
    diagnostica) usa il buffer in-memory corrente — comportamento legacy.
    """
    global _SNAPSHOT_HISTORY

    # v1.3: carica history persistita se mem fornita e buffer ancora vuoto
    _load_history(mem)

    if not _ATTIVO:
        return {"active": False, "version": _SOMATIC_VERSION}
    out = {
        "ts": datetime.now().isoformat(),
        "version": _SOMATIC_VERSION,
        "active": True,
    }
    nvml = _read_nvml()
    if nvml:
        out.update(nvml)
        # Normalizzazioni istantanee
        temp_norm = max(0.0, min(1.0, (nvml["gpu_temp_c"] - _TEMP_FLOOR) / (_TEMP_CEIL - _TEMP_FLOOR)))
        vram_norm = nvml["vram_util"]
        out["temp_norm"] = round(temp_norm, 4)
        out["vram_norm"] = round(vram_norm, 4)
        out["somatic_entropy"] = round((temp_norm + vram_norm) / 2.0, 4)
        out["somatic_arousal"] = round(vram_norm, 4)
        out["somatic_threat"]  = round(max(0.0, (temp_norm - 0.7) * 3.0), 4)

        # Thermal persistence (v1.2): EWMA su finestra 30 min
        # Aggiunge il campione corrente alla storia e mantiene max _MAX_HISTORY entry.
        _SNAPSHOT_HISTORY.append({"temp_norm": temp_norm, "vram_norm": vram_norm})
        if len(_SNAPSHOT_HISTORY) > _MAX_HISTORY:
            _SNAPSHOT_HISTORY = _SNAPSHOT_HISTORY[-_MAX_HISTORY:]

        if len(_SNAPSHOT_HISTORY) >= 2:
            _temps = [s["temp_norm"] for s in _SNAPSHOT_HISTORY]
            _vrams = [s["vram_norm"] for s in _SNAPSHOT_HISTORY]
            out["temp_persistence"] = _compute_persistence(_temps)
            out["vram_persistence"] = _compute_persistence(_vrams)
            # Allostatic load: media pesata tra picco istantaneo e carica accumulata.
            # temp_persistence > temp_norm → sistema in stress prolungato (più pericoloso).
            out["allostatic_load"] = round(
                0.40 * temp_norm + 0.60 * out["temp_persistence"], 4
            )
        else:
            out["temp_persistence"] = temp_norm
            out["vram_persistence"] = vram_norm
            out["allostatic_load"]  = temp_norm
    else:
        out["error"] = "nvml_unavailable"

    cpu = _read_cpu_load()
    if cpu is not None:
        out["cpu_load_norm"] = cpu

    # Segnale circadiano — indipendente da GPU, sempre disponibile
    h = datetime.now().hour
    out["circadian_arousal"] = _circadian_arousal(h)
    out["circadian_phase"]   = _circadian_phase(h)

    # v1.3: persiste history aggiornata se mem fornita
    _save_history(mem)

    return out


def compute_modulation(somatic_state: dict) -> dict:
    """Calcola modulazione effettiva da somatic_state — wiring v1.3.

    Mapping (deterministico, peer-reviewable):
      - allostatic_load > 0.85 → temperature_reduce 0.10 (tono asciutto, stress acuto)
      - allostatic_load > 0.70 → block_proactive (no proattivi sotto carico sostenuto)
      - somatic_threat  > 0.50 → block_proactive (GPU > ~85°C: emergenza termica)
      - vram_norm       > 0.95 → length_reduce 0.75 (VRAM saturo: risposte più brevi)

    Output (sempre presente, default neutro):
      {block_proactive: bool, temperature_delta: float|None,
       length_multiplier: float, reason: str}
    """
    out = {
        "block_proactive": False,
        "temperature_delta": None,
        "length_multiplier": 1.0,
        "reason": "",
    }
    if not somatic_state:
        return out

    allo = somatic_state.get("allostatic_load") or 0.0
    threat = somatic_state.get("somatic_threat") or 0.0
    vram = somatic_state.get("vram_norm") or 0.0

    reasons = []
    if allo > _ALLOSTATIC_TEMP_REDUCE:
        out["temperature_delta"] = -0.10
        out["block_proactive"]   = True
        reasons.append(f"allostatic_acute({allo:.2f})")
    elif allo > _ALLOSTATIC_BLOCK_PROACTIVE:
        out["block_proactive"] = True
        reasons.append(f"allostatic_sustained({allo:.2f})")

    if threat > _SOMATIC_THREAT_BLOCK:
        out["block_proactive"] = True
        reasons.append(f"thermal_threat({threat:.2f})")

    if vram > 0.95:
        out["length_multiplier"] = 0.75
        reasons.append(f"vram_saturated({vram:.2f})")

    out["reason"] = "+".join(reasons) if reasons else "neutral"
    return out


def apply_to_homeostatic(mem: dict, snap: Optional[dict] = None) -> dict:
    """Inietta somatic_state in mem['homeostatic_state'] e ritorna le modulazioni effettive.

    v1.3 — wiring effettivo: oltre a scrivere somatic_state (canale additivo),
    calcola e scrive `somatic_modulation` con effetti deterministici su
    block_proactive / temperature / length. agent.py legge questa chiave per
    applicare la modulazione alla prossima generazione.

    Layer 1 originale (umano + rule-based) resta firmato/versionato; la
    modulazione somatica si COMPONE con quella di Layer 1 (OR logico su
    block_proactive, somma su delta — vedi agent._apply_modulazione).

    Pre-registrazione: dichiarato in addendum a PRE_REGISTRATION_v3.
    """
    if not isinstance(mem, dict):
        return {}
    # v1.3: passa mem a snapshot per load/save persistente
    s = snap or snapshot(mem)
    if not s.get("active") or "somatic_entropy" not in s:
        return {"applied": False, "reason": "no_signal"}

    state = mem.setdefault("homeostatic_state", {})
    state["somatic_state"] = {
        "ts": s["ts"],
        "version": s["version"],
        "temp_norm":        s.get("temp_norm"),
        "vram_norm":        s.get("vram_norm"),
        "somatic_entropy":  s.get("somatic_entropy"),
        "somatic_arousal":  s.get("somatic_arousal"),
        "somatic_threat":   s.get("somatic_threat"),
        "temp_persistence": s.get("temp_persistence"),
        "vram_persistence": s.get("vram_persistence"),
        "allostatic_load":  s.get("allostatic_load"),
        "circadian_arousal": s.get("circadian_arousal"),
        "circadian_phase":   s.get("circadian_phase"),
    }
    # v1.3: wiring effettivo a Layer 1
    state["somatic_modulation"] = compute_modulation(state["somatic_state"])
    return {"applied": True, **state["somatic_state"],
            "modulation": state["somatic_modulation"]}


def status() -> dict:
    return {
        "active": _ATTIVO,
        "version": _SOMATIC_VERSION,
        "snapshot": snapshot(),
    }
