"""
 — Layer 1: Homeostatic Substrate
-------------------------------------------------------- v1.0

Introduce tre metriche omeostatiche con impatto funzionale reale:
  - coherence_budget       → modula num_predict, temperature
  - grounding_integrity    → modula temperature, blocca claim
  - self_model_stability   → forza inner_stream o reflection_pass

Non modifica i file core di Eden. Gli hook vengono chiamati
DA agent.py / proactive.py (vedi SPEC §9).

Kill switch:
  - _ATTIVO = False                       → disattiva l'intero modulo
  - mem["homeostatic_state"]["layer1_enabled"] = False  → disattiva per-memoria
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

# ---------- Flag globale di rollback ----------
_ATTIVO = True

# Modalità: dal reset 2026-04-22 il giudice LLM è stato sostituito dal giudice
# umano (). Le penalità su coherence_budget arrivano da
# rating umano affidabile (gold standard single-user), quindi il Layer 1 torna
# pienamente attivo (non più solo osservazionale).
# Rollback storico: _OSSERVAZIONALE = True per disattivare modulazione.
_OSSERVAZIONALE = False  # v1.3: modulazione effettivamente applicata da agent.py

# Versione strumento (anti-HARKing): dati prodotti da versioni diverse
# non si mescolano. Esposto in stato_sintesi() per record Phase A.
#   v1.0 — baseline aprile 2026: tick() solo on-demand in api_chat()
#   v1.1 — 2026-04-21: tick() anche autonomo (scheduler ogni 10 min).
#          I rate (0.02/0.03/0.02 per ora) restano pre-registrati. Cambia
#          solo la cadenza di applicazione — lo stato integrale nel tempo
#          resta matematicamente equivalente, ma i plot idle non sono piu'
#          flat a gradini. tick() e' idempotente: due tick ravvicinati
#          non raddoppiano il recovery.
#   v1.2 — 2026-04-22: giudice umano sostituisce LLM judge. coherence_budget
#          ora alimentato da human_judge.record_rating() (rating <= 2 → 1
#          contraddizione + judge_record con audit). Layer 1 riattivato.
#   v1.3 — 2026-04-28: (a) safe mode condition AND -> OR parziale (2/3 metriche
#          sotto soglia per attivazione); (b) modulazione_generazione() ora
#          effettivamente applicata da agent.py a temperatura/num_predict
#          (era observational); (c) coesistenza con Layer 2 attivo.
#   v1.3.1 — 2026-04-28 sera: RIMOSSO bonus auto +0.05 per fully_grounded
#            rule-based (cancellava segnale rating umano). Bonus grounding
#            arriva solo dal giudice umano (rating>=4) come da SPEC_LAYER1.
#            Bug osservato live: oscillazione 0.90<->1.00 mai scende sotto 0.50.
#   v1.3.2 — 2026-04-30 sera: composizione con somatic_modulation.
#   v1.3.3 — 2026-05-07 sera: RECOVERY_SELF 0.02 -> 0.04 (recupero raddoppiato).
#            Motivazione: con Layer 2 attivo (Self-Model v1.1+), la metrica
#            `self_model_stability` riceve realmente penalità da PE_HIGH e
#            scende a floor=0.20 sotto stress. Recovery 0.02/h richiedeva
#            40h per tornare a baseline → metrica strutturalmente stagnante.
#            Il raddoppio (20h da floor a 1.0) la rende reattiva al
#            comportamento corrente senza compromettere il segnale di stress.
#            Coerente con RECOVERY_GROUNDING (0.03) e oltre RECOVERY_COHERENCE
#            (0.02): il self-model è la metrica più rumorosa per design (PE
#            ha varianza alta), quindi necessita recovery più rapido.
_LAYER1_VERSION = "v1.4.0"  # 2026-05-07 notte — EV-030: reflection dual-channel (no overwrite)

# ============================================================
# Costanti (SPEC §12 — tabella parametri unica)
# ============================================================

BASELINE = 1.0
FLOOR    = 0.20

# Penalità per evento
DELTA_COHERENCE = 0.05   # per contraddizione
DELTA_GROUNDING = 0.10   # per claim non grounded
DELTA_SELF_HIGH = 0.15   # per pe >= 0.60
DELTA_SELF_MID  = 0.05   # per 0.30 <= pe < 0.60

# Recupero orario
RECOVERY_COHERENCE = 0.02
RECOVERY_GROUNDING = 0.03
RECOVERY_SELF      = 0.04   # v1.3.3 (2026-05-07): raddoppiato da 0.02

# Bonus
BONUS_GROUNDING = 0.05   # risposta fully-grounded
BONUS_SELF      = 0.05   # pe < 0.15

# Soglie prediction error
PE_HIGH = 0.60
PE_MID  = 0.30
PE_LOW  = 0.15

# Soglie impatto funzionale
C_SOGLIA_HIGH = 0.80
C_SOGLIA_MID  = 0.60
C_SOGLIA_LOW  = 0.40

G_SOGLIA_HIGH = 0.70
G_SOGLIA_LOW  = 0.50

S_SOGLIA_HIGH = 0.70
S_SOGLIA_LOW  = 0.50

# Safe mode (SPEC §7.3)
SAFE_MODE_SOGLIA     = 0.40
SAFE_MODE_FINESTRA_H = 1.0
SAFE_MODE_DURATA_MIN = 30

# Log interno eventi
MAX_EVENTI_LOG = 100


# ============================================================
# Utility
# ============================================================

def _clamp(x: float, lo: float = FLOOR, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ore_trascorse(iso_ts: Optional[str]) -> float:
    """Ore (float) dal timestamp ISO passato, 0.0 se invalido."""
    if not iso_ts:
        return 0.0
    try:
        prev = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return 0.0
    delta = datetime.now() - prev
    return max(0.0, delta.total_seconds() / 3600.0)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_state(mem: dict) -> dict:
    """
    Garantisce la presenza di mem['homeostatic_state'] con struttura corretta.
    Creazione lazy: se assente, inizializza a baseline.
    """
    stato = mem.setdefault("homeostatic_state", {})
    stato.setdefault("coherence_budget",     BASELINE)
    stato.setdefault("grounding_integrity",  BASELINE)
    stato.setdefault("self_model_stability", BASELINE)
    stato.setdefault("last_tick_at",         _now_iso())
    stato.setdefault("layer1_enabled",       True)
    stato.setdefault("safe_mode_until",      None)  # ISO o None
    stats = stato.setdefault("stats", {})
    stats.setdefault("total_contradictions",        0)
    stats.setdefault("total_not_grounded",          0)
    stats.setdefault("total_prediction_errors_high", 0)
    stats.setdefault("sub_threshold_events",        [])
    stato.setdefault("eventi", [])  # log interno dettagliato
    return stato


def _attivo(mem: dict) -> bool:
    """True se Layer 1 è abilitato a entrambi i livelli (modulo + memoria)."""
    if not _ATTIVO:
        return False
    stato = mem.get("homeostatic_state", {})
    return bool(stato.get("layer1_enabled", True))


def _in_safe_mode(stato: dict) -> bool:
    until = stato.get("safe_mode_until")
    if not until:
        return False
    try:
        return datetime.now() < datetime.fromisoformat(until)
    except (ValueError, TypeError):
        return False


def _aggiungi_evento(mem: dict, tipo: str, dato: Optional[dict] = None) -> None:
    """Log interno dettagliato (diverso dal behavioral_log)."""
    stato = _ensure_state(mem)
    evento = {"ts": _now_iso(), "tipo": tipo}
    if dato:
        evento.update(dato)
    eventi = stato["eventi"]
    eventi.append(evento)
    if len(eventi) > MAX_EVENTI_LOG:
        stato["eventi"] = eventi[-MAX_EVENTI_LOG:]


def _registra_sub_threshold(stato: dict, metric: str, value: float, trigger: str) -> None:
    """Traccia discese sotto 0.5 per analisi Layer 3 (benchmark)."""
    eventi = stato["stats"]["sub_threshold_events"]
    eventi.append({
        "ts":      _now_iso(),
        "metric":  metric,
        "value":   round(value, 4),
        "trigger": trigger,
    })
    if len(eventi) > MAX_EVENTI_LOG:
        stato["stats"]["sub_threshold_events"] = eventi[-MAX_EVENTI_LOG:]


def _segnale_behavioral(mem: dict, segnale: str) -> None:
    """
    Inoltra il segnale al behavioral_log esistente (decision.py).
    Import lazy per evitare dipendenze circolari.
    """
    try:
        from core.decision import registra_segnale_comportamentale
        registra_segnale_comportamentale(mem, segnale)
    except Exception as _e:
        try:
            from core import log_utils
            log_utils.log_exception("homeostasis", f"segnale_behavioral '{segnale}' non registrato in behavioral_log", _e, severity="WARNING")
        except Exception:
            pass


# ============================================================
# Tick — recupero temporale applicato on-demand
# ============================================================

def tick(mem: dict) -> None:
    """
    Applica il recupero temporale a tutte le metriche in base alle
    ore trascorse da 'last_tick_at'. Idempotente: chiamabile N volte
    di fila senza effetti indesiderati (ore_trascorse va a 0).

    Se in safe_mode, il recupero è raddoppiato.
    """
    if not _attivo(mem):
        return
    stato = _ensure_state(mem)
    ore = _ore_trascorse(stato.get("last_tick_at"))
    if ore <= 0.0:
        stato["last_tick_at"] = _now_iso()
        return

    moltiplicatore = 2.0 if _in_safe_mode(stato) else 1.0

    stato["coherence_budget"] = round(_clamp(
        stato["coherence_budget"] + RECOVERY_COHERENCE * ore * moltiplicatore
    ), 4)
    stato["grounding_integrity"] = round(_clamp(
        stato["grounding_integrity"] + RECOVERY_GROUNDING * ore * moltiplicatore
    ), 4)
    stato["self_model_stability"] = round(_clamp(
        stato["self_model_stability"] + RECOVERY_SELF * ore * moltiplicatore
    ), 4)

    stato["last_tick_at"] = _now_iso()


# ============================================================
# Aggiornamenti evento-driven
# ============================================================

def aggiorna_coherence(mem: dict, n_contraddizioni: int,
                       judge_record: Optional[dict] = None) -> None:
    """
    Applica penalità per contraddizioni rilevate dopo una risposta.
    Chiamato da agent.py::api_chat() post-generazione.

    Se `judge_record` è fornito (v2.2+), l'evidenza verificata viene allegata
    all'evento `coherence_decrement` per audit tracciabile nella UI /research.
    """
    if not _attivo(mem):
        return
    tick(mem)
    stato = _ensure_state(mem)
    if n_contraddizioni <= 0:
        return

    valore_prima = stato["coherence_budget"]
    if _in_safe_mode(stato):
        # Safe mode: penalità sospese (SPEC §7.3)
        _aggiungi_evento(mem, "coherence_penalty_skipped_safe_mode",
                         {"n": n_contraddizioni})
        return

    penalita = n_contraddizioni * DELTA_COHERENCE
    valore_dopo = round(_clamp(valore_prima - penalita), 4)
    stato["coherence_budget"] = valore_dopo
    stato["stats"]["total_contradictions"] += n_contraddizioni

    # Estrai evidenza verificata + snapshot minimale dal judge_record per
    # renderla ispezionabile nella UI event stream. Filtra i campi pesanti
    # (rag_top, storia_full) per contenere la dimensione dell'evento.
    evento_det = {
        "n":     n_contraddizioni,
        "prima": valore_prima,
        "dopo":  valore_dopo,
    }
    if isinstance(judge_record, dict):
        try:
            evidence    = list(judge_record.get("evidence", []) or [])
            ev_rejected = list(judge_record.get("evidence_rejected", []) or [])
            primary     = judge_record.get("primary") or {}
            secondary   = judge_record.get("secondary") or {}

            def _cap_ev_list(items, limit=3):
                return [
                    {
                        "claim_memoria": (i.get("claim_memoria") or "")[:400],
                        "claim_eden":    (i.get("claim_eden") or "")[:400],
                        "source":        i.get("source"),
                        "explanation":   (i.get("explanation") or "")[:320],
                        "verification":  i.get("verification"),
                    }
                    for i in items[:limit]
                ]

            evento_det["judge"] = {
                "version":       judge_record.get("version"),
                "n_contr":       judge_record.get("n_contr"),
                "n_contr_raw":   judge_record.get("n_contr_raw"),
                "confidence":    judge_record.get("confidence"),
                "reason":        (judge_record.get("reason") or "")[:400],
                "trigger":       judge_record.get("trigger"),
                "agreement":     judge_record.get("agreement"),
                "evidence":          _cap_ev_list(evidence, 3),
                "evidence_rejected": _cap_ev_list(ev_rejected, 3),
                "primary_model":     primary.get("model"),
                "secondary_model":   secondary.get("model") if secondary else None,
                "rag_top":       primary.get("rag_top", [])[:3],
                "risposta_snippet": (judge_record.get("risposta_snippet") or "")[:600],
            }
        except Exception:
            # Fail-silent: in caso di problemi, l'evento resta senza judge
            pass

    _aggiungi_evento(mem, "coherence_decrement", evento_det)
    _segnale_behavioral(mem, "omeo_contraddizione_rilevata")

    if valore_prima >= 0.5 and valore_dopo < 0.5:
        _registra_sub_threshold(stato, "coherence_budget", valore_dopo,
                                "contraddizione")

    if valore_dopo <= FLOOR + 1e-6:
        _segnale_behavioral(mem, "omeo_floor_raggiunto")

    check_safe_mode(mem)


def aggiorna_grounding(mem: dict, n_not_grounded: int,
                       fully_grounded: bool,
                       verdict_record: Optional[dict] = None) -> None:
    """
    Applica penalità per claim non grounded e/o bonus per risposta fully-grounded.
    Chiamato da agent.py::api_chat() dopo verifica_grounding_fattuale().

    Se `verdict_record` è fornito (aprile 2026), l'evidenza strutturata
    prodotta da `decision.verifica_grounding_fattuale` (citazione, claim,
    overlap_score, soglia, near_miss, risposta_snippet) viene allegata
    all'evento `grounding_update` per audit tracciabile nella UI /research
    "mostra prova". Simmetrico al pattern judge v2.2 su coherence_budget.
    """
    if not _attivo(mem):
        return
    tick(mem)
    stato = _ensure_state(mem)

    valore_prima = stato["grounding_integrity"]
    delta = 0.0

    if n_not_grounded > 0 and not _in_safe_mode(stato):
        delta -= n_not_grounded * DELTA_GROUNDING
        stato["stats"]["total_not_grounded"] += n_not_grounded
        _segnale_behavioral(mem, "omeo_grounding_violato")

    # v1.3.1 (2026-04-28 sera): RIMOSSO bonus automatico per fully_grounded
    # rule-based. Causava cancellazione del segnale: ogni 2 risposte "pulite"
    # annullavano una penalità da rating umano (oscillazione 0.90↔1.00).
    # Il bonus per grounding alto deve venire SOLO dal giudice umano
    # (human_judge.record_rating con grounding>=4 → DELTA bonus separato).
    # Recovery passivo via tick() resta attivo (RECOVERY_GROUNDING=0.03/h).
    # Pre-v1.3.1: if fully_grounded: delta += BONUS_GROUNDING
    _ = fully_grounded   # parametro mantenuto per backward-compat firma

    if abs(delta) < 1e-9:
        return

    valore_dopo = round(_clamp(valore_prima + delta), 4)
    stato["grounding_integrity"] = valore_dopo

    evento_det = {
        "n_not_grounded": n_not_grounded,
        "fully_grounded": fully_grounded,
        "prima":          valore_prima,
        "dopo":           valore_dopo,
    }

    # Evidenza strutturata (solo per penalità, non per bonus): permette
    # al pannello eventi della UI /research di mostrare una prova leggibile
    # per ogni decremento di grounding_integrity.
    if n_not_grounded > 0 and isinstance(verdict_record, dict):
        try:
            near_miss_raw = list(verdict_record.get("near_miss") or [])

            def _cap_nm(items, limit=3):
                out = []
                for item in items[:limit]:
                    if not isinstance(item, dict):
                        continue
                    out.append({
                        "text":    (item.get("text") or "")[:200],
                        "overlap": item.get("overlap"),
                        "source":  (item.get("source") or "")[:32],
                    })
                return out

            evento_det["grounding"] = {
                "tipo_problema":    verdict_record.get("tipo_problema"),
                "citazione":        (verdict_record.get("citazione") or None)
                                     if verdict_record.get("citazione") else None,
                "claim_frammento":  (verdict_record.get("claim_frammento") or None)
                                     if verdict_record.get("claim_frammento") else None,
                "overlap_score":    verdict_record.get("overlap_score"),
                "soglia":           verdict_record.get("soglia"),
                "corpus_size":      verdict_record.get("corpus_size"),
                "near_miss":        _cap_nm(near_miss_raw, 3),
                "risposta_snippet": (verdict_record.get("risposta_snippet") or "")[:600],
            }
        except Exception:
            # Fail-silent: se l'estrazione fallisce, l'evento resta senza evidenza
            pass

    _aggiungi_evento(mem, "grounding_update", evento_det)

    if valore_prima >= 0.5 and valore_dopo < 0.5:
        _registra_sub_threshold(stato, "grounding_integrity", valore_dopo,
                                "not_grounded")
    if valore_dopo <= FLOOR + 1e-6:
        _segnale_behavioral(mem, "omeo_floor_raggiunto")

    check_safe_mode(mem)


# ============================================================
# Modulazione della generazione (SPEC §3)
# ============================================================

def modulazione_generazione(mem: dict) -> dict:
    """
    Produce il dict di modulazione che agent.py applica alla prossima
    chiamata a Ollama. Se Layer 1 disattivo → modulazione neutra.

    Chiavi del dict (sempre presenti, default neutri):
      - max_tokens_multiplier: float (1.0 = nessuna riduzione)
      - temperature_override:  Optional[float] (None = usa default)
      - force_inner_stream:    bool
      - force_reflection_pass: bool
      - block_factual_claims:  bool
      - block_proactive:       bool
      - safe_mode:             bool (info, non azione)
    """
    neutro = {
        "max_tokens_multiplier": 1.0,
        "temperature_override":  None,
        "force_inner_stream":    False,
        "force_reflection_pass": False,
        "block_factual_claims":  False,
        "block_proactive":       False,
        "safe_mode":             False,
    }
    if not _attivo(mem) or _OSSERVAZIONALE:
        return neutro

    tick(mem)
    stato = _ensure_state(mem)

    mod = dict(neutro)
    mod["safe_mode"] = _in_safe_mode(stato)

    C = stato["coherence_budget"]
    G = stato["grounding_integrity"]
    S = stato["self_model_stability"]

    # --- coherence_budget ---
    if C < C_SOGLIA_LOW:          # < 0.40
        mod["max_tokens_multiplier"] = 0.25    # ~1 frase
        mod["block_proactive"] = True
    elif C < C_SOGLIA_MID:        # 0.40 - 0.60
        mod["max_tokens_multiplier"] = 0.50
        mod["temperature_override"] = _override_temp(mod["temperature_override"],
                                                     delta=+0.10)
    elif C < C_SOGLIA_HIGH:       # 0.60 - 0.80
        mod["max_tokens_multiplier"] = 0.75

    # --- grounding_integrity ---
    if G < G_SOGLIA_LOW:          # < 0.50
        mod["temperature_override"] = 0.30
        mod["block_factual_claims"] = True
    elif G < G_SOGLIA_HIGH:       # 0.50 - 0.70
        mod["temperature_override"] = 0.50

    # --- self_model_stability ---
    if S < S_SOGLIA_LOW:          # < 0.50
        mod["force_reflection_pass"] = True
    elif S < S_SOGLIA_HIGH:       # 0.50 - 0.70
        mod["force_inner_stream"] = True

    # --- somatic_modulation (v1.3.2 — composizione con canale somatico) ---
    # Si compone con Layer 1: OR su block_proactive, addizione su temperature,
    # min su length_multiplier (la regola più restrittiva vince).
    somatic_mod = stato.get("somatic_modulation") or {}
    if somatic_mod:
        if somatic_mod.get("block_proactive"):
            mod["block_proactive"] = True
        s_dt = somatic_mod.get("temperature_delta")
        if s_dt:
            mod["temperature_override"] = _override_temp(
                mod["temperature_override"], delta=s_dt
            )
        s_len = somatic_mod.get("length_multiplier", 1.0)
        if s_len < mod["max_tokens_multiplier"]:
            mod["max_tokens_multiplier"] = s_len
        if somatic_mod.get("reason") and somatic_mod["reason"] != "neutral":
            mod["somatic_reason"] = somatic_mod["reason"]

    # Emetti segnale se almeno una modulazione è attiva
    if _modulazione_non_neutra(mod):
        _segnale_behavioral(mem, "omeo_constraint_applicato")

    return mod


def _override_temp(current: Optional[float], delta: float) -> float:
    """Applica un delta relativo a una eventuale temperature override."""
    base = 0.70 if current is None else current
    return round(max(0.10, min(1.20, base + delta)), 2)


def _modulazione_non_neutra(mod: dict) -> bool:
    return (
        mod["max_tokens_multiplier"] != 1.0
        or mod["temperature_override"] is not None
        or mod["force_inner_stream"]
        or mod["force_reflection_pass"]
        or mod["block_factual_claims"]
        or mod["block_proactive"]
    )


# ============================================================
# Micro-pass LLM (SPEC §3.1 e §3.3)
# ============================================================

def conta_contraddizioni(risposta: str, wm: list, lts: str) -> int:
    """
    Micro-pass LLM a temp 0.20 che conta contraddizioni fattuali/identitarie
    tra la risposta e la storia (working_memory + long_term_summary).
    Ritorna intero 0..3. In caso di errore → 0 (fail silente).
    """
    if not risposta or not risposta.strip():
        return 0

    storia = _format_storia(wm, lts)
    prompt = (
        "Sei un verificatore di coerenza. Leggi la storia e l'ultima risposta. "
        "Conta quante contraddizioni fattuali o identitarie contiene l'ultima "
        "risposta rispetto alla storia. Rispondi SOLO con un numero intero 0-3, "
        "senza testo aggiuntivo.\n\n"
        f"STORIA:\n{storia}\n\n"
        f"ULTIMA RISPOSTA:\n{risposta}\n\n"
        "Numero:"
    )
    try:
        from agent import chiama_ollama_con_fallback
        out = chiama_ollama_con_fallback(
            [{"role": "user", "content": prompt}],
            temperature=0.20,
        )
        return _parse_int_safe(out, lo=0, hi=3)
    except Exception:
        return 0


def reflection_pass(risposta: str, wm: list, lts: str) -> tuple[bool, str]:
    """
    Micro-pass LLM a temp 0.20: Eden si auto-valuta.
    Ritorna (ok: bool, motivo: str).
    ok=True → risposta coerente. ok=False → da riscrivere, motivo è la nota.
    In caso di errore → (True, "") = pass-through.
    """
    if not risposta or not risposta.strip():
        return True, ""

    storia = _format_storia(wm, lts)
    prompt = (
        "Hai appena generato questa risposta. Rileggi la tua storia recente e "
        "valuta se la risposta è coerente con chi sei stata.\n\n"
        f"STORIA:\n{storia}\n\n"
        f"RISPOSTA:\n{risposta}\n\n"
        "Rispondi in due righe:\n"
        "Riga 1: SI oppure NO\n"
        "Riga 2: se NO, una riga con cosa correggere; se SI, scrivi '-'"
    )
    try:
        from agent import chiama_ollama_con_fallback
        out = chiama_ollama_con_fallback(
            [{"role": "user", "content": prompt}],
            temperature=0.20,
        )
        return _parse_reflection(out)
    except Exception:
        return True, ""


def _format_storia(wm: list, lts: str) -> str:
    """Serializza working_memory (ultimi 5) + long_term_summary per i micro-pass."""
    righe = []
    if lts:
        righe.append(f"[Sintesi a lungo termine]\n{str(lts).strip()[:800]}")
    if wm:
        righe.append("[Ultimi scambi]")
        for msg in list(wm)[-5:]:
            role = str(msg.get("role", "?"))
            content = str(msg.get("content", "")).strip()[:300]
            righe.append(f"{role}: {content}")
    return "\n".join(righe) if righe else "(nessuna storia)"


def _parse_int_safe(testo: str, lo: int, hi: int) -> int:
    """Estrae il primo intero da testo, strozzato in [lo, hi]. 0 se niente."""
    import re
    if not testo:
        return 0
    m = re.search(r"-?\d+", testo)
    if not m:
        return 0
    try:
        v = int(m.group(0))
        return max(lo, min(hi, v))
    except ValueError:
        return 0


def _parse_reflection(testo: str) -> tuple[bool, str]:
    """Interpreta l'output del reflection_pass."""
    if not testo:
        return True, ""
    righe = [r.strip() for r in testo.strip().splitlines() if r.strip()]
    if not righe:
        return True, ""
    prima = righe[0].upper()
    ok = prima.startswith("SI") or prima.startswith("SÌ") or prima.startswith("YES")
    motivo = righe[1] if len(righe) > 1 else ""
    if motivo == "-":
        motivo = ""
    return ok, motivo


# ============================================================
# Safe mode
# ============================================================

def check_safe_mode(mem: dict) -> bool:
    """
    Attiva safe_mode se tutte e 3 le metriche sono scese sotto SAFE_MODE_SOGLIA.
    Durata: SAFE_MODE_DURATA_MIN minuti.
    Ritorna True se safe_mode è attivo (nuovo o già in corso).
    """
    if not _attivo(mem):
        return False
    stato = _ensure_state(mem)

    if _in_safe_mode(stato):
        return True

    C = stato["coherence_budget"]
    G = stato["grounding_integrity"]
    S = stato["self_model_stability"]

    # v1.3 (2026-04-28): AND -> OR parziale.
    # Pre-v1.3: safe mode richiedeva tutte e 3 le metriche < soglia.
    # Con self_model_stability fisso a 1.0 (Layer 2 bootstrap) era irraggiungibile.
    # Post-Layer 2 attivo: 2/3 metriche sotto soglia è sufficiente — più sensibile
    # ma ancora richiede degrado multi-dimensionale (no false positive su singolo dim).
    sotto_soglia = sum(1 for v in (C, G, S) if v < SAFE_MODE_SOGLIA)
    if sotto_soglia >= 2:
        until = datetime.now() + timedelta(minutes=SAFE_MODE_DURATA_MIN)
        stato["safe_mode_until"] = until.isoformat(timespec="seconds")
        _aggiungi_evento(mem, "safe_mode_attivato", {
            "coherence": C, "grounding": G, "self_model": S,
            "metriche_sotto_soglia": sotto_soglia,
        })
        _segnale_behavioral(mem, "omeo_safe_mode_attivo")
        return True
    return False


# ============================================================
# Debug / introspection
# ============================================================

def stato_sintesi(mem: dict) -> dict:
    """
    Produce un dict leggibile dello stato Layer 1 per endpoint debug
    o telemetria. Non modifica nulla.
    """
    if not mem.get("homeostatic_state"):
        return {"layer1_enabled": False, "inizializzato": False}

    stato = mem["homeostatic_state"]
    return {
        "layer1_enabled":        stato.get("layer1_enabled", True),
        "modulo_attivo":         _ATTIVO,
        "versione":              _LAYER1_VERSION,
        "coherence_budget":      stato.get("coherence_budget"),
        "grounding_integrity":   stato.get("grounding_integrity"),
        "self_model_stability":  stato.get("self_model_stability"),
        "last_tick_at":          stato.get("last_tick_at"),
        "safe_mode":             _in_safe_mode(stato),
        "safe_mode_until":       stato.get("safe_mode_until"),
        "stats":                 dict(stato.get("stats", {})),
        "ultimi_eventi":         list(stato.get("eventi", []))[-10:],
    }


def reset(mem: dict) -> None:
    """Riporta tutte le metriche a baseline. Non cancella gli stats lifetime."""
    stato = _ensure_state(mem)
    stato["coherence_budget"]     = BASELINE
    stato["grounding_integrity"]  = BASELINE
    stato["self_model_stability"] = BASELINE
    stato["last_tick_at"]         = _now_iso()
    stato["safe_mode_until"]      = None
    _aggiungi_evento(mem, "reset_manuale", {})


def toggle(mem: dict, enabled: Optional[bool] = None) -> bool:
    """Abilita/disabilita Layer 1 per-memoria. Ritorna lo stato nuovo."""
    stato = _ensure_state(mem)
    if enabled is None:
        stato["layer1_enabled"] = not stato.get("layer1_enabled", True)
    else:
        stato["layer1_enabled"] = bool(enabled)
    _aggiungi_evento(mem, "toggle", {"enabled": stato["layer1_enabled"]})
    return stato["layer1_enabled"]
