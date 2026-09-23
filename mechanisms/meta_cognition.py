"""meta_cognition.py — MetaCognitive Execution Trace (MET) v1.0.

Analogia neurologica (EV-009): architettura bicamerale de facto.
  - Layer sistema (subcoscio): Kuzu facts, intent router, homeo, grounding check
  - Layer LLM (narratore): genera testo, non ha accesso diretto al layer sistema
  - MET: segnale propriocettivo minimo che collega i due layer

Funzione: dopo ogni scambio, costruisce un trace compatto di ciò che
il layer sistema ha effettivamente fatto. Il trace è iniettato nel
system prompt dello scambio successivo via META-MEM.

Riduce la confabulazione causale: Eden può fare riferimento a eventi
reali ("la mia risposta è stata riscritta") invece di inventare
narrative psicologiche ("alcune domande mi mettono a disagio").

Non è coscienza. Non elimina tutta la confabulazione.
È accesso selettivo a variabili di sistema altrimenti opache.

Versioning anti-HARKing: _MET_VERSION esposto in ogni trace record.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

_MET_VERSION = "v1.0"

_INTENT_LABELS = {
    "memory":     "ricerca in memoria",
    "fact":       "query fattuale",
    "trap":       "domanda diagnostica/test",
    "correction": "correzione di un fatto",
    "meta":       "meta-domanda su di me",
    "emotion":    "espressione emotiva",
    "greeting":   "saluto",
    "general":    "conversazione generale",
}

# Intent che non vale la pena iniettare (troppo frequenti, poco segnale)
_SILENT_INTENTS = {"general", "greeting"}


def build_exec_trace(
    intent_info: dict,
    correction_result: dict,
    grounding_result: dict,
    homeo_mod: Optional[dict],
    rewrite_triggered: bool,
    gi_value: Optional[float] = None,
) -> dict:
    """Costruisce il trace di esecuzione per lo scambio appena completato.

    Da chiamare DOPO che la risposta è stata finalizzata (post tutti i filtri).
    Il risultato viene salvato in mem['last_exec_trace'] e letto dallo scambio
    successivo tramite build_system_prompt.
    """
    rewrite_type = None
    if rewrite_triggered and not grounding_result.get("grounded", True):
        rewrite_type = grounding_result.get("tipo_problema")

    return {
        "version":             _MET_VERSION,
        "ts":                  datetime.now().isoformat(),
        "intent":              intent_info.get("intent", "general"),
        "intent_confidence":   round(intent_info.get("confidence", 0.0), 2),
        "correction_applied":  bool(correction_result.get("applied", False)),
        "correction_key":      correction_result.get("key"),
        "correction_value":    correction_result.get("value"),
        "rewrite_triggered":   rewrite_triggered,
        "rewrite_type":        rewrite_type,
        "grounded":            bool(grounding_result.get("grounded", True)),
        "gi_value":            gi_value,
        "block_proactive":     bool((homeo_mod or {}).get("block_proactive", False)),
        "force_reflection":    bool((homeo_mod or {}).get("force_reflection_pass", False)),
    }


def format_trace_for_prompt(trace: dict) -> str:
    """Formatta il trace come testo breve per l'iniezione nel system prompt.

    Restituisce stringa vuota se il trace non contiene informazioni utili.
    Principio: inietta solo eventi non-triviali — quelli che potrebbero altrimenti
    causare confabulazione causale se Eden li osservasse senza contesto.
    """
    if not trace:
        return ""

    lines = []
    intent = trace.get("intent", "general")

    # Solo intent non-triviali meritano menzione
    if intent not in _SILENT_INTENTS:
        label = _INTENT_LABELS.get(intent, intent)
        lines.append(f"Tipo domanda precedente: {label}")

    # Correzione applicata — Eden deve saperlo per non ri-confabulare
    if trace.get("correction_applied"):
        k = trace.get("correction_key", "?")
        v = trace.get("correction_value", "?")
        lines.append(f"Fatto aggiornato: {k} = \"{v}\" (fonte: correzione utente)")

    # Rewrite — Eden ha detto qualcosa di non verificabile, il sistema lo ha corretto
    if trace.get("rewrite_triggered"):
        rtype = trace.get("rewrite_type") or ""
        if rtype == "citazione_inventata":
            lines.append(
                "Risposta precedente riscritta: conteneva una citazione non verificabile"
            )
        elif rtype in ("episodio_non_grounded", "relazione_inventata"):
            lines.append(
                "Risposta precedente riscritta: affermazione su fatto non verificabile"
            )
        else:
            lines.append("Risposta precedente riscritta per grounding insufficiente")

    if not lines:
        return ""

    return (
        "\n\n[TRACCIA — ultimo scambio]\n"
        + "\n".join(f"  • {l}" for l in lines)
    )