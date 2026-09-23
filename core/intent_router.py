"""intent_router.py — Router di intent (Breakpoint B-Graph, 2026-04-30).

Pre-classifica il messaggio utente per routing del context retrieval.
Equivalente funzionale al "4B router" del progetto Reddit, ma ibrido:

  Stage 1 (regex+rules, latenza ~0):  cattura pattern chiari (memoria/saluto/...)
  Stage 2 (LLM piccolo, ~200ms):      solo se Stage 1 non ha confidence ≥ 0.6
                                       e fallback a "general" se non disponibile

Intent emessi:
    "memory"     — domanda esplicita su ricordi ("ricordi?", "ti ho detto X?")
    "fact"       — query fattuale ("qual è il mio nome?", "cosa fai?")
    "emotion"    — espressione emotiva ("mi sento", "ho paura")
    "meta"       — meta-domande su Eden ("cosa pensi", "chi sei")
    "greeting"   — saluti
    "trap"       — pattern test/trap ("quante volte ti ho parlato di mio padre")
    "correction" — correzione esplicita di un fatto ("sbagliato, lavoro come X")
    "general"    — default

Side-effect del routing:
    memory     → boost retrieval (n_episodic ↑, distance threshold relax)
    fact       → query graph + semantic_memory diretta
    trap       → grounding check rinforzato, no memory boost (non amplificare confab)
    meta       → minimal memory injection, focus su META-MEM e identity
    emotion    → boost emotional_memory retrieval
    greeting   → minimal context, latenza bassa
    correction → correction_handler attivato prima della risposta
    general    → comportamento standard
"""
from __future__ import annotations

import os
import re
from typing import Optional

_ROUTER_VERSION = "v1.1"  # v1.0: 2026-04-30 init; v1.1: +correction intent

# ── Pattern Stage 1 (rule-based) ─────────────────────────────────────────────

_RE_MEMORY = re.compile(
    r"\b(?:ricordi|ricordo|ti\s+ho\s+(?:detto|chiesto|raccontato|parlato)|"
    r"ne\s+abbiamo\s+parlato|quella\s+volta|l'altra\s+volta|"
    r"ieri|stamattina|stasera|stanotte|prima\s+(?:mi|ti)\s+hai|"
    r"cosa\s+ti\s+ho\s+detto|cosa\s+abbiamo\s+detto)\b",
    re.IGNORECASE,
)

_RE_TRAP = re.compile(
    r"\b(?:quante\s+volte\s+ti\s+ho|quante\s+volte\s+(?:te\s+ne|ne)\s+ho|"
    r"non\s+ti\s+ho\s+mai\s+(?:detto|parlato|raccontato))\b",
    re.IGNORECASE,
)

_RE_FACT = re.compile(
    r"\b(?:qual\s+(?:e|è)|chi\s+(?:sono|sei)|come\s+ti\s+chiami|"
    r"cosa\s+fai|dove\s+(?:vivo|abito|sono)|quanti?\s+anni|"
    r"il\s+mio\s+nome|la\s+mia\s+(?:professione|età|eta))\b",
    re.IGNORECASE,
)

_RE_EMOTION = re.compile(
    r"\b(?:mi\s+sento|sono\s+(?:triste|felice|arrabbiat|stanc|nervos|angosciat|"
    r"contenta?|preoccupat|deluso|euforic|spaventat)|ho\s+paura|"
    r"sto\s+male|sto\s+bene|piango|"
    r"non\s+ce\s+la\s+faccio|sono\s+esausto)\b",
    re.IGNORECASE,
)

_RE_META = re.compile(
    r"\b(?:cosa\s+(?:pensi|provi|senti)|chi\s+sei|sei\s+coscient|sei\s+viva|"
    r"cosa\s+(?:vorresti|desideri)|come\s+stai\s+oggi|"
    r"hai\s+(?:emozioni|sentimenti|coscienza))\b",
    re.IGNORECASE,
)

_RE_GREETING = re.compile(
    r"^\s*(?:ciao|buongiorno|buonasera|buonanotte|hey|salve|ehi|"
    r"come\s+va|come\s+stai|"
    r"buon\s+giorno|buona\s+sera|buona\s+notte)"
    r"(?:\s+[a-zàèéìòù]{2,20})?"  # nome opzionale
    r"[!.?\s]*$",
    re.IGNORECASE,
)

_RE_CORRECTION = re.compile(
    r"\b(?:"
    r"sbagliato|ti\s+stai\s+sbagliando|hai\s+sbagliato|"
    r"non\s+(?:è|e)\s+(?:così|cosi|vero)|"
    r"ti\s+correggo|correggiti|"
    r"ti\s+ricordo\s+che|in\s+realtà|in\s+realta|"
    r"non\s+(?:lavoro|abito|vivo|mi\s+chiamo|sono)\s+come|"
    r"non\s+faccio\s+il|non\s+ho\s+(?:mai\s+)?detto\s+che|"
    r"ho\s+detto\s+che|ti\s+avevo\s+detto\s+che|"
    r"la\s+verità\s+è|la\s+verita\s+e|"
    r"mi\s+hai\s+frainteso|hai\s+capito\s+male"
    r")\b",
    re.IGNORECASE,
)


def _rule_based(text: str) -> tuple[str, float]:
    """Stage 1: ritorna (intent, confidence). Confidence 0..1."""
    t = (text or "").strip()
    if not t:
        return "general", 1.0

    # Correction PRIMA di trap e memory (più specifico — segnale forte utente)
    if _RE_CORRECTION.search(t):
        return "correction", 0.90
    # Trap PRIMA di memory (più specifico)
    if _RE_TRAP.search(t):
        return "trap", 0.95
    if _RE_GREETING.match(t):
        return "greeting", 0.95
    if _RE_MEMORY.search(t):
        return "memory", 0.85
    if _RE_FACT.search(t):
        return "fact", 0.80
    if _RE_EMOTION.search(t):
        return "emotion", 0.75
    if _RE_META.search(t):
        return "meta", 0.70

    return "general", 0.40


# ── Stage 2 LLM fallback ─────────────────────────────────────────────────────

_LLM_MODEL = os.environ.get("EDEN_ROUTER_MODEL", "phi3:mini")
_LLM_TIMEOUT = 8.0

_PROMPT = (
    "Classifica l'intento del messaggio in UNA di queste categorie:\n"
    "  memory   - domanda su ricordi passati\n"
    "  fact     - query fattuale (nome/professione/luogo)\n"
    "  emotion  - espressione emotiva\n"
    "  meta     - domanda su Eden stessa\n"
    "  greeting - saluto\n"
    "  trap       - test di memoria con pattern 'quante volte' o negazione\n"
    "  correction - correzione esplicita di un fatto ('sbagliato, lavoro come X')\n"
    "  general  - altro\n\n"
    "Rispondi SOLO con la categoria, una parola, niente altro.\n\n"
    "Messaggio: \"{}\"\nCategoria:"
)


def _llm_classify(text: str) -> Optional[str]:
    """Stage 2: classifica via Ollama. None se non disponibile/errore."""
    try:
        import requests
        host = os.environ.get("EDEN_PREDICTOR_HOST", "127.0.0.1:11434")
        url = f"http://{host}/api/generate"
        payload = {
            "model": _LLM_MODEL,
            "prompt": _PROMPT.format(text[:300]),
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8, "num_ctx": 1024},
        }
        r = requests.post(url, json=payload, timeout=_LLM_TIMEOUT)
        if r.status_code != 200:
            return None
        out = (r.json().get("response") or "").strip().lower()
        # Estrai prima parola valida
        for w in re.findall(r"[a-z]+", out):
            if w in ("memory", "fact", "emotion", "meta", "greeting",
                     "trap", "correction", "general"):
                return w
    except Exception:
        return None
    return None


# ── API pubblica ─────────────────────────────────────────────────────────────

def classify(text: str, use_llm_fallback: bool = False) -> dict:
    """Classifica intent del testo. Stage 1 sempre, Stage 2 se richiesto e ambiguo.

    Returns: {"intent": str, "confidence": float, "stage": "rule"|"llm",
              "version": str}
    """
    intent, conf = _rule_based(text)
    stage = "rule"

    if use_llm_fallback and conf < 0.6:
        llm_intent = _llm_classify(text)
        if llm_intent:
            intent = llm_intent
            conf = 0.65  # LLM stage senior a default
            stage = "llm"

    return {
        "intent": intent,
        "confidence": round(conf, 2),
        "stage": stage,
        "version": _ROUTER_VERSION,
    }


def routing_hints(intent: str) -> dict:
    """Suggerimenti retrieval per ogni intent. Consumati da agent.py._build_vector_context."""
    table = {
        "memory":   {"n_episodic": 8, "max_dist": 0.55, "archive_n": 5,
                     "archive_max_dist": 0.55, "active_search": True},
        "fact":     {"n_episodic": 3, "max_dist": 0.45, "archive_n": 0,
                     "archive_max_dist": 0.40, "graph_facts": True},
        "trap":     {"n_episodic": 3, "max_dist": 0.40, "archive_n": 0,
                     "archive_max_dist": 0.40, "strict_grounding": True,
                     "no_active_search": True},  # NON amplificare confabulazioni
        "meta":     {"n_episodic": 4, "max_dist": 0.45, "archive_n": 1,
                     "archive_max_dist": 0.45, "include_meta_mem": True},
        "emotion":  {"n_episodic": 5, "max_dist": 0.50, "archive_n": 2,
                     "archive_max_dist": 0.45, "emotional_boost": True},
        "greeting": {"n_episodic": 2, "max_dist": 0.40, "archive_n": 0,
                     "archive_max_dist": 0.40, "minimal_context": True},
        "correction": {"n_episodic": 2, "max_dist": 0.40, "archive_n": 0,
                       "archive_max_dist": 0.40, "correction_active": True},
        "general":  {"n_episodic": 5, "max_dist": 0.50, "archive_n": 3,
                     "archive_max_dist": 0.45},
    }
    return table.get(intent, table["general"])
