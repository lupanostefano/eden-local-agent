"""correction_handler.py — Reconsolidamento esplicito (2026-04-30).

Analogia neurologica: finestra di labilità post-riattivazione (Nader 2000).
Quando l'utente corregge un fatto, il sistema:
  1. Rileva la correzione — 3 livelli in cascata:
       L1 Esplicito  (conf 0.90): "sbagliato / ti correggo / in realtà / ..."
       L2 Implicito  (conf 0.75): "no, / già te l'ho detto / non ricordi che / ..."
       L3 Context    (conf 0.70): estrai Fact → confronta Kuzu → se diverso = correzione
                                   deterministico, zero parole chiave richieste
  2. Estrae (chiave, valore) MULTIPLI dalla correzione — v1.2
  3. Aggiorna tutti i Fact trovati nel Kuzu con confidence=0.99 (override manuale)
  4. Inietta coppia Q/A di correzione nella working_memory
     → rompe il ciclo vizioso (history contaminata replicava errore)
  5. Logga su correction_log.jsonl (anti-HARKing)

Chirurgicità:
  - Fail-silent su ogni step: se LLM fallisce, niente viene aggiornato
  - Solo Fact chiave:valore atomici
  - Non modifica episodi storici in ChromaDB
  - L3 attivo solo se Kuzu disponibile e Fact già esistente (no false positive su info nuove)
  - Threshold confidence_correction ≥ 0.70 per evitare falsi positivi

_CORRECTION_VERSION  v1.0: 2026-04-30 init
                     v1.1: +L2 implicito +L3 context-aware
                     v1.2: multi-fact extraction + 9 nuovi pattern
                     v1.3: +M2 Behavioral Constraint Register (EV-015)
                     v1.4: M2 regex esteso + pre-scan simboli noti (<3)
                     v1.5: M2 architettura 3 tier, dati in meta_instruction_library.py
                     v1.6: fix _RE_NOME/_RE_CITTA — inline flag (?i:...) solo prefisso
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

_CORRECTION_VERSION = "v1.6"

# ── EV-015 fix: Behavioral Constraint Register (M2) ──────────────────────────
# v1.5: architettura 3 tier — dati delegati a meta_instruction_library.py.
# Nessun pattern hardcoded qui. Per aggiungere pattern/trigger/simboli,
# modificare core/meta_instruction_library.py.

from core.meta_instruction_library import (
    TRIGGER_KEYWORDS     as _TRIGGER_KEYWORDS,
    TRIGGER_VERBS_WEAK   as _TRIGGER_VERBS_WEAK,
    KNOWN_BAN_SYMBOLS    as _KNOWN_BAN_SYMBOLS,
    TARGET_SKIP_WORDS    as _META_BAN_SKIP,
    PATTERN_GROUPS_T1    as _PATTERN_GROUPS,
    META_BAN_EXPIRY      as _META_BAN_EXPIRY,
    LLM_TIER2            as _LLM_TIER2_CONFIG,
    _META_LIBRARY_VERSION,
)


def _compile_meta_ban_regex(groups: list) -> re.Pattern:
    """Compila PATTERN_GROUPS_T1 in un unico regex al momento dell'import."""
    all_patterns = []
    for group in groups:
        all_patterns.extend(group["patterns"])
    combined = "|".join(f"(?:{p})" for p in all_patterns)
    return re.compile(rf"(?:{combined})(.{{1,80}})(?:[,\.!?]|$)", re.IGNORECASE)


_RE_META_BAN: re.Pattern = _compile_meta_ban_regex(_PATTERN_GROUPS)


def _build_tier2_prompt(user_msg: str) -> str:
    """Costruisce il prompt few-shot per Tier 2 LLM."""
    examples_lines = []
    for msg_ex, is_ban, target in _LLM_TIER2_CONFIG["prompt_examples"]:
        tgt = f'"{target}"' if target else "null"
        examples_lines.append(
            f'  Msg: "{msg_ex}" -> {{"is_ban": {str(is_ban).lower()}, "target": {tgt}}}'
        )
    examples_str = "\n".join(examples_lines) + "\n"
    return _LLM_TIER2_CONFIG["prompt_template"].format(
        msg=user_msg[:200],
        examples=examples_str,
    )


def _extract_ban_target(raw_target: str) -> str:
    """Estrae il target da vietare dal gruppo regex catturato."""
    # Pre-scan simboli noti: bypass del filtro lunghezza
    for sym in _KNOWN_BAN_SYMBOLS:
        if sym in raw_target:
            return sym
    words = [
        w.strip(".,!? ").lower()
        for w in raw_target.split()
        if w.strip(".,!? ").lower() not in _META_BAN_SKIP
        and len(w.strip(".,!? ")) >= 3
    ]
    return words[0] if words else raw_target.strip().rstrip(".,!? ").lower()


def _tier2_llm_classify(user_msg: str, mem: dict) -> bool:
    """
    Tier 2: micro-pass phi3:mini. Fail-silent su qualsiasi errore.
    Attivato solo se Tier 0 matcha ma Tier 1 non ha trovato pattern.
    """
    if not _LLM_TIER2_CONFIG.get("enabled", False):
        return False
    try:
        import requests as _req
        import os as _os
        port = int(_os.environ.get("EDEN_META_OLLAMA_PORT",
                                   _LLM_TIER2_CONFIG["ollama_port"]))
        prompt = _build_tier2_prompt(user_msg)
        resp = _req.post(
            f"http://127.0.0.1:{port}/api/generate",
            json={
                "model":  _LLM_TIER2_CONFIG["model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": _LLM_TIER2_CONFIG["temperature"],
                    "num_predict": _LLM_TIER2_CONFIG["max_tokens"],
                },
            },
            timeout=_LLM_TIER2_CONFIG["timeout_seconds"],
        )
        if resp.status_code != 200:
            return False
        text = resp.json().get("response", "").strip()
        # Estrai JSON (il modello può premettere testo)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            return False
        data = json.loads(text[start:end])
        if not data.get("is_ban"):
            return False
        target = data.get("target")
        if not target or not isinstance(target, str) or len(target.strip()) < 2:
            return False
        target = target.lower().strip()
        exchange_count = mem.get("exchange_count", 0)
        constraints    = mem.setdefault("behavioral_constraints", [])
        constraints[:] = [c for c in constraints if c.get("expires_at", 0) > exchange_count]
        if any(c.get("target") == target for c in constraints):
            return False
        constraints.append({
            "type":       "style_ban",
            "target":     target,
            "expires_at": exchange_count + _META_BAN_EXPIRY,
            "added_at":   exchange_count,
            "tier":       2,
        })
        return True
    except Exception:
        return False


def detect_meta_instruction(user_msg: str, mem: dict) -> bool:
    """
    Rileva meta-istruzioni comportamentali (EV-015 M2) e le registra in
    mem["behavioral_constraints"]. Restituisce True se almeno un vincolo aggiunto.

    Architettura 3 tier (pattern e config in meta_instruction_library.py):
      Tier 0 — keyword pre-filter O(1): skip immediato se nessun trigger
      Tier 1 — regex compilato da PATTERN_GROUPS_T1: ~80% copertura
      Tier 2 — phi3:mini LLM micro-pass: ~15-18% residuo, fail-silent
    """
    msg_lower = user_msg.lower()

    # ── Tier 0: keyword pre-filter ──────────────────────────────────────────
    # Strong triggers: singola parola sufficiente (smetti, odio, fastidio…)
    # Weak triggers: "non" + verbo d'azione ("non usare", "non fare"…)
    _t0 = (
        any(kw in msg_lower for kw in _TRIGGER_KEYWORDS)
        or ("non" in msg_lower and any(vb in msg_lower for vb in _TRIGGER_VERBS_WEAK))
    )
    if not _t0:
        return False

    # ── Tier 1: regex pattern library ──────────────────────────────────────
    matches = _RE_META_BAN.findall(user_msg)
    if matches:
        exchange_count = mem.get("exchange_count", 0)
        constraints    = mem.setdefault("behavioral_constraints", [])
        constraints[:] = [c for c in constraints if c.get("expires_at", 0) > exchange_count]
        added = False
        for raw_target in matches:
            target = _extract_ban_target(raw_target)
            if not target or len(target) < 2:
                continue
            if any(c.get("target") == target for c in constraints):
                continue
            constraints.append({
                "type":       "style_ban",
                "target":     target,
                "expires_at": exchange_count + _META_BAN_EXPIRY,
                "added_at":   exchange_count,
                "tier":       1,
            })
            added = True
        return added

    # ── Tier 2: LLM micro-pass (solo se Tier 0 matchò ma Tier 1 fallì) ─────
    return _tier2_llm_classify(user_msg, mem)


# ── L1: Pattern espliciti di correzione (conf=0.90) ──────────────────────────

_RE_CORRECTION_L1 = re.compile(
    r"\b(?:"
    r"sbagliato|ti\s+stai\s+sbagliando|hai\s+sbagliato|"
    r"non\s+(?:è|e)\s+(?:così|cosi|vero)|"
    r"ti\s+correggo|correggiti|"
    r"ti\s+ricordo\s+che|in\s+realtà|in\s+realta|"
    r"non\s+(?:lavoro|abito|vivo|mi\s+chiamo|sono)\s+come|"
    r"non\s+faccio\s+il|non\s+ho\s+detto|"
    r"ho\s+detto\s+che|ti\s+avevo\s+detto|"
    r"la\s+verità\s+è|la\s+verita\s+e|"
    r"non\s+è\s+(?:quello|ciò)\s+che|"
    r"mi\s+hai\s+frainteso|hai\s+capito\s+male|"
    r"perciò\s+hai\s+detto\s+(?:una\s+cosa\s+)?errat|"
    r"hai\s+detto\s+(?:una\s+cosa\s+)?errat|"
    r"hai\s+errato|hai\s+sbagliato"
    r")\b",
    re.IGNORECASE,
)

# ── L2: Pattern impliciti-segnale (conf=0.75) ────────────────────────────────

_RE_L2_NO_INIZIO = re.compile(r"^\s*no\s*[,\.!]", re.IGNORECASE)

_RE_L2_KEYWORDS = re.compile(
    r"\b(?:"
    r"quante\s+volte\s+(?:te\s+lo\s+devo|ti\s+devo)\s+(?:dire|ripetere)|"
    r"gi[aà]\s+te\s+l.?ho\s+detto|"
    r"te\s+l.?(?:avevo|ho)\s+(?:gi[aà]\s+)?detto|"
    r"l.?ho\s+gi[aà]\s+detto|"
    r"non\s+ricordi\s+(?:che|quando)|"
    r"hai\s+dimenticato\s+che|"
    r"sempre\s+dimentichi|"
    r"stai\s+confondendo|"
    r"ti\s+stai\s+confondendo|"
    r"non\s+\w+(?:\s+\w+){0,3},\s+(?:ma\s+)?(?:sono|faccio|lavoro|abito|mi\s+chiamo)"
    r")\b",
    re.IGNORECASE,
)


# ── Regex di estrazione — pattern per chiave:valore ──────────────────────────
# Ogni pattern estrae UN valore; _extract_all_via_regex prova tutti in sequenza.

# Professione / lavoro
_RE_PROFESSIONE = re.compile(
    r"(?:lavoro\s+come|faccio\s+(?:il|la|lo)|sono\s+(?:un|una|un')|"
    r"lavoro\s+nel|lavoro\s+in|mi\s+occupo\s+di)\s+"
    r"([a-zàèéìòù][a-zàèéìòùA-Z\s\-]{2,50}?)(?:[,\.!\n]|$)",
    re.IGNORECASE,
)

# Nome
# Fix 2026-05-12: inline flag (?i:...) solo prefisso — IGNORECASE (10/05) rompeva il
# guard uppercase: "sono così" → nome=così (falso positivo scritto in Kuzu).
_RE_NOME = re.compile(
    r"(?i:(?:mi\s+chiamo|sono|il\s+mio\s+nome\s+[eè])\s+)([A-Z][a-zàèéìòù]{1,30})",
)

# Città / residenza
# Fix 2026-05-12: stessa radice di _RE_NOME — inline flag mantiene uppercase guard.
_RE_CITTA = re.compile(
    r"(?i:(?:abito\s+a|vivo\s+a|sono\s+di|risiedo\s+a|la\s+mia\s+(?:città|citta)\s+[eè])\s+)"
    r"([A-Z][a-zàèéìòùA-Z\s]{1,40}?)(?:[,\.!\n]|$)",
)

# Età  — "ho 33 anni" / "faccio 33 anni" / "compio 33 anni" / "ho compiuto 33 anni"
_RE_ETA = re.compile(
    r"(?:ho|faccio|compio|avrò|compirò|ho\s+compiuto)\s+(\d{1,3})\s+ann[io]",
    re.IGNORECASE,
)

# Anno di nascita — "sono nato nel 1993" / "nato nel 1993" / "nascita 1993"
_RE_ANNO_NASCITA = re.compile(
    r"(?:sono\s+nato\s+(?:nel\s+)?|nato\s+(?:nel\s+)?|anno\s+di\s+nascita\s*[:\s]+)(\d{4})",
    re.IGNORECASE,
)

# Data di nascita completa — "sono nato il 15 marzo 1993" / "nato il 3/5/1993"
_RE_DATA_NASCITA = re.compile(
    r"(?:sono\s+nato\s+il\s+|nato\s+il\s+)"
    r"(\d{1,2}[\s/\-]+\w+[\s/\-]+\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)

# Compleanno — "compio gli anni il 15 ottobre" / "il mio compleanno è il 15 ottobre"
# Anche: "compio 33 anni il 15 ottobre" → estrae la data
_RE_COMPLEANNO = re.compile(
    r"(?:"
    r"il\s+mio\s+compleanno\s+[eè]\s+(?:il\s+)?|"
    r"compio\s+(?:gli?\s+)?anni\s+il\s+|"
    r"il\s+(?:mio\s+)?compleanno\s+cade\s+il\s+|"
    r"sono\s+nato\s+il\s+\d{1,2}\s+\w+\s*(?:\d{4})?\s*(?:,\s*)?"
    r")"
    r"(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
    r"settembre|ottobre|novembre|dicembre)(?:\s+\d{4})?)",
    re.IGNORECASE,
)
# Catch-all compleanno con mese: "il 15 ottobre" in contesto compleanno
_RE_COMPLEANNO_DATE = re.compile(
    r"(?:compio\s+\d+\s+anni\s+(?:il\s+)?|"
    r"festeggio\s+(?:il\s+)?(?:mio\s+)?compleanno\s+(?:il\s+)?)"
    r"(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
    r"settembre|ottobre|novembre|dicembre)(?:\s+\d{4})?)",
    re.IGNORECASE,
)

# Interessi / passioni / hobby
_RE_INTERESSI = re.compile(
    r"(?:i\s+miei\s+(?:principali\s+)?interess[io]|le\s+mie\s+passioni|"
    r"i\s+miei\s+hobby|mi\s+appassiona|sono\s+appassionato\s+di)\s+"
    r"(?:sono\s+)?(.{5,150}?)(?:[\.!\n]|$)",
    re.IGNORECASE,
)

# Parente — "mio padre si chiama X" / "mia madre si chiama X"
_RE_PARENTE = re.compile(
    r"(?:mio\s+padre|mia\s+madre|mio\s+fratello|mia\s+sorella|"
    r"mio\s+figlio|mia\s+figlia|mio\s+nonno|mia\s+nonna)\s+"
    r"(?:si\s+chiama|è|era|ha\s+nome)\s+"
    r"([A-Z][a-zàèéìòù]{1,30})",
    re.IGNORECASE,
)

# Nazione / paese
_RE_NAZIONE = re.compile(
    r"(?:sono\s+(?:italiano|francese|spagnolo|tedesco|inglese|americano|"
    r"svizzero|portoghese|[a-zàèéìòù]{4,20})|"
    r"vengo\s+dal[la]?\s+(?:Italia|Francia|Spagna|Germania|[A-Z][a-zàèéìòù]{3,30}))",
    re.IGNORECASE,
)

# ── Mappa parente → chiave Kuzu ──────────────────────────────────────────────
_PARENTE_TO_KEY = {
    "padre": "padre", "madre": "madre",
    "fratello": "fratello", "sorella": "sorella",
    "figlio": "figlio", "figlia": "figlia",
    "nonno": "nonno", "nonna": "nonna",
}


# ── Estrazione LLM (fallback) ─────────────────────────────────────────────────

_LLM_EXTRACT_PROMPT = """\
Il messaggio seguente contiene una o più correzioni di fatti da parte dell'utente.
Estrai TUTTI i fatti corretti come array JSON. Ogni elemento ha "key" e "value".
Keys valide: nome, professione, età, compleanno, anno_nascita, data_nascita,
             città, residenza, interessi, padre, madre, fratello, sorella,
             nazione, (qualsiasi altra chiave semplice in italiano).
Se non riesci a estrarre nessun fatto chiaro, rispondi: []

Messaggio: "{msg}"

Rispondi SOLO con l'array JSON, nient'altro. Esempi validi:
[{{"key":"età","value":"33"}},{{"key":"anno_nascita","value":"1993"}}]
[]"""


# Blacklist per professione: prima parola del valore estratto.
# Se inizia con queste, è un'espressione idiomatica, non una professione.
_NON_PROFESSIONI: set = {
    "portento", "appassionato", "fan", "tipo", "disastro",
    "essere", "patito", "principiante", "casino", "pezzo",
    "caso", "po'", "po", "macello", "sognatore", "po'.",
}

# Nomi dell'agente — non possono essere "nome utente"
_NOMI_AGENTE_ESCLUSI: set = {"eden"}


def _extract_all_via_regex(text: str) -> list[tuple[str, str]]:
    """Estrae TUTTI i fatti trovabili tramite regex.
    Restituisce lista di (chiave, valore), deduplicata per chiave.
    """
    results: list[tuple[str, str]] = []
    seen_keys: set[str] = set()

    def _add(key: str, value: str) -> None:
        v = (value or "").strip().rstrip(".,! ")
        if v and key not in seen_keys:
            seen_keys.add(key)
            results.append((key, v))

    # Professione (scarta match preceduti da "non" e parole non-lavoro idiomatiche)
    # Fix 2026-05-10: il pattern "sono un X" matchava espressioni idiomatiche
    # (es. "sono un portento nel cucinarla") che NON sono professioni.
    for m in _RE_PROFESSIONE.finditer(text):
        prefix = text[max(0, m.start() - 6): m.start()].lower()
        if "non" in prefix:
            continue
        valore = (m.group(1) or "").strip().lower()
        prima_parola = valore.split()[0] if valore else ""
        if prima_parola in _NON_PROFESSIONI:
            continue
        _add("professione", m.group(1))

    # Nome (scarta nome agente — evita falso positivo su "sono Eden" citato)
    m = _RE_NOME.search(text)
    if m and m.group(1).lower() not in _NOMI_AGENTE_ESCLUSI:
        _add("nome", m.group(1))

    # Città
    m = _RE_CITTA.search(text)
    if m:
        _add("città", m.group(1))

    # Età
    m = _RE_ETA.search(text)
    if m:
        _add("età", m.group(1))

    # Anno nascita
    m = _RE_ANNO_NASCITA.search(text)
    if m:
        _add("anno_nascita", m.group(1))

    # Data nascita completa (ha priorità su anno_nascita se entrambi matchano)
    m = _RE_DATA_NASCITA.search(text)
    if m:
        _add("data_nascita", m.group(1))

    # Compleanno
    for pattern in (_RE_COMPLEANNO, _RE_COMPLEANNO_DATE):
        m = pattern.search(text)
        if m:
            _add("compleanno", m.group(1))
            break

    # Interessi
    m = _RE_INTERESSI.search(text)
    if m:
        _add("interessi", m.group(1))

    # Parenti — cerca la relazione specifica
    m = _RE_PARENTE.search(text)
    if m:
        testo_low = text.lower()
        for parola, chiave in _PARENTE_TO_KEY.items():
            if parola in testo_low:
                _add(chiave, m.group(1))
                break

    return results


def _extract_all_via_llm(text: str, ollama_fn) -> list[tuple[str, str]]:
    """Estrazione LLM leggera per casi non catturati dal regex."""
    if ollama_fn is None:
        return []
    try:
        prompt = _LLM_EXTRACT_PROMPT.format(msg=text[:400])
        risposta = ollama_fn(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            num_predict=120,
        )
        if not risposta:
            return []
        # Cerca array JSON nella risposta
        m = re.search(r'\[.*?\]', risposta, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        if not isinstance(data, list):
            return []
        results = []
        seen = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            k = str(item.get("key") or "").strip().lower()[:60]
            v = str(item.get("value") or "").strip()[:200]
            if k and v and k != "null" and v != "null" and k not in seen:
                seen.add(k)
                results.append((k, v))
        return results
    except Exception:
        return []


# ── Pulizia working_memory ────────────────────────────────────────────────────

def _inietta_correzioni_in_wm(mem: dict, facts: list[tuple[str, str]]) -> None:
    """Inietta le coppie di correzione nella working_memory.
    Fix ghost Q/A (EV-009): usa marker [SISTEMA] invece di duplicare il testo utente.
    """
    if not facts:
        return
    wm = mem.setdefault("working_memory", [])
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    # Una voce compatta per tutte le correzioni dello stesso scambio
    facts_str = " · ".join(f'{k} = "{v}"' for k, v in facts)
    wm.append({
        "role": "user",
        "content": f"[SISTEMA — FATTI CORRETTI]: {facts_str} (fonte: utente)",
        "ts": ts,
        "correction": True,
    })
    if len(facts) == 1:
        k, v = facts[0]
        conferma = f"Registrato. {k.capitalize()} aggiornato: {v}."
    else:
        conferma = "Registrato. " + ", ".join(f"{k}={v}" for k, v in facts) + " aggiornati."
    wm.append({
        "role": "assistant",
        "content": conferma,
        "ts": ts,
        "correction": True,
    })


# ── Log ───────────────────────────────────────────────────────────────────────

def _log_correction(
    user_msg: str,
    facts: list[tuple[str, str]],
    method: str,
    kuzu_updated: bool,
    wm_injected: bool,
) -> None:
    try:
        from eden_paths import DATA_DIR
        log_path = DATA_DIR / "logs" / "correction_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Backward-compat: mantieni extracted_key/value (primo fatto)
        first_key = facts[0][0] if facts else None
        first_val = facts[0][1] if facts else None
        record = {
            "ts": datetime.now().isoformat(),
            "version": _CORRECTION_VERSION,
            "user_msg_preview": user_msg[:200],
            "extracted_key": first_key,
            "extracted_value": first_val,
            "extracted_facts": [{"key": k, "value": v} for k, v in facts],
            "method": method,
            "kuzu_updated": kuzu_updated,
            "wm_injected": wm_injected,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── L3: context-aware check ───────────────────────────────────────────────────

def _context_aware_check(text: str) -> bool:
    """L3: estrai Fact dal testo → confronta con Kuzu → se diverso = correzione.
    Zero falsi positivi su info nuove (Kuzu non ha ancora quel Fact → False).
    """
    facts = _extract_all_via_regex(text)
    if not facts:
        return False
    try:
        from mechanisms.graph_memory import get_graph
        g = get_graph()
        if not g.disponibile:
            return False
        for chiave, valore in facts:
            existing = g.get_fact(f"utente.{chiave}")
            if existing and existing["value"].strip().lower() != valore.strip().lower():
                return True
    except Exception:
        pass
    return False


# ── API pubblica ──────────────────────────────────────────────────────────────

def is_correction(text: str) -> tuple[bool, float]:
    """Rileva se il messaggio contiene una correzione. 3 livelli in cascata."""
    t = (text or "").strip()
    if not t:
        return False, 0.0
    if _RE_CORRECTION_L1.search(t):
        return True, 0.90
    if _RE_L2_NO_INIZIO.search(t) or _RE_L2_KEYWORDS.search(t):
        return True, 0.75
    if _context_aware_check(t):
        return True, 0.70
    return False, 0.0


def detect_and_apply(
    mem: dict,
    user_msg: str,
    ollama_fn=None,
    confidence_threshold: float = 0.70,
) -> dict:
    """Rileva correzione, estrae tutti i Fact, aggiorna Kuzu, inietta in WM.

    Returns:
        {
          "applied": bool,
          "key": str|None,          — primo fatto (backward compat)
          "value": str|None,        — primo fatto (backward compat)
          "facts": list,            — tutti i fatti estratti [(key, value), ...]
          "method": str,            — "regex" | "llm" | "none"
          "confidence": float,
        }
    """
    result: dict = {
        "applied": False,
        "key": None, "value": None,
        "facts": [],
        "method": "none", "confidence": 0.0,
    }

    # M2: rileva meta-istruzioni comportamentali indipendentemente dalla correzione
    detect_meta_instruction(user_msg, mem)

    detected, conf = is_correction(user_msg)
    if not detected or conf < confidence_threshold:
        return result

    result["confidence"] = conf

    # Estrazione: prima regex (veloce), poi LLM fallback sui gap
    facts = _extract_all_via_regex(user_msg)
    method = "regex" if facts else None

    if not facts:
        facts = _extract_all_via_llm(user_msg, ollama_fn)
        method = "llm" if facts else None

    if not facts:
        _log_correction(user_msg, [], "none", False, False)
        return result

    result["facts"] = facts
    result["key"] = facts[0][0]
    result["value"] = facts[0][1]
    result["method"] = method

    # Aggiornamento Kuzu — tutti i fatti trovati
    kuzu_updated = False
    try:
        from mechanisms.graph_memory import get_graph
        g = get_graph()
        if g.disponibile:
            for chiave, valore in facts:
                g.upsert_fact(
                    f"utente.{chiave}",
                    valore,
                    confidence=0.99,
                    source="user_explicit_correction",
                )
            kuzu_updated = True
    except Exception:
        pass

    # Iniezione in working_memory
    wm_injected = False
    try:
        _inietta_correzioni_in_wm(mem, facts)
        wm_injected = True
    except Exception:
        pass

    _log_correction(user_msg, facts, method, kuzu_updated, wm_injected)

    result["applied"] = kuzu_updated or wm_injected
    return result
