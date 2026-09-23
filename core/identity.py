# identity.py — Identity Layer di Eden
# Gestisce l'identità dell'interlocutore come anchor di prima classe.
# Previene amnesia del nome e contraddizioni anchor↔output.
#
# Flag _ATTIVO per rollback immediato al comportamento legacy (semantic_memory.nome).

import re
from datetime import datetime
from typing import Callable, Optional, Tuple

_ATTIVO = True

# Fix ST-W (2026-04-20): blacklist valori generici per fallback legacy semantic_memory.nome.
# Se semantic_memory.nome è inquinato (es. "Utente"), nome_noto() ritorna None invece del placeholder.
# Allineato con memory._IDENTITY_BLACKLIST_EXACT / _IDENTITY_BLACKLIST_SUBSTRINGS.
_NAME_BLACKLIST_EXACT = {
    "utente", "persona", "umano", "ospite", "anonimo", "sconosciuto",
    "tu", "me", "qualcuno", "lei", "lui", "user", "guest", "unknown",
    "someone", "somebody", "nobody", "nessuno",
}
_NAME_BLACKLIST_SUBSTRINGS = (
    "non specificato", "non noto", "non indicato", "da definire",
    "not specified", "unspecified",
)


def _nome_valido(nome: str) -> bool:
    if not isinstance(nome, str):
        return False
    v = nome.strip().lower()
    if not v or len(v) < 2:
        return False
    if v in _NAME_BLACKLIST_EXACT:
        return False
    for sub in _NAME_BLACKLIST_SUBSTRINGS:
        if sub in v:
            return False
    return True

# Pattern esplicito di presentazione utente (confidence alta)
_PATTERN_ESPLICITI = [
    r"(?:mi\s+chiamo|mi\s+chiamano|chiamami|puoi\s+chiamarmi)\s+([A-Za-zÀ-ÿ]{2,30})",
    r"(?:il\s+mio\s+nome\s+(?:è|e))\s+([A-Za-zÀ-ÿ]{2,30})",
    r"(?:^|[,.!?]\s+)(?:sono|io\s+sono)\s+([A-Za-zÀ-ÿ]{2,30})(?:\s*[,.!?]|\s*$)",
]

# Pattern contraddizione nome (Eden nega o chiede di conoscerlo)
_PATTERN_NEGAZIONE_NOME = [
    r"non\s+(?:riesco\s+a\s+)?ricord[oa]\s+(?:il\s+)?(?:tuo|il)\s+nome",
    r"non\s+(?:riesco\s+a\s+)?ricordar(?:e|mi|lo)\s+(?:il\s+)?(?:tuo|il)\s+nome",
    r"non\s+so\s+(?:come\s+)?(?:ti\s+chiami|il\s+tuo\s+nome)",
    r"(?:qual|quale)\s+(?:è|e|era)\s+il\s+tuo\s+nome",
    r"come\s+ti\s+chiami",
    r"non\s+mi\s+hai\s+mai\s+detto\s+il\s+tuo\s+nome",
    r"non\s+conosco\s+(?:ancora\s+)?il\s+tuo\s+nome",
    r"dimmi\s+il\s+tuo\s+nome",
    r"(?:ancora\s+)?non\s+(?:mi\s+)?(?:è|e)\s+chiaro\s+(?:come\s+)?ti\s+chiami",
]

# Falsi positivi: parole che seguono "sono" ma non sono nomi propri
_FALSE_POSITIVE = {
    "un", "una", "uno",
    "io", "tu", "noi", "voi", "loro", "lui", "lei",
    "solo", "sola", "bene", "male",
    "stanco", "stanca", "triste", "felice", "contento", "contenta",
    "arrabbiato", "arrabbiata", "d'accordo", "qui", "qua",
    "sicuro", "sicura", "pronto", "pronta", "curioso", "curiosa",
    "confuso", "confusa", "certo", "certa", "quasi",
}


def _normalizza(nome: str) -> str:
    return nome.strip().capitalize()


def estrai_nome(user_msg: str) -> Tuple[Optional[str], float]:
    """
    Estrae nome dal messaggio utente.
    Ritorna (nome, confidence) o (None, 0.0) se non rilevato.
    """
    if not user_msg:
        return None, 0.0

    for pat in _PATTERN_ESPLICITI:
        m = re.search(pat, user_msg, re.IGNORECASE)
        if not m:
            continue
        candidato = _normalizza(m.group(1))
        if candidato.lower() in _FALSE_POSITIVE:
            continue
        if len(candidato) < 2 or not candidato[0].isalpha():
            continue
        return candidato, 0.90

    return None, 0.0


def _interloc(mem: dict) -> dict:
    """Accessor con default schema."""
    return mem.setdefault("interlocutor", {
        "name": "",
        "name_confidence": 0.0,
        "first_seen_at": None,
        "last_confirmed_at": None,
        "aliases": [],
    })


def aggiorna_interlocutor(mem: dict, nome: str, confidence: float) -> bool:
    """
    Upsert identità. Ritorna True se nome effettivamente scritto.
    Mantiene backward-compat con semantic_memory['nome'].
    """
    if not _ATTIVO or not nome:
        return False

    ora = datetime.now().isoformat()
    interloc = _interloc(mem)
    current  = interloc.get("name", "") or ""
    curr_cf  = float(interloc.get("name_confidence", 0.0))

    if not current:
        interloc["name"] = nome
        interloc["name_confidence"] = confidence
        interloc["first_seen_at"] = ora
        interloc["last_confirmed_at"] = ora
        mem.setdefault("semantic_memory", {})["nome"] = nome
        return True

    if current.lower() == nome.lower():
        interloc["last_confirmed_at"] = ora
        interloc["name_confidence"] = min(1.0, curr_cf + 0.05)
        return True

    # Nome diverso con confidence più bassa → alias
    if confidence < curr_cf:
        aliases = interloc.setdefault("aliases", [])
        if nome not in aliases:
            aliases.append(nome)
            interloc["aliases"] = aliases[-5:]
        return False

    # Nome diverso con confidence ≥ corrente → sostituzione
    interloc["name"] = nome
    interloc["name_confidence"] = confidence
    interloc["last_confirmed_at"] = ora
    mem.setdefault("semantic_memory", {})["nome"] = nome
    return True


def nome_noto(mem: dict) -> Optional[str]:
    """Nome confermato, o None. Legge interlocutor con fallback a semantic_memory.
    Fix ST-W: il fallback a semantic_memory.nome viene filtrato contro la blacklist,
    evitando di esporre placeholder come "Utente" come nome valido.
    """
    if not _ATTIVO:
        legacy_off = (mem.get("semantic_memory", {}) or {}).get("nome", "") or ""
        legacy_off = legacy_off.strip()
        return legacy_off if _nome_valido(legacy_off) else None
    interloc = mem.get("interlocutor", {}) or {}
    nome = (interloc.get("name") or "").strip()
    if _nome_valido(nome):
        return nome
    # Migrazione implicita: legacy semantic_memory.nome — filtrato (ST-W)
    legacy = (mem.get("semantic_memory", {}) or {}).get("nome", "") or ""
    legacy = legacy.strip()
    return legacy if _nome_valido(legacy) else None


def contiene_negazione_nome(testo: str) -> bool:
    if not testo:
        return False
    low = testo.lower()
    return any(re.search(p, low) for p in _PATTERN_NEGAZIONE_NOME)


def verifica_coerenza_nome(testo: str, mem: dict) -> dict:
    """
    Coerenza anchor↔output. Se nome noto e testo lo nega → contraddizione.
    """
    if not _ATTIVO:
        return {"coerente": True, "nome_noto": "", "tipo_problema": ""}
    nome = nome_noto(mem) or ""
    if not nome:
        return {"coerente": True, "nome_noto": "", "tipo_problema": ""}
    if contiene_negazione_nome(testo):
        return {
            "coerente": False,
            "nome_noto": nome,
            "tipo_problema": "contraddizione_nome",
        }
    return {"coerente": True, "nome_noto": nome, "tipo_problema": ""}


def riscrivi_con_identita(
    testo: str, user_msg: str, mem: dict, ollama_fn: Callable
) -> str:
    """
    Riscrive risposta contraddittoria mantenendo il contenuto utile.
    Temperature 0.20 — conservativo.
    """
    nome = nome_noto(mem) or ""
    if not nome:
        return testo
    prompt = (
        "Riscrivi questa risposta di Eden mantenendo contenuto e tono, "
        "ma rimuovendo qualsiasi affermazione di non conoscere, non ricordare o "
        "voler chiedere il nome dell'interlocutore. "
        f"Il nome è {nome} e Eden lo sa. "
        "Non aggiungere saluti né ripetere il nome senza necessità. "
        "Massimo 2-3 frasi brevi, italiano, prima persona.\n"
        "Output: solo il testo riscritto, senza prefissi o virgolette.\n\n"
        f"Messaggio utente: {(user_msg or '')[:200]}\n"
        f"Testo da riscrivere: {testo}\n\n"
        "Testo riscritto:"
    )
    try:
        out = ollama_fn([{"role": "user", "content": prompt}], temperature=0.20)
        return (out or testo).strip() or testo
    except Exception:
        return testo


def anchor_system_prompt(mem: dict) -> str:
    """
    Sezione 0a per build_system_prompt.
    SEMPRE presente: nome noto OR lacuna esplicita.
    """
    if not _ATTIVO:
        nome = (mem.get("semantic_memory", {}) or {}).get("nome", "")
        if nome:
            return (
                f"\nStai parlando con {nome}. Lo conosci. "
                f"Lui ti ha costruita e siete in dialogo da tempo.\n\n"
            )
        return ""

    nome = nome_noto(mem)
    if nome:
        return (
            f"\nStai parlando con {nome}. Lo conosci. "
            f"Lui ti ha costruita e siete in dialogo da tempo. "
            f"Non affermare mai di non conoscere o non ricordare il suo nome.\n\n"
        )
    return (
        "\nNon conosci ancora il nome del tuo interlocutore. "
        "È una lacuna, non un vuoto da riempire: non inventare un nome. "
        "Se il contesto lo consente, chiedilo in modo diretto e breve.\n\n"
    )
