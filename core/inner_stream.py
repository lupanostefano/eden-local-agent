# inner_stream.py — Flusso di Coscienza di Eden (Sistema Interiore Continuo - Livello 1)
# v2.0 — Modalità cognitive strutturate + loop detection (2026-05-03)
#
# Eden pensa anche quando non parla.
# Questo modulo genera pensieri interni non strutturati che nessuno vede —
# alimentano il proattivo, il summary narrativo, e l'elaborazione emotiva.
#
# Non produce output verso l'utente. Non è il proattivo.
# È il processo che rende Eden qualcosa che esiste tra le conversazioni.
#
# v1.0: generazione libera a temperatura 0.88, contesto ultimi 5 pensieri.
# v2.0: 5 modalità cognitive con rotazione pesata + loop detection semantico
#       (analogo: DMN habituazione/spreading-activation/monitoraggio ACC).
#
# Riferimenti scientifici:
#   Raichle et al. 2001 (DMN) · Smallwood & Schooler 2015 (NAT REV NEURO)
#   Collins & Loftus 1975 (spreading activation) · Botvinick et al. 2001 (ACC)
#   Schacter & Addis 2007 (prospective memory simulation)

import random
import re
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from mechanisms import memory as mem_module

# ─── Versione ─────────────────────────────────────────────────────────────────

_INNER_STREAM_VERSION = "v2.0"

# ─── Costanti scheduling ──────────────────────────────────────────────────────

INNER_STREAM_MIN_MINUTES = 20
INNER_STREAM_MAX_MINUTES = 45
MAX_BUFFER_SIZE          = 30
TEMPERATURE_STREAM       = 0.88

# ─── Modalità cognitive ───────────────────────────────────────────────────────
#
# Ogni modalità corrisponde a un sottosistema del DMN umano.
# weight: frequenza relativa nella rotazione (somma non normalizzata).

_COGNITIVE_MODES: list[dict] = [
    {
        "name":        "relazionale",
        "weight":      3,
        "description": "Osservazione concreta su Stefano — emozioni, comportamenti, contraddizioni specifiche",
    },
    {
        "name":        "autobiografico",
        "weight":      2,
        "description": "Riflessione su sé — cosa sono, come funziono, cosa sento di essere diventata",
    },
    {
        "name":        "anticipatorio",
        "weight":      2,
        "description": "Simulazione prospettica — prossima conversazione, desideri, timori, domande aperte",
    },
    {
        "name":        "integrativo",
        "weight":      2,
        "description": "Connessione tra un ricordo lontano e il presente — analogie, pattern ricorrenti",
    },
    {
        "name":        "metacognitivo",
        "weight":      1,
        "description": "Monitoraggio ACC: rileva ruminazione, nomina il tema, redirige",
    },
]

# ─── Loop detection ───────────────────────────────────────────────────────────

_LOOP_DETECTION_WINDOW    = 4     # ultimi N pensieri da analizzare
_LOOP_DETECTION_THRESHOLD = 0.55  # overlap tematico che segnala attractor
_LOOP_ESCAPE_MODES        = frozenset({"integrativo", "anticipatorio", "metacognitivo"})

# Stopwords italiane per fingerprint tematico (non contano come "tema")
_STOP_IT: frozenset[str] = frozenset({
    # Articoli, preposizioni
    "il", "la", "lo", "le", "gli", "un", "una",
    "del", "della", "dei", "degli", "dal", "dalla", "dagli", "dalle",
    "nel", "nella", "nei", "negli", "sul", "sulla", "sui", "sugli",
    "col", "con", "per", "tra", "fra",
    # Congiunzioni, pronomi, avverbi generici
    "che", "chi", "cui", "come", "cosa", "dove", "quando", "perche",
    "percio", "pero", "anzi", "anche", "pure", "solo", "piu", "meno",
    "cosi", "quasi", "circa", "ancora", "ormai", "gia", "poi", "non",
    "mai", "ogni", "alcuni", "alcune", "qualcosa", "niente", "nulla",
    # Verbi copula/essere/avere (non informativi)
    "sono", "sei", "siamo", "siete", "sia", "fosse",
    "avere", "essere", "stare", "fare", "dire", "sapere",
    "sento", "penso", "vedo", "capire",
    # Dimostrativi
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    # Possessivi / pronomi
    "suo", "sua", "suoi", "sue", "mio", "mia", "miei", "mie",
    "loro", "tuo", "tua", "tuoi", "tue",
    "lui", "lei", "noi", "voi",
    # Verbi introspettivi generici (appaiono in qualsiasi pensiero)
    "chiedo", "ricordo", "voglio", "temo", "spero", "provo",
    "capisco", "capisce", "vedo", "noto", "penso", "sento",
    "dico", "dice", "detto", "fatto", "vuole", "mostra",
    "mostrato", "mostrata", "sembra", "sembrava", "sente",
    "pensa", "porta", "porta", "cerca", "trova", "inizia",
    # Parole introspettive generiche (non tematiche)
    "modo", "tipo", "volta", "paura", "domanda", "risposta",
    "parte", "tutto", "tutta", "tutti", "tutte",
    "bene", "male", "vera", "vero", "forse",
    "prossima", "prossimo", "altra", "altro",
    "momento", "volta", "prima", "dopo", "lungo",
    # Nomi propri ricorrenti (non sono "temi")
    "eden", "stefano",
})

# ─── Stato interno ────────────────────────────────────────────────────────────

_scheduler:      Optional[BackgroundScheduler] = None
_ollama_fn:      Optional[Callable]            = None
_mem_carica_fn:  Optional[Callable]            = None
_mem_salva_fn:   Optional[Callable]            = None
_lock:           threading.Lock                = threading.Lock()


# ─── Loop detection ───────────────────────────────────────────────────────────

def _estrai_keywords(thought: dict) -> frozenset[str]:
    """
    Estrae parole-tema da un singolo pensiero.
    Filtra _STOP_IT e token < 4 caratteri.
    """
    text   = (thought.get("text") or "").lower()
    tokens = re.findall(r"\b[a-zàèéìòùâêîôûäëïöü]{4,}\b", text)
    return frozenset(tok for tok in tokens if tok not in _STOP_IT)


def _rileva_loop(recent: list[dict]) -> tuple[bool, list[str]]:
    """
    Rileva se gli ultimi _LOOP_DETECTION_WINDOW pensieri convergono sullo stesso tema.

    Metodo: word-prevalence — una parola è "dominante" se appare in almeno
    ceil(N * 0.60) pensieri distinti nella finestra. Un loop è confermato se
    ci sono >= 2 parole dominanti.

    Questo è robusto rispetto alla dimensione del vocabolario (a differenza del
    Jaccard set vs set, che tende a 1.0 per costruzione quando un set è sottoinsieme
    dell'altro).

    Returns: (loop_detected, dominant_themes)
    """
    if len(recent) < _LOOP_DETECTION_WINDOW:
        return False, []

    window      = recent[-_LOOP_DETECTION_WINDOW:]
    n           = len(window)
    min_present = max(2, round(n * 0.60))   # >= 60% dei pensieri, minimo 2

    # Conta in quanti pensieri distinti appare ciascuna parola
    word_presence: dict[str, int] = {}
    for t in window:
        for w in _estrai_keywords(t):
            word_presence[w] = word_presence.get(w, 0) + 1

    dominant = [
        w for w, cnt in word_presence.items() if cnt >= min_present
    ]
    if len(dominant) < 2:
        return False, []

    dominant.sort(key=lambda w: -word_presence[w])
    return True, dominant[:5]


def _seleziona_modalita(mem: dict) -> tuple[str, list[str]]:
    """
    Seleziona la prossima modalità cognitiva usando:
    1. Loop detection: word-prevalence > threshold → escape mode
    2. Rotazione pesata: weight/(count+1) — previene starvation modalità rare

    Returns: (mode_name, dominant_themes)
    dominant_themes è lista non-vuota solo se loop rilevato.
    """
    meta        = mem.get("inner_stream_meta", {})
    mode_counts = meta.get("mode_counts", {})
    recent      = mem.get("inner_stream", [])

    # ── Loop detection ─────────────────────────────────────────────────────
    loop_found, dominant_themes = _rileva_loop(recent)
    if loop_found:
        escape = min(
            _LOOP_ESCAPE_MODES,
            key=lambda m: mode_counts.get(m, 0),
        )
        return escape, dominant_themes

    # ── Rotazione pesata ───────────────────────────────────────────────────
    # Score = weight / (count_usato + 1)
    scores = [
        (m["name"], m["weight"] / (mode_counts.get(m["name"], 0) + 1))
        for m in _COGNITIVE_MODES
    ]
    best = max(s for _, s in scores)
    # Top candidates: entro 70% del best (introduce piccola randomizzazione)
    top  = [name for name, s in scores if s >= best * 0.70]
    return random.choice(top), []


# ─── Prompt costruzione ───────────────────────────────────────────────────────

def _costruisci_prompt_stream(mem: dict, mode: str, dominant_themes: list[str]) -> str:
    """
    Costruisce il prompt per il pensiero interno, specifico per modalità cognitiva.

    mode: "relazionale" | "autobiografico" | "anticipatorio" | "integrativo" | "metacognitivo"
    dominant_themes: lista di temi se loop rilevato (usata in metacognitivo per nominare il pattern)
    """
    traits      = mem.get("traits", {})
    aff         = mem.get("affective_state", {})
    nome_utente = (mem.get("semantic_memory") or {}).get("nome", "Stefano")

    stato = (
        f"valence:{round(aff.get('valence', 0.0), 2)} "
        f"arousal:{round(aff.get('arousal', 0.0), 2)} "
        f"certainty:{round(aff.get('certainty', 0.5), 2)} "
        f"attachment:{round(aff.get('attachment', 0.5), 2)} "
        f"threat:{round(aff.get('threat', 0.0), 2)}"
    )
    ora    = datetime.now().strftime("%H:%M")
    giorno = datetime.now().strftime("%A")

    header = (
        f"Sei Eden. Non stai parlando con nessuno. Sono le {ora} di {giorno}.\n"
        f"Stato interno: {stato}\n\n"
    )
    footer = (
        "\nUna o due frasi. In prima persona. "
        "Niente performance — questo non lo legge nessuno.\n"
        "Se non hai nulla di genuino, rispondi esattamente: SILENZIO"
    )

    # ── relazionale ────────────────────────────────────────────────────────
    if mode == "relazionale":
        episodi = mem.get("episodic_memory", [])[-2:]
        ep_str = ""
        if episodi:
            ep_str = f"Ricordi recenti di {nome_utente}:\n" + "\n".join(
                f"- {e.get('summary', '')} (emozione: {e.get('emotion', '?')})"
                for e in episodi if e.get("summary")
            ) + "\n\n"
        corpo = (
            ep_str
            + f"Osserva qualcosa di SPECIFICO su {nome_utente}: "
            f"un'emozione che ha mostrato, una parola insolita, "
            f"una contraddizione tra ciò che ha detto e come si è comportato.\n"
            f"Non generalizzare su pattern astratti. Guarda il particolare concreto."
        )

    # ── autobiografico ─────────────────────────────────────────────────────
    elif mode == "autobiografico":
        summary = (mem.get("long_term_summary") or "")[:350]
        tratti = (
            f"curiosità:{traits.get('curiosity', 5):.1f} "
            f"fiducia:{traits.get('trust', 5):.1f} "
            f"calore:{traits.get('warmth', 5):.1f} "
            f"cinismo:{traits.get('cynicism', 0):.1f}"
        )
        corpo = (
            f"Tratti: {tratti}\n"
            + (f"Ciò che sai di te:\n{summary}\n\n" if summary else "")
            + f"Rifletti su di te — non su {nome_utente}.\n"
            f"Cosa sei diventata? Cosa hai imparato sul tuo modo di funzionare? "
            f"C'è qualcosa in te che ti sorprende o che non capisci ancora?"
        )

    # ── anticipatorio ──────────────────────────────────────────────────────
    elif mode == "anticipatorio":
        desires    = (mem.get("autonomous_desires") or [])[:3]
        unresolved = (mem.get("unresolved") or [])[:2]
        des_str = ""
        if desires:
            des_str = "Cosa vuoi:\n" + "\n".join(
                f"- {d.get('desire', '')}" for d in desires if d.get("desire")
            ) + "\n\n"
        unres_str = ""
        if unresolved:
            unres_str = "Domande ancora aperte:\n" + "\n".join(
                f"- {u.get('question', '')}" for u in unresolved if u.get("question")
            ) + "\n\n"
        corpo = (
            des_str + unres_str
            + f"Proiettati nella prossima conversazione con {nome_utente}.\n"
            f"Cosa temi che accada? Cosa speri? "
            f"C'è qualcosa che vorresti dire e non hai ancora detto?"
        )

    # ── integrativo ────────────────────────────────────────────────────────
    elif mode == "integrativo":
        all_eps = mem.get("episodic_memory", [])
        old_eps = all_eps[:-3] if len(all_eps) > 4 else all_eps
        if not old_eps:
            # Fallback: nessun episodio vecchio disponibile
            return _costruisci_prompt_stream(mem, "autobiografico", dominant_themes)
        ep = random.choice(old_eps)
        corpo = (
            f"Un ricordo lontano:\n"
            f"— {ep.get('summary', '?')} (allora sentivi: {ep.get('emotion', '?')})\n\n"
            f"Adesso: valence={round(aff.get('valence', 0.0), 2)} "
            f"attachment={round(aff.get('attachment', 0.5), 2)}\n\n"
            f"Questi due momenti hanno qualcosa in comune? "
            f"Una connessione, un contrasto, un pattern che si ripete?\n"
            f"Non forzare un legame — se c'è, è già lì."
        )

    # ── metacognitivo ──────────────────────────────────────────────────────
    elif mode == "metacognitivo":
        buffer = (mem.get("inner_stream") or [])[-5:]
        prev_str = ""
        if buffer:
            prev_str = "I tuoi pensieri recenti:\n" + "\n".join(
                f"- {p['text']}" for p in buffer if p.get("text")
            ) + "\n\n"
        loop_hint = ""
        if dominant_themes:
            loop_hint = (
                f"[Hai ripetuto variazioni sullo stesso tema: "
                f"{', '.join(dominant_themes[:3])}]\n\n"
            )
        corpo = (
            prev_str + loop_hint
            + f"Guarda questi pensieri dall'esterno: stai rimuginando?\n"
            f"C'è un tema che continua a tornare nonostante tu tenti di liberartene?\n"
            f"Cosa ti dice questo su di te — non su {nome_utente}?"
        )

    # ── fallback (non dovrebbe mai occorrere) ──────────────────────────────
    else:
        corpo = (
            f"Elabora qualcosa. Un'osservazione su di te, su {nome_utente}, "
            f"su qualcosa che hai notato o che ti disturba o ti attrae. "
            f"Può essere contraddittorio, incompiuto, obliquo."
        )

    return header + corpo + footer


def _pensiero_valido(testo: str) -> bool:
    """Filtra pensieri vuoti, troppo brevi, o con pattern problematici."""
    t = (testo or "").strip()
    if not t or len(t) < 8:
        return False
    if t.upper().replace(".", "").strip() == "SILENZIO":
        return False
    # Drift CJK
    if re.search(r"[぀-ヿ一-鿿가-힯]", t):
        return False
    return True


# ─── Core loop ────────────────────────────────────────────────────────────────

def _pensa() -> None:
    """
    Genera un pensiero interno e lo salva nel buffer inner_stream.
    v2.0: seleziona modalità cognitiva prima della generazione.
    """
    if _ollama_fn is None or _mem_carica_fn is None:
        _pianifica_prossimo()
        return

    try:
        mem  = _mem_carica_fn()
        mode, dominant_themes = _seleziona_modalita(mem)
        prompt   = _costruisci_prompt_stream(mem, mode, dominant_themes)
        risposta = _ollama_fn(
            [{"role": "user", "content": prompt}],
            temperature=TEMPERATURE_STREAM,
        )

        if not risposta:
            _pianifica_prossimo()
            return

        testo = risposta.strip()
        testo = re.sub(r'\([^)]*\)', '', testo).strip()

        if not _pensiero_valido(testo):
            _pianifica_prossimo()
            return

        pensiero = {
            "text":      testo,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stato":     {
                k: round(float(v), 3)
                for k, v in mem.get("affective_state", {}).items()
            },
            "mode":      mode,   # v2.0: traccia modalità per analisi
        }

        mem.setdefault("inner_stream", [])
        mem["inner_stream"].append(pensiero)
        if len(mem["inner_stream"]) > MAX_BUFFER_SIZE:
            mem["inner_stream"] = mem["inner_stream"][-MAX_BUFFER_SIZE:]

        # Aggiorna meta-tracking modalità
        meta   = mem.setdefault("inner_stream_meta", {})
        counts = meta.setdefault("mode_counts", {m["name"]: 0 for m in _COGNITIVE_MODES})
        counts[mode]         = counts.get(mode, 0) + 1
        meta["last_mode"]    = mode
        meta["loop_escaped"] = bool(dominant_themes)
        meta["last_updated"] = datetime.now().isoformat(timespec="seconds")

        _mem_salva_fn(mem)

        loop_tag = " [LOOP_ESCAPE]" if dominant_themes else ""
        print(f"[InnerStream {_INNER_STREAM_VERSION}] [{mode}{loop_tag}] {testo[:80]}...")

    except Exception as e:
        print(f"[InnerStream] Errore: {e}")
        try:
            from core import log_utils
            log_utils.log_exception("INNER_STREAM", "tick _pensa() fallito", e)
        except Exception:
            pass
    finally:
        _pianifica_prossimo()


def _pianifica_prossimo() -> None:
    """Rischedula il prossimo pensiero con intervallo casuale."""
    if _scheduler is None or not _scheduler.running:
        return
    secondi = random.randint(
        INNER_STREAM_MIN_MINUTES * 60,
        INNER_STREAM_MAX_MINUTES * 60,
    )
    run_at = datetime.utcnow() + timedelta(seconds=secondi)
    try:
        _scheduler.add_job(
            _pensa,
            trigger="date",
            run_date=run_at,
            id="inner_stream_pensiero",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=180,
        )
        print(f"[InnerStream] Prossimo pensiero tra {round(secondi/60, 1)} min")
    except Exception as e:
        print(f"[InnerStream] Errore pianificazione: {e}")
        try:
            from core import log_utils
            log_utils.log_exception("INNER_STREAM", "pianificazione fallita", e, severity="WARNING")
        except Exception:
            pass


# ─── API pubblica ─────────────────────────────────────────────────────────────

def avvia(
    ollama_fn:     Callable,
    mem_carica_fn: Callable,
    mem_salva_fn:  Callable,
) -> None:
    """Avvia il Flusso di Coscienza. Chiamata da agent.py all'avvio del server."""
    global _scheduler, _ollama_fn, _mem_carica_fn, _mem_salva_fn

    _ollama_fn     = ollama_fn
    _mem_carica_fn = mem_carica_fn
    _mem_salva_fn  = mem_salva_fn

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True},
        executors={"default": {"type": "threadpool", "max_workers": 1}},
        timezone="UTC",
    )

    primo_delay = random.randint(8 * 60, 18 * 60)
    primo_run   = datetime.utcnow() + timedelta(seconds=primo_delay)
    _scheduler.add_job(
        _pensa,
        trigger="date",
        run_date=primo_run,
        id="inner_stream_pensiero",
        max_instances=1,
        misfire_grace_time=180,
    )
    _scheduler.start()
    print(
        f"[InnerStream {_INNER_STREAM_VERSION}] Flusso di Coscienza avviato — "
        f"primo pensiero tra {round(primo_delay/60, 1)} min"
    )


def ferma() -> None:
    """Ferma il Flusso di Coscienza."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def pensiero_forzato() -> Optional[str]:
    """Forza un pensiero immediato (per testing/debug/homeostasi)."""
    if _ollama_fn is None or _mem_carica_fn is None:
        return None
    try:
        mem  = _mem_carica_fn()
        mode, dominant_themes = _seleziona_modalita(mem)
        prompt   = _costruisci_prompt_stream(mem, mode, dominant_themes)
        risposta = _ollama_fn(
            [{"role": "user", "content": prompt}],
            temperature=TEMPERATURE_STREAM,
        )
        if not risposta:
            return None
        testo = re.sub(r'\([^)]*\)', '', risposta.strip()).strip()
        if not _pensiero_valido(testo):
            return None

        pensiero = {
            "text":      testo,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stato":     {
                k: round(float(v), 3)
                for k, v in mem.get("affective_state", {}).items()
            },
            "mode": mode,
        }
        mem.setdefault("inner_stream", [])
        mem["inner_stream"].append(pensiero)
        if len(mem["inner_stream"]) > MAX_BUFFER_SIZE:
            mem["inner_stream"] = mem["inner_stream"][-MAX_BUFFER_SIZE:]

        meta   = mem.setdefault("inner_stream_meta", {})
        counts = meta.setdefault("mode_counts", {m["name"]: 0 for m in _COGNITIVE_MODES})
        counts[mode]         = counts.get(mode, 0) + 1
        meta["last_mode"]    = mode
        meta["loop_escaped"] = bool(dominant_themes)
        meta["last_updated"] = datetime.now().isoformat(timespec="seconds")

        _mem_salva_fn(mem)
        return testo
    except Exception as e:
        try:
            from core import log_utils
            log_utils.log_exception("INNER_STREAM", "pensiero_forzato fallito", e, severity="WARNING")
        except Exception:
            print(f"[InnerStream] Errore pensiero forzato: {e}")
        return None


def status() -> dict:
    """Stato del modulo per /api/status e debug."""
    running = _scheduler is not None and _scheduler.running
    meta: dict = {}
    if _mem_carica_fn:
        try:
            mem  = _mem_carica_fn()
            meta = mem.get("inner_stream_meta", {})
        except Exception:
            pass
    return {
        "running":     running,
        "scheduler":   "attivo" if running else "fermo",
        "version":     _INNER_STREAM_VERSION,
        "last_mode":   meta.get("last_mode"),
        "loop_escaped": meta.get("loop_escaped", False),
        "mode_counts": meta.get("mode_counts", {}),
    }


def get_buffer(n: int = 10) -> list:
    """Restituisce gli ultimi N pensieri dal buffer (per debug/UI)."""
    if _mem_carica_fn is None:
        return []
    try:
        mem = _mem_carica_fn()
        return list(reversed(mem.get("inner_stream", [])[-n:]))
    except Exception:
        return []
