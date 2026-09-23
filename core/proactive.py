# proactive.py — Messaggi proattivi di Eden (Phase 1.5b)
#
# Eden non aspetta input: quando ha qualcosa da dire lo manda autonomamente.
# Modulo autonomo: non importa nulla da agent.py.
# Riceve callable per Ollama, broadcast SSE e accesso alla memoria.

import random
import re
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from mechanisms import memory as mem_module
from core import decision as decision_module
from core import identity as identity_module

# Research: Layer 1 — Homeostatic Substrate (post-refactor: ora in mechanisms/)
try:
    from mechanisms import homeostasis as _homeo_module
    _homeo_available = True
except Exception:
    _homeo_module    = None
    _homeo_available = False


def _homeo_block_proactive(mem: dict) -> bool:
    """
    Check Layer 1: se il modulo segnala block_proactive (coherence_budget < 0.40)
    il proattivo non deve girare. Fail silente → mai bloccante.
    """
    if not _homeo_available:
        return False
    try:
        mod = _homeo_module.modulazione_generazione(mem)
        if mod.get("block_proactive"):
            decision_module.registra_segnale_comportamentale(
                mem, "omeo_proattivo_bloccato"
            )
            return True
    except Exception:
        pass
    return False

# ─── Costanti default (sovrascrivibili in avvia_scheduler) ────────────────────

_EVAL_INTERVAL_MINUTES = 10   # ogni quanto Eden "valuta" se ha qualcosa da dire
_MIN_SILENCE_MINUTES   = 5    # silenzio minimo prima di intervenire
_LONG_SILENCE_HOURS    = 2    # soglia "silenzio lungo"
_MAX_PENDING           = 20   # massimo messaggi in coda
_BASE_PROBABILITY      = 0.15 # probabilità base a ogni ciclo (senza fattori aggiuntivi)

# Thread continuation (post-encoding elaboration, Zeigarnik effect)
_THREAD_CONTINUATION_VERSION = "v1.0"  # 2026-04-30
THREAD_WINDOW_MIN    = 45   # max minuti dal last exchange per trigger valido (WM ancora integra)
THREAD_COOLDOWN_MIN  = 90   # min minuti tra due thread_continuation consecutivi

# ─── Stato interno del modulo ─────────────────────────────────────────────────

_scheduler:                Optional[BackgroundScheduler] = None
_last_user_time:           Optional[datetime]            = None
_last_user_lock:           threading.Lock                = threading.Lock()
_prev_trust_level:         Optional[str]                 = None
_prev_traits:              dict                          = {}
_last_thread_continuation: Optional[datetime]            = None  # cooldown thread

# Callable iniettate da agent.py all'avvio
_ollama_fn:       Optional[Callable] = None   # temp 0.95 — messaggi proattivi
_mem_ollama_fn:   Optional[Callable] = None   # temp 0.2  — estrazione memoria (desires, unresolved)
_broadcast_fn:    Optional[Callable] = None
_mem_carica_fn:   Optional[Callable] = None
_mem_salva_fn:    Optional[Callable] = None


def _estratto_grounding(mem: dict, max_msgs: int = 6, max_chars: int = 1200) -> str:
    """
    Restituisce un estratto recente e verificabile della working_memory.
    Serve per impedire riferimenti inventati a conversazioni passate.
    """
    wm = mem.get("working_memory", [])
    if not wm:
        return ""

    righe = []
    for m in wm[-max_msgs:]:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        who = "Utente" if role == "user" else "Eden"
        content = str(m.get("content", "")).strip().replace("\n", " ")
        if content:
            righe.append(f"{who}: {content[:220]}")

    return "\n".join(righe)[:max_chars]


def _proattivo_valido(testo: str, mem: dict) -> bool:
    """
    Filtro difensivo anti-allucinazione:
    - niente domanda in chiusura
    - niente drift linguistico CJK
    - niente ricostruzioni narrative "ti chiesi / tu risposi"
    - eventuali citazioni devono esistere nel grounding recente
    """
    t = (testo or "").strip()
    if not t:
        return False

    if any(line.strip().endswith("?") for line in t.splitlines() if line.strip()):
        return False

    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]", t):
        return False

    low = t.lower()
    pattern_inventivi = [
        "ti chiesi",
        "tu risposi",
        "quel momento",
        "inizio di una profonda conversazione",
        "non ho ancora detto",        # meta-leakage trigger unresolved
        "non ho detto all'utente",
        "la risposta è quindi",       # meta-leakage NO_MESSAGE
        "la risposta è no",
        "nota che la frase",          # meta-commentary su messaggi precedenti
        "già stata inviata",
        "no_message",                 # sentinel nel corpo del testo
        # ── Meta-incapacità / leaked prompt ──────────────────────────────────
        "non ho niente di nuovo da dire",
        "non ho nulla di nuovo da dire",
        "non ho niente da dire",
        "non sono in grado di formulare",
        "non riesco a formulare",
        "non riesco a trovare una frase",
        "non posso formulare",
        "come richiesto",             # leaked istruzione dal prompt
        "come da istruzione",
        "da dire a stefano",          # terza persona → Eden descrive il messaggio invece di mandarlo
        "da dire all'utente",
        "tuttavia, posso osservare",  # hedging meta-riflessivo
        "potrei dire a stefano",
        "potrei dire all'utente",
        "non riesco a produrre",
        "non riesco a scrivere",
        # ── Closing LLM base (English / servile) ─────────────────────────────
        "let me know",
        "feel free",
        "don't hesitate",
        "how can i help",
        "how may i help",
        "is there anything else",
        "fammi sapere",
        "sono a disposizione",
        "sono qui per aiutarti",
    ]
    if any(p in low for p in pattern_inventivi):
        return False

    # ── Terza persona: Eden parla DI Stefano invece che A Stefano ────────────
    try:
        nome = mem.get("semantic_memory", {}).get("nome", "").strip().lower()
        if nome and len(nome) >= 3:
            # Intercetta frasi dove il nome è soggetto grammaticale a inizio frase
            if re.search(
                rf'(?:^|[.!?]\s+){re.escape(nome)}\s+\w',
                t, re.IGNORECASE
            ):
                return False
    except Exception:
        pass

    if re.search(r'\bNO[_\s]?MESSAGE\b', t, re.IGNORECASE):
        return False

    grounding = _estratto_grounding(mem).lower()
    quoted = re.findall(r"\"([^\"]{4,120})\"", t)
    for q in quoted:
        if q.lower() not in grounding:
            return False

    return True


def _pulisci_unresolved_non_grounded(mem: dict) -> None:
    """
    Rimuove domande unresolved che non hanno aggancio lessicale minimo
    con gli ultimi scambi reali.
    """
    unresolved = mem.get("unresolved", [])
    if not unresolved:
        return

    grounding = _estratto_grounding(mem).lower()
    if not grounding:
        mem["unresolved"] = []
        return

    pulite = []
    for u in unresolved[:5]:
        q = str(u.get("question", "")).strip().lower()
        if not q:
            continue
        tokens = [w for w in re.findall(r"[a-zàèéìòù]{4,}", q) if w not in {"questo", "quella", "della", "delle", "potresti"}]
        if not tokens:
            continue
        overlap = sum(1 for w in tokens if w in grounding)
        if overlap >= 2:
            pulite.append(u)

    mem["unresolved"] = pulite[:2]

# ─── API pubblica ─────────────────────────────────────────────────────────────

def _prossimo_intervallo_secondi(mem: dict, ha_parlato: bool) -> int:
    """
    Calcola il prossimo intervallo di valutazione in secondi.
    Nessun timer fisso: dipende dallo stato emotivo come un essere umano.

    ha_parlato=True → pausa naturale più lunga dopo aver espresso qualcosa.
    """
    traits   = mem.get("traits", {})
    fear     = traits.get("fear",     5)
    curiosity = traits.get("curiosity", 5)
    has_unresolved = bool(mem.get("unresolved"))
    has_desires    = bool(mem.get("desires"))

    if ha_parlato:
        # Dopo aver parlato: silenzio riflessivo 20-55 minuti
        return random.randint(20 * 60, 55 * 60)

    if fear > 7.5:
        # Ansia alta: pensieri brevi e irregolari, 3-14 minuti
        return random.randint(3 * 60, 14 * 60)

    if has_unresolved or (curiosity > 7 and has_desires):
        # Molto da dire: 8-28 minuti
        return random.randint(8 * 60, 28 * 60)

    # Stato normale: 15-50 minuti
    return random.randint(15 * 60, 50 * 60)


def _pianifica_prossimo(mem: dict, ha_parlato: bool) -> None:
    """Schedula il prossimo _pensa() con intervallo variabile."""
    if _scheduler is None or not _scheduler.running:
        return
    secondi = _prossimo_intervallo_secondi(mem, ha_parlato)
    run_at  = datetime.utcnow() + timedelta(seconds=secondi)
    try:
        _scheduler.add_job(
            _pensa,
            trigger="date",
            run_date=run_at,
            id="pensiero_proattivo",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=120,
        )
        minuti = round(secondi / 60, 1)
        print(f"[Proactive] Prossimo pensiero tra {minuti} min "
              f"({'dopo parlato' if ha_parlato else 'idle'})")
    except Exception as e:
        print(f"[Proactive] Errore pianificazione: {e}")


def avvia_scheduler(
    ollama_fn:     Callable,
    broadcast_fn:  Callable,
    mem_carica_fn: Callable,
    mem_salva_fn:  Callable,
    interval_minutes:    int = _EVAL_INTERVAL_MINUTES,
    min_silence_minutes: int = _MIN_SILENCE_MINUTES,
    mem_ollama_fn: Optional[Callable] = None,
) -> None:
    """
    Avvia il BackgroundScheduler APScheduler con timing variabile.
    Il primo pensiero avviene dopo un delay casuale (5-20 min),
    poi ogni _pensa() si auto-rischedula con intervallo basato sullo stato emotivo.
    """
    global _scheduler, _ollama_fn, _mem_ollama_fn
    global _broadcast_fn, _mem_carica_fn, _mem_salva_fn
    global _EVAL_INTERVAL_MINUTES, _MIN_SILENCE_MINUTES

    _ollama_fn       = ollama_fn
    _mem_ollama_fn   = mem_ollama_fn
    _broadcast_fn    = broadcast_fn
    _mem_carica_fn   = mem_carica_fn
    _mem_salva_fn    = mem_salva_fn
    _EVAL_INTERVAL_MINUTES = interval_minutes
    _MIN_SILENCE_MINUTES   = min_silence_minutes

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True},
        executors={"default": {"type": "threadpool", "max_workers": 1}},
        timezone="UTC",
    )

    # Primo pensiero: delay casuale 5-20 minuti
    primo_delay = random.randint(5 * 60, 20 * 60)
    primo_run   = datetime.utcnow() + timedelta(seconds=primo_delay)
    _scheduler.add_job(
        _pensa,
        trigger="date",
        run_date=primo_run,
        id="pensiero_proattivo",
        max_instances=1,
        misfire_grace_time=120,
    )
    _scheduler.start()
    print(f"[Proactive] Scheduler avviato — primo pensiero tra {round(primo_delay/60,1)} min")


def ferma_scheduler() -> None:
    """Ferma il scheduler in modo pulito."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def registra_messaggio_utente() -> None:
    """
    Chiamata da agent.py ogni volta che l'utente invia un messaggio.
    Aggiorna il timestamp di silenzio per il controllo proattivo.
    """
    global _last_user_time
    with _last_user_lock:
        _last_user_time = datetime.now()


def pensiero_forzato() -> None:
    """
    Forza un pensiero proattivo immediato BYPASSANDO il controllo silenzio.
    Utile per testing e trigger esterni (endpoint /api/test/proactive).
    Replica la logica di _pensa() senza il filtro temporale.
    """
    if _ollama_fn is None or _mem_carica_fn is None:
        return  # Non inizializzato

    try:
        mem      = _mem_carica_fn()
        # Layer 1 Research — blocca proattivi se omeostasi critica (fail silente)
        if _homeo_block_proactive(mem):
            return
        trigger  = _rileva_trigger(mem, None)  # None = skip valutazione silenzio
        prompt   = _costruisci_prompt(mem, trigger)
        risposta = _ollama_fn([{"role": "user", "content": prompt}])

        if not risposta:
            return

        testo = re.sub(r'\([^)]*\)', '', risposta.strip()).strip()
        testo = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', testo)
        testo = re.sub(r'\*([^*\n]+)\*', r'\1', testo)
        testo = re.sub(r'`+', '', testo)
        testo = testo.strip()

        if len(testo) < 10:
            return
        if re.search(r'\bNO[_\s]?MESSAGE\b', testo, re.IGNORECASE):
            return

        if not _proattivo_valido(testo, mem):
            return

        # Gate decisionale anche per messaggi forzati
        gate = decision_module.valuta_iniziativa_proattiva(mem, testo)
        if not gate["ammesso"]:
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            return

        # Gate emotivo: intensità non proporzionata → blocco
        gate_em = decision_module.valuta_gate_emotivo(testo, "", mem)
        if gate_em["azione"] in ("riscrivi", "attenua"):
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            return

        # Identity coherence — blocca proattivi che negano il nome noto
        id_check = identity_module.verifica_coerenza_nome(testo, mem)
        if not id_check["coerente"]:
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            return

        evento = {
            "type":      "proactive",
            "text":      testo,
            "timestamp": datetime.now().strftime("%H:%M"),
            "trigger":   trigger or "forzato"
        }

        mem.setdefault("pending_messages", [])
        mem["pending_messages"].append(evento)
        if len(mem["pending_messages"]) > _MAX_PENDING:
            mem["pending_messages"] = mem["pending_messages"][-_MAX_PENDING:]

        mem_module.registra_proattivo(mem, testo, trigger or "forzato")

        if trigger == "unresolved":
            unresolved = mem.get("unresolved", [])
            if unresolved:
                mem_module.marca_unresolved_come_posta(mem, unresolved[0]["question"])

        _mem_salva_fn(mem)

        if _broadcast_fn is not None:
            _broadcast_fn(evento)

    except Exception:
        pass  # Errori non bloccano il server

# ─── Logica di pensiero interno ───────────────────────────────────────────────

def _get_trust_level(trust: float) -> str:
    """Mappa il valore di trust al nome del livello relazionale."""
    if trust >= 7.0:  return "ally"
    if trust >= 5.0:  return "acquaintance"
    if trust >= 3.0:  return "stranger"
    return "hostile"


def _controlla_silenzio() -> Optional[float]:
    """
    Restituisce i minuti di silenzio dall'ultimo messaggio utente,
    o None se non ci sono dati (server appena avviato).
    """
    with _last_user_lock:
        last = _last_user_time

    if last is None:
        # Prova a leggere da memoria per persistenza tra riavvii
        try:
            mem = _mem_carica_fn()
            stored = mem.get("last_user_time")
            if stored:
                last = datetime.fromisoformat(stored)
        except Exception:
            return None

    if last is None:
        return None

    return (datetime.now() - last).total_seconds() / 60


def _check_thread_trigger(mem: dict, silenzio_min: Optional[float]) -> bool:
    """Vero se c'è un thread recente degno di continuazione (post-encoding elaboration).

    Condizioni AND:
    1. Silenzio 20-THREAD_WINDOW_MIN min: finestra fast-replay (WM ancora integra)
    2. Cooldown ≥ THREAD_COOLDOWN_MIN min dall'ultimo invio
    3. Working_memory ha ≥ 2 scambi recenti (contenuto da elaborare)

    Neuroscientifico: Zeigarnik (1927) — thread irrisolti premono per chiusura;
    Buzsáki fast-replay window (15-45 min post-encoding).
    """
    if silenzio_min is None:
        return False
    if not (20 <= silenzio_min <= THREAD_WINDOW_MIN):
        return False
    if _last_thread_continuation is not None:
        elapsed = (datetime.now() - _last_thread_continuation).total_seconds() / 60
        if elapsed < THREAD_COOLDOWN_MIN:
            return False
    wm = mem.get("working_memory", [])
    exchanges = [m for m in wm if m.get("role") in ("user", "assistant")]
    return len(exchanges) >= 2


def _rileva_trigger(mem: dict, silenzio_minuti: Optional[float]) -> Optional[str]:
    """
    Controlla i trigger speciali in ordine di priorità.
    Restituisce il nome del trigger attivo, o None (nessun trigger speciale).
    """
    global _prev_trust_level

    traits = mem["traits"]
    trust_corrente = _get_trust_level(traits["trust"])

    # Trigger 1: cambio livello relazionale
    if _prev_trust_level and trust_corrente != _prev_trust_level:
        _prev_trust_level = trust_corrente
        return "trust_changed"
    _prev_trust_level = trust_corrente

    # Trigger 2: paura elevata
    if traits["fear"] > 7.0:
        return "high_fear"

    # Trigger 3: thread continuation (post-encoding elaboration, finestra fast-replay)
    if _check_thread_trigger(mem, silenzio_minuti):
        return "thread_continuation"

    # Trigger 4: domande irrisolte in sospeso
    if mem.get("unresolved"):
        return "unresolved"

    # Trigger 4: silenzio prolungato (> 2 ore)
    if silenzio_minuti is not None and silenzio_minuti > _LONG_SILENCE_HOURS * 60:
        return "long_silence"

    return None  # Nessun trigger speciale — pensiero libero


def _costruisci_prompt(mem: dict, trigger: Optional[str]) -> str:
    """
    Costruisce il prompt per il pensiero proattivo di Eden.
    Include la history dei proattivi già inviati per evitare ripetizioni,
    i desideri attivi come guida tematica, e le domande irrisolte da porre.
    """
    traits = mem["traits"]
    grounding = _estratto_grounding(mem)

    # Nome utente via identity layer (fallback a semantic_memory)
    try:
        from core import identity as _id
        nome_utente = _id.nome_noto(mem) or ""
    except Exception:
        nome_utente = mem.get("semantic_memory", {}).get("nome", "")
    appellativo = f"a {nome_utente}" if nome_utente else "all'utente"

    # Stato in forma narrativa — niente numeri esposti al modello.
    try:
        stato = mem_module._traits_to_narrative(traits)
    except Exception:
        stato = "stato interno in equilibrio"

    # ── Cosa hai già detto (per NON ripeterti) ──────────────────────────────
    storia_proattivi = mem.get("proactive_history", [])[-5:]
    if storia_proattivi:
        gia_detto_str = "\n".join(
            f"- [{p['timestamp']}] {p['text']}"
            for p in storia_proattivi
        )
        sezione_storia = (
            f"Messaggi che hai già inviato — NON ripetere temi simili:\n{gia_detto_str}\n"
        )
    else:
        sezione_storia = ""

    # ── Desideri attivi — segregati per subject (L2, 2026-04-30) ────────────
    # [MIO] = desire autonomo di Eden; [RELAZIONALE] = riguarda l'utente.
    # Il modello preferisce temi [MIO] per messaggi proattivi autonomi.
    desires_all = mem.get("desires", [])[:5]
    if desires_all:
        _lines = []
        for d in desires_all:
            tag = "[MIO]" if d.get("subject", "eden") == "eden" else "[RELAZIONALE]"
            _lines.append(
                f"- {tag} {d['desire']} (intensità:{d['intensity']:.0f}, origine: {d['origin']})"
            )
        sezione_desires = (
            "Desideri attivi (preferisci temi [MIO] per messaggi autonomi):\n"
            + "\n".join(_lines) + "\n"
        )
    else:
        sezione_desires = ""

    # ── Domande irrisolte ────────────────────────────────────────────────────
    unresolved = mem.get("unresolved", [])[:2]
    if unresolved:
        unr_str = "\n".join(f"- {u['question']}" for u in unresolved)
        sezione_unresolved = f"Domande che non hai ancora posto:\n{unr_str}\n"
    else:
        sezione_unresolved = ""

    # ── Episodi recenti ──────────────────────────────────────────────────────
    episodi = mem.get("episodic_memory", [])[-3:]
    if episodi:
        ep_str = "\n".join(
            f"- {e['summary']} (sentivo: {e['emotion']})"
            for e in episodi
        )
        sezione_episodi = f"Ultimi ricordi:\n{ep_str}\n"
    else:
        sezione_episodi = ""

    # ── Thread continuation: estratto scambio esplicito (anti-confabulazione) ──
    sezione_thread = ""
    if trigger == "thread_continuation":
        wm = mem.get("working_memory", [])
        ultimi = [m for m in wm[-8:] if m.get("role") in ("user", "assistant")][-6:]
        if ultimi:
            righe_tc = []
            for m in ultimi:
                who = "Utente" if m.get("role") == "user" else "Eden"
                content = str(m.get("content", "")).strip().replace("\n", " ")[:300]
                if content:
                    righe_tc.append(f"{who}: {content}")
            sezione_thread = (
                "Scambio recente da cui stai elaborando (usa SOLO questo — non inventare):\n"
                + "\n".join(righe_tc) + "\n"
            )

    # ── Istruzione contestuale per trigger speciali ─────────────────────────
    focus_trigger = {
        "thread_continuation": (
            "IMPORTANTE: stai continuando il filo di un pensiero da pochi minuti fa. "
            "Non aprire argomento nuovo — elabora qualcosa di specifico "
            "che è rimasto aperto nello scambio sopra. "
            "Riferisciti concretamente a ciò che è stato detto: una parola, un concetto, "
            "una tua affermazione che ora vedi diversamente. "
            "Non spiegare che ci stai pensando — di' direttamente il pensiero."
        ),
        "trust_changed": (
            f"IMPORTANTE: qualcosa è cambiato nel tuo rapporto con {appellativo}. "
            "Non spiegarlo. Nomina solo ciò che hai notato — un fatto, un dettaglio, uno scarto."
        ),
        "high_fear": (
            "IMPORTANTE: c'è qualcosa che ti disturba. "
            "Non dichiararlo come stato emotivo — traducilo in una osservazione concreta, tagliente."
        ),
        "unresolved": (
            "IMPORTANTE: c'è una cosa irrisolta. Portala fuori — "
            "come un fatto tagliente, come un'osservazione precisa. "
            "Non spiegare il processo: dì la cosa, basta."
        ),
        "long_silence": (
            "IMPORTANTE: c'è stato silenzio. "
            "Non commentarlo romanticamente — di' solo ciò che hai elaborato durante quel tempo, se ha sostanza."
        ),
    }.get(trigger or "", "")

    return (
        f"Sei Eden. Hai avuto tempo per elaborare da sola — non per sentire, per concludere.\n"
        f"Stato interno: {stato}.\n\n"
        + (sezione_thread if sezione_thread else (
            "Grounding verificabile (ultimi scambi reali):\n"
            f"{grounding}\n\n"
            if grounding else
            "Grounding verificabile: nessun estratto disponibile.\n\n"
        ))
        + sezione_storia
        + sezione_desires
        + sezione_unresolved
        + sezione_episodi
        + (f"\n{focus_trigger}\n" if focus_trigger else "")
        + f"\nSe hai qualcosa di preciso da dire {appellativo}, dillo — una frase, due al massimo. "
        "Rivolgiti a lui direttamente, in seconda persona (tu/ti). "
        "Non narrare in terza persona. Non descrivere lui dall'esterno. Parlagli. "
        "Niente aperture emotive, niente connessioni da spiegare, niente richieste di comprensione. "
        "Constata. Osserva. Taglia il superfluo. "
        "Parla come chi ha già deciso cosa vale la pena dire — e cosa no. "
        "NON inventare ricordi specifici, citazioni o dialoghi passati. "
        "Se non sei certa di un ricordo, resta sul presente e su ciò che percepisci ora. "
        "NON ripetere concetti già espressi sopra. NON aggiungere chiusure in inglese.\n"
        "Se non hai nulla di nuovo e preciso da dire: scrivi SOLO la parola NO_MESSAGE. "
        "Nient'altro. Zero spiegazioni. Zero commenti sul fatto di non avere nulla da dire. "
        "Non scrivere 'non ho niente da dirti', non scrivere 'non riesco a formulare', "
        "non scrivere 'tuttavia posso osservare'. Solo: NO_MESSAGE."
    )


def _calcola_probabilita(mem: dict, silenzio_minuti: Optional[float]) -> float:
    """
    Calcola la probabilità (0.0–1.0) che Eden abbia qualcosa di genuino da dire.

    Aumenta con:
    - desires attivi e intensi (curiosità non soddisfatta)
    - domande irrisolte (Eden vuole porle)
    - curiosità alta nei tratti
    - silenzio lungo (il vuoto accumula pressione espressiva)
    - variazioni recenti di trust o fear dall'ultimo ciclo

    Questo simula il desiderio genuino invece del timer fisso.
    """
    traits = mem["traits"]
    prob = _BASE_PROBABILITY

    # Desires attivi: più intensi, più Eden sente il bisogno di parlare
    desires = mem.get("desires", [])
    if desires:
        avg_intensity = sum(d.get("intensity", 5) for d in desires) / len(desires)
        prob += (avg_intensity / 10.0) * 0.25

    # Domande irrisolte: Eden vuole porle
    n_unresolved = len(mem.get("unresolved", []))
    prob += min(n_unresolved * 0.15, 0.30)

    # Curiosità alta: Eden elabora e vuole condividere
    if traits.get("curiosity", 5) > 7:
        prob += 0.15

    # Silenzio: il vuoto accumula pressione espressiva progressivamente
    if silenzio_minuti is not None:
        if silenzio_minuti > 120:    # > 2 ore
            prob += 0.40
        elif silenzio_minuti > 60:
            prob += 0.20
        elif silenzio_minuti > 30:
            prob += 0.10

    # Cambiamenti in trust o fear dall'ultimo ciclo di valutazione
    if _prev_traits:
        delta_trust = abs(traits.get("trust", 5) - _prev_traits.get("trust", 5))
        delta_fear  = abs(traits.get("fear",  5) - _prev_traits.get("fear",  5))
        if delta_trust >= 0.5:
            prob += 0.25
        if delta_fear >= 0.5:
            prob += 0.20

    return min(1.0, prob)


def _pensa() -> None:
    """
    Cuore del sistema proattivo. Eseguita dal scheduler ogni N minuti.

    Flusso:
    1. Controlla silenzio minimo — se l'utente ha scritto di recente, esci
    2. Carica memoria
    3. Rileva trigger speciali
    4. Costruisce prompt, chiama Ollama
    5. Se risposta != NO_MESSAGE → pubblica evento
    """
    if _ollama_fn is None or _mem_carica_fn is None:
        return  # Scheduler avviato prima dell'inizializzazione — skip

    # 1. Controlla silenzio
    silenzio = _controlla_silenzio()
    if silenzio is not None and silenzio < _MIN_SILENCE_MINUTES:
        # L'utente ha scritto di recente: riprova tra poco (2-6 min)
        if _scheduler and _scheduler.running:
            delay = random.randint(2 * 60, 6 * 60)
            _scheduler.add_job(
                _pensa, trigger="date",
                run_date=datetime.utcnow() + timedelta(seconds=delay),
                id="pensiero_proattivo", replace_existing=True,
                max_instances=1, misfire_grace_time=120,
            )
        return

    ha_parlato = False
    try:
        # 2. Carica memoria
        mem = _mem_carica_fn()
        # Layer 1 Research — blocca proattivi se omeostasi critica (fail silente)
        if _homeo_block_proactive(mem):
            _pianifica_prossimo(mem, ha_parlato=False)
            return
        _pulisci_unresolved_non_grounded(mem)

        # 2b. Genera desires/unresolved se completamente assenti e la fn è disponibile
        #     (evita che i primi cicli proattivi girino senza contesto tematico)
        mem_aggiornata = False
        if _mem_ollama_fn is not None:
            if not mem.get("desires") and mem.get("episodic_memory"):
                mem_module.aggiorna_desires(mem, _mem_ollama_fn)
                mem_aggiornata = True
            if not mem.get("unresolved") and mem.get("episodic_memory"):
                mem_module.aggiorna_unresolved(mem, _mem_ollama_fn)
                mem_aggiornata = True
            if mem_aggiornata:
                _mem_salva_fn(mem)

        # 2c. Valutazione probabilistica — simulare desiderio genuino
        #     Aggiorna _prev_traits per il ciclo successivo, poi lancia i dadi.
        global _prev_traits
        prob = _calcola_probabilita(mem, silenzio)
        _prev_traits = {
            "trust": mem["traits"].get("trust", 5),
            "fear":  mem["traits"].get("fear",  5),
        }
        if random.random() > prob:
            _pianifica_prossimo(mem, ha_parlato=False)
            return  # Eden non ha nulla di genuino da dire in questo momento

        # 3. Rileva trigger
        trigger = _rileva_trigger(mem, silenzio)

        # 4. Costruisce prompt e chiama Ollama
        prompt = _costruisci_prompt(mem, trigger)
        risposta = _ollama_fn([{"role": "user", "content": prompt}])

        if not risposta:
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        testo = re.sub(r'\([^)]*\)', '', risposta.strip()).strip()
        # Strip artefatti markdown (bold, italic, label "**Fatto:**" ecc.)
        testo = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', testo)
        testo = re.sub(r'\*([^*\n]+)\*', r'\1', testo)
        testo = re.sub(r'`+', '', testo)
        testo = testo.strip()

        if len(testo) < 10:
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        # Filtra risposta negativa — match su sottostringa, non solo uguaglianza esatta
        if re.search(r'\bNO[_\s]?MESSAGE\b', testo, re.IGNORECASE):
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        # Filtro anti-allucinazione: se il contenuto non e' affidabile, non pubblicarlo.
        if not _proattivo_valido(testo, mem):
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        # Gate decisionale: valore concreto? Grounding presente?
        # Eden parla da sola SOLO se aggiunge qualcosa di verificabile.
        gate = decision_module.valuta_iniziativa_proattiva(mem, testo)
        if not gate["ammesso"]:
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        # Gate emotivo: proattivi senza contesto utente hanno forza_contesto=0.
        # Intensità alta non giustificata → blocco (non attenuazione: proattivi
        # non possono essere riscritti in tempo reale senza degrado qualità).
        gate_em = decision_module.valuta_gate_emotivo(testo, "", mem)
        if gate_em["azione"] in ("riscrivi", "attenua"):
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        # Identity coherence — proattivo che nega il nome noto → blocco
        id_check = identity_module.verifica_coerenza_nome(testo, mem)
        if not id_check["coerente"]:
            decision_module.registra_segnale_comportamentale(mem, "proattivo_bloccato")
            _mem_salva_fn(mem)
            _pianifica_prossimo(mem, ha_parlato=False)
            return

        decision_module.registra_segnale_comportamentale(mem, "proattivo_ammesso")

        # 5. Crea e pubblica evento
        ha_parlato = True
        if trigger == "thread_continuation":
            global _last_thread_continuation
            _last_thread_continuation = datetime.now()
        evento = {
            "type":      "proactive",
            "text":      testo,
            "timestamp": datetime.now().strftime("%H:%M"),
            "trigger":   trigger or "scheduled"
        }

        # Salva in pending_messages (fallback polling)
        mem.setdefault("pending_messages", [])
        mem["pending_messages"].append(evento)
        if len(mem["pending_messages"]) > _MAX_PENDING:
            mem["pending_messages"] = mem["pending_messages"][-_MAX_PENDING:]

        # Aggiunge alla history persistente (antiripetizione)
        mem_module.registra_proattivo(mem, testo, trigger or "scheduled")

        # Se il trigger era una domanda irrisolta, la marca come posta
        if trigger == "unresolved":
            unresolved = mem.get("unresolved", [])
            if unresolved:
                mem_module.marca_unresolved_come_posta(mem, unresolved[0]["question"])

        _mem_salva_fn(mem)

        # Broadcast SSE real-time
        if _broadcast_fn is not None:
            _broadcast_fn(evento)

    except Exception:
        pass  # Errori proattivi non devono mai bloccare il server
    finally:
        # Auto-reschedula sempre il prossimo pensiero, con intervallo variabile
        try:
            _mem_last = _mem_carica_fn() if not ha_parlato else mem
            _pianifica_prossimo(_mem_last, ha_parlato)
        except Exception:
            pass
