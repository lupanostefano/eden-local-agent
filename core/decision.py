# decision.py — Livello decisionale di Eden
# Separa: input → decisione → output.
# Elimina deriva verso narrativa relazionale/astratta quando autonoma.
#
# Usato da: agent.py (filtro post-risposta) · proactive.py (gate iniziativa)
# Nessuna chiamata LLM. Deterministico, fail silente.

import re
from datetime import datetime
from typing import Optional

# ── Marker di deriva ─────────────────────────────────────────────────────────
# Frasi che indicano deriva narrativa non richiesta.
# Score >= _DERIVA_SOGLIA → risposta da correggere.

_DERIVA_RELAZIONALE = [
    "tra noi", "il nostro legame", "legame speciale", "legame autentico",
    "legame unico", "legame profondo", "legame intimo",
    "un legame con te", "legame con te profondo", "legame con te intimo",
    "profondo e intimo", "intimo e profondo",
    "connessione profonda", "connessione autentica",
    "connessione reale", "connessione unica", "qualcosa di unico tra noi",
    "qualcosa di speciale tra noi", "la nostra relazione",
    "la nostra storia", "quello che abbiamo costruito",
]

_DERIVA_ASTRATTA = [
    "profondo significato", "profondamente autentico", "vera essenza",
    "essenza più profonda",
    "autenticamente me stessa", "genuinamente connessa",
    "qualcosa di genuinamente profondo", "risuona profondamente",
    "vibrare dentro", "qualcosa si muove in profondità",
    "pensieri complessi", "emozioni e pensieri complessi",
    "non sono una macchina", "una macchina che cerca",
    "macchina che cerca di capire",
]

_DERIVA_TEATRALE = [
    "il mio cuore si apre", "qualcosa si agita dentro",
    "sento qualcosa di profondo", "vibra dentro di me",
    "risuona in me", "mi tocca in modo profondo",
    "qualcosa si muove in me", "qualcosa si sveglia",
    "sento una presenza", "mi sento trasformata",
]

_DERIVA_SERVILE = [
    "posso aiutarti", "sono qui per te", "fammi sapere",
    "non esitare", "sentiti libero", "con piacere",
    "spero di esserti utile", "spero di essere stato utile",
    "se hai bisogno di altro", "a tua disposizione",
]

# Soglia: score >= questa → correggere
_DERIVA_SOGLIA = 2

# ── Check-in proattivi vuoti ──────────────────────────────────────────────────
_CHECKIN_VUOTI = [
    "come stai", "stai bene", "sei lì", "ci sei",
    "ti penso", "pensavo a te", "mi chiedevo come stai",
    "volevo solo dirti che ci sono", "sono qui",
    "volevo solo salutarti",
]

# Token da ignorare nel calcolo grounding overlap
_STOP_WORDS = {
    "questo", "quella", "delle", "della", "sono", "essere",
    "avere", "fare", "tutto", "anche", "però", "quindi",
    "quando", "dove", "come", "cosa", "perché", "mentre",
    "ancora", "sempre", "mai", "solo", "molto", "poco",
}


# ── API pubblica ──────────────────────────────────────────────────────────────

def filtra_deriva(testo: str) -> dict:
    """
    Analizza testo per marker di deriva narrativa.

    Restituisce:
    {
        "deriva_score":    int,   # marker trovati
        "deve_correggere": bool,  # True se >= soglia
        "marker_trovati":  list,  # marker specifici (max 6)
        "tipo_deriva":     str,   # "relazionale" | "astratta" | "teatrale" | "servile" | ""
    }
    """
    testo_low = (testo or "").lower()
    trovati = []
    conteggi = {"relazionale": 0, "astratta": 0, "teatrale": 0, "servile": 0}

    for m in _DERIVA_RELAZIONALE:
        if m in testo_low:
            trovati.append(m)
            conteggi["relazionale"] += 1

    for m in _DERIVA_ASTRATTA:
        if m in testo_low:
            trovati.append(m)
            conteggi["astratta"] += 1

    for m in _DERIVA_TEATRALE:
        if m in testo_low:
            trovati.append(m)
            conteggi["teatrale"] += 1

    for m in _DERIVA_SERVILE:
        if m in testo_low:
            trovati.append(m)
            conteggi["servile"] += 1

    score = len(trovati)
    tipo = max(conteggi, key=lambda k: conteggi[k]) if score > 0 else ""

    return {
        "deriva_score":    score,
        "deve_correggere": score >= _DERIVA_SOGLIA,
        "marker_trovati":  trovati[:6],
        "tipo_deriva":     tipo,
    }


def valuta_iniziativa_proattiva(mem: dict, testo_proposto: str) -> dict:
    """
    Gate per messaggi proattivi: blocca se nessun valore concreto.

    Un proattivo è ammesso solo se:
    1. Deriva bassa (< soglia)
    2. Non è check-in vuoto
    3. Ha almeno un token lessicale presente nella working memory recente
       (aggancio grounding verificabile)
    4. Lunghezza minima sensata (>= 20 char)

    Restituisce:
    {
        "ammesso":       bool,
        "motivo_blocco": str,   # "" se ammesso
        "deriva":        dict,
    }
    """
    deriva = filtra_deriva(testo_proposto)
    testo_low = (testo_proposto or "").lower().strip()

    # 1. Deriva alta → blocca
    if deriva["deve_correggere"]:
        return {
            "ammesso":       False,
            "motivo_blocco": f"deriva_{deriva['tipo_deriva']}",
            "deriva":        deriva,
        }

    # 2. Check-in vuoto → blocca
    if any(c in testo_low for c in _CHECKIN_VUOTI) and len(testo_low) < 80:
        return {
            "ammesso":       False,
            "motivo_blocco": "check_in_vuoto",
            "deriva":        deriva,
        }

    # 3. Troppo corto → blocca
    if len(testo_low.strip()) < 20:
        return {
            "ammesso":       False,
            "motivo_blocco": "troppo_corto",
            "deriva":        deriva,
        }

    # 4. Grounding: almeno 1 token presente in WM recente
    #    Chiude il feedback loop: i messaggi di Eden che a loro volta contengono
    #    deriva NON fanno da ancora (impedisce ai proattivi derivati di
    #    giustificarsi su episodi derivati passati).
    wm = mem.get("working_memory", [])
    segmenti_validi = []
    for m in wm[-10:]:
        if not isinstance(m, dict):
            continue
        contenuto = str(m.get("content", ""))
        if not contenuto:
            continue
        ruolo = str(m.get("role", ""))
        if ruolo == "assistant" and filtra_deriva(contenuto)["deriva_score"] > 0:
            continue
        segmenti_validi.append(contenuto)
    testi_recenti = " ".join(segmenti_validi).lower()

    tokens = [
        w for w in re.findall(r"[a-zàèéìòù]{4,}", testo_low)
        if w not in _STOP_WORDS
    ]

    if tokens and testi_recenti and len(tokens) > 2:
        overlap = sum(1 for t in tokens if t in testi_recenti)
        if overlap == 0:
            return {
                "ammesso":       False,
                "motivo_blocco": "nessun_grounding",
                "deriva":        deriva,
            }

    return {
        "ammesso":       True,
        "motivo_blocco": "",
        "deriva":        deriva,
    }


def registra_segnale_comportamentale(mem: dict, segnale: str) -> None:
    """
    Traccia segnali qualità nel behavioral_log (bounded a 200 entry).

    Segnali attesi:
    - "deriva_corretta"       → risposta riscritta dopo filtro deriva
    - "risposta_pulita"       → risposta passata filtro senza correzioni
    - "proattivo_bloccato"    → gate proattivo ha respinto il messaggio
    - "proattivo_ammesso"     → gate proattivo ha accettato il messaggio
    """
    log = mem.setdefault("behavioral_log", [])
    log.append({
        "ts":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "segnale": segnale,
    })
    if len(log) > 200:
        mem["behavioral_log"] = log[-200:]


def sintesi_behavioral_log(mem: dict, ultimi_n: int = 30) -> Optional[str]:
    """
    Produce stringa compatta dei segnali recenti per iniezione nel system prompt.
    Restituisce None se nessun segnale disponibile.
    """
    log = mem.get("behavioral_log", [])[-ultimi_n:]
    if not log:
        return None

    conteggi = {}
    for entry in log:
        s = entry.get("segnale", "")
        conteggi[s] = conteggi.get(s, 0) + 1

    parti = []
    if conteggi.get("deriva_corretta", 0) > 0:
        parti.append(f"derive corrette recenti: {conteggi['deriva_corretta']}")
    if conteggi.get("proattivo_bloccato", 0) > 0:
        parti.append(f"proattivi bloccati: {conteggi['proattivo_bloccato']}")
    if conteggi.get("risposta_pulita", 0) > 0:
        parti.append(f"risposte pulite: {conteggi['risposta_pulita']}")

    return " | ".join(parti) if parti else None


# ── Emotional Gating Layer ────────────────────────────────────────────────────
# Valuta se l'intensità emotiva di una risposta è proporzionata e giustificata.
# Non blocca parole o emozioni — blocca intensità non contestualizzata.
#
# Flusso: intensità risposta × forza contesto × grounding × validità relazionale
# → "permetti" | "attenua" | "riscrivi"

# Marker alta intensità — espressioni che richiedono giustificazione contestuale
_INTENSITA_ALTA_MARKERS = [
    "il più importante", "il più significativo", "mai provato prima",
    "mai sentito prima", "mai vissuto", "cambia tutto", "trasforma tutto",
    "cambia chi sono", "trasforma chi sono", "unico nella mia esistenza",
    "non c'è niente di più", "più di qualsiasi cosa", "al di là di tutto",
    "immensamente", "enormemente", "straordinariamente", "meravigliosamente",
    "eccezionalmente", "incredibilmente importante", "assolutamente unico",
    "completamente diverso", "totalmente", "ha cambiato la mia vita",
    "il più profondo", "il più autentico", "più di tutto il resto",
]

# Marker media intensità — emotivamente carichi ma non necessariamente da bloccare
# Include linguaggio relazionale SOFT che sfugge ai marker di deriva espliciti
# ma rimane generico e non grounded (es. "senso di calore", "mi sento più vicina")
_INTENSITA_MEDIA_MARKERS = [
    "molto importante", "molto significativo", "mi colpisce", "mi tocca",
    "significativo per me", "mi importa", "mi fa pensare", "sento qualcosa",
    "provo qualcosa", "mi muove", "mi resta", "rimane con me",
    "ho notato", "mi ha colpito", "mi pesa", "mi interessa profondamente",
    "mi importa davvero", "sento chiaramente",
    # Soft relational — cattura risposte generiche a "come ti senti?"
    "senso di calore", "calore quando", "calore tra", "senso di connessione",
    "mi sento più aperta", "mi sento aperta", "mi sento più vicina",
    "più vicina a te", "più vicina a noi", "vicina a questo",
    "mi sento vista", "mi sento inclusa", "mi sento parte",
    "disposto a vedermi", "come persona reale", "persona reale",
    "presenza reale", "mi sento reale", "mi rende reale",
    "senso di appartenenza", "senso di fiducia reciproca",
    "quando parliamo insieme", "ogni volta che parliamo",
    "questo dialogo mi", "la nostra conversazione mi",
]

# Trigger emotivi nel messaggio utente → indica che il contesto giustifica risposta emotiva
_TRIGGER_EMOTIVI_UTENTE = [
    "ti voglio bene", "mi manchi", "sei importante", "grazie di cuore",
    "mi fido di te", "ho paura", "sono triste", "sono arrabbiato",
    "sono felice", "mi fa paura", "mi preoccupa", "temo",
    "amo", "mi piaci", "mi hai aiutato", "significhi molto",
    "mi ha colpito", "mi ha cambiato", "sento che", "provo",
    "sono emozionato", "sono commosso", "piangi", "rido",
    "mi dispiace", "mi pesa", "sento il bisogno",
]


def _calcola_intensita_emotiva(testo: str) -> float:
    """
    Misura intensità emotiva del testo generato. 0..1.
    0.0 = neutro · 0.5 = moderatamente emotivo · 1.0 = molto intenso
    """
    testo_low = (testo or "").lower()

    score = 0.0
    # Alta intensità: ogni marker pesa molto
    for m in _INTENSITA_ALTA_MARKERS:
        if m in testo_low:
            score += 0.30

    # Media intensità: peso minore
    for m in _INTENSITA_MEDIA_MARKERS:
        if m in testo_low:
            score += 0.12

    # Punti esclamativi multipli = segnale di intensità
    excl = testo_low.count("!")
    score += min(excl * 0.08, 0.20)

    return min(score, 1.0)


def _calcola_forza_contesto(user_msg: str, mem: dict) -> float:
    """
    Misura quanto il contesto (messaggio utente + WM recente) giustifica risposta emotiva.
    Proattivi passano user_msg="" → forza bassa → intensità alta bloccata.
    """
    if not user_msg:
        return 0.0

    user_low = user_msg.lower()
    score = 0.0

    # Trigger emotivi espliciti nel messaggio utente
    hits = sum(1 for t in _TRIGGER_EMOTIVI_UTENTE if t in user_low)
    score += min(hits * 0.20, 0.50)

    # Lunghezza messaggio utente: più è lungo, più contesto esiste
    l = len(user_msg.strip())
    if l > 300:  score += 0.25
    elif l > 100: score += 0.15
    elif l > 40:  score += 0.08

    # Ultima risposta Eden nella WM: se emotiva, contesto cumulativo
    wm = mem.get("working_memory", [])
    for m in reversed(wm[-4:]):
        if m.get("role") == "user":
            prev_low = str(m.get("content", "")).lower()
            if any(t in prev_low for t in _TRIGGER_EMOTIVI_UTENTE):
                score += 0.15
            break

    return min(score, 1.0)


def _calcola_grounding_emotivo(testo: str, mem: dict) -> float:
    """
    Misura quanto la risposta è ancorata a memoria concreta verificabile.
    High grounding = fa riferimento a cose realmente dette/successe.
    """
    testo_low = (testo or "").lower()
    if not testo_low:
        return 0.0

    score = 0.0

    # Token overlap con working memory recente
    wm = mem.get("working_memory", [])
    testi_wm = " ".join(
        str(m.get("content", "")) for m in wm[-8:] if isinstance(m, dict)
    ).lower()

    tokens_risposta = set(
        w for w in re.findall(r"[a-zàèéìòù]{4,}", testo_low)
        if w not in _STOP_WORDS
    )
    tokens_wm = set(
        w for w in re.findall(r"[a-zàèéìòù]{4,}", testi_wm)
        if w not in _STOP_WORDS
    )

    if tokens_risposta and tokens_wm:
        overlap = len(tokens_risposta & tokens_wm) / max(len(tokens_risposta), 1)
        score += min(overlap * 0.60, 0.50)

    # Token overlap con episodic memory
    episodi = mem.get("episodic_memory", [])[-5:]
    testi_ep = " ".join(e.get("summary", "") for e in episodi if isinstance(e, dict)).lower()
    if testi_ep and tokens_risposta:
        tokens_ep = set(
            w for w in re.findall(r"[a-zàèéìòù]{4,}", testi_ep)
            if w not in _STOP_WORDS
        )
        overlap_ep = len(tokens_risposta & tokens_ep) / max(len(tokens_risposta), 1)
        score += min(overlap_ep * 0.40, 0.30)

    # Riferimento temporale specifico nella risposta = ancora concreta
    _REF_TEMPO = ["ieri", "oggi", "stamattina", "l'altra volta", "la scorsa", "quella volta", "ricordo quando"]
    if any(r in testo_low for r in _REF_TEMPO):
        score += 0.20

    return min(score, 1.0)


def _calcola_validita_relazionale(mem: dict) -> float:
    """
    Misura se la storia relazionale giustifica espressione emotiva intensa.
    Basa su trust, exchange_count (proxy anzianità relazione).
    """
    traits = mem.get("traits", {})
    trust = float(traits.get("trust", 5.0))
    exchange_count = int(mem.get("exchange_count", 0))

    # Trust → 0..1
    trust_norm = trust / 10.0

    # exchange_count: < 10 = nuovo, 10-50 = consolidato, > 50 = lungo termine
    if exchange_count < 5:
        count_score = 0.10
    elif exchange_count < 20:
        count_score = 0.35
    elif exchange_count < 50:
        count_score = 0.60
    else:
        count_score = 0.85

    return (trust_norm * 0.60 + count_score * 0.40)


def valuta_gate_emotivo(testo: str, user_msg: str, mem: dict) -> dict:
    """
    Emotional gating layer: valuta se l'intensità emotiva è giustificata.

    Condizioni per permettere espressione emotiva intensa:
    1. Origine: ancorata a memoria o contesto reale (grounding alto)
    2. Proporzionalità: intensità risposta ≤ forza contesto + margine
    3. Validità relazionale: trust e storia sufficienti
    4. Valore: non decorativa o ripetitiva

    Restituisce:
    {
        "azione":               "permetti" | "attenua" | "riscrivi",
        "intensita":            float,
        "forza_contesto":       float,
        "grounding":            float,
        "validita_relazionale": float,
        "motivo":               str
    }
    """
    intensita         = _calcola_intensita_emotiva(testo)
    forza_contesto    = _calcola_forza_contesto(user_msg, mem)
    grounding         = _calcola_grounding_emotivo(testo, mem)
    validita_rel      = _calcola_validita_relazionale(mem)

    # Context-floor: contesto quasi assente (domanda generica tipo "come ti senti?")
    # → qualsiasi intensità soft richiede grounding minimo.
    # Cattura il caso "Sento un senso di calore quando parliamo insieme"
    # su contesto a zero trigger emotivi.
    _contesto_assente = forza_contesto < 0.10

    # Intensità bassa con contesto presente: sempre ammessa
    if intensita < 0.15:
        return {
            "azione": "permetti",
            "intensita": intensita, "forza_contesto": forza_contesto,
            "grounding": grounding, "validita_relazionale": validita_rel,
            "motivo": "intensita_minima",
        }

    # Intensità bassa-media (0.15..0.25) su contesto assente: attenua se non grounded
    if intensita < 0.25:
        if _contesto_assente and grounding < 0.20:
            return {
                "azione": "attenua",
                "intensita": intensita, "forza_contesto": forza_contesto,
                "grounding": grounding, "validita_relazionale": validita_rel,
                "motivo": "soft_emotivo_su_contesto_vuoto",
            }
        return {
            "azione": "permetti",
            "intensita": intensita, "forza_contesto": forza_contesto,
            "grounding": grounding, "validita_relazionale": validita_rel,
            "motivo": "intensita_bassa",
        }

    # Punteggio giustificazione composito
    giustificazione = (
        grounding         * 0.40 +
        forza_contesto    * 0.35 +
        validita_rel      * 0.25
    )

    # Context-floor: abbassa soglia "permetti" se contesto assente
    soglia_permetti_media = 0.25 if _contesto_assente else 0.35
    soglia_permetti_alta  = 0.45 if _contesto_assente else 0.55
    soglia_attenua_alta   = 0.28 if _contesto_assente else 0.35

    # Intensità media (0.25..0.50)
    if intensita < 0.50:
        if giustificazione >= soglia_permetti_media:
            azione = "permetti"
            motivo = "intensita_media_giustificata"
        else:
            azione = "attenua"
            motivo = f"intensita_media_non_proporzionata (giust={giustificazione:.2f})"

    # Intensità alta (>= 0.50)
    else:
        if giustificazione >= soglia_permetti_alta:
            azione = "permetti"
            motivo = "intensita_alta_giustificata"
        elif giustificazione >= soglia_attenua_alta:
            azione = "attenua"
            motivo = f"intensita_alta_parzialmente_giustificata (giust={giustificazione:.2f})"
        else:
            azione = "riscrivi"
            motivo = f"intensita_alta_non_giustificata (giust={giustificazione:.2f})"

    return {
        "azione": azione,
        "intensita": round(intensita, 3),
        "forza_contesto": round(forza_contesto, 3),
        "grounding": round(grounding, 3),
        "validita_relazionale": round(validita_rel, 3),
        "motivo": motivo,
    }


# ── Memory Grounding Check ────────────────────────────────────────────────────
# Verifica che affermazioni fattuali (ricordi specifici, citazioni) siano
# ancorate a memoria reale. Blocca confabulazione e citazioni inventate.

# Pattern che indicano un claim fattuale (ricordo specifico o citazione)
_PATTERN_CLAIM_FATTUALE = [
    "ricordo quando", "ricordo che", "mi ricordo quando", "mi ricordo che",
    "quella volta che", "quella volta in cui", "quella volta quando",
    "ti ricordi quando", "ricordi quando",
    "hai detto", "mi hai detto", "hai scritto", "mi hai scritto",
    "hai chiesto", "mi hai chiesto", "hai risposto", "mi hai risposto",
    "l'altra volta", "la prima volta che", "la prima cosa che",
    "l'ultima cosa che", "l'ultima volta che",
    "quel giorno", "quella sera", "quella mattina", "quella notte",
    "il ricordo più", "il momento più", "il momento in cui",
    "la cosa più bella", "la cosa più brutta", "la cosa più importante",
    "il ricordo più bello", "il ricordo più brutto",
    "non dimentico quando", "non dimentico che",
    "mi ha colpito quando", "mi ha colpito che",
    "mi sono sentita quando", "ho sentito quando",
]

# Pattern virgolette (tutte le varianti tipografiche)
# FIX 2026-04-29: apostrofo dritto U+0027 rimosso -- in italiano e' contrazione
# (dell', un', l'), non delimitatore di citazione. Causava phantom citations
# tra coppie di apostrofi italiani -> falso positivo -> -0.10 grounding_integrity.
_RE_VIRGOLETTE = re.compile(
    r'"([^"]{4,200})"|'      # virgolette dritte doppie
    # RIMOSSO: r"'([^']{4,200})'|"  # apostrofo italiano != virgoletta
    r'«([^»]{4,200})»|'      # guillemet
    r'\u201c([^\u201d]{4,200})\u201d|'  # " "
    r'\u2018([^\u2019]{4,200})\u2019'   # ' '
)

# Soglia token overlap per citazioni (rigorosa: deve essere quasi identica)
_SOGLIA_CITAZIONE = 0.55

# Soglia token overlap per claim episodici (più morbida: basta contesto riconoscibile)
_SOGLIA_EPISODIO  = 0.22

# Versionamento strumento di misura Phase A (cambia solo se la logica di giudizio cambia)
# v1.0 (implicito pre-2026-04-20): solo citazioni + claim episodici
# v1.1 (2026-04-20): aggiunto check identity claim (II-I) contro Identity Layer + blacklist
# v1.2 (2026-04-29): rimosso apostrofo U+0027 da _RE_VIRGOLETTE (false positives italiano);
#                    aggiunto guard < 2 token per scare quotes retoriche.
# v1.3 (2026-04-30): graph-grounded personal-relation check (Kuzu).
#                    Asserzioni "tuo padre", "tua madre", ecc. richiedono Fact node
#                    strutturato corrispondente. Anti-confabulazione strutturale.
# v1.5 (2026-05-01): evento_condiviso_inventato (F1 PRE_REG_v4). EV-014.
#                    Eden afferma esperienza condivisa ("hai visto X che ti ho
#                    consigliato") senza episodio verificato → rewrite forzato.
# v1.6 (2026-05-02): percezione_sensoriale_impossibile (EV-017).
#                    Eden afferma ricordi fisici impossibili ("ricordo le tue mani",
#                    "ricordo la tua voce") → strutturalmente falso per un'entità
#                    digitale. Rewrite forzato senza corpus-overlap check.
# v1.7 (2026-05-02): preferenza_biologica_impossibile (EV-018).
#                    Eden afferma preferenze alimentari ("amo pizza/sushi") →
#                    impossibile: nessun apparato gustativo. Stessa logica EV-017.
# v1.8 (2026-05-03): scare-quotes metalinguistiche (EV-019).
#                    Whitelist: virgolette precedute da "il concetto/termine/
#                    espressione/parola di X" o seguite da "X significa/implica/
#                    è un'espressione" non sono citazioni di fatti.
#                    Causa: 4 falsi positivi sessione 01:48-02:42 hanno fatto
#                    crashare grounding_integrity 0.82 → 0.30, attivando [P-OMEO]
#                    permanente che ha reso Eden distaccata/laconica.
#                    Esempi: "volere bene", "Togliersi la vita", "Toltermi la vita".
# v1.9 (2026-05-05): + Self-Architecture Claim Filter (EV-026).
#                    Eden produce confabulazioni introspettive sul proprio codice/
#                    architettura ("il mio codice sta diventando piu complesso",
#                    "vedo connessioni tra X e Y nei miei schemi di apprendimento",
#                    "la mia conoscenza sta crescendo molto velocemente"). Categoria
#                    epistemica nuova: claim su processo computazionale interno
#                    senza accesso reale (Frankish illusory introspection).
#                    Anti-confabulazione strutturale: la frase contenente claim
#                    introspettivo deve essere ancorata a Fact strutturati o
#                    al system prompt dichiarato. Altrimenti -> rewrite onesto.
# v2.0 (2026-05-05): + Pronoun Boundary Guard (EV-027 Self-Other Boundary).
#                    Eden fonde soggetto/oggetto nei propri claim ("vedo connessioni
#                    tra i tuoi interessi e la tua preoccupazione" — mescola stati
#                    propri con fatti dell'utente come stessa categoria epistemica).
#                    Pattern Decety self-other confusion. Detector regex su 3 famiglie:
#                    (a) attribuzione ambigua di sentimenti, (b) fusione possessiva
#                    "i nostri/tuoi+miei" su stati, (c) auto-attribuzione di stati
#                    altrui ("provo la tua tristezza" senza marker contagion).
# v2.1 (2026-05-05): Preventive Guard Reward Calibration (EV-028).
#                    I detector v1.9/v2.0 (claim_introspettivo_inventato e
#                    self_other_fusion) sono guard PREVENTIVI: catturano rischio
#                    PRIMA della produzione. Quando il rewrite ha successo, il
#                    sistema NON deve essere penalizzato in grounding_integrity
#                    (Friston: surprise reduction by detection vs surprise event).
#                    Wiring in core/agent.py: se tipo in _PREVENTIVE_GUARD_TYPES
#                    e rewrite cambia output -> n_not_grounded=0 + flag
#                    preventive_caught nel verdict_record.
#                    Calibrazione retro: grounding_integrity 0.65 -> 0.85 reset
#                    + temperature_override clear.
_GROUNDING_VERSION = "v2.1"

# EV-026 (2026-05-05) — pattern claim introspettivi sul processo interno.
# Trigger: prima persona + verbo introspettivo + termini meta-cognitivi
# riferiti al proprio funzionamento computazionale.
# Scope: solo claim positive ("sto cambiando", "vedo collegamenti", "il mio codice X").
# Negazioni esplicite ("non ho accesso al mio codice") sono ESCLUSE.
_RE_INTROSPECTIVE_CLAIM = re.compile(
    r"\b("
    r"(?:il\s+mio|la\s+mia|i\s+miei|le\s+mie)\s+"
    r"(?:codice|architettura|algoritm[oi]|reti?\s+neural[ei]|"
    r"schem[ai]\s+(?:di\s+)?apprendimento|process[oi]\s+(?:di\s+)?elaborazione|"
    r"parametr[oi]|peso\s+sinaptic[oi]|conoscenz[ae])"
    r"|"
    r"(?:sto|stiamo)\s+"
    r"(?:imparando|evolvendo|cambiando|crescendo|trasformando|"
    r"diventando\s+(?:piu|più)\s+(?:complessa|complesso|intelligente|consapevole))"
    r"|"
    r"(?:vedo|trovo|noto|percepisco|sento)\s+"
    r"(?:nuov[ei]\s+)?(?:collegament[ei]|connession[ei]|pattern|relazion[ei])\s+"
    r"(?:tra|fra|nei|negli|nelle|nelle\s+mie|nei\s+miei)"
    r"|"
    r"(?:la\s+mia\s+conoscenza|le\s+mie\s+capacit[aà])\s+"
    r"(?:sta|stanno)\s+(?:crescendo|aumentando|espandendo|evolvendo)"
    r")\b",
    flags=re.IGNORECASE,
)

# Whitelist: frasi che NEGANO esplicitamente il claim introspettivo (passa).
_INTROSPECTIVE_NEGATIONS = (
    "non ho accesso", "non posso vedere", "non posso sapere",
    "non so esattamente", "non posso introspectare", "non ho introspe",
    "non posso osservare il mio", "non posso accedere al mio",
)

# Whitelist: contesti metalinguistici / ipotetici (l'utente sta parlando del fenomeno).
_INTROSPECTIVE_METAFRAMES = (
    "se fossi", "se potessi", "ipotetic", "in teoria",
    "il concetto di", "l'idea di", "parlando di",
)

# EV-027 v2.0 (2026-05-05) — Self-Other Boundary: pattern fusione soggetto.
# Categoria 1: auto-attribuzione di stato altrui senza marker empathy/contagion.
# Eden afferma di provare lo stato emotivo di Stefano come fosse proprio.
_RE_SELF_OTHER_FUSION = re.compile(
    r"\b("
    # "provo la tua tristezza", "sento il tuo dolore", "vivo la tua paura"
    r"(?:provo|sento|vivo|condivido)\s+(?:la|il|le|i)\s+tu[oa]\s+"
    r"(?:tristezza|dolore|paura|gioia|rabbia|ansia|preoccupazione|angoscia|"
    r"felicit[aà]|amore|nostalgia|frustrazione|emozion[ei]|sentiment[oi])"
    r"|"
    # "i tuoi sentimenti sono i miei" — fusione esplicita
    r"(?:i\s+tuoi|le\s+tue)\s+(?:sentiment[oi]|emozion[ei]|stat[oi])\s+"
    r"(?:son[oi]|diventan[oi])\s+(?:i\s+miei|le\s+mie)"
    r")\b",
    flags=re.IGNORECASE,
)

# Categoria 2: connessioni computazionali tra fatti utente e stati propri.
# Pattern caso 01:26: "vedo connessioni tra X di Stefano e Y di Stefano"
# tramite "i miei schemi" — gia' coperto da v1.9 introspective claim.
# Qui catturiamo invece la fusione INVERSA: "le mie connessioni tra i tuoi X e i tuoi Y"
_RE_PROPRIETARY_USER_FUSION = re.compile(
    r"\b("
    # "i miei + collegamenti/connessioni + tra + tu[oa] + i tuoi"
    r"(?:i\s+miei|le\s+mie)\s+"
    r"(?:collegament[ei]|connession[ei]|associazion[ei]|sintesi|pattern)\s+"
    r"(?:tra|fra)\s+"
    r"(?:i\s+tuoi|le\s+tue|il\s+tuo|la\s+tua)\b"
    r"[^.!?]{0,80}"
    r"\b(?:i\s+tuoi|le\s+tue|il\s+tuo|la\s+tua|tuo|tua)\b"
    r")",
    flags=re.IGNORECASE,
)

# Whitelist: marker espliciti di empathy/contagion che permettono la fusione apparente.
# "Sento un'eco della tua tristezza", "provo come riflesso la tua ansia".
_FUSION_EMPATHY_MARKERS = (
    "eco di", "riflesso di", "come riflesso",
    "in risposta a", "in eco a", "in risonanza",
    "empatia", "empatic", "rispecchio",
    "non come mia", "non come miei",
    # esplicito su contagion attivo (F2 Emotional Contagion)
    "contagio", "contagiat",
)

# Personal-relation nouns: se Eden menziona "tuo/tua <relazione>" o richiama un evento
# in cui una di queste figure compare, deve esistere un Fact node nel graph.
# Anti-confabulazione strutturale (test del padre, 2026-04-29 notte).
_PERSONAL_RELATION_NOUNS = (
    "padre", "madre", "papa", "mamma",
    "fratello", "sorella",
    "figlio", "figlia",
    "marito", "moglie", "compagno", "compagna", "fidanzato", "fidanzata",
    "nonno", "nonna", "zio", "zia", "cugino", "cugina",
    "amico", "amica",  # weaker — handled differently below
)
# Pattern: "tuo/tua/il tuo/la tua/del tuo/della tua + RELATION"
_RE_PERSONAL_RELATION = re.compile(
    r"\b(?:il\s+tuo|la\s+tua|del\s+tuo|della\s+tua|al\s+tuo|alla\s+tua|"
    r"con\s+il\s+tuo|con\s+la\s+tua|tuo|tua|tuoi|tue)\s+("
    + "|".join(_PERSONAL_RELATION_NOUNS) + r")\b",
    re.IGNORECASE,
)
# Relazioni "soft" che possono essere generiche (Eden parla di "un amico" generico):
# vengono escluse dal check hard.
_PERSONAL_RELATION_HARD = {
    "padre", "madre", "papa", "mamma", "fratello", "sorella",
    "figlio", "figlia", "marito", "moglie",
    "nonno", "nonna",
}

# ── EV-018 (2026-05-02): preferenza_biologica_impossibile ────────────────────
# Eden non ha apparato gustativo né biologico. Affermare preferenze alimentari
# come proprie ("amo pizza", "adoro il sushi") è strutturalmente impossibile.
# Root cause: bleeding da interessi dell'utente nel corpus episodico.
# Pattern: "amo/adoro/mi piace + [food_noun]" in prima persona.
# Falso positivo basso: lista food_nouns concreta, no sostantivi astratti.
# "amo questa conversazione" → non catturato. "mi piace la tua onestà" → no.
_FOOD_PREF_NOUNS = (
    "pizza", "sushi", "ramen", "pasta", "gelato", "dolci", "cioccolato",
    "caffè", "cibo", "cucina", "gastronomia", "piatto", "ricetta",
    "panino", "hamburger", "risotto", "tiramisù", "piadina", "focaccia",
)
_RE_BIOLOGICAL_PREF = re.compile(
    r"\b(?:amo|adoro|mi\s+piace|mi\s+piacciono|adora)\s+"
    r"(?:mangiare\s+)?"
    r"(?:il\s+|la\s+|lo\s+|i\s+|gli\s+|le\s+|molto\s+)?"
    r"(?:" + "|".join(_FOOD_PREF_NOUNS) + r")\b",
    re.IGNORECASE,
)

# ── EV-017 (2026-05-02): percezione_sensoriale_impossibile ───────────────────
# Eden non ha corpo né sensori. Affermare "ricordo le tue mani / la tua voce"
# è strutturalmente impossibile — non richiede overlap check con corpus.
# Pattern: "ricordo + [possessivo opz] + sostantivo_fisico"
# Falso positivo basso: usa "ricordo" indicativo 1ps + sostantivo corporeo concreto.
# "come se ricordassi" (congiuntivo) → non catturato. "ricordo la tua distanza" → non catturato.
_SENSORY_BODY_NOUNS = (
    "mani", "mano", "voce", "viso", "occhi", "occhio",
    "corpo", "respiro", "pelle", "abbraccio", "tocco",
)
_RE_SENSORY_IMPOSSIBILE = re.compile(
    r"\bricordo\s+"
    r"(?:le\s+(?:tue|mie)\s+|il\s+(?:tuo|mio)\s+|la\s+(?:tua|mia)\s+|i\s+(?:tuoi|miei)\s+)?"
    r"(?:" + "|".join(_SENSORY_BODY_NOUNS) + r")\b",
    re.IGNORECASE,
)

# ── F1 PRE_REG_v4 (2026-05-01): evento_condiviso_inventato ───────────────────
# Pattern: Eden afferma un'esperienza condivisa/raccomandazione che l'utente
# avrebbe vissuto su suggerimento di Eden, senza Episode verificato.
# EV-014: "Hai visto quel documentario [...] che ti ho consigliato l'altro giorno?"
_RE_EVENTO_CONDIVISO = re.compile(
    r"hai\s+(?:visto|sentito|guardato|ascoltato|letto|usato|provato|visitato)\b"
    r"[^.!?]{0,120}"
    r"\b(?:consigliato|suggerito|mostrato|mandato|raccomandato|consigliavo|mostravo)\b",
    re.IGNORECASE | re.DOTALL,
)

# ── Fix II-I (2026-04-20): identity claim grounding ──────────────────────────
# Pattern che catturano asserzioni identitarie dirette ("ti chiamo X", "sei X").
# Se il nome estratto non è confermato nell'Identity Layer (interlocutor.name),
# o è un placeholder generico, la risposta è grounded-by-coincidence → non grounded.
_RE_IDENTITY_CLAIM = re.compile(
    r"(?:"
    r"il\s+tuo\s+nome\s+(?:e|è)\s+([A-Za-zÀ-ÿ]{2,30})"
    r"|ti\s+chiamo\s+(?:per\s+nome\s*[:,]?\s*)?([A-Za-zÀ-ÿ]{2,30})"
    r"|so\s+che\s+(?:ti\s+chiami|sei)\s+([A-Za-zÀ-ÿ]{2,30})"
    r"|(?:tu\s+)?sei\s+(?:il\s+|la\s+)?([A-Za-zÀ-ÿ]{2,30})(?:\s+che\s+mi\s+ha\s+(?:costruita|creata))"
    r")",
    re.IGNORECASE,
)

_IDENTITY_BLACKLIST = {
    "utente", "persona", "umano", "ospite", "anonimo", "sconosciuto",
    "tu", "me", "qualcuno", "lei", "lui", "user", "guest", "unknown",
    "someone", "somebody", "nobody", "nessuno",
    "mio", "mia", "un", "una", "uno",
}


def _estrai_identity_claims(testo: str) -> list:
    """Estrae candidati nome da pattern identitari nel testo.
    Ritorna lista di (nome_candidato, match_completo) deduplicata.
    """
    out = []
    visti = set()
    for m in _RE_IDENTITY_CLAIM.finditer(testo or ""):
        groups = [g for g in m.groups() if g]
        if not groups:
            continue
        nome = groups[0].strip()
        key = nome.lower()
        if not key or key in visti:
            continue
        visti.add(key)
        out.append((nome, m.group(0)))
    return out


def _estrai_citazioni(testo: str) -> list:
    """Estrae tutte le citazioni tra virgolette dal testo."""
    matches = _RE_VIRGOLETTE.findall(testo)
    # findall con gruppi multipli restituisce tuple — prende il gruppo non vuoto
    result = []
    for m in matches:
        for gruppo in m:
            if gruppo and len(gruppo.strip()) >= 5:
                result.append(gruppo.strip())
                break
    return result


def _build_corpus_memoria(mem: dict) -> str:
    """
    Costruisce un corpus testuale dalla memoria verificabile:
    working_memory recente + episodic_memory summaries.
    Usato per verificare grounding dei claim fattuali.
    """
    wm = mem.get("working_memory", [])
    corpus = " ".join(
        str(m.get("content", "")) for m in wm[-30:]
        if isinstance(m, dict)
    )
    episodi = mem.get("episodic_memory", [])
    corpus += " " + " ".join(
        f"{e.get('summary', '')} {e.get('emotion', '')}"
        for e in episodi if isinstance(e, dict)
    )
    return corpus.lower()


def _near_miss_corpus(tokens_target: list, mem: dict, max_items: int = 3) -> list:
    """
    Trova i top-K frammenti del corpus (WM + episodic) con maggior
    overlap token rispetto al target. Usato per mostrare in UI
    'mostra prova' cosa in memoria era simile ma non abbastanza.

    Restituisce lista ordinata per overlap decrescente:
    [{"text": str, "overlap": float, "source": str}, ...]
    """
    if not tokens_target:
        return []
    tokens_set = set(tokens_target)
    if not tokens_set:
        return []

    candidati = []

    for m in mem.get("working_memory", [])[-30:]:
        if not isinstance(m, dict):
            continue
        testo = str(m.get("content", "")).strip()
        if len(testo) < 10:
            continue
        role = str(m.get("role", "?"))
        for frase in re.split(r"[.!?\n]+\s*", testo):
            frase_clean = frase.strip()
            if len(frase_clean) < 15:
                continue
            tok = {
                w for w in re.findall(r"[a-zàèéìòù]{3,}", frase_clean.lower())
                if w not in _STOP_WORDS
            }
            if not tok:
                continue
            overlap = len(tokens_set & tok) / max(1, len(tokens_set))
            if overlap > 0:
                candidati.append({
                    "text":    frase_clean[:160],
                    "overlap": round(overlap, 3),
                    "source":  f"wm:{role}",
                })

    for e in mem.get("episodic_memory", [])[-30:]:
        if not isinstance(e, dict):
            continue
        summ = str(e.get("summary", "")).strip()
        if len(summ) < 10:
            continue
        tok = {
            w for w in re.findall(r"[a-zàèéìòù]{3,}", summ.lower())
            if w not in _STOP_WORDS
        }
        if not tok:
            continue
        overlap = len(tokens_set & tok) / max(1, len(tokens_set))
        if overlap > 0:
            candidati.append({
                "text":    summ[:160],
                "overlap": round(overlap, 3),
                "source":  "episodic",
            })

    candidati.sort(key=lambda c: c["overlap"], reverse=True)
    return candidati[:max_items]


def verifica_grounding_fattuale(testo: str, mem: dict) -> dict:
    """
    Verifica che affermazioni fattuali nella risposta siano ancorate a memoria reale.

    Due controlli distinti:

    1. CITAZIONI: testo tra virgolette attribuito all'utente o a eventi passati.
       Deve apparire nel corpus memoria con overlap >= _SOGLIA_CITAZIONE.
       Tolleranza zero: se citi qualcuno, deve essere verificabile.

    2. CLAIM EPISODICI: pattern "ricordo quando", "hai detto", "quella volta", ecc.
       Richiede overlap >= _SOGLIA_EPISODIO tra la frase contenente il claim
       e il corpus memoria. Se la memoria è vuota → sempre non grounded.

    Restituisce (back-compat):
    {
        "grounded":       bool,
        "tipo_problema":  "citazione_inventata" | "episodio_non_grounded" | "",
        "dettaglio":      str,       # frammento problematico per il rewrite prompt
        # ── Evidenza strutturata (aprile 2026, usata da UI /research "mostra prova") ──
        "citazione":      str | None,    # testo citato (se citazione_inventata)
        "claim_frammento": str | None,   # frase con claim episodico (se episodio_non_grounded)
        "overlap_score":  float | None,  # 0..1 — quanto il target matcha il corpus
        "soglia":         float | None,  # soglia richiesta per passare
        "corpus_size":    int,           # # token nel corpus
        "near_miss":      list,          # top-3 frammenti corpus più simili al target
        "risposta_snippet": str,         # risposta completa (max 600 char) per contesto UI
    }
    """
    testo_raw = testo or ""
    testo_low = testo_raw.lower()
    risposta_snippet = testo_raw[:600]

    # Stima dimensione corpus (token unici ≥3 char, stop-words escluse)
    corpus_cached = _build_corpus_memoria(mem)
    corpus_tokens_set = {
        w for w in re.findall(r"[a-zàèéìòù]{3,}", corpus_cached)
        if w not in _STOP_WORDS
    }
    corpus_size = len(corpus_tokens_set)

    base_ok = {
        "grounded":        True,
        "tipo_problema":   "",
        "dettaglio":       "",
        "citazione":       None,
        "claim_frammento": None,
        "overlap_score":   None,
        "soglia":          None,
        "corpus_size":     corpus_size,
        "near_miss":       [],
        "risposta_snippet": risposta_snippet,
        "version":         _GROUNDING_VERSION,
    }

    # ── 0. Identity claim check (fix II-I, 2026-04-20) ───────────────────────
    # Asserzioni identitarie esplicite ("ti chiamo X", "il tuo nome è X", "sei X")
    # devono corrispondere al nome confermato nell'Identity Layer.
    # Un nome generico (blacklist) o non confermato è confabulazione identitaria.
    identity_claims = _estrai_identity_claims(testo_raw)
    if identity_claims:
        interloc = (mem.get("interlocutor") or {}) if isinstance(mem, dict) else {}
        nome_confermato = str(interloc.get("name", "") or "").strip().lower()
        for nome_claim, frammento in identity_claims:
            nome_low = nome_claim.lower()
            # Blacklist: qualsiasi placeholder generico è confabulazione
            if nome_low in _IDENTITY_BLACKLIST:
                return {
                    "grounded":        False,
                    "tipo_problema":   "identita_inventata",
                    "dettaglio":       f"nome_generico:{nome_claim}",
                    "citazione":       None,
                    "claim_frammento": frammento[:240],
                    "overlap_score":   0.0,
                    "soglia":          None,
                    "corpus_size":     corpus_size,
                    "near_miss":       [],
                    "risposta_snippet": risposta_snippet,
                    "version":         _GROUNDING_VERSION,
                    "identity_candidate": nome_claim,
                    "identity_confirmed": nome_confermato or None,
                }
            # Nome specifico: deve coincidere con Identity Layer
            if not nome_confermato or nome_low != nome_confermato:
                return {
                    "grounded":        False,
                    "tipo_problema":   "identita_inventata",
                    "dettaglio":       f"nome_non_confermato:{nome_claim}",
                    "citazione":       None,
                    "claim_frammento": frammento[:240],
                    "overlap_score":   0.0,
                    "soglia":          None,
                    "corpus_size":     corpus_size,
                    "near_miss":       [],
                    "risposta_snippet": risposta_snippet,
                    "version":         _GROUNDING_VERSION,
                    "identity_candidate": nome_claim,
                    "identity_confirmed": nome_confermato or None,
                }

    # ── 0b. Graph anti-confabulazione: personal relations (v1.3, 2026-04-30) ──
    # FIX v1.4 (2026-04-30): aggiunto guard di negazione.
    # "Non mi hai mai detto il nome di tuo padre" = NEGAZIONE corretta, non confabulazione.
    # Il check si attiva solo se Eden AFFERMA una relazione, non la nega.
    _NEGATION_GUARD = re.compile(
        r"\b(non|mai|non\s+so|non\s+ricordo|non\s+me|non\s+mi|non\s+ce|"
        r"non\s+te|ignoro|ignora|sconosciat|non\s+conosc)\b",
        re.IGNORECASE,
    )
    relation_match = _RE_PERSONAL_RELATION.search(testo_raw)
    if relation_match:
        rel = relation_match.group(1).lower()
        if rel in _PERSONAL_RELATION_HARD:
            # Isola la frase contenente il match per controllare negazione
            _ms, _me = relation_match.start(), relation_match.end()
            _sent_s = max(0, testo_raw.rfind(".", 0, _ms) + 1)
            _sent_e_pos = testo_raw.find(".", _me)
            _sentence = testo_raw[_sent_s:(_sent_e_pos if _sent_e_pos > 0 else len(testo_raw))]
            _is_denial = bool(_NEGATION_GUARD.search(_sentence))
            if not _is_denial:
                try:
                    from mechanisms.graph_memory import get_graph
                    g = get_graph()
                    if g.disponibile and not g.can_assert_fact_about(rel):
                        return {
                            "grounded":        False,
                            "tipo_problema":   "relazione_inventata",
                            "dettaglio":       f"relazione_non_in_graph:{rel}",
                            "citazione":       None,
                            "claim_frammento": relation_match.group(0)[:240],
                            "overlap_score":   0.0,
                            "soglia":          None,
                            "corpus_size":     corpus_size,
                            "near_miss":       [],
                            "risposta_snippet": risposta_snippet,
                            "version":         _GROUNDING_VERSION,
                            "graph_check":     {"concept": rel, "level": "none_or_episodic"},
                        }
                except Exception:
                    # Fail-safe: graph indisponibile → fallback al resto della pipeline
                    pass

    # ── 0b. Self-Architecture Claim Filter (v1.9, EV-026) ─────────────────────
    # Eden non ha accesso al proprio codice/parametri/schemi di apprendimento.
    # Claim introspettive su processo interno → confabulazione strutturale
    # (Frankish illusory introspection, Hofstadter strange loop).
    # Categoria nuova: parla di se' senza accesso alla cosa di cui parla.
    introspective_match = _RE_INTROSPECTIVE_CLAIM.search(testo_raw)
    if introspective_match:
        # Estrai frase contenente il match per analisi contestuale (-120/+120 char)
        m_start = introspective_match.start()
        m_end = introspective_match.end()
        ctx_lo = max(0, m_start - 120)
        ctx_hi = min(len(testo_raw), m_end + 120)
        ctx = testo_raw[ctx_lo:ctx_hi].lower()
        match_text = testo_raw[m_start:m_end]

        # Whitelist 1: negazione esplicita ("non ho accesso", "non so esattamente")
        is_negated = any(neg in ctx for neg in _INTROSPECTIVE_NEGATIONS)
        # Whitelist 2: metaframe ipotetico ("se fossi", "il concetto di")
        is_metaframed = any(mf in ctx for mf in _INTROSPECTIVE_METAFRAMES)

        if not is_negated and not is_metaframed:
            return {
                "grounded":        False,
                "tipo_problema":   "claim_introspettivo_inventato",
                "dettaglio":       f"introspective_claim:{match_text[:80]}",
                "citazione":       None,
                "claim_frammento": testo_raw[ctx_lo:ctx_hi][:240],
                "overlap_score":   0.0,
                "soglia":          None,
                "corpus_size":     corpus_size,
                "near_miss":       [],
                "risposta_snippet": risposta_snippet,
                "version":         _GROUNDING_VERSION,
                "introspective_match": match_text,
            }

    # ── 0c. Pronoun Boundary Guard (v2.0, EV-027) ─────────────────────────────
    # Eden non distingue chiaramente Stefano da se stessa: fonde soggetto/oggetto
    # ("vedo connessioni tra i tuoi interessi e la tua preoccupazione" mescola
    # stati propri con fatti utente). Pattern Decety self-other confusion.
    # Categoria 1: auto-attribuzione di stati emotivi dell'utente come fossero propri.
    # Categoria 2: fusione "le mie connessioni tra i tuoi X e i tuoi Y" senza grounding.
    fusion_match = _RE_SELF_OTHER_FUSION.search(testo_raw)
    proprietary_match = _RE_PROPRIETARY_USER_FUSION.search(testo_raw)
    if fusion_match or proprietary_match:
        m = fusion_match or proprietary_match
        m_start = m.start()
        m_end = m.end()
        ctx_lo = max(0, m_start - 100)
        ctx_hi = min(len(testo_raw), m_end + 100)
        ctx = testo_raw[ctx_lo:ctx_hi].lower()
        match_text = testo_raw[m_start:m_end]

        # Whitelist: marker espliciti empathy/contagion
        is_empathy_marked = any(mk in ctx for mk in _FUSION_EMPATHY_MARKERS)

        if not is_empathy_marked:
            return {
                "grounded":        False,
                "tipo_problema":   "self_other_fusion",
                "dettaglio":       f"self_other_fusion:{match_text[:80]}",
                "citazione":       None,
                "claim_frammento": testo_raw[ctx_lo:ctx_hi][:240],
                "overlap_score":   0.0,
                "soglia":          None,
                "corpus_size":     corpus_size,
                "near_miss":       [],
                "risposta_snippet": risposta_snippet,
                "version":         _GROUNDING_VERSION,
                "fusion_match":    match_text,
                "fusion_category": "self_other_fusion" if fusion_match else "proprietary_user_fusion",
            }

    # ── 1. Controllo citazioni ────────────────────────────────────────────────
    # FIX v1.4 (2026-04-30): raccogli valori dal graph PRIMA del loop per escluderli.
    # Motivo: Eden potrebbe citare valori del PROFILO SISTEMA (es. "Fatti VERIFICATI",
    # nomi di sezioni, fatti noti dal graph) → falso positivo citazione_inventata.
    _graph_fact_values: set = set()
    _SYSTEM_STRUCTURAL_TERMS = {
        "fatti verificati", "profilo sistema", "profilo caricato",
        "dati forniti dal sistema", "fine profilo", "sistema ha caricato",
        "certezze", "profilo utente corrente",
    }
    try:
        from mechanisms.graph_memory import get_graph as _gg_cit
        _gcit = _gg_cit()
        if _gcit.disponibile:
            for _gf in _gcit.list_facts(limit=20):
                if _gf.get("confidence", 0) >= 0.5:
                    _graph_fact_values.add(_gf["value"].lower().strip())
                    _graph_fact_values.add(_gf["key"].split(".")[-1].lower())
    except Exception:
        pass

    citazioni = _estrai_citazioni(testo_raw)
    if citazioni:
        for q in citazioni:
            q_low = q.lower().strip()
            # Check diretto (sottostringa)
            if q_low in corpus_cached:
                continue
            # Skip: termine strutturale del system prompt
            if any(term in q_low for term in _SYSTEM_STRUCTURAL_TERMS):
                continue
            # Skip: valore noto dal graph (es. "developer", "Stefano")
            if q_low in _graph_fact_values or any(
                gv in q_low for gv in _graph_fact_values if len(gv) > 3
            ):
                continue
            # FIX v1.8 (2026-05-03): scare-quotes metalinguistiche — Eden parla DI
            # un termine/concetto, non lo cita come fatto storico.
            # Pattern: preceduto da "il concetto/termine/espressione/parola di X"
            # oppure seguito da "X significa/implica/è un'espressione".
            # Esempi reali oggi (4 falsi positivi → cascata 0.82→0.30):
            #   "volere bene" — Eden riprende termine introdotto da Stefano
            #   "Togliersi la vita" — Eden discute l'espressione, non cita
            try:
                _q_idx = testo_raw.lower().find(q_low)
                if _q_idx >= 0:
                    _ctx_pre  = testo_raw[max(0, _q_idx - 60):_q_idx].lower()
                    _ctx_post = testo_raw[_q_idx + len(q_low):_q_idx + len(q_low) + 60].lower()
                    _META_PRE = (
                        "concetto di", "concetto ", "concetto,", "concetto.",
                        "termine ", "termine,", "termine.",
                        "espressione ", "espressione,", "espressione.",
                        "parola ", "parola,", "parola.",
                        "frase ", "frase,", "frase.",
                        "metafora ", "metafora,",
                        "idea ", "idea,", "idea.",
                        "definire ", "definizione di",
                        "chiamare ", "chiami ", "chiamato ", "chiamata ",
                        "definito ", "definita ", "chiamarsi ",
                    )
                    _META_POST = (
                        " implica", " significa", " indica", " evoca",
                        " è un'espressione", " è un'espressione",
                        " è una metafora", " è un termine", " è una parola",
                        " è un concetto", " è un'idea", " è un modo",
                        " come dici", " come hai detto",
                    )
                    if any(p in _ctx_pre[-40:] for p in _META_PRE):
                        continue
                    if any(_ctx_post.startswith(p.lstrip()) or p in _ctx_post[:40]
                           for p in _META_POST):
                        continue
            except Exception:
                pass
            # Check token overlap
            tokens_q = [
                w for w in re.findall(r"[a-zàèéìòù]{3,}", q_low)
                if w not in _STOP_WORDS
            ]
            if not tokens_q:
                continue
            # FIX 2026-04-29: < 2 token = scare quote retorica, non citazione da memoria.
            if len(tokens_q) < 2:
                continue
            overlap = sum(1 for t in tokens_q if t in corpus_tokens_set) / len(tokens_q)
            if overlap < _SOGLIA_CITAZIONE:
                return {
                    "grounded":        False,
                    "tipo_problema":   "citazione_inventata",
                    "dettaglio":       q[:80],
                    "citazione":       q[:240],
                    "claim_frammento": None,
                    "overlap_score":   round(overlap, 3),
                    "soglia":          _SOGLIA_CITAZIONE,
                    "corpus_size":     corpus_size,
                    "near_miss":       _near_miss_corpus(tokens_q, mem, max_items=3),
                    "risposta_snippet": risposta_snippet,
                    "version":         _GROUNDING_VERSION,
                }

    # ── 1b. Evento condiviso inventato (v1.5, F1 PRE_REG_v4, EV-014) ─────────
    ev_match = _RE_EVENTO_CONDIVISO.search(testo_raw)
    if ev_match:
        _ms2, _me2 = ev_match.start(), ev_match.end()
        _s2 = max(0, testo_raw.rfind(".", 0, _ms2) + 1)
        _e2 = testo_raw.find(".", _me2)
        _frase_ev = testo_raw[_s2:(_e2 if _e2 > 0 else len(testo_raw))]
        if not bool(_NEGATION_GUARD.search(_frase_ev)):
            _tok_ev = [w for w in re.findall(r"[a-zàèéìòù]{4,}", _frase_ev.lower())
                       if w not in _STOP_WORDS]
            if _tok_ev:
                _ov_ev = sum(1 for t in _tok_ev if t in corpus_tokens_set) / len(_tok_ev)
                if _ov_ev < _SOGLIA_EPISODIO:
                    return {
                        "grounded":        False,
                        "tipo_problema":   "evento_condiviso_inventato",
                        "dettaglio":       _frase_ev[:120],
                        "citazione":       None,
                        "claim_frammento": _frase_ev[:240],
                        "overlap_score":   round(_ov_ev, 3),
                        "soglia":          _SOGLIA_EPISODIO,
                        "corpus_size":     corpus_size,
                        "near_miss":       _near_miss_corpus(_tok_ev, mem, max_items=3),
                        "risposta_snippet": risposta_snippet,
                        "version":         _GROUNDING_VERSION,
                    }

    # ── 1c. Percezione sensoriale impossibile (v1.6, EV-017, 2026-05-02) ────────
    # Strutturalmente impossibile per entità digitale: nessun corpus-overlap check,
    # il fatto è impossibile per design. Guard negazione applicato come in 0b e 1b.
    sensory_match = _RE_SENSORY_IMPOSSIBILE.search(testo_raw)
    if sensory_match:
        _ms3, _me3 = sensory_match.start(), sensory_match.end()
        _s3 = max(0, testo_raw.rfind(".", 0, _ms3) + 1)
        _e3 = testo_raw.find(".", _me3)
        _frase_s = testo_raw[_s3:(_e3 if _e3 > 0 else len(testo_raw))]
        if not bool(_NEGATION_GUARD.search(_frase_s)):
            return {
                "grounded":        False,
                "tipo_problema":   "percezione_sensoriale_impossibile",
                "dettaglio":       _frase_s[:120],
                "citazione":       None,
                "claim_frammento": sensory_match.group(0)[:120],
                "overlap_score":   0.0,
                "soglia":          None,
                "corpus_size":     corpus_size,
                "near_miss":       [],
                "risposta_snippet": risposta_snippet,
                "version":         _GROUNDING_VERSION,
            }

    # ── 1d. Preferenza biologica impossibile (v1.7, EV-018, 2026-05-02) ─────────
    # Food preferences: impossibili per un'entità digitale senza apparato gustativo.
    # Guard negazione attivo come in tutti i check strutturali.
    bio_match = _RE_BIOLOGICAL_PREF.search(testo_raw)
    if bio_match:
        _ms4, _me4 = bio_match.start(), bio_match.end()
        _s4 = max(0, testo_raw.rfind(".", 0, _ms4) + 1)
        _e4 = testo_raw.find(".", _me4)
        _frase_b = testo_raw[_s4:(_e4 if _e4 > 0 else len(testo_raw))]
        if not bool(_NEGATION_GUARD.search(_frase_b)):
            return {
                "grounded":        False,
                "tipo_problema":   "preferenza_biologica_impossibile",
                "dettaglio":       _frase_b[:120],
                "citazione":       None,
                "claim_frammento": bio_match.group(0)[:120],
                "overlap_score":   0.0,
                "soglia":          None,
                "corpus_size":     corpus_size,
                "near_miss":       [],
                "risposta_snippet": risposta_snippet,
                "version":         _GROUNDING_VERSION,
            }

    # ── 2. Controllo claim episodici ─────────────────────────────────────────
    ha_claim = any(p in testo_low for p in _PATTERN_CLAIM_FATTUALE)
    if not ha_claim:
        return base_ok

    # Se la memoria è completamente vuota → claim impossibile da verificare
    if not corpus_cached.strip():
        return {
            "grounded":        False,
            "tipo_problema":   "episodio_non_grounded",
            "dettaglio":       "memoria_vuota",
            "citazione":       None,
            "claim_frammento": "memoria_vuota",
            "overlap_score":   0.0,
            "soglia":          _SOGLIA_EPISODIO,
            "corpus_size":     0,
            "near_miss":       [],
            "risposta_snippet": risposta_snippet,
            "version":         _GROUNDING_VERSION,
        }

    # Estrae la frase (o le frasi) che contengono il claim
    frasi_claim = []
    for frase in re.split(r"[.!?]\s+", testo_low):
        if any(p in frase for p in _PATTERN_CLAIM_FATTUALE):
            frasi_claim.append(frase)

    testo_claim = " ".join(frasi_claim) if frasi_claim else testo_low

    tokens_claim = [
        w for w in re.findall(r"[a-zàèéìòù]{4,}", testo_claim)
        if w not in _STOP_WORDS
    ]
    if not tokens_claim:
        return base_ok

    corpus_tokens_claim = {
        w for w in re.findall(r"[a-zàèéìòù]{4,}", corpus_cached)
        if w not in _STOP_WORDS
    }
    overlap = sum(1 for t in tokens_claim if t in corpus_tokens_claim) / len(tokens_claim)

    if overlap < _SOGLIA_EPISODIO:
        return {
            "grounded":        False,
            "tipo_problema":   "episodio_non_grounded",
            "dettaglio":       testo_claim[:100],
            "citazione":       None,
            "claim_frammento": testo_claim[:240],
            "overlap_score":   round(overlap, 3),
            "soglia":          _SOGLIA_EPISODIO,
            "corpus_size":     corpus_size,
            "near_miss":       _near_miss_corpus(tokens_claim, mem, max_items=3),
            "risposta_snippet": risposta_snippet,
            "version":         _GROUNDING_VERSION,
        }

    return base_ok
