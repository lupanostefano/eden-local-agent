"""
peso.py — quanto un ricordo ha contato: emozione, intensità, segno (stile amigdala) e quante volte è tornato utile.

Ogni scambio ha una riga in `peso`, sul suo primo messaggio. La stima a regole è veloce e grezza: marca subito ogni
scambio, come l'amigdala; il sonno (step 6) la rifinisce con il modello e scrive `fonte = 'sonno …'`, che non si
sovrascrive più. Il richiamo (`ricerca.py`) usa intensità, segno, recenza e rinforzo per riordinare i risultati;
niente si cancella mai: ciò che non torna più utile scende soltanto di posizione.
"""
from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime

from nucleo import registro
from nucleo.testo import norm

FONTE_REGOLE = "regole"

# (espressione sul testo normalizzato, emozione, segno da -1 a +1, peso)
LESSICO: list[tuple[str, str, float, float]] = [
    (r"\b(felice|felicita|contento|contenta|gioia|gioioso|entusiast\w+|euforic\w+)\b", "gioia", 1, .35),
    (r"\b(bellissim\w+|meraviglios\w+|fantastic\w+|stupend\w+|magnific\w+|splendid\w+|adoro)\b", "gioia", 1, .25),
    (r"\b(evviva|urra|finalmente|ce l ho fatta|ce l abbiamo fatta|ho vinto|promosso|ho superato)\b", "gioia", 1, .40),
    (r"\b(orgoglios\w+|fiero|fiera|orgoglio)\b", "orgoglio", 1, .40),
    (r"\b(emozionat\w+|commoss\w+|commuov\w+|mi emoziona|lacrime di gioia)\b", "commozione", 1, .40),
    (r"\bti (voglio bene|amo|adoro)\b", "affetto", 1, .70),
    (r"\b(grazie di cuore|grazie mille|sei speciale|sei importante|mi fai sentire|mi fido di te|sei preziosa)\b", "affetto", 1, .45),
    (r"\b(abbraccio|abbracci|abbracciar\w+|coccol\w+|tenerezza|dolcezza)\b", "affetto", 1, .30),
    (r"\b(gratitudine|grato|grata|riconoscente|grazie)\b", "gratitudine", .8, .15),
    (r"\bmi (manchi|sei mancat\w+|mancherai|manca)\b", "nostalgia", .2, .40),
    (r"\b(nostalgia|nostalgic\w+)\b", "nostalgia", .2, .35),
    (r"\b(triste|tristezza|sconforto|sconfortat\w+|abbattut\w+|giu di morale|malinconi\w+)\b", "tristezza", -1, .40),
    (r"\b(piango|piangere|piangevo|pianto|lacrime|singhioz\w+)\b", "tristezza", -1, .50),
    (r"\b(depress\w+|disperat\w+|senza speranza|non ce la faccio|non ne posso piu|sto male|mi sento male|mi sento vuot\w+)\b",
     "tristezza", -1, .55),
    (r"\b(mi sento sol[oa]|sono sol[oa]|solitudine|isolat\w+)\b", "solitudine", -1, .40),
    (r"\b(delus\w+|delusione)\b", "delusione", -1, .40),
    (r"\b(morto|morta|morte|morire|deceduto|scomparso|scomparsa|lutto|funerale|salma|defunt\w+)\b", "lutto", -1, .45),
    (r"\b(malattia|malato|malata|ospedale|terminale|tumore|cancro|sla|diagnosi|chemio\w*)\b", "preoccupazione", -1, .30),
    (r"\b(arrabbiat\w+|incazzat\w+|furios\w+|furibond\w+|infuriat\w+|rabbia|odio|odiare|schifo|schifos\w+)\b", "rabbia", -1, .45),
    (r"\b(mi da fastidio|non sopporto|che palle|sono stuf[oa]|mi hai stufato|insopportabile)\b", "rabbia", -1, .30),
    (r"\b(frustrat\w+|esasperat\w+|vergogn\w+)\b", "frustrazione", -1, .35),
    (r"\b(paura|terrore|terrorizzat\w+|spavent\w+|panico|angoscia|angoscios\w+)\b", "paura", -1, .50),
    (r"\b(ansia|ansios\w+|agitat\w+|preoccupat\w+|preoccupazione|inquiet\w+|nervos\w+)\b", "ansia", -1, .35),
    (r"\b(stanc\w+|esaust\w+|stress\w*|insonnia|a pezzi|burnout|sfinit\w+)\b", "stanchezza", -.6, .20),
    (r"\b(mia figlia|mia moglie|mio padre|mia madre|mio figlio|mio fratello|mia sorella|i miei genitori|mio marito)\b",
     "legami", 0, .20),
]
_LESSICO = [(re.compile(rx), emozione, segno, peso) for rx, emozione, segno, peso in LESSICO]

# Parole con cui Stefano chiede un ricordo "di quel tipo" senza dire una cosa precisa: ("emozione", segno) o solo segno
_RICHIESTE = [
    (re.compile(r"\b(piu bello|piu belli|bellissim\w+|migliore|migliori|piu felice|felici|meraviglios\w+)\b"), None, 1.0),
    (re.compile(r"\b(piu brutt\w+|peggior\w+|piu dur\w+|piu difficil\w+|piu giu|piu doloros\w+|piu tristi?)\b"), None, -1.0),
    (re.compile(r"\b(important\w+|special\w+|significativ\w+|toccant\w+|commovent\w+|emozionant\w+|indimenticabil\w+)\b"), None, None),
]

W_EMOZIONE_AFFETTIVA = .9    # quanto pesa l'intensità quando la richiesta è affettiva ("il momento più bello")
W_EMOZIONE = .15             # … e quando non lo è: solo un piccolo vantaggio a parità di somiglianza
W_RINFORZO = .4
W_RECENZA = .15
RICHIAMI_SATURA = 5          # dopo tanti richiami il rinforzo non cresce più (evita che i soliti ricordi coprano tutto)
MEZZA_VITA_RINFORZO = 60     # giorni: un ricordo non richiamato da un po' perde il rinforzo (non il peso emotivo)
MEZZA_VITA_RECENZA = 45
SOGLIA_CANALE = .3           # sotto questa intensità un ricordo non si propone da solo per la sua emozione


# ---------------------------------------------------------------- stima a regole

_PRIMA_PERSONA = re.compile(r"\b(io|ho|sono|mi sento|mi sentivo|provo|sento|mi fa|mi fai|mi manca|mi ha|mi sei)\b")
DOMANDA, ASTRATTO = .4, .6   # una domanda "sulla paura" o un'osservazione in generale pesano meno di "ho paura"


def _frasi(testo: str) -> list[tuple[str, float]]:
    """(frase normalizzata, quanto conta): una domanda o un discorso in astratto contano meno di una cosa vissuta."""
    out = []
    for f in re.findall(r"[^.!?\n]+[.!?]*", testo):
        n = norm(f)
        if n:
            out.append((n, DOMANDA if f.rstrip().endswith("?") else 1.0 if _PRIMA_PERSONA.search(n) else ASTRATTO))
    return out


def _punti(testo: str) -> tuple[float, float, dict[str, float]]:
    """(somma dei pesi, somma pesata dei segni, peso per emozione) dei segnali trovati nel testo."""
    frasi = _frasi(testo)
    somma = segno = 0.0
    per_emozione: dict[str, float] = {}
    for rx, emozione, s, peso in _LESSICO:
        quanto = max((q for f, q in frasi if rx.search(f)), default=0.0)   # ogni segnale conta una volta, nel caso migliore
        if quanto:
            somma += peso * quanto
            segno += s * peso * quanto
            per_emozione[emozione] = per_emozione.get(emozione, 0) + peso * quanto
    if re.search(r"!{2,}", testo):
        somma += .10
    if len(re.findall(r"\b[A-ZÀ-Ý]{3,}\b", testo)) >= 2:
        somma += .10
    return somma, segno, per_emozione


def stima(u: str | None, a: str | None) -> tuple[str | None, float, float]:
    """(emozione, intensità 0..1, segno -1..+1) di uno scambio. Conta soprattutto ciò che scrive Stefano."""
    su, gu, eu = _punti(u or "")
    sa, ga, ea = _punti(a or "")
    somma = su + .5 * sa
    if somma <= 0:
        return None, 0.0, 0.0
    somma *= 1 + min(.5, len(u or "") / 1500)                # un messaggio lungo e personale pesa di più
    per_emozione = dict(eu)
    for k, v in ea.items():
        per_emozione[k] = per_emozione.get(k, 0) + .5 * v
    emozione = max((k for k in per_emozione if k != "legami"), key=per_emozione.get, default=None)
    segno = (gu + .5 * ga) / (su + .5 * sa)
    return emozione, round(1 - math.exp(-1.5 * somma), 3), round(max(-1, min(1, segno)), 3)


def richiesta_affettiva(domanda: str) -> tuple[bool, str | None, float | None]:
    """La domanda chiede un ricordo per come è stato vissuto? (sì/no, emozione cercata, segno cercato)"""
    t = norm(domanda)
    emozioni: dict[str, float] = {}
    segno = 0.0
    for rx, emozione, s, peso in _LESSICO:
        if emozione != "legami" and rx.search(t):
            emozioni[emozione] = emozioni.get(emozione, 0) + peso
            segno += s * peso
    affettiva = bool(emozioni)
    for rx, _, s in _RICHIESTE:
        if rx.search(t):
            affettiva = True
            segno += s or 0
    return affettiva, (max(emozioni, key=emozioni.get) if emozioni else None), (1.0 if segno > 0 else -1.0 if segno < 0 else None)


# ---------------------------------------------------------------- registro

def carica(con: sqlite3.Connection) -> dict[int, dict]:
    """Il peso di ogni scambio, per numero del suo primo messaggio."""
    return {mid: {"emozione": e, "intensita": i or 0.0, "segno": s or 0.0, "richiami": r, "ultimo_richiamo": u, "fonte": f}
            for mid, e, i, s, r, u, f in con.execute(
                "SELECT messaggio_id, emozione, intensita, segno, richiami, ultimo_richiamo, fonte FROM peso")}


def riempi_mancanti(con: sqlite3.Connection) -> int:
    """Stima a regole per gli scambi senza peso. Non tocca mai le righe che ci sono già (né quelle del sonno)."""
    ha = {r[0] for r in con.execute("SELECT messaggio_id FROM peso")}
    nuovi = [(s["ids"][0], *stima(s["u"], s["a"]), FONTE_REGOLE) for s in registro.carica_scambi(con)
             if s["ids"][0] not in ha]
    con.executemany("INSERT INTO peso(messaggio_id, emozione, intensita, segno, fonte) VALUES (?,?,?,?,?)", nuovi)
    con.commit()
    return len(nuovi)


def rinforza(con: sqlite3.Connection, scambio: int, quando: str) -> None:
    """Il ricordo è tornato utile (una citazione verificata l'ha usato): un richiamo in più."""
    con.execute("INSERT INTO peso(messaggio_id, richiami, ultimo_richiamo, fonte) VALUES (?, 1, ?, ?) "
                "ON CONFLICT(messaggio_id) DO UPDATE SET richiami = richiami + 1, ultimo_richiamo = excluded.ultimo_richiamo",
                (scambio, quando, FONTE_REGOLE))
    con.commit()


# ---------------------------------------------------------------- richiamo

def _giorni(da: str | None, a: str) -> float | None:
    return None if not da else max(0.0, (datetime.fromisoformat(a) - datetime.fromisoformat(da)).total_seconds() / 86400)


def moltiplicatore(p: dict | None, ts: str, adesso: str, richiesta: tuple[bool, str | None, float | None]) -> float:
    """Di quanto il peso del ricordo alza (o lascia com'è) il suo punteggio di somiglianza. Sempre ≥ 1: niente si sotterra."""
    affettiva, emozione, segno = richiesta
    emo = rin = 0.0
    if p:
        emo = p["intensita"]
        if affettiva and segno:
            emo *= max(0.0, 1 + .5 * segno * p["segno"])
        if affettiva and emozione and p["emozione"] == emozione:
            emo *= 1.5
        giorni = _giorni(p["ultimo_richiamo"], adesso)
        if p["richiami"] and giorni is not None:
            rin = min(1.0, math.log1p(p["richiami"]) / math.log1p(RICHIAMI_SATURA)) * math.exp(-giorni / MEZZA_VITA_RINFORZO)
    recenza = math.exp(-(_giorni(ts, adesso) or 0) / MEZZA_VITA_RECENZA)
    return (1 + (W_EMOZIONE_AFFETTIVA if affettiva else W_EMOZIONE) * emo) * (1 + W_RINFORZO * rin) * (1 + W_RECENZA * recenza)
