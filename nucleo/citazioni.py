"""
citazioni.py — le fonti che Eden cita nelle risposte ([[#1234]]) e la loro verifica.

Una citazione regge se: il numero è uno scambio dei nostri ricordi che Eden ha davvero avuto sotto gli occhi in questo
turno (storia lunga, risultati, strumenti); le parole tra «» che la precedono compaiono davvero in quello scambio; la
frase che la precede ha qualcosa in comune con lo scambio (numeri, nomi, parole non comuni).
Le fonti web si citano con [[@n]] (n = numero nel risultato di cerca_web/leggi_pagina): deve essere una fonte vista in
questo turno, le parole tra «» devono comparire nella pagina (o nell'estratto), e per valere del tutto la pagina deve
essere stata aperta (solo l'estratto → `debole`).
Esiti: `ok` · `debole` (il numero esiste ma non sembra reggere la frase: si mostra, non si corregge) · `falsa`
(numero inventato, mai visto, o parole tra «» che non ci sono: si fa correggere a Eden).
Senza modelli: regole sul testo, come `passato.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nucleo.ricerca import STOP, norm
from nucleo.strumenti import Turno

# Uno o più marcatori attaccati ([[#3]][[#4]] = [[#3, #4]]): reggono insieme la stessa frase. Identico a quello della pagina.
_UNO = r"\[\[\s*[#@]?\d+(?:\s*[,;]\s*[#@]?\d+)*\s*\]\]"
MARCATORE = re.compile(rf"{_UNO}(?:\s*{_UNO})*")
SOGLIA_SOSTEGNO = 0.3        # quota di parole "distintive" della frase che devono ritrovarsi nello scambio
MIN_DISTINTIVE = 2           # con meno parole distintive non si può giudicare: la citazione vale se esiste
VUOTE = set("""ricordo ricordi ricordare ricordato dicesti disse detto dissi detti raccontato raccontasti raccontai parlato
parlammo parlavamo volta giorno giorni settimana mese stesso stessa ancora proprio quindi anche molto sempre certo davvero
eravamo abbiamo stavamo avevi avevo allora insieme conversazione messaggio scambio memoria esatto esattamente
qualcosa ultima ultimo prima primo dopo tutto tutti tanto quanto adesso ormai invece ovvero cioe dunque perche""".split())


@dataclass
class Citazione:
    inizio: int                  # posizione del marcatore nel testo
    fine: int
    ids: list[int]               # i numeri citati
    stato: str = "ok"            # ok · debole · falsa
    motivo: str = ""
    scambio: int | None = None   # numero (primo messaggio) dello scambio che regge la citazione
    quando: str | None = None    # data e ora di quello scambio (ISO)
    url: str | None = None       # fonte web: indirizzo e titolo (per i ricordi restano vuoti)
    titolo: str | None = None
    web: bool = False            # è una fonte [[@n]] (anche se non esiste: allora url resta vuoto)

    @property
    def etichetta(self) -> str:
        """Il marcatore com'è scritto nel testo, per ricordarlo a Eden."""
        segno = "@" if self.web else "#"
        return "[[" + ", ".join(f"{segno}{i}" for i in self.ids) + "]]"

    def per_registro(self) -> dict:
        """Come si salva nel registro e si manda alla pagina."""
        d = {"ids": self.ids, "stato": self.stato, "motivo": self.motivo or None, "scambio": self.scambio,
             "quando": self.quando}
        return {**d, "url": self.url, "titolo": self.titolo} if self.web else d


def _piatto(s: str) -> str:
    """Minuscolo, senza accenti né punteggiatura: per confrontare parole senza badare a come sono scritte."""
    return re.sub(r"[^a-z0-9]+", " ", norm(s)).strip()


def _confini(testo: str) -> list[int]:
    """Dove finisce una frase (dopo . ! ? o a capo), fuori dalle virgolette «»."""
    fuori, out = True, []
    for i, c in enumerate(testo):
        if c == "«":
            fuori = False
        elif c == "»":
            fuori = True
        elif c in ".!?\n" and fuori:
            out.append(i + 1)
    return out


def _segmento(testo: str, m: re.Match, marcatori: list[re.Match], confini: list[int]) -> str:
    """Le parole a cui il marcatore si riferisce: dall'inizio della frase (o dal marcatore precedente) fino a lui.
    Se il marcatore sta dopo il punto ("… gatti. [[#3]]") vale la frase prima."""
    prima = [c for c in confini if c <= m.start()]
    inizio = prima[-1] if prima else 0
    prec = [x.end() for x in marcatori if x.end() <= m.start() and x.end() > inizio]
    seg = testo[max([inizio] + prec):m.start()]
    if not seg.strip() and prima:
        fine = prima[-1]
        inizio = prima[-2] if len(prima) > 1 else 0
        seg = testo[inizio:fine]
    return MARCATORE.sub(" ", seg)


def _frammenti(citazione: str) -> list[str]:
    """Una citazione può saltare dei pezzi con «…» o «[...]»: ogni pezzo deve comparire."""
    pezzi = re.split(r"…|\[\s*\.{3}\s*\]|\.{3}", citazione)
    return [p for p in (_piatto(x) for x in pezzi) if len(p) >= 3]


def _radici(testo: str) -> set[str]:
    return {w[:5] for w in _piatto(testo).split()}


def _distintive(frase: str) -> list[str]:
    """Numeri, nomi e parole non comuni della frase, ridotti alla radice (gatti/gatto → gatt)."""
    parole = re.findall(r"[a-z0-9]+", norm(re.sub(r"«[^»]*»", " ", frase)))
    return list(dict.fromkeys(w[:5] for w in parole
                              if (len(w) >= 5 or any(c.isdigit() for c in w)) and w not in STOP and w not in VUOTE))


def _verifica_id(mid: int, citate: list[str], frase: str, turno: Turno) -> tuple[str, str, int | None]:
    """(stato, motivo, numero dello scambio) per un singolo numero."""
    i = turno.ricerca.msg2sc.get(mid)
    if i is None or mid >= turno.prima_di_id:
        return "falsa", f"il numero {mid} non è uno scambio dei nostri ricordi", None
    s = turno.ricerca.scambi[i]
    if not turno.ha_visto(mid):
        return "falsa", f"lo scambio {s['ids'][0]} non l'hai letto in questo turno: non è una fonte", s["ids"][0]
    testo = _piatto(" ".join(x for x in (s["u"], s["a"]) if x))
    for q in citate:
        for f in _frammenti(q):
            if f not in testo:
                return "falsa", f"le parole «{q[:60]}» non compaiono nello scambio {s['ids'][0]}", s["ids"][0]
    parole = _distintive(frase)
    if len(parole) >= MIN_DISTINTIVE:
        radici = _radici(testo)
        if sum(w in radici for w in parole) / len(parole) < SOGLIA_SOSTEGNO:
            return "debole", f"lo scambio {s['ids'][0]} non sembra parlare di ciò che dici", s["ids"][0]
    return "ok", "", s["ids"][0]


def _verifica_web(n: int, citate: list[str], frase: str, turno: Turno) -> tuple[str, str, dict | None]:
    """(stato, motivo, fonte) per una fonte web [@n]."""
    if not 1 <= n <= len(turno.web):
        return "falsa", f"la fonte web @{n} non esiste: nessuna ricerca l'ha mostrata in questo turno", None
    f = turno.web[n - 1]
    letta = f["testo"] is not None
    base = _piatto(" ".join(x for x in (f["titolo"], f["testo"] if letta else f["estratto"]) if x))
    for q in citate:
        for fr in _frammenti(q):
            if fr not in base:
                return "falsa", f"le parole «{q[:60]}» non compaiono nella fonte @{n}" + ("" if letta else " (hai letto solo l'estratto)"), f
    if not letta:
        return "debole", f"della fonte @{n} hai visto solo l'estratto della ricerca, non la pagina", f
    parole = _distintive(frase)
    if len(parole) >= MIN_DISTINTIVE and sum(w in _radici(base) for w in parole) / len(parole) < SOGLIA_SOSTEGNO:
        return "debole", f"la fonte @{n} non sembra parlare di ciò che dici", f
    return "ok", "", f


def _voci(marcatore: str) -> tuple[list[int], list[int]]:
    """(numeri dei ricordi, numeri delle fonti web) di un marcatore."""
    ricordi, fonti = [], []
    for segno, n in re.findall(r"([#@]?)\s*(\d+)", marcatore):
        (fonti if segno == "@" else ricordi).append(int(n))
    return ricordi, fonti


def verifica(testo: str, turno: Turno) -> list[Citazione]:
    """Tutte le citazioni del testo, nell'ordine in cui compaiono, con il loro esito."""
    marcatori = list(MARCATORE.finditer(testo))
    confini = _confini(testo)
    out = []
    for m in marcatori:
        ricordi, fonti = _voci(m.group(0))
        frase = _segmento(testo, m, marcatori, confini)
        citate = re.findall(r"«([^»]+)»", frase)
        esiti = [(*_verifica_id(mid, citate, frase, turno), None, False) for mid in ricordi]   # (stato, motivo, scambio, fonte, è web)
        esiti += [(e[0], e[1], None, e[2], True) for e in (_verifica_web(n, citate, frase, turno) for n in fonti)]
        stato, motivo, scambio, fonte, e_web = min(esiti, key=lambda e: ("ok", "debole", "falsa").index(e[0]))
        quando = turno.ricerca.scambi[turno.ricerca.msg2sc[scambio]]["ts"] if scambio else None
        web = {"url": fonte["url"], "titolo": fonte["titolo"]} if fonte else {}
        out.append(Citazione(m.start(), m.end(), fonti if e_web else ricordi, stato, motivo, scambio, quando, web=e_web, **web))
    return out


def numeri(testo: str) -> set[int]:
    """I numeri dei ricordi citati in un testo (le risposte di Eden dei turni già passati restano sotto i suoi occhi)."""
    return {n for m in MARCATORE.finditer(testo) for n in _voci(m.group(0))[0]}


def da_correggere(cit: list[Citazione]) -> list[Citazione]:
    return [c for c in cit if c.stato == "falsa"]


def nota_correzione(sbagliate: list[Citazione]) -> str:
    """Il messaggio con cui si chiede a Eden di rifare la risposta: le fonti sbagliate e cosa non va."""
    righe = "\n".join(f"- {c.etichetta}: {c.motivo}" for c in sbagliate)
    return ("Controllo automatico delle tue fonti: alcune non reggono.\n" + righe +
            "\nRiscrivi la risposta usando solo fonti che risultano davvero (i ricordi puoi rileggerli con cerca_ricordi, le "
            "pagine con leggi_pagina); se una cosa non risulta, dillo. Rispondi a Stefano come se fosse la prima volta: "
            "non nominare questo controllo.")
