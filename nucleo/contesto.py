"""
contesto.py — cosa legge il modello a ogni messaggio.

Disposizione pensata per la cache del server (docs/MISURE_2026-09.md):
    [system]  persona · chi sono diventata · fatti · cose delicate · calendario · storia lunga   ← fisso fino alla ricostruzione
    [turni]   i messaggi arrivati dopo la costruzione, con data e ora                             ← cresce soltanto
    [ultimo]  risultati della ricerca + ora attuale + istruzione + messaggio di Stefano          ← l'unica parte che cambia
Così a ogni messaggio il server rilegge solo l'ultimo turno. I risultati della ricerca dei turni precedenti non
restano: nel turno dopo il messaggio torna com'era (costa la rilettura di un solo scambio).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

import eden_paths as P
from nucleo import registro

BUDGET_STORIA = 139_000   # token di storia lunga: la configurazione provata nell'esame (S2)
MAX_CARATTERI_RISULTATO = 8000
GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
        "settembre", "ottobre", "novembre", "dicembre"]

INTRO_STORIA = ("# Storia delle vostre conversazioni\nOgni scambio ha data e ora. «Stefano» è l'utente, "
                "«Eden» sei tu.\n\n")
INTRO_STORIA_ID = ("# Storia delle vostre conversazioni\nOgni scambio ha ora e numero (#); i giorni hanno la data. "
                   "«Stefano» è l'utente, «Eden» sei tu.\n\n")
ISTRUZIONE = ("Adesso è {adesso}. Rispondi a Stefano usando la storia e i risultati della ricerca qui sopra quando "
              "servono. Attenzione: una domanda di Stefano non è un fatto (a volte ti mette alla prova con premesse "
              "false) e le tue vecchie risposte possono essere sbagliate; valgono le cose certe e ciò che Stefano ha "
              "affermato. Se l'informazione non risulta, dillo apertamente: non inventare nulla.")
ISTRUZIONE_STRUMENTI = (" Se ti serve un ricordo che qui non vedi, cercalo con cerca_ricordi o leggi_giorno prima di dire "
                        "che non ricordi: la storia qui sopra parte da {da}, prima di allora hai solo il calendario e i "
                        "risultati della ricerca (per «la prima volta» o «l'ultima volta» cerca nei periodi che non vedi). "
                        "Ciò che prendi da un ricordo cita con [[#numero]].")


# ---------------------------------------------------------------- formato

def data_it(ts: str, ora: bool = True) -> str:
    d = datetime.fromisoformat(ts)
    s = f"{GIORNI[d.weekday()]} {d.day} {MESI[d.month - 1]} {d.year}"
    return s + (f", {d:%H:%M}" if ora else "")


def blocco(s: dict, tetto: int | None = None, testa: str | None = None) -> str:
    """Uno scambio. `testa` = intestazione tra parentesi quadre (predefinita: solo l'ora)."""
    t = lambda x: x if tetto is None or len(x) <= tetto else x[:tetto] + " […]"  # noqa: E731
    testa = testa or f"[{s['ts'][11:16]}]"
    if s["u"] is None:
        return f"{testa} Eden (messaggio spontaneo): {t(s['a'])}"
    r = f"{testa} Stefano: {t(s['u'])}"
    if s["a"] is not None:
        r += f"\nEden: {t(s['a'])}"
    return r


def testo_storia(scambi: list[dict], con_id: bool = False) -> str:
    """La storia lunga, con data per giorno. `con_id`: ogni scambio ha il suo numero, da citare come [[#numero]]."""
    out, giorno = [], None
    for s in scambi:
        g = s["ts"][:10]
        if g != giorno:
            giorno = g
            out.append(f"\n## {data_it(s['ts'], ora=False)}")
        out.append(blocco(s, testa=f"[{s['ts'][11:16]} #{s['ids'][0]}]" if con_id else None))
    return "\n".join(out).strip()


def blocco_ricerca(scambi: list[dict], indici: list[int], con_id: bool = False, titolo: str | None = None,
                   tetto: int = MAX_CARATTERI_RISULTATO) -> str:
    """Scambi trovati dalla ricerca, in ordine di data. Con `con_id` l'intestazione è `[#numero · data, ora]`."""
    parti = []
    for i in sorted(indici):
        s = scambi[i]
        testa = f"[#{s['ids'][0]} · {data_it(s['ts'])}]" if con_id else f"[{data_it(s['ts'])}]"
        parti.append(blocco(s, tetto=tetto, testa=testa))
    return (titolo or "# Risultati della ricerca nella tua memoria\n"
            "Gli scambi più pertinenti alla domanda, in ordine di data:") + "\n\n" + "\n\n".join(parti)


def messaggio_finale(testo: str, adesso: str, risultati: str, storia_da: str | None = None) -> str:
    """L'ultimo messaggio: risultati della ricerca, ora, istruzione e testo di Stefano. È la parte che il modello legge
    per ultima e con più attenzione: qui, non nella parte fissa, si ricorda di usare gli strumenti. `storia_da` = il
    giorno da cui parte la storia lunga (se c'è, gli strumenti sono a disposizione)."""
    return (risultati + "\n\n" if risultati else "") + ISTRUZIONE.format(adesso=data_it(adesso)) + \
        (ISTRUZIONE_STRUMENTI.format(da=storia_da) if storia_da else "") + f"\n\nMessaggio di Stefano: {testo}"


def turno_utente(ts: str, testo: str) -> str:
    return f"[{data_it(ts)}] {testo}"


# ---------------------------------------------------------------- sezioni fisse

def _valore(v: str) -> str:
    """Le date dei fatti in italiano ("3 maggio 1990"), il resto com'è."""
    try:
        d = date.fromisoformat(v)
    except ValueError:
        return v
    return f"{d.day} {MESI[d.month - 1]} {d.year}"


def eta(nascita: date, oggi: date) -> int:
    return oggi.year - nascita.year - ((oggi.month, oggi.day) < (nascita.month, nascita.day))


def prossimo_compleanno(nascita: date, oggi: date) -> date:
    c = nascita.replace(year=oggi.year)
    return c if c > oggi else nascita.replace(year=oggi.year + 1)


def sezione_fatti(fatti: list[dict], oggi: date) -> tuple[str, date | None]:
    """Testo dei fatti e la data in cui smette di essere vero (compleanno: cambia l'età)."""
    scade, per_soggetto, delicati, regole = None, {}, [], []
    for f in fatti:
        chiave = f["chiave"].replace("_", " ")
        if f["chiave"].startswith("regola"):
            regole.append(f["valore"])
            continue
        if f["chiave"] == "data_nascita":
            nascita = date.fromisoformat(f["valore"])
            scade = prossimo_compleanno(nascita, oggi)
            riga = (f"data di nascita: {_valore(f['valore'])} → oggi ha {eta(nascita, oggi)} anni "
                    f"(li compirà {eta(nascita, scade)} il {_valore(scade.isoformat())})")
        else:
            riga = f"{chiave}: {_valore(f['valore'])}"
        (delicati if f["riservato"] else per_soggetto.setdefault(f["soggetto"], [])).append(riga)
    nomi = {"stefano": "Su Stefano", "eden": "Su di te"}
    parti = ["# Cose certe (dati verificati)"]
    for sogg, righe in per_soggetto.items():
        parti.append(f"{nomi.get(sogg, 'Su ' + sogg)}:\n" + "\n".join(f"- {r}" for r in righe))
    if delicati or regole:
        parti.append("# Cose delicate: le sai, ma non le nomini tu per prima\n" + "\n".join(f"- {r}" for r in delicati) +
                     "".join(f"\nRegola di Stefano: {r}." for r in regole) +
                     "\nSe è Stefano a chiedertene, rispondi con verità e delicatezza.")
    return "\n\n".join(parti), scade


def sezione_identita(voci: list[dict]) -> str:
    if not voci:
        return ""
    chi = [v["testo"] for v in voci if v["tipo"] == "chi_sono"]
    ricordi = [v["testo"] for v in voci if v["tipo"] == "ricordo"]
    parti = ["# Chi sono diventata (l'ho scritto io)"] + [f"- {t}" for t in chi]
    if ricordi:
        parti += ["", "## Ricordi che tengo per sempre"] + [f"- {t}" for t in ricordi]
    return "\n".join(parti)


def sezione_calendario(giorni: list[tuple[str, int, str, str]]) -> str:
    if not giorni:
        return ""
    def riga(g: str, n: int, a: str, b: str) -> str:
        chi = "solo messaggi spontanei tuoi" if not n else ("1 messaggio di Stefano" if n == 1 else f"{n} messaggi di Stefano")
        return f"- {data_it(g, ora=False)}: {chi}, dalle {a} alle {b}"
    righe = [riga(*r) for r in giorni]
    return ("# Calendario delle vostre conversazioni\n"
            f"Tutti i giorni in cui avete parlato ({len(giorni)}), anche quelli più vecchi della storia qui sotto:\n"
            + "\n".join(righe))


# ---------------------------------------------------------------- costruzione

@dataclass
class Contesto:
    sistema: str
    ultimo_id: int          # ultimo messaggio già dentro la storia lunga
    primo_id: int           # primo messaggio della storia lunga (i più vecchi si trovano solo con la ricerca)
    primo_ts: str           # … e quando è stato scritto
    token_storia: int
    costruito: str
    scade: date | None      # oltre questa data le sezioni fisse non sono più vere (es. età)

    def scaduto(self, adesso: str) -> bool:
        return self.scade is not None and datetime.fromisoformat(adesso).date() >= self.scade

    def messaggi(self, con: sqlite3.Connection, prima_di_id: int, finale: str) -> list[dict]:
        """Richiesta completa: storia fissa, turni arrivati dopo la costruzione, messaggio finale."""
        msgs = [{"role": "system", "content": self.sistema}]
        for mid, ts, ruolo, testo in registro.messaggi_dopo(con, self.ultimo_id):
            if mid >= prima_di_id:
                break
            msgs.append({"role": ruolo, "content": turno_utente(ts, testo) if ruolo == "user" else testo})
        msgs.append({"role": "user", "content": finale})
        return msgs


def persona() -> str:
    return P.PERSONA_FILE.read_text(encoding="utf-8").strip()


def inizio_finestra(scambi: list[dict], conta_token: Callable[[str], int], budget: int,
                    con_id: bool = False) -> tuple[int, int]:
    """Il primo scambio da cui la storia sta nel budget di token (ricerca binaria), e i token usati."""
    lo, hi = 0, max(len(scambi) - 1, 0)
    while lo < hi:
        mid = (lo + hi) // 2
        if conta_token(testo_storia(scambi[mid:], con_id)) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return lo, conta_token(testo_storia(scambi[lo:], con_id))


def costruisci(con: sqlite3.Connection, adesso: str, conta_token: Callable[[str], int],
               budget: int = BUDGET_STORIA, finestra: tuple[int, int] | None = None,
               prima_di_id: int | None = None, con_id: bool = False) -> Contesto:
    """Storia = i messaggi prima di `adesso` e (se dato) prima del messaggio `prima_di_id`: il confine si fissa
    col numero del messaggio, perché più messaggi possono avere lo stesso secondo.
    `finestra` = (primo scambio, token) già calcolati, per non rifare la ricerca binaria (esame)."""
    limite = prima_di_id or registro.ultimo_id(con) + 1
    scambi = [s for s in registro.carica_scambi(con) if s["ts"] <= adesso and max(s["ids"]) < limite]
    primo, token = finestra or inizio_finestra(scambi, conta_token, budget, con_id)
    oggi = datetime.fromisoformat(adesso).date()
    fatti, scade = sezione_fatti(registro.fatti_validi(con, adesso), oggi)
    intro = INTRO_STORIA_ID if con_id else INTRO_STORIA
    parti = [persona(), sezione_identita(registro.identita_attiva(con)), fatti,
             sezione_calendario(registro.giorni(con, adesso, limite)), intro + testo_storia(scambi[primo:], con_id)]
    return Contesto(sistema="\n\n".join(p for p in parti if p), ultimo_id=max(scambi[-1]["ids"]) if scambi else 0,
                    primo_id=scambi[primo]["ids"][0] if scambi else 0, primo_ts=scambi[primo]["ts"] if scambi else adesso,
                    token_storia=token, costruito=adesso, scade=scade)
