"""
diario.py — il diario di Eden: un file al giorno (data/logs/diario/AAAA-MM-GG.md), da leggere come un racconto.

Per ogni scambio: cosa ha scritto Stefano, cosa ha fatto Eden (ricordi cercati, siti cercati e pagine aperte con
indirizzo, no ricevuti dal filtro o dalla privacy), l'inizio del suo ragionamento, la risposta, le fonti citate.
Il testo intero resta nel registro (data/eden.db); qui ci sono estratti e il numero del messaggio.
Un errore di scrittura non deve mai fermare una risposta: si annota nel log tecnico e si va avanti.
"""
from __future__ import annotations

import logging
import threading
from urllib.parse import urlparse

import eden_paths as P
from nucleo import contesto

log = logging.getLogger("nucleo")
_scrittura = threading.Lock()
MAX_TESTO = 700            # caratteri di un messaggio riportato nel diario
MAX_PENSIERO = 900         # caratteri del ragionamento riportati


def _scrivi(ts: str, blocco: str) -> None:
    giorno = ts[:10]
    try:
        with _scrittura:
            P.DIARIO_DIR.mkdir(parents=True, exist_ok=True)
            f = P.DIARIO_DIR / f"{giorno}.md"
            nuovo = not f.exists()
            with f.open("a", encoding="utf-8") as fh:
                if nuovo:
                    fh.write(f"# Diario di Eden — {contesto.data_it(giorno + 'T00:00:00', ora=False)}\n\n")
                fh.write(blocco.rstrip() + "\n\n")
    except OSError:
        log.exception("diario: scrittura fallita")


def _cita(testo: str, massimo: int = MAX_TESTO) -> str:
    """Il testo come citazione markdown, accorciato."""
    t = testo.strip()
    if len(t) > massimo:
        t = t[:massimo].rstrip() + f" […+{len(t) - massimo} caratteri]"
    return "\n".join("> " + r if r.strip() else ">" for r in t.splitlines())


def _sito(url: str) -> str:
    return (urlparse(url).hostname or url).removeprefix("www.")


def azione(c: dict) -> list[str]:
    """Le righe del diario per una chiamata di strumento (`Turno.chiamate`)."""
    nome, a = c["nome"], c.get("argomenti") or {}
    if c.get("errore"):
        cosa = {"cerca_web": f"cercare sul web «{a.get('domanda', '')}»", "leggi_pagina": f"aprire {a.get('dove', '')}",
                "cerca_ricordi": f"cercare nei ricordi «{a.get('domanda', '')}»",
                "leggi_giorno": f"rileggere il giorno {a.get('giorno', '')}"}.get(nome, nome)
        return [f"- **Non riuscito** — voleva {cosa}: {c['errore']}"]
    d = c.get("dettaglio") or {}
    if nome == "cerca_web":
        righe = [f"- **Cerca sul web** «{a.get('domanda', '')}»{' (notizie)' if a.get('notizie') else ''} → "
                 f"{len(d.get('risultati', []))} risultati" + (f" (più {d['scartati']} scartati dal filtro sugli argomenti esclusi)" if d.get("scartati") else "")]
        return righe + [f"  - {r['titolo']} — {r['url']}" for r in d.get("risultati", [])]
    if nome == "leggi_pagina":
        return [f"- **Apre la pagina** «{d.get('titolo') or _sito(d.get('url', ''))}» — {d.get('url', a.get('dove', ''))} "
                f"({d.get('caratteri', '?')} caratteri{', solo l’inizio' if d.get('tagliata') else ''})"]
    if nome == "cerca_ricordi":
        periodo = (f" dal {a['dal']}" if a.get("dal") else "") + (f" al {a['al']}" if a.get("al") else "")
        return [f"- **Cerca nei ricordi** «{a.get('domanda', '')}»{periodo} → {c.get('scambi', 0)} scambi"]
    if nome == "leggi_giorno":
        return [f"- **Rilegge il giorno** {a.get('giorno', '')} → {c.get('scambi', 0)} scambi"]
    if nome == "rileggi_pensiero":
        return [f"- **Rilegge il proprio ragionamento** ({'scambio #' + str(a['scambio']) if a.get('scambio') else 'l’ultima risposta'})"]
    return [f"- {nome}"]


def _fonti(citazioni: list) -> str:
    voci = []
    for c in citazioni:
        segno = {"ok": "✓", "debole": "~", "falsa": "✗"}.get(c.stato, "?")
        voci.append(f"{segno} {_sito(c.url)}" if c.url else f"{segno} ricordo #{c.scambio or c.ids[0]}")
    return ", ".join(voci)


def scrivi_turno(ts: str, domanda: str, risposta: str, r, t, id_eden: int | None = None, interrotta: bool = False) -> None:
    """Uno scambio con Stefano. `r` = turno.Risultato, `t` = strumenti.Turno."""
    b = [f"## {ts[11:16]} · Stefano scrive", _cita(domanda), ""]
    if t.chiamate:
        b += ["**Cosa ha fatto**"] + [riga for c in t.chiamate for riga in azione(c)] + [""]
    if r.pensiero.strip():
        p = r.pensiero.strip()
        resto = f" (ragionamento intero: messaggio #{id_eden})" if id_eden and len(p) > MAX_PENSIERO else ""
        b += ["**Cosa ha pensato** (l’inizio)", _cita(p, MAX_PENSIERO) + resto, ""]
    if r.bozza:
        b += ["**Prima versione, scartata** (una fonte non reggeva)", _cita(r.bozza, 400), ""]
    b += ["**Risposta**" + (" (interrotta)" if interrotta else ""), _cita(risposta), ""]
    if r.citazioni:
        b += [f"**Fonti citate:** {_fonti(r.citazioni)}", ""]
    nota = [f"{r.sec:.0f} s", f"{r.giri} giri"] + ([f"{r.ripieghi} ripiego senza ragionamento"] if r.ripieghi else []) + \
           [f"contesto {r.contesto:,} token".replace(",", ".")]
    b.append("_" + " · ".join(nota) + (f" · messaggio #{id_eden}" if id_eden else "") + "_")
    _scrivi(ts, "\n".join(b))


def scrivi_evento(ts: str, titolo: str, testo: str = "") -> None:
    """Un fatto che non è uno scambio: risveglio, storia riletta, errore."""
    _scrivi(ts, f"## {ts[11:16]} · {titolo}" + (f"\n{testo}" if testo else ""))
