"""
turno.py — un turno di Eden: il modello risponde; se chiede uno strumento lo si esegue e riprende.

Serve sia a `eden.py` (la risposta vera) sia all'esame (stessa strada, senza scrivere nel registro).
Il modello viene chiamato sempre con le stesse definizioni di strumenti: così la parte fissa del prompt resta in cache.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

from nucleo import citazioni, modello, strumenti
from nucleo.citazioni import Citazione
from nucleo.strumenti import Turno

MAX_STRUMENTI = 5     # giri con strumenti per turno (una ricerca sul web e due pagine ne usano 3); dopo si risponde con ciò che c'è
AVVISO_ULTIMO = ("(Hai usato tutti gli strumenti che questo messaggio consente: rispondi ora a Stefano con ciò che hai trovato, "
                 "senza chiamare altri strumenti.)")
CHIAMATA_IN_CHIARO = "<tool_call>"     # con gli strumenti vietati il modello può scriverla come testo


@dataclass
class Risultato:
    """Cosa è uscito dal turno. Lo riempie `genera` man mano: se il modello si interrompe resta il testo arrivato."""
    testo: str = ""
    pensiero: str = ""
    giri: int = 0                            # richieste fatte al modello
    ripieghi: int = 0                        # giri rifatti senza ragionamento (girava a vuoto o risposta vuota)
    sec: float = 0.0                         # tempo di tutti i giri
    token_scritti: int = 0
    primo_testo_sec: float | None = None     # dall'inizio del turno alla prima parola della risposta
    prompt_token: int = 0                    # dell'ultima richiesta: quanti token nuovi ha letto…
    cache_token: int = 0                     # …e quanti erano già in cache (insieme: la dimensione del contesto)
    contesto_iniziale: int = 0               # dimensione del contesto alla prima richiesta, prima dei risultati degli strumenti
    citazioni: list[Citazione] = field(default_factory=list)   # le fonti della risposta finale, verificate
    bozza: str | None = None                 # la prima risposta, se una fonte non reggeva e Eden l'ha rifatta

    def statistiche(self) -> dict:
        return {"sec": round(self.sec, 2), "primo_testo_sec": self.primo_testo_sec, "giri": self.giri, "ripieghi": self.ripieghi,
                "prompt_token": self.prompt_token, "cache_token": self.cache_token, "token_scritti": self.token_scritti}

    @property
    def contesto(self) -> int:
        return self.prompt_token + self.cache_token


def gira_a_vuoto(pensiero: str) -> bool:
    """Il ragionamento ripete la stessa frase lunga: il modello non ne esce da solo (visto: 100 s e nessuna risposta)."""
    coda = pensiero[-160:]
    return len(pensiero) > 640 and pensiero.count(coda) >= 3


def genera(msgs: list[dict], turno: Turno, pensa: bool, r: Risultato) -> Iterator[dict]:
    """Eventi per la pagina: pensa · testo · scarta (il testo arrivato era un preambolo prima di uno strumento) · cerca.
    `msgs` cresce con le chiamate di strumenti e i loro risultati. Se il ragionamento gira a vuoto o la risposta esce
    vuota, si rifà il giro una volta senza ragionamento (risponde subito); se anche quello è vuoto, `r.testo` resta vuoto."""
    usati, ripiego, avvisato = 0, False, False
    while True:
        ultimo = usati >= MAX_STRUMENTI       # niente altri strumenti: ora si risponde
        if ultimo and not avvisato:
            msgs.append({"role": "user", "content": AVVISO_ULTIMO})
            avvisato = True
        pensiero, chiamate, stat, t0 = "", [], {}, time.time()
        r.testo, r.giri = "", r.giri + 1
        controllato = 0
        flusso = modello.genera(msgs, pensa and not ripiego, strumenti=strumenti.DEFINIZIONI, senza_strumenti=ultimo)
        for tipo, dato in flusso:
            if tipo == "testo":
                r.testo += dato
                yield {"tipo": "testo", "testo": dato}
            elif tipo == "pensiero":
                if not pensiero:
                    yield {"tipo": "pensa"}
                pensiero += dato
                if len(pensiero) - controllato >= 320:
                    controllato = len(pensiero)
                    if gira_a_vuoto(pensiero):
                        flusso.close()        # chiude la connessione: il server smette di generare
                        break
            elif tipo == "chiamata":
                chiamate.append(dato)
            else:
                stat = dato
        r.pensiero += pensiero
        if CHIAMATA_IN_CHIARO in r.testo:     # una chiamata scritta come testo non è una risposta
            r.testo = r.testo.split(CHIAMATA_IN_CHIARO)[0]
            yield {"tipo": "scarta"}
        if stat.get("primo_testo_sec") and not chiamate:
            r.primo_testo_sec = round(r.sec + stat["primo_testo_sec"], 2)
        r.sec += stat.get("sec") or (time.time() - t0)
        r.token_scritti += stat.get("token_scritti") or 0
        if stat:
            r.prompt_token, r.cache_token = stat.get("prompt_token") or 0, stat.get("cache_token") or 0
            r.contesto_iniziale = r.contesto_iniziale or r.contesto
        if chiamate and not ultimo:
            if r.testo.strip():
                yield {"tipo": "scarta"}
            msgs.append({"role": "assistant", "content": r.testo, "reasoning_content": pensiero,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["nome"], "arguments": c["argomenti"] or "{}"}}
                                        for c in chiamate]})
            for c in chiamate:
                yield {"tipo": "cerca", "testo": strumenti.descrivi(c["nome"], _argomenti(c["argomenti"]))}
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": strumenti.esegui(turno, c["nome"], c["argomenti"])})
            usati += 1
            continue
        r.testo = r.testo.strip()
        if r.testo or ripiego:
            return
        r.ripieghi, ripiego = r.ripieghi + 1, True     # giro a vuoto o risposta vuota: si riprova senza ragionamento
        yield {"tipo": "cerca", "testo": "ci ripenso"}


def _argomenti(grezzi: str) -> dict:
    try:
        a = json.loads(grezzi or "{}")
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def rispondi(msgs: list[dict], turno: Turno, pensa: bool, r: Risultato) -> Iterator[dict]:
    """Un turno completo: risposta (con gli strumenti che vuole), verifica delle fonti che cita e, se una non regge
    (numero inventato, parole tra «» che non ci sono), una sola correzione. Stessi eventi di `genera`."""
    yield from genera(msgs, turno, pensa, r)
    r.citazioni = citazioni.verifica(r.testo, turno)
    sbagliate = citazioni.da_correggere(r.citazioni)
    if sbagliate:
        r.bozza = r.testo
        yield {"tipo": "scarta"}
        yield {"tipo": "cerca", "testo": "ricontrollo le mie fonti"}
        msgs += [{"role": "assistant", "content": r.bozza},
                 {"role": "user", "content": citazioni.nota_correzione(sbagliate)}]
        yield from genera(msgs, turno, pensa, r)
        r.citazioni = citazioni.verifica(r.testo, turno)
