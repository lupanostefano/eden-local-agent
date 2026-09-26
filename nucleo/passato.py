"""
passato.py — il messaggio chiede di ricordare? Allora più risultati di ricerca (12 invece di 6); il ragionamento
è sempre acceso per scelta di Stefano (`eden.PENSA_SEMPRE`), altrimenti si accende solo qui.

Regole sul testo, senza chiamare modelli (una sola chiamata per messaggio). Un falso positivo costa ~5 s di
ragionamento; un falso negativo costa una risposta sul passato meno precisa (esame: 74 → 78/83 con ragionamento).
"""
from __future__ import annotations

import re

from nucleo.ricerca import norm

_MESI = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"

_RICORDO = re.compile("|".join([
    r"\bricord",                                                             # ricordi, ti ricordi, ricordami
    r"\b(ti|te|t) (ho|avevo|avrei|dissi|diedi|chiesi|scrissi|raccontai|feci|mandai|promisi)\b",
    r"\bti (ho|avevo) (mai )?(detto|chiesto|scritto|raccontato|parlato|promesso|citato|dato|mandato|fatto|confidato|consigliato|incollato)",
    r"\bmi (hai|avevi) (mai )?(detto|chiesto|scritto|raccontato|parlato|promesso|risposto|dato)",
    r"\b(mi |ti )?(dicesti|rispondesti|scegliesti|chiedesti)\b",
    r"\babbiamo (mai )?(parlato|discusso|detto|fatto|litigato)", r"\bci siamo (parlati|sentiti|conosciuti|visti)",
    r"\b(prima|ultima) volta\b", r"\bl altro giorno\b", r"\bieri\b", r"\b(settimana|mese) scors",
    rf"\b\d{{1,2}} ({_MESI})\b", rf"\bprimo ({_MESI})\b", rf"\b(a|ad|di|in|fine|inizio|meta) ({_MESI})\b",
    r"\bquanti (anni|giorni|mesi|settimane)\b",
    r"\b(giusto|vero|no|corretto)\s*\?", r"\bo e un ricordo\b",
    r"\b(mio|mia|miei|mie)\b[^.!]*\?", r"\b(parlami|dimmi|raccontami)\b[^.!?]*\b(mio|mia|miei|mie|di me)\b",
    r"\b(cosa|che cosa) (sai|pensi di sapere) (di|su) me\b",
    r"\b(faccio|sono nato|abito|guido|preferisco|tifo|cucinare|lavoro faccio)\b[^.!]*\?",
    r"\bda chi\b[^.!]*\?", r"\bprogett\w*\b[^.!]*\?", r"\bnel testo\b[^.!]*\?", r"\b(ti ho|ho) (citato|detto)\b",
    r"\bsecondo me\s*\?", r"\bsono cresciuto\b",
    r"\b(che|quale|quali|quanti|quante|come|dove|con cui)\b[^?]{0,60}\b(ho|uso|valuto)\b[^?]*\?",  # "che animali ho?"
    # imperfetto in una domanda ("come mi sentivo?", "dove eravamo rimasti?"): verbi comuni e plurali in -vamo/-vano
    r"\b(avev|facev|dicev|stav|pensav|sapev|volev|potev|dovev|sentiv|parlav|chiamav|piacev|er)(o|i|a|amo|ate|ano)\b[^.!]*\?",
    r"\b\w{2,}[aei]va(mo|no)\b[^.!]*\?",
]))


def sul_passato(testo: str) -> bool:
    # il punto nei numeri ("0.30") non chiude la frase
    return bool(_RICORDO.search(re.sub(r"(\d)\.(\d)", r"\1,\2", norm(testo))))
