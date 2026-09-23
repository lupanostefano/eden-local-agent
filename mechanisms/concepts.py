"""concepts.py — estrazione concetti rule-based dal testo.

Usato da memory.py per collegare episodi ai Concept nel grafo Kuzu.
Estratto da experiment/migrate_to_graph.py (2026-09-03, potatura ricerca).
"""
from __future__ import annotations

import re


# Stop-list italiana per concept extraction.
# v2.0 (2026-05-04, F4): triplicata per impedire ingresso di concetti spazzatura
# nel grafo Kuzu. Allineata con _DNA_SEMANTIC_BLACKLIST in digital_sleep.py
# (filtro a due livelli: estrazione + cristallizzazione).
STOPWORDS = {
    # ─ Articoli, preposizioni, congiunzioni
    "il", "la", "lo", "le", "gli", "i", "un", "una", "uno", "del", "della",
    "delle", "dei", "degli", "dal", "dalla", "dallo", "dalle", "dagli",
    "che", "cui", "non", "per", "con", "tra", "fra",
    "sul", "sulla", "sullo", "sulle", "sugli",
    "alla", "allo", "alle", "agli", "all", "all'",
    "ed", "od", "ma", "se", "ne", "ci", "vi",
    # ─ Dimostrativi e indefiniti
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
    "stesso", "stessa", "stessi", "stesse",
    "ogni", "alcun", "alcuno", "alcuna", "alcuni", "alcune",
    "tutto", "tutti", "tutta", "tutte", "altro", "altra", "altri", "altre",
    "qualche", "qualcuno", "qualcosa", "nulla", "nessuno", "nessuna",
    # ─ Verbi ausiliari/modali
    "essere", "avere", "fare", "stare", "andare", "venire", "dire", "dare",
    "potere", "dovere", "volere", "sapere",
    "siamo", "sono", "sei", "era", "erano", "ero", "eri", "fosse", "fossero",
    "hai", "ha", "ho", "abbiamo", "avete", "hanno", "avere",
    "fa", "faccio", "fai", "fanno", "facciamo", "fate",
    "stato", "stata", "stati", "state", "essendo", "essere",
    # ─ Avverbi e quantificatori
    "molto", "poco", "tanto", "troppo", "abbastanza", "anche", "ancora",
    "sempre", "mai", "spesso", "talvolta", "qui", "lì", "là", "qua", "ci",
    "sopra", "sotto", "dentro", "fuori", "vicino", "lontano",
    "prima", "dopo", "presto", "tardi", "subito",
    "più", "meno", "bene", "male",
    # ─ Avverbi-rinforzo
    "veramente", "davvero", "proprio", "certamente", "sicuramente",
    "probabilmente", "forse", "magari", "ovviamente",
    # ─ Pronomi personali e possessivi
    "io", "tu", "lui", "lei", "noi", "voi", "loro", "me", "te",
    "mio", "mia", "miei", "mie", "tuo", "tua", "tuoi", "tue",
    "suo", "sua", "suoi", "sue", "nostro", "nostra", "vostro", "vostra",
    # ─ Interrogativi e relativi
    "perché", "perche", "perchè", "quando", "dove", "come",
    "quale", "quali", "quanto", "quanta", "quanti", "quante",
    # ─ Riferimenti progetto (sempre presenti, non discriminanti)
    "eden", "stefano", "ciao", "ora", "oggi", "ieri", "domani",
    # ─ Categorie temporali generiche
    "giorno", "giorni", "tempo", "tempi", "ora", "ore", "minuto", "minuti",
    "anno", "anni", "mese", "mesi", "settimana", "settimane",
    "primo", "prima", "ultimo", "ultima", "secondo", "seconda",
    # ─ Categorie generiche prive di valore identitario
    "modo", "modi", "cosa", "cose", "tema", "temi", "punto", "punti",
    "volta", "volte", "caso", "casi", "tipo", "tipi", "forma", "forme",
    "parte", "parti", "esempio", "esempi", "situazione", "situazioni",
    "aspetto", "aspetti", "elemento", "elementi", "fattore", "fattori",
    # ─ NUOVO v2.0: meta-discorso (Eden parla di parlare)
    "riassunto", "riassunti", "messaggio", "messaggi", "conversazione",
    "conversazioni", "scambio", "scambi", "risposta", "risposte",
    "domanda", "domande", "frase", "frasi", "discorso", "discorsi",
    "narrazione", "narrazioni", "contenuto", "contenuti",
    # ─ NUOVO v2.0: verbi descrittivi del dialogo
    "propone", "propongo", "propongono", "proponi", "proporre",
    "spiega", "spiego", "spiegano", "spiegare", "spiegazione", "spiegazioni",
    "afferma", "affermo", "affermano", "affermazione", "affermazioni", "affermare",
    "suggerisce", "suggerisco", "suggeriscono", "suggerimento", "suggerire",
    "riconosce", "riconosco", "riconoscono", "riconoscere",
    "considera", "considero", "considerano", "considerazione",
    "discute", "discuto", "discutono", "discussione", "discussioni",
    "menziona", "menziono", "menzionano",
    "indica", "indico", "indicano", "indicazione",
    "osserva", "osservo", "osservano", "osservazione", "osservazioni",
    "esplora", "esploro", "esplorano", "esplorazione",
    "analizza", "analizzo", "analizzano", "analisi",
    "valuta", "valuto", "valutano", "valutazione",
    "esprime", "esprimo", "esprimono", "espressione", "esprimere",
    "scrive", "scrivo", "scrivono", "scrivere",
    "racconta", "racconto", "raccontano", "raccontare",
    "chiede", "chiedo", "chiedono", "chiedere",
    "dice", "dico", "dicono", "dire", "detto",
    "parla", "parlo", "parlano", "parlare",
    "risponde", "rispondo", "rispondono", "rispondere",
    # ─ NUOVO v2.0: sostantivi astratti vacui
    "concetto", "concetti", "concettuale",
    "interesse", "interessi", "interessante", "interessato", "interessata",
    "metafore", "metafora", "metaforica", "metaforico",
    "processo", "processi", "approccio", "approcci", "metodo", "metodi",
    "qualità", "qualita", "contesto", "contesti",
    # ─ Dominio progetto (non distintivi se Eden parla di emozioni di default)
    "emozione", "emozioni", "emozionale", "emozionali",
    "emotivo", "emotiva", "emotivi", "emotive", "emotivamente",
    "sentimento", "sentimenti", "sensazione", "sensazioni",
    # ─ NUOVO v2.0: meta-memoria (auto-riferimento di default)
    "ricordi", "ricordo", "ricorda", "ricordano", "ricordare", "ricordata",
    "memoria", "memorie", "pensieri", "pensiero", "pensa", "penso", "pensare",
    # ─ Valutazioni morali generiche
    "vero", "falso", "giusto", "sbagliato", "buono", "cattivo",
    # ─ Frammenti tronchi da tokenizzazione regex senza apostrofo
    "sull", "dell", "nell", "all", "coll", "dall",
    "quest", "quel", "tant", "molt", "tutt",
}

WORD_RE = re.compile(r"[a-zàèéìòù]{5,}", re.IGNORECASE)
# v2.0: lunghezza minima 5 (era 4) per ridurre cattura di frammenti corti.

# Whitelist soft v2.0: concetti tematici che meritano di passare anche se
# non in stopword. Lasciato vuoto per default — la blacklist è l'unica
# barriera, ma struttura predisposta per future SPEC.
_WHITELIST_THEMATIC: set[str] = set()


def extract_concepts(text: str, top_k: int = 5) -> list:
    """Estrae fino a top_k token candidati come concetti dal testo.

    v2.0 (2026-05-04, F4): stopword triplicata + lunghezza min 5 +
    filtro frammenti tronchi per evitare DNA spazzatura.
    """
    if not text:
        return []
    seen = set()
    out = []
    for m in WORD_RE.findall(text.lower()):
        if m in STOPWORDS or m in seen:
            continue
        # Frammenti tronchi: parole che terminano in patterns sospetti.
        # "sull" non triggera la stopword se passa da forme come "sulle"->"sull"
        # ma la guard di lunghezza ≥ 5 già lo blocca. Doppia sicurezza:
        if len(m) < 5:
            continue
        seen.add(m)
        out.append(m)
        if len(out) >= top_k:
            break
    return out
