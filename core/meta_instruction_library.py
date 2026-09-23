"""
meta_instruction_library.py — Libreria universale pattern M2 (EV-015)
Versione: v1.0  |  Data: 2026-05-02  |  Usata da: correction_handler.py

Motivazione
-----------
Il Behavioral Constraint Register (M2) richiede il rilevamento di
meta-istruzioni di stile nell'italiano naturale. Un singolo blocco regex
hardcoded è fragile: ogni frase non contemplata apre un buco silenzioso.

Questo modulo centralizza TUTTI i dati di M2 in un unico punto:
pattern, parole-trigger, simboli noti, skip-words, configurazione LLM.
correction_handler.py importa questi dati e li usa — non definisce pattern.

ARCHITETTURA A 3 TIER
----------------------

  Tier 0 — Keyword pre-filter (O(1), zero latenza)
    Frozenset ~35 parole-trigger. Se nessuna è presente nel messaggio
    dell'utente, M2 termina immediatamente. Elimina ~85% dei messaggi
    normali senza elaborazione ulteriore. Alta recall intenzionale —
    la precision è demandata ai tier successivi.

  Tier 1 — Regex pattern library (compilato da PATTERN_GROUPS_T1)
    ~65 pattern in 6 categorie semantiche:
      1. imperative_direct   — smetti di, basta con, togli, stop
      2. modal_prohibition   — non devi, non puoi, non dovresti
      3. preference_explicit — preferisco che tu, preferirei
      4. negative_desire     — non voglio, non mi va
      5. dislike_expression  — mi dà fastidio, odio quando
      6. future_instruction  — d'ora in poi, da questo momento
    Copertura stimata: ~80% dei casi reali.
    Zero latenza, zero dipendenze.

  Tier 2 — LLM micro-pass (phi3:mini, fail-silent)
    Attivato SOLO se Tier 0 matcha ma Tier 1 non trova nulla.
    Gestisce costruzioni indirette, idiomatiche, non-standard.
    Latenza ~80-150ms. Fail-silent se phi3:mini non disponibile.
    Copertura residua stimata: ~15-18% (totale ~95-98%).

ESTENSIONE (senza toccare correction_handler.py)
-------------------------------------------------
  - Aggiungere parole a TRIGGER_KEYWORDS  → estende Tier 0
  - Aggiungere pattern a PATTERN_GROUPS_T1 → estende Tier 1
  - Aggiungere esempi a LLM_TIER2["prompt_examples"] → calibra Tier 2
  - correction_handler.py compila il regex da PATTERN_GROUPS_T1 all'import

ANTI-HARKING
------------
Ogni modifica a questo file deve:
  1. Incrementare _META_LIBRARY_VERSION
  2. Aggiungere una riga al changelog in fondo al file
  3. Essere

CHANGELOG
---------
  v1.0  2026-05-02  Creazione. Migrazione da regex hardcoded in
                    correction_handler.py v1.4. Aggiunta architettura
                    3 tier. ~65 pattern, ~35 trigger keywords.
"""

_META_LIBRARY_VERSION = "v1.1"


# ═══════════════════════════════════════════════════════════════════════════
# TIER 0 — KEYWORD PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════
#
# Parole-trigger: presenza NECESSARIA (non sufficiente) per meta-istruzione.
# Se nessuna di queste è nel testo → return False immediato in M2.
#
# Criteri di inclusione:
#   - Parola che appare in quasi tutte le forme di meta-istruzione italiana
#   - Falsi positivi accettabili (la precision è compito di Tier 1)
#   - Esclude parole così comuni da rendere Tier 0 inutile (es. "non", "il")
#
# Per aggiungere: inserire la forma base minuscola. Il check usa msg.lower().

TRIGGER_KEYWORDS: frozenset[str] = frozenset({
    # ── Imperativi diretti ──────────────────────────────────────────────────
    "smetti",       # smetti di, smettila di, smetta di
    "smettila",     # smettila con
    "basta",        # basta con
    "evita",        # evita di usare
    "togli",        # togli le metafore
    "elimina",      # elimina queste frasi
    "abbandona",    # abbandona questo stile
    "rinuncia",     # rinuncia a usare
    "lascia",       # lascia perdere
    "stop",         # stop con
    # ── Modali proibitivi ───────────────────────────────────────────────────
    "devi",         # non devi scrivere (con "non" nel contesto)
    "puoi",         # non puoi usare
    "dovresti",     # non dovresti fare
    "permesso",     # non ti è permesso
    "permetti",     # non ti permetto
    # ── Preferenza / desiderio ──────────────────────────────────────────────
    "preferisco",   # preferisco che tu smetta
    "preferirei",   # preferirei che tu non usassi
    "vorrei",       # vorrei che tu non usassi più
    "voglio",       # non voglio più
    "piacerebbe",   # mi piacerebbe che tu smettessi
    # ── Fastidio / avversione ───────────────────────────────────────────────
    "fastidio",     # mi dà fastidio
    "infastidisce", # mi infastidisce
    "irrita",       # mi irrita
    "stanca",       # mi stanca (nel senso di annoiare)
    "annoia",       # mi annoia
    "tollero",      # non tollero
    "tollerare",    # non riesco a tollerare
    "sopporto",     # non sopporto
    "sopportare",   # non riesco a sopportare
    "detesto",      # detesto quando
    "odio",         # odio quando
    # ── Gradimento ─────────────────────────────────────────────────────────
    "piace",        # non mi piace quando (serve anche per "non mi piace")
    # ── Istruzioni temporali ────────────────────────────────────────────────
    "futuro",       # in futuro, per il futuro
    "momento",      # da questo momento
})

# Verbi d'azione deboli: trigger SOLO in coppia con "non" nel messaggio.
# Troppo comuni come trigger singoli, ma "non usare/fare/scrivere/mettere"
# è quasi sempre una forma di divieto. Tier 0 li controlla separatamente
# (vedi correction_handler.py::_tier0_check).
TRIGGER_VERBS_WEAK: frozenset[str] = frozenset({
    "usare", "usarmi",
    "scrivere", "scrivermi",
    "mettere", "mettermi",
    "fare",
    "dire", "dirmi",
    # Infiniti azione-generici — presenti in istruzioni come "non concludere con X"
    "concludere", "aggiungere", "inserire", "includere",
    "mandare", "mostrare", "iniziare", "finire", "terminare",
    "ripetere", "continuare", "usarla", "farlo", "farne",
})


# ═══════════════════════════════════════════════════════════════════════════
# SIMBOLI NOTI — bypass del filtro lunghezza parola
# ═══════════════════════════════════════════════════════════════════════════
#
# Simboli che vengono riconosciuti direttamente prima del word-split.
# Necessari perché i simboli non-alfabetici (es. <3, len=2) verrebbero
# filtrati dal check len>=3 nella target extraction.

KNOWN_BAN_SYMBOLS: frozenset[str] = frozenset({
    "<3",    # simbolo cuore — il caso che ha motivato questa libreria
    ":)",    # sorriso
    ":(",    # tristezza
    ":-)",   # sorriso classico
    ":-(",   # tristezza classica
    ":D",    # risata
    "xD",    # risata
    ";)",    # strizzatina d'occhio
    "...",   # puntini di sospensione ripetuti
})


# ═══════════════════════════════════════════════════════════════════════════
# TARGET SKIP WORDS — parole da ignorare nell'estrazione del target
# ═══════════════════════════════════════════════════════════════════════════
#
# Dopo il match del pattern, la funzione _extract_ban_target cerca la prima
# parola significativa nel gruppo catturato. Queste parole vengono saltate
# perché non rappresentano il target reale del vincolo.
#
# Categorie incluse:
#   - Dimostrativi e aggettivi indefiniti
#   - Avverbi modali/temporali
#   - Preposizioni articolate (che passano len>=3)
#   - Congiunzioni e pronomi relativi
#   - Forme verbali comuni della 2a persona
#   - Sostantivi generici non utili come target

TARGET_SKIP_WORDS: frozenset[str] = frozenset({
    # Dimostrativi
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "tali", "simili",
    # Aggettivi quantitativi / indefiniti
    "certo", "certa", "certi", "certe",
    "grande", "grandi", "piccolo", "piccola",
    "troppo", "troppa", "troppi", "troppe",
    "molto", "molta", "molti", "molte",
    "stesso", "stessa", "stessi", "stesse",
    "ogni", "qualsiasi", "qualunque", "alcun", "alcuna",
    # Avverbi temporali / modali
    "sempre", "mai", "ancora", "già", "poi", "quando",
    "spesso", "raramente", "continuamente", "costantemente",
    "più", "meno", "circa", "quasi", "solo", "anche",
    "pure", "anzi", "però", "invece",
    "cosi", "così",
    # Preposizioni articolate (3 char, passano il filtro base)
    "nel", "nella", "nelle", "negli",
    "del", "della", "delle", "degli",
    "dal", "dalla", "dalle", "dagli",
    "sul", "sulla", "sulle", "sugli",
    "col", "colla", "colle", "cogli",
    # Congiunzioni / pronomi
    "che", "chi", "cui", "quale", "quali",
    "dove", "come", "perché", "poiché",
    # Forme verbali 2a persona (da skippare come target)
    "sei", "stai", "fai", "vai", "hai", "dici",
    "metti", "usi", "scrivi", "porti", "dai",
    # Gerundi comuni
    "mettendo", "scrivendo", "usando", "facendo",
    "dicendo", "portando",
    # Infiniti che appaiono nel raw_target
    "usare", "scrivere", "mettere", "fare",
    "dire", "portare", "essere",
    # Preposizioni semplici (len 3, non filtrate dal check lunghezza)
    "con", "per", "tra", "fra",
    # Sostantivi generici non utili come target
    "frase", "frasi", "messaggio", "messaggi",
    "risposta", "risposte", "modo", "modi",
    "cosa", "cose", "parola", "parole",
    "testo", "contenuto", "contesto",
    "volta", "volte", "fine", "inizio",
})


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — PATTERN GROUPS
# ═══════════════════════════════════════════════════════════════════════════
#
# Lista ordinata di gruppi di pattern.
#
# Schema di ogni gruppo:
#   name        (str)   — ID machine-readable, usato nei log
#   description (str)   — Descrizione italiana del gruppo
#   confidence  (float) — Stima di precision (P(vera meta-istruzione | match))
#   examples    (list)  — Frasi reali/attese che il gruppo deve catturare
#   patterns    (list)  — Regex prefix: tutto ciò che PRECEDE il target
#
# COME SCRIVERE UN PATTERN:
#   - Il pattern cattura il prefisso della meta-istruzione
#   - Il target (.{1,80}) viene catturato automaticamente DOPO il pattern
#   - Usare \s+ tra parole (non spazi letterali)
#   - Terminare con \s+ se il target è una parola, senza spazio se è un simbolo
#   - Usare (?:...)? per parti opzionali
#   - Usare [uù] per accenti opzionali (italiano digitato vs corretto)
#   - Usare re.IGNORECASE (già impostato nel compilatore)
#
# COPERTURA TARGET:
#   Tier 1 cattura ~80% dei casi strutturati. Il restante ~20% (frasi
#   indirette, costruzioni idiomatiche) è gestito da Tier 2 LLM.

META_BAN_EXPIRY: int = 30  # scambi prima della scadenza del vincolo

PATTERN_GROUPS_T1: list[dict] = [

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "imperative_direct",
        "description": "Imperativo diretto — smetti di, basta con, togli, stop",
        "confidence": 0.95,
        "examples": [
            "smettila di usare le metafore astronomiche",
            "smetti di scrivere <3",
            "smettila con i cuori",
            "smettila con le domande retoriche",
            "basta con le analogie cosmiche",
            "basta usare il simbolo del cuore",
            "togli i puntini di sospensione",
            "elimina le frasi poetiche",
            "lascia perdere le analogie con l'universo",
            "stop con le domande alla fine",
            "stop di usare le metafore",
            "abbandona questo stile narrativo",
            "rinuncia a usare questi simboli",
        ],
        "patterns": [
            r"smetti(?:la)?\s+di\s+(?:usare\s+|scrivere\s+|mettere\s+|fare\s+)?",
            r"smetti(?:la)?\s+con\s+(?:le\s+|i\s+|il\s+|lo\s+|la\s+|gli\s+|queste\s+|questi\s+)?",
            r"basta\s+con\s+(?:le\s+|i\s+|il\s+|lo\s+|la\s+|gli\s+|questi\s+|queste\s+|ogni\s+)?",
            r"basta\s+(?:usare\s+|scrivere\s+|mettere\s+|fare\s+)",
            r"togli\s+(?:le\s+|i\s+|il\s+|questi\s+|queste\s+|ogni\s+)?",
            r"elimina\s+(?:le\s+|i\s+|il\s+|questi\s+|queste\s+)?",
            r"lascia\s+perdere\s+(?:le\s+|i\s+|il\s+|questi\s+|queste\s+)?",
            r"abbandona\s+(?:le\s+|i\s+|il\s+|questi\s+|queste\s+|questo\s+stile\s+)?",
            r"rinuncia\s+a\s+(?:usare\s+|scrivere\s+)?",
            r"stop\s+con\s+(?:le\s+|i\s+|il\s+|lo\s+)?",
            r"stop\s+(?:di\s+)?(?:usare\s+|scrivere\s+|mettere\s+)?",
            r"toglimi\s+(?:le\s+|questi\s+|queste\s+)?",
            r"evita\s+(?:di\s+)?(?:usare\s+|scrivere\s+|mettere\s+|fare\s+)?",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "modal_prohibition",
        "description": "Modale proibitivo — non devi, non puoi, non dovresti",
        "confidence": 0.90,
        "examples": [
            "non devi scrivermi <3 ogni volta",
            "non puoi usare le metafore continuamente",
            "non dovresti usare sempre lo stesso tono",
            "non ti è permesso inventare ricordi",
            "non fare più domande alla fine",
            "non usare più il simbolo del cuore",
            "non scrivere più queste frasi lunghe",
            "non mettere più puntini di sospensione",
        ],
        "patterns": [
            # "non devi + qualsiasi infinito" — copre "concludere", "aggiungere", ecc.
            r"non\s+devi\s+(?:pi[uù]\s+)?\w+(?:are|ere|ire)\s+",
            r"non\s+devi\s+(?:pi[uù]\s+)?(?:scrivermi|usarmi|mettermi|dirmi|farlo|farla|fare)\s+",
            r"non\s+puoi\s+(?:pi[uù]\s+)?(?:scrivere|usare|mettere|fare|dire)\s+",
            r"non\s+dovresti\s+(?:pi[uù]\s+)?(?:scrivere|usare|mettere|fare|dire)\s+",
            r"non\s+ti\s+(?:[èe]\s+permesso|conviene|si\s+addice)\s+(?:di\s+)?(?:usare\s+|scrivere\s+|fare\s+)?",
            r"non\s+fare\s+(?:pi[uù]\s+)?",
            r"non\s+usare\s+(?:pi[uù]\s+)?",
            r"non\s+scrivere\s+(?:pi[uù]\s+)?",
            r"non\s+scrivermi\s+(?:mai\s+|pi[uù]\s+)?",
            r"non\s+mettere\s+(?:pi[uù]\s+)?",
            r"non\s+dire\s+(?:pi[uù]\s+)?(?:quella\s+|queste\s+|questo\s+|frasi\s+)?",
            r"non\s+voglio\s+(?:pi[uù]\s+)?(?:che\s+tu\s+)?(?:scriva|usi|metta|faccia|dica)\s+",
            # bare imperativo negato — "non concludere ogni X con Y"
            r"non\s+\w+(?:are|ere|ire)\s+(?:\w+\s+){0,3}(?:la|il|lo|le|i|gli|ogni|con|la)\s+",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "preference_explicit",
        "description": "Preferenza esplicita — preferisco, preferirei, mi piacerebbe",
        "confidence": 0.85,
        "examples": [
            "preferisco tu smetta di usare il simbolo <3",
            "preferisco che tu non usi le metafore astronomiche",
            "preferirei che tu non scrivessi sempre lo stesso",
            "mi piacerebbe che tu smettessi con i cuori",
            "sarebbe meglio che tu non usassi frasi così lunghe",
            "sarebbe opportuno che tu evitassi queste ripetizioni",
        ],
        "patterns": [
            r"preferisco\s+(?:che\s+tu\s+|tu\s+)?smetta\s+di\s+(?:usare\s+|scrivere\s+|fare\s+)?",
            r"preferisco\s+(?:che\s+tu\s+|tu\s+)?non\s+(?:usi|scriva|metta|faccia|dica)\s+",
            r"preferisco\s+(?:che\s+tu\s+|tu\s+)?eviti\s+",
            r"preferirei\s+(?:che\s+tu\s+)?(?:non\s+)?(?:usassi|scrivessi|mettessi|facessi|dicessi)\s+",
            r"preferirei\s+(?:che\s+tu\s+)?smettessi\s+di\s+(?:usare\s+|scrivere\s+)?",
            r"mi\s+piacerebbe\s+che\s+tu\s+(?:smettessi|non)\s+(?:di\s+|di\s+usare\s+)?",
            r"mi\s+piacerebbe\s+(?:che\s+tu\s+)?(?:smettessi|evitassi)\s+",
            r"sarebbe\s+meglio\s+(?:che\s+tu\s+non\s+|se\s+tu\s+non\s+)",
            r"sarebbe\s+(?:meglio|opportuno)\s+(?:che\s+tu\s+)?(?:evitassi|smettessi)\s+",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "negative_desire",
        "description": "Desiderio negativo — non voglio, non mi va, vorrei che non",
        "confidence": 0.85,
        "examples": [
            "non voglio più che tu usi <3",
            "non mi va che tu scriva sempre così",
            "non voglio più vedere queste frasi",
            "vorrei che tu non usassi più le metafore",
            "non mi va bene che tu usi sempre lo stesso schema",
            "non vorrei più sentire questa struttura ripetitiva",
        ],
        "patterns": [
            r"non\s+voglio\s+(?:pi[uù]\s+)?(?:che\s+tu\s+)?(?:usare|scrivere|mettere|vedere|sentire|leggere)\s+",
            r"non\s+voglio\s+pi[uù]\s+",
            r"non\s+mi\s+va\s+(?:che\s+tu\s+|di\s+)?",
            r"non\s+mi\s+va\s+bene\s+(?:che\s+tu\s+)?",
            r"vorrei\s+che\s+tu\s+(?:smettessi|non)\s+(?:di\s+|di\s+usare\s+|di\s+scrivere\s+)?",
            r"non\s+vorrei\s+(?:pi[uù]\s+)?(?:sentire|vedere|leggere)\s+",
            r"non\s+voglio\s+(?:pi[uù]\s+)?(?:sentire|vedere|leggere)\s+",
            r"non\s+mi\s+piace\s+(?:pi[uù]\s+)?",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "dislike_expression",
        "description": "Espressione di fastidio/avversione — mi dà fastidio, odio quando",
        "confidence": 0.80,
        # NOTA TARGET: per questo gruppo il target è spesso preceduto da una
        # forma verbale (usi, scrivi, metti) o da "quando". _extract_ban_target
        # salta queste parole via TARGET_SKIP_WORDS. In caso di target
        # impreciso, Tier 2 LLM può essere usato come override.
        "examples": [
            "mi dà fastidio quando scrivi <3 sempre",
            "mi infastidisce quando usi queste metafore",
            "odio quando metti il cuore alla fine",
            "non sopporto le domande retoriche alla fine",
            "mi irrita che tu usi sempre le stesse frasi",
            "mi stanca questa ripetizione continua",
            "detesto quando aggiungi il simbolo del cuore",
            "non tollero che tu scriva sempre così",
        ],
        # NOTA: tutti i pattern richiedono un elemento NON-opzionale dopo
        # il verbo di fastidio (quando/che/articolo/forma verbale), per evitare
        # falsi positivi su frasi come "odio io simbolo del cuore".
        "patterns": [
            # mi dà fastidio: richiede "quando/che/se" o articolo
            r"mi\s+d[aà]\s+fastidio\s+(?:quando|che|se)\s+(?:tu\s+)?",
            r"mi\s+d[aà]\s+fastidio\s+(?:il|la|lo|i|gli|le)\s+",
            r"mi\s+d[aà]\s+fastidio\s+(?:questa|questo|questi|queste)\s+",
            # mi infastidisce / irrita / stanca / annoia: richiedono "quando/che"
            r"mi\s+infastidisce\s+(?:quando|che)\s+(?:tu\s+)?",
            r"mi\s+irrita\s+(?:quando|che)\s+(?:tu\s+)?",
            r"mi\s+stanca\s+(?:quando|che|questa|questo)\s+",
            r"mi\s+annoia\s+(?:quando|che|questa|questo)\s+",
            # non sopporto / non tollero: richiedono "quando/che" o articolo
            r"non\s+sopporto\s+(?:quando|che)\s+(?:tu\s+)?",
            r"non\s+sopporto\s+(?:il|la|lo|i|gli|le|questa|questo)\s+",
            r"non\s+tollero\s+(?:quando|che)\s+(?:tu\s+)?",
            r"non\s+tollero\s+(?:il|la|lo|i|gli|le|questa|questo)\s+",
            # non mi piace: richiede forma verbale esplicita
            r"non\s+mi\s+piace\s+(?:quando|che)\s+(?:tu\s+)?(?:usi|scrivi|metti|fai|dici)\s+",
            r"non\s+mi\s+piace\s+(?:il|la|lo|i|gli|le|questa|questo)\s+",
            # detesto / odio: richiedono "quando/che" o articolo (NO optional puro)
            r"detesto\s+(?:quando|che)\s+(?:tu\s+)?",
            r"detesto\s+(?:il|la|lo|i|gli|le)\s+",
            r"odio\s+quando\s+(?:tu\s+)?",
            r"odio\s+che\s+(?:tu\s+)?",
            r"odio\s+(?:il|la|lo|i|gli|le)\s+",
            r"mi\s+d[aà]\s+noia\s+(?:quando|che)\s+",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "future_instruction",
        "description": "Istruzione temporale — d'ora in poi, da questo momento, in futuro",
        "confidence": 0.90,
        "examples": [
            "d'ora in poi non usare <3",
            "da questo momento evita le metafore",
            "in futuro non scrivere sempre lo stesso",
            "da ora in poi cerca di dosare i cuori",
            "per il futuro evita queste strutture ripetitive",
        ],
        "patterns": [
            r"d['']\s*ora\s+in\s+poi\s+(?:non\s+|evita\s+)?",
            r"da\s+(?:questo\s+|ora\s+|adesso\s+)?momento\s+(?:in\s+poi\s+)?(?:non\s+|evita\s+)?",
            r"da\s+ora\s+(?:in\s+poi\s+)?(?:non\s+|evita\s+)?",
            r"in\s+futuro\s+(?:non\s+|evita\s+)?",
            r"per\s+(?:il\s+futuro|sempre)\s+(?:non\s+|evita\s+)?",
            r"d['']\s*ora\s+(?:in\s+poi\s+)?(?:non\s+|evita\s+)?",
            r"da\s+questo\s+momento\s+(?:in\s+poi\s+)?",
            # "a partire da ora/adesso" — forma comune nell'italiano parlato
            r"a\s+partire\s+da\s+(?:ora|adesso|questo\s+momento)\s+(?:non\s+)?",
            r"da\s+adesso\s+(?:in\s+poi\s+)?(?:non\s+)?",
        ],
    },

]


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — LLM MICRO-PASS (phi3:mini)
# ═══════════════════════════════════════════════════════════════════════════
#
# Configurazione per la chiamata LLM di fallback.
# Attivato SOLO quando Tier 0 matcha ma Tier 1 non trova pattern.
#
# MODELLO: phi3:mini — già in stack per intent_router. Temperatura 0.0
# per output JSON deterministico. num_predict basso (80) per latenza minima.
#
# PROMPT: few-shot con esempi positivi e negativi. I "prompt_examples"
# vengono iniettati nel template. Aggiungere esempi qui per calibrare
# il classificatore senza toccare correction_handler.py.
#
# FAIL-SILENT: qualsiasi errore (Ollama non disponibile, timeout,
# JSON malformato, phi3:mini non trovato) → return False silenzioso.

LLM_TIER2: dict = {
    "enabled": True,

    # Modello da usare (deve essere disponibile su ollama_port)
    "model": "phi3:mini",

    # Porta Ollama. Default 11434. In dual-GPU usa 11435 (secondary).
    # Override via env: EDEN_META_OLLAMA_PORT
    "ollama_port": 11434,

    # Generazione
    "temperature": 0.0,   # deterministico
    "max_tokens": 80,     # sufficiente per JSON breve
    "timeout_seconds": 8,

    # Template prompt. {msg} viene sostituito con il messaggio utente
    # (troncato a 200 char). {examples} viene sostituito con prompt_examples.
    "prompt_template": (
        "Sei un classificatore di meta-istruzioni per un'IA italiana.\n"
        "L'utente sta chiedendo all'IA di smettere di usare o fare qualcosa?\n"
        "Rispondi SOLO con JSON valido su una riga.\n\n"
        "Esempi:\n"
        "{examples}"
        "\nMessaggio: \"{msg}\"\n"
        "JSON:"
    ),

    # Esempi few-shot iniettati nel prompt.
    # Formato: (messaggio, is_ban, target)
    # Aggiungere esempi qui per migliorare la classificazione.
    "prompt_examples": [
        # Positivi — vari formati
        ("smettila con le metafore",                    True,  "metafore"),
        ("odio quando usi il simbolo cuore",             True,  "simbolo cuore"),
        ("non voglio più frasi lunghissime",             True,  "frasi lunghissime"),
        ("mi dà fastidio questa struttura ripetitiva",  True,  "struttura ripetitiva"),
        ("potresti evitare le domande alla fine?",       True,  "domande"),
        ("ti chiedo di non usare più i puntini",         True,  "puntini"),
        ("mi irrita quando sei così prolissa",           True,  "prolissa"),
        ("non mi piace questo tono passivo",             True,  "tono passivo"),
        ("cerca di non fare sempre le stesse domande",  True,  "stesse domande"),
        ("tieni a mente: niente cuori d'ora in poi",    True,  "cuori"),
        # Negativi — messaggi normali che contengono trigger-words
        ("come stai?",                                  False, None),
        ("cosa pensi di questo argomento?",             False, None),
        ("mi piace la pizza",                           False, None),
        ("devi venire a trovarmi",                      False, None),
        ("odio l'estate, fa troppo caldo",              False, None),
        ("voglio raccontarti una cosa",                 False, None),
        ("mi fa piacere che tu sia qui",                False, None),
    ],
}
