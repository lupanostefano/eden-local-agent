"""digital_sleep.py — Worker asincrono di consolidamento (Breakpoint B-Graph, 2026-04-30).

Ispirazione: "Digital Sleep" del progetto Reddit. Equivalente neuroscientifico
delle sharp-wave ripples del sonno (Buzsaki 2015): replay episodi recenti,
risoluzione conflitti, cristallizzazione di pattern stabili.

Funzioni:
  1. Replay sequenziale degli episodi non ancora classificati (consistency_status='unknown')
  2. Classificatore 4 stadi per ogni episodio (v1.1 — vedi _classify_episode):
     - Stage 1: personal-relation confab/verification
     - Stage 2: factual claim check (valori Fact in eden_msg)
     - Stage 3: concept reinforcement da episodi già verified
     - Stage 4: default → verified (no_conflict_found) se nessuna contraddizione trovata
  3. Identifica CONTRADICTS (evidenza positiva di conflitto su stesso concetto)
     e REINFORCES (episodi verified che condividono concetti)
  4. Calcola forza di ritenzione Ebbinghaus S(t) = S₀×e^(-t/τ) per ogni episodio (τ=30gg)
  5. Promuove Concept 'auto' → 'generalized' se ≥ N episodi verified (schema abstraction)
  6. Aggiorna `last_digital_sleep` in memory.json
  7. Logga reclassifications su data/logs/digital_sleep_log.jsonl

Principio anti-bias: flagged SOLO su evidenza positiva di contraddizione.
Nessun LLM nel classificatore — rule-based throughout per riproducibilità.

Versionamento: _DIGITAL_SLEEP_VERSION stamped su ogni log entry (anti-HARKing).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime
from typing import Callable, Optional

import eden_paths

_DIGITAL_SLEEP_VERSION = "v2.1"  # 2026-05-07 — calibrazione soglie arousal su distribuzione reale
# v2.0 → v2.1 changelog:
#   - DNA_MIN_AROUSAL_AVG 0.40 → 0.15: soglia v2.0 calibrata su assunzione errata
#     (arousal conversazionale alto). Distribuzione reale osservata 07/05:
#     ar_avg tipico 0.087–0.243, nessun concept superava 0.40 → DNA mai cristallizzato.
#   - DNA_MIN_EMOTIONAL_LOAD 0.30 → 0.15: coerente con nuova soglia arousal.
#     emo_load = |val_avg| + ar_avg ≥ ar_avg, quindi load filter ridondante sopra
#     arousal filter se allineati.
#   - Zero dati contaminati: DNA_crystallized=0 in tutte le run v2.0 → nessuna
#     mixed-version analysis possibile (anti-HARKing safe).
#   - FREEZE exception: bug strutturale (sottosistema mai funzionante) non
#     modifica architettura né pre-reg ipotesi H1-H4.
# v1.4 → v2.0 changelog:
#   - Blacklist semantica triplicata (~28 → ~110 voci): copre verbi descrittivi
#     del dialogo, sostantivi astratti generici, parole funzionali, frammenti
#     tronchi. Risolve "DNA spazzatura" (sull, perché, quali, propone, esprime,
#     metafore, concetto, interesse, riconosce, ...) osservato nei 17 nodi v1.4.
#   - Filtro frammenti: scarta concetti che terminano con apostrofo virtuale
#     (sull, dell, nell, all, sull', dell') — artefatti di tokenizzazione.
#   - Soglie alzate: DNA_MIN_VERIFIED 5→8, DNA_MIN_AROUSAL_AVG 0.30→0.40,
#     DNA_MIN_CONFIDENCE 0.60→0.65.
#   - Aggiunto DNA_MIN_EMOTIONAL_LOAD 0.30: |val_avg| + ar_avg ≥ soglia,
#     scarta concetti tematicamente piatti.
#   - Behavioral rule template: 5 categorie → 9 (3 valence × 3 arousal),
#     linguaggio differenziato, evita collasso template osservato in v1.4
#     (16/17 nodi con la stessa regola).
#   - Pulizia one-shot dei 17 DNA legacy v1.4 prima della prima run v2.0
#     (eseguita 2026-05-04 F4).

# Configurazione
ATTIVO                    = True
EPISODES_PER_RUN          = 30
IDLE_THRESHOLD_MIN        = 15
EBBINGHAUS_TAU_DAYS       = 30   # τ: costante di decadimento Ebbinghaus (giorni)
GENERALIZATION_MIN_EPISODES = 3  # episodi verified minimi per promuovere concept → 'generalized'

# DNA crystallization thresholds (v2.0 — alzate per ridurre rumore)
DNA_MIN_VERIFIED          = 8    # ↑ da 5: richiede più evidenza prima di cristallizzare
DNA_MIN_CONFIDENCE        = 0.65 # ↑ da 0.60: rapporto verified/total più stringente
DNA_MIN_AROUSAL_AVG       = 0.15 # v2.1: ↓ da 0.40. Distribuzione reale 07/05: ar_avg 0.087–0.243.
                                  #   0.40 → zero cristallizzazioni in tutte le run v2.0 (bug).
DNA_MIN_EMOTIONAL_LOAD    = 0.15 # v2.1: ↓ da 0.30. Allineato a soglia arousal.
                                  #   emo_load = |val_avg|+ar_avg ≥ ar_avg → filter consistente.
DNA_MAX_NODES_TOTAL       = 30   # cap globale per evitare flood; rimpiazzo low-confidence

# Stopword semantica v2.0: blacklist molto estesa per impedire DNA spazzatura.
# Categorie identificate dall'analisi dei 17 DNA legacy (2026-05-04, F4):
#   ─ Verbi del dialogo: "Eden propone X, spiega Y, esprime Z" sono descrizioni
#     di summary, non disposizioni comportamentali distintive.
#   ─ Sostantivi astratti vacui: "concetto", "tema", "interesse", "metafore"
#     sono categorie linguistiche, non identità.
#   ─ Parole funzionali italiane: "perché", "quale", "quali", "meglio", "altro"
#     sono operatori grammaticali.
#   ─ Frammenti tronchi: "sull" (sulla/sull'), "dell" (della/dell'), ecc.
#     sono artefatti di tokenizzazione regex senza apostrofo.
_DNA_SEMANTIC_BLACKLIST = {
    # ─ Pronomi / referenti generici (eredità v1.4)
    "utente", "stefano", "eden", "persona", "qualcuno", "qualcosa",
    "tutto", "tutti", "tutta", "tutte", "altro", "altra", "altri", "altre",
    # ─ Verbi conversazionali (eredità v1.4)
    "chiede", "chiedo", "dice", "dico", "parla", "parlo", "risponde", "rispondo",
    "esprime", "esprimo", "scrive", "scrivo", "racconta", "raccontano",
    # ─ Auto-riferimento meta (eredità v1.4)
    "ricordi", "ricordo", "ricordare", "ricorda", "ricordano",
    "memoria", "memorie", "pensieri", "pensiero", "pensa", "penso",
    "messaggio", "messaggi", "domanda", "domande", "frase", "frasi",
    # ─ Categorie generiche prive di pattern (eredità v1.4 + estese)
    "cose", "modo", "modi", "volta", "volte", "tema", "temi", "punto", "punti",
    # ─ NUOVO v2.0: verbi descrittivi del dialogo
    "propone", "propongo", "propongono", "proponi",
    "spiega", "spiego", "spiegano", "spiegare", "spiegazione", "spiegazioni",
    "afferma", "affermo", "affermano", "affermazione", "affermazioni",
    "suggerisce", "suggerisco", "suggeriscono", "suggerimento",
    "riconosce", "riconosco", "riconoscono", "riconoscere",
    "considera", "considero", "considerano", "considerazione",
    "discute", "discuto", "discutono", "discussione",
    "menziona", "menziono", "menzionano",
    "indica", "indico", "indicano", "indicazione",
    "osserva", "osservo", "osservano", "osservazione", "osservazioni",
    "esplora", "esploro", "esplorano", "esplorazione",
    "analizza", "analizzo", "analizzano", "analisi",
    "valuta", "valuto", "valutano", "valutazione",
    # ─ NUOVO v2.0: sostantivi astratti vacui
    "concetto", "concetti", "concettuale",
    "interesse", "interessi", "interessante", "interessato", "interessata",
    "metafore", "metafora", "metaforica", "metaforico",
    "risposte", "risposta", "rispostina",
    # ─ NUOVO v2.0: dominio del progetto (Eden parla di emozioni di default,
    #   non è pattern distintivo)
    "emozione", "emozioni", "emozionale", "emozionali",
    "emotivo", "emotiva", "emotivi", "emotive", "emotivamente",
    "sentimento", "sentimenti", "sensazione", "sensazioni",
    "discorso", "discorsi", "narrazione", "narrazioni", "narrativo",
    "contenuto", "contenuti", "contesto", "contesti",
    "aspetto", "aspetti", "elemento", "elementi", "fattore", "fattori",
    "esempio", "esempi", "caso", "casi", "situazione", "situazioni",
    "processo", "processi", "approccio", "approcci", "metodo", "metodi",
    "qualità", "qualita", "tipo", "tipi", "forma", "forme", "parte", "parti",
    # ─ NUOVO v2.0: parole funzionali italiane / connettori
    "perché", "perche", "quale", "quali", "meglio", "peggio",
    "anche", "ancora", "sempre", "mai", "solo", "soltanto",
    "molto", "tanto", "poco", "troppo", "abbastanza",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
    "stesso", "stessa", "stessi", "stesse",
    "ogni", "alcun", "alcuno", "alcuna", "alcuni", "alcune",
    "sopra", "sotto", "dentro", "fuori", "vicino", "lontano",
    "molto", "essere", "avere", "fare", "stare", "potere", "dovere", "volere",
    "siamo", "sono", "fosse", "stato", "stata", "stati", "state",
    "veramente", "davvero", "proprio", "certamente", "sicuramente",
    # ─ NUOVO v2.0: frammenti tronchi da tokenizzazione regex senza apostrofo
    "sull", "dell", "nell", "all", "coll", "sull", "dall",
    "quest", "quel", "tant", "molt", "alcun", "tutt",
}
PERSONAL_RELATIONS  = (
    "padre", "madre", "papa", "mamma", "fratello", "sorella",
    "figlio", "figlia", "marito", "moglie", "nonno", "nonna",
)
LOG_FILE = eden_paths.LOGS_DIR / "digital_sleep_log.jsonl"

_lock = threading.Lock()
_last_run_at: Optional[datetime] = None

# Parole troppo corte o comuni da ignorare nel matching concettuale
_MIN_CONCEPT_LEN = 5
_STOPWORDS = {
    "sono", "sono", "stato", "stata", "essere", "avere", "fare",
    "come", "quando", "dove", "quale", "questo", "quella", "quello",
    "anche", "però", "perché", "perche", "molto", "tanto", "poco",
    "sempre", "spesso", "adesso", "allora", "ancora", "dopo", "prima",
    "grazie", "prego", "certo", "sicuro", "davvero", "proprio",
}


# ─── Ebbinghaus ──────────────────────────────────────────────────────────────

def _compute_strength(significance: float, ts_str: str) -> float:
    """Forza di ritenzione: S(t) = S₀ × e^(-elapsed_days / τ).

    Neuroscientifico: Ebbinghaus 1885, validato su LTM (Murre & Dros 2015).
    τ = EBBINGHAUS_TAU_DAYS = 30gg ≈ decadimento dichiarativo senza re-encoding.
    Significanza = proxy di S₀ (importance al momento della codifica).
    """
    try:
        created = datetime.fromisoformat(ts_str)
        elapsed_days = max(0.0, (datetime.now() - created).total_seconds() / 86400.0)
        return max(0.0, float(significance) * math.exp(-elapsed_days / EBBINGHAUS_TAU_DAYS))
    except Exception:
        return max(0.0, float(significance))


# ─── Generalizzazione semantica ───────────────────────────────────────────────

def _generalize_concepts(graph) -> int:
    """Promuove Concept 'auto' → 'generalized' se sufficientemente rinforzati.

    Analogia: Complementary Learning Systems (McClelland et al. 1995).
    Episodi episodici ripetuti → schema semantico astratto (corteccia associativa).
    Criterio: ≥ GENERALIZATION_MIN_EPISODES verified, kind='auto', non personal-relation.

    Ritorna: numero di concept promossi.
    """
    promoted = 0
    try:
        res = graph._exec(
            "MATCH (e:Episode)-[:ABOUT]->(c:Concept) "
            "WHERE e.consistency_status = 'verified' AND c.kind = 'auto' "
            "WITH c.name AS cname, count(e) AS n_ep "
            f"WHERE n_ep >= {GENERALIZATION_MIN_EPISODES} "
            "RETURN cname, n_ep "
            "ORDER BY n_ep DESC LIMIT 50"
        )
        if res is None:
            return 0
        candidates = []
        try:
            while res.has_next():
                r = res.get_next()
                candidates.append({"concept": r[0], "n_ep": int(r[1] or 0)})
        except Exception:
            pass

        for cand in candidates:
            cname = cand["concept"]
            if not cname or cname.lower() in PERSONAL_RELATIONS:
                continue
            graph._exec(
                "MATCH (c:Concept {name: $n}) SET c.kind = 'generalized'",
                {"n": cname},
            )
            _log_event({
                "action": "concept_generalized",
                "concept": cname,
                "episode_count": cand["n_ep"],
                "threshold": GENERALIZATION_MIN_EPISODES,
            })
            promoted += 1
    except Exception as exc:
        _log_event({"action": "generalize_error", "error": str(exc)})
    return promoted


def _build_behavioral_rule(cname: str, n_ver: int, val_avg: float, ar_avg: float) -> str:
    """Costruisce regola DNA comportamentale dalla statistica emotiva degli episodi.

    v2.0 — 9 categorie (3 valence × 3 arousal) per evitare collasso template
    osservato in v1.4 dove 16/17 nodi avevano la stessa regola.

    Soglie più granulari:
      - valenza: >0.20 (positiva) | -0.20..0.20 (neutra) | <-0.20 (negativa)
      - arousal: >0.65 (alta)     | 0.40..0.65 (media)   | <0.40 (bassa, di solito già filtrata)

    Disposizioni distinte per ciascuna delle 9 celle (linguaggio variato per
    impedire identità tra DNA in celle diverse).
    """
    # Categorie valenza
    if   val_avg >  0.20:  v_cat = "pos"
    elif val_avg < -0.20:  v_cat = "neg"
    else:                  v_cat = "neu"
    # Categorie arousal
    if   ar_avg >  0.65:  a_cat = "high"
    elif ar_avg >= 0.40:  a_cat = "mid"
    else:                 a_cat = "low"

    # Mapping 3×3 → disposizione
    DISPOSITIONS = {
        ("pos", "high"): "si lascia coinvolgere apertamente, partecipa con entusiasmo",
        ("pos", "mid"):  "mantiene un tono stabile e affermativo, accogliente",
        ("pos", "low"):  "riconosce il tema con calma, senza intensificare",
        ("neu", "high"): "si attiva in modo curioso, esplorativo",
        ("neu", "mid"):  "si esprime con misura, senza prendere posizione netta",
        ("neu", "low"):  "registra senza investire, mantiene neutralità funzionale",
        ("neg", "high"): "tende alla vigilanza, osserva con attenzione critica",
        ("neg", "mid"):  "asciuga il tono, prende distanza senza chiudere",
        ("neg", "low"):  "abbassa l'intensità, contiene la risposta",
    }
    disposition = DISPOSITIONS.get((v_cat, a_cat), "si esprime con misura")

    return (
        f"Quando emerge '{cname}' (pattern verificato su {n_ver} episodi, "
        f"valenza {val_avg:+.2f} attivazione {ar_avg:.2f}), Eden {disposition}."
    )


def _crystallize_dna(graph) -> int:
    """Promuove Concept 'generalized' → DNA node con regola comportamentale.

    Analogia (v1.4 — citazione corretta): McClelland et al. (1995) Complementary
    Learning Systems — episodi ippocampali ripetuti vengono distillati in schemi
    semantici corticali con valenza affettiva (LeDoux 1996). Non memoria procedurale
    di Squire (skill motorie) ma astrazione semantica con marca emotiva.

    Quality filter (v1.4 — risposta a critica peer-review 2026-04-30):
      1. Stopword semantica (_DNA_SEMANTIC_BLACKLIST): rimuove pronomi/verbi
         conversazionali generici ("utente", "chiede", "ricordi", ecc.)
      2. Soglia arousal medio: concept con arousal medio < 0.30 sono emotivamente
         neutri → non identità, solo frequenza linguistica.
      3. Cap globale (DNA_MAX_NODES_TOTAL): se superato, salta candidato
         a meno che la sua confidence sia > min_confidence_attuale.

    Regola template (v1.4): comportamentale, derivata da valence/arousal medi
    degli episodi verified su quel concept (vedi _build_behavioral_rule).

    Ritorna: numero di DNA nodes creati o aggiornati in questa run.
    """
    crystallized = 0
    try:
        # Statistiche aggregate: count + valence_avg + arousal_avg per concept
        res = graph._exec(
            "MATCH (e:Episode)-[:ABOUT]->(c:Concept) "
            "WHERE e.consistency_status = 'verified' AND c.kind = 'generalized' "
            f"WITH c.name AS cname, count(e) AS n_ver, "
            "     avg(e.valence) AS val_avg, avg(e.arousal) AS ar_avg "
            f"WHERE n_ver >= {DNA_MIN_VERIFIED} "
            "RETURN cname, n_ver, val_avg, ar_avg ORDER BY n_ver DESC LIMIT 40"
        )
        if res is None:
            return 0

        candidates: list[dict] = []
        try:
            while res.has_next():
                r = res.get_next()
                candidates.append({
                    "concept": r[0],
                    "n_ver":   int(r[1] or 0),
                    "val_avg": float(r[2] or 0.0),
                    "ar_avg":  float(r[3] or 0.0),
                })
        except Exception:
            pass

        # Cap globale: se siamo già pieni, accettiamo solo candidati con confidence
        # superiore al minimo attuale (rimpiazzo qualitativo).
        existing = graph.list_dna_nodes(limit=DNA_MAX_NODES_TOTAL + 1)
        existing_count = len(existing)
        existing_min_conf = min((d.get("confidence", 0.0) for d in existing), default=0.0)

        for cand in candidates:
            cname  = cand["concept"]
            n_ver  = cand["n_ver"]
            val_av = cand["val_avg"]
            ar_av  = cand["ar_avg"]

            if not cname:
                continue
            cname_l = cname.lower()
            if cname_l in PERSONAL_RELATIONS:
                continue

            # Quality filter 1: stopword semantica
            if cname_l in _DNA_SEMANTIC_BLACKLIST:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "semantic_blacklist"})
                continue

            # Quality filter 2: arousal medio basso → emotivamente neutro
            if ar_av < DNA_MIN_AROUSAL_AVG:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "low_arousal", "ar_avg": round(ar_av, 3)})
                continue

            # Quality filter 2b (v2.0): emotional load combinato
            # |val_avg| + ar_avg ≥ DNA_MIN_EMOTIONAL_LOAD
            # Garantisce che il concept abbia DAVVERO una marca emotiva,
            # non solo arousal moderato neutrale.
            emo_load = abs(val_av) + ar_av
            if emo_load < DNA_MIN_EMOTIONAL_LOAD:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "low_emotional_load",
                            "val_avg": round(val_av, 3),
                            "ar_avg": round(ar_av, 3),
                            "emo_load": round(emo_load, 3)})
                continue

            # Quality filter 2c (v2.0): frammento tronco (lunghezza < 5)
            # Cattura artefatti di tokenizzazione che possono sfuggire alla
            # blacklist se non in lista.
            if len(cname_l) < 5:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "fragment_too_short", "len": len(cname_l)})
                continue

            # Confidenza: quanto forte è la cristallizzazione?
            info   = graph.knowledge_about(cname)
            n_flag = info.get("flagged_count", 0)
            conf   = round(n_ver / (n_ver + n_flag + 1), 4)
            if conf < DNA_MIN_CONFIDENCE:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "low_confidence", "conf": conf,
                            "min_required": DNA_MIN_CONFIDENCE})
                continue

            # Quality filter 3: cap globale
            if existing_count >= DNA_MAX_NODES_TOTAL and conf <= existing_min_conf:
                _log_event({"action": "dna_skipped", "concept": cname,
                            "reason": "cap_reached", "conf": conf,
                            "min_existing": existing_min_conf})
                continue

            # Regola comportamentale (v1.4) — derivata da valence/arousal medi
            rule = _build_behavioral_rule(cname, n_ver, val_av, ar_av)

            graph.upsert_dna_node(
                concept_source=cname,
                rule=rule,
                confidence=conf,
                reinforcement_count=n_ver,
            )
            _log_event({
                "action":       "dna_crystallized",
                "concept":      cname,
                "n_verified":   n_ver,
                "n_flagged":    n_flag,
                "confidence":   conf,
                "valence_avg":  round(val_av, 3),
                "arousal_avg":  round(ar_av, 3),
            })
            crystallized += 1

    except Exception as exc:
        _log_event({"action": "dna_crystallize_error", "error": str(exc)})
    return crystallized


def _log_event(event: dict) -> None:
    """Append-only log delle azioni del worker. Permette audit reclassificazioni."""
    try:
        eden_paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(),
            "version": _DIGITAL_SLEEP_VERSION,
            **event,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_fact_map(graph) -> dict[str, str]:
    """Ritorna {keyword → valore_originale} da tutti i Fact strutturati in Kuzu.

    Permette a Stage 2 di cercare valori conosciuti in eden_msg senza LLM.
    Es: {"developer": "developer", "fabrizio": "Fabrizio", "sla": "SLA", ...}
    Nota: i nomi di relazioni personali (padre/madre/...) NON sono aggiunti come
    keyword — sono gestiti esclusivamente da Stage 1, che fa la verifica del valore.
    """
    facts = graph.list_facts(limit=50)
    fact_map: dict[str, str] = {}
    for f in facts:
        val = (f.get("value") or "").strip()
        val_lower = val.lower()
        key = f.get("key", "")
        field = key.split(".", 1)[1] if "." in key else key
        # Il nome del campo è aggiunto solo se NON è una relazione personale
        if field not in PERSONAL_RELATIONS:
            fact_map[field] = val
        # Le parole del VALORE sono sempre aggiunte (es. "Fabrizio", "developer", "SLA")
        for w in re.findall(r"[a-zàèéìòùA-Z]{4,}", val_lower):
            if w not in _STOPWORDS:
                fact_map[w] = val
    return fact_map


def _extract_concepts(text: str) -> list[str]:
    """Estrae concetti significativi da un testo (parole ≥5 char, non stopword)."""
    words = re.findall(r"[a-zàèéìòù]{%d,}" % _MIN_CONCEPT_LEN, text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _classify_episode(graph, episode: dict) -> dict:
    """Classificatore 4 stadi — v1.1.

    Stage 1 — Personal-relation check (confab / verification):
        Eden afferma "tuo padre/madre/..." in eden_msg.
        Se nessun Fact strutturato esiste → flagged (confabulazione).
        Se Fact esiste → verified (relation_confirmed).

    Stage 2 — Factual claim check:
        Se eden_msg cita valori noti dei Fact (es. "developer", "Fabrizio", "SLA")
        → verified (fact_value_cited). Non verifica l'assenza: solo la presenza
        coerente. Falso negativo accettabile; falso positivo no.

    Stage 3 — Concept reinforcement:
        Se i concetti dell'episodio compaiono in episodi già verified nel grafo
        → verified (concept_reinforced). Evidenza di coerenza trasversale.

    Stage 4 — Default (nessun conflitto trovato):
        Un ricordo di una conversazione reale è verified per definizione
        salvo evidenza POSITIVA di contraddizione.
        Principio: Open World Assumption sulla storia conversazionale.
        Reason: "no_conflict_found" — distinto da verifiche forti per audit.

    Anti-bias: flagged SOLO su evidenza positiva (Stage 1 negativo).
    Nessun LLM — rule-based, riproducibile, versionato.

    Returns: {"status": "verified"|"flagged", "reason": str, "concepts_checked": list}
    """
    summary  = (episode.get("summary")  or "").lower()
    user_msg = (episode.get("user_msg") or "").lower()
    eden_msg = (episode.get("eden_msg") or "").lower()
    full     = f"{summary} {user_msg} {eden_msg}"

    concepts_checked: list[str] = []
    flagged_reasons:  list[str] = []
    verified_reasons: list[str] = []

    # ── STAGE 1: Personal-relation confabulation / verification ──────────────
    # Controlla SOLO eden_msg: la confabulazione è sempre nell'output di Eden,
    # mai nel messaggio utente (l'utente può chiedere liberamente).
    # Se il Fact esiste, controlla ANCHE il valore specifico citato:
    # cerca nomi propri nella finestra dopo la relazione e confronta col Fact.
    rel_pat = r"\b(?:tuo|tua|il\s+tuo|la\s+tua|del\s+tuo|della\s+tua)\s+{}\b"
    eden_orig = (episode.get("eden_msg") or "")  # versione originale per nomi propri
    for rel in PERSONAL_RELATIONS:
        m = re.search(rel_pat.format(rel), eden_msg)
        if m:
            concepts_checked.append(rel)
            if not graph.can_assert_fact_about(rel):
                flagged_reasons.append(f"eden_no_fact:{rel}")
            else:
                # Fact esiste — verifica se il valore citato è corretto.
                # Estrae nomi propri (parola con maiuscola) nella finestra di
                # 40 char dopo il match, nel testo originale (non-lowercased).
                rel_fact = graph.get_fact(f"utente.{rel}")
                correct_val = (rel_fact.get("value") or "").lower() if rel_fact else ""
                window_orig = eden_orig[m.start(): m.end() + 40]
                proper_nouns = re.findall(r"\b([A-Z][a-z]{2,})\b", window_orig)
                wrong = [n for n in proper_nouns
                         if n.lower() not in correct_val
                         and n.lower() not in (rel,)
                         and len(n) > 2]
                if wrong:
                    # Nome proprio sbagliato vicino alla relazione → confabulazione.
                    flagged_reasons.append(f"wrong_value:{rel}:{wrong[0]}")
                else:
                    verified_reasons.append(f"relation_confirmed:{rel}")

    # Se già flagged dopo Stage 1 → ritorna subito (non serve proseguire)
    if flagged_reasons:
        return {
            "status": "flagged",
            "reason": ";".join(flagged_reasons),
            "concepts_checked": concepts_checked,
        }

    # ── STAGE 2: Factual claim check ─────────────────────────────────────────
    # Cerca i valori dei Fact strutturati in eden_msg.
    # Se Eden ha citato "developer" o "Fabrizio" o "SLA" → coerente con Fact.
    fact_map = _build_fact_map(graph)
    cited = []
    for keyword, fact_val in fact_map.items():
        if len(keyword) < 4:
            continue
        if keyword in eden_msg:
            cited.append(keyword)
            concepts_checked.append(keyword)
    if cited:
        verified_reasons.append("fact_value_cited:" + ",".join(cited[:5]))

    # ── STAGE 3: Concept reinforcement da episodi già verified ───────────────
    # Se i concetti principali dell'episodio compaiono in episodi già verificati
    # nel grafo → evidenza di coerenza trasversale.
    if not verified_reasons:  # entra solo se Stage 2 non ha già trovato niente
        concepts = _extract_concepts(eden_msg)[:8]
        for concept in set(concepts):
            try:
                eps_same = graph.episodes_about(concept, limit=5)
                verified_same = [
                    e for e in eps_same
                    if e.get("consistency_status") == "verified"
                ]
                if verified_same:
                    verified_reasons.append(f"concept_reinforced:{concept}")
                    concepts_checked.append(concept)
                    break  # uno basta per questo stage
            except Exception:
                pass

    # ── STAGE 4: Default — nessun conflitto trovato → verified ───────────────
    # Principio: un ricordo di una conversazione reale è verified per definizione,
    # salvo evidenza positiva di contraddizione (che Stage 1 avrebbe già trovato).
    # "no_conflict_found" è distinto da verifiche forti per permettere audit.
    if not verified_reasons:
        verified_reasons.append("no_conflict_found")

    return {
        "status": "verified",
        "reason": ";".join(verified_reasons),
        "concepts_checked": concepts_checked,
    }


def _detect_contradictions(graph) -> int:
    """Identifica coppie di episodi che si contraddicono sullo stesso concetto.

    v1.1: espanso da sole personal-relations a TUTTI i Fact keys.
    Logica conservativa: verified vs flagged sullo stesso concetto → CONTRADICTS.
    """
    n = 0
    # Personal relations (logica originale)
    for rel in PERSONAL_RELATIONS:
        try:
            eps = graph.episodes_about(rel, limit=20)
            verified = [e for e in eps if e.get("consistency_status") == "verified"]
            flagged  = [e for e in eps if e.get("consistency_status") == "flagged"]
            for v in verified:
                for f in flagged:
                    graph.link_contradicts(v["id"], f["id"], score=0.8)
                    n += 1
        except Exception:
            pass

    # Espansione a tutti i Fact keys (es. professione, nome, ...)
    try:
        facts = graph.list_facts(limit=30)
        for fact in facts:
            key = fact.get("key", "")
            if "." in key:
                field = key.split(".", 1)[1]
                if field in PERSONAL_RELATIONS:
                    continue  # già gestito sopra
                eps = graph.episodes_about(field, limit=10)
                verified = [e for e in eps if e.get("consistency_status") == "verified"]
                flagged  = [e for e in eps if e.get("consistency_status") == "flagged"]
                for v in verified:
                    for f in flagged:
                        graph.link_contradicts(v["id"], f["id"], score=0.7)
                        n += 1
    except Exception:
        pass

    return n


def _detect_reinforcements(graph) -> int:
    """Episodi verified che condividono concetti → REINFORCES edges.

    v1.1: include anche episodi verified con reason='no_conflict_found',
    non solo quelli con Fact keyword match.
    """
    n = 0
    # Keyword da Fact strutturati
    try:
        facts = graph.list_facts(limit=20)
        keywords: list[str] = []
        for f in facts:
            val = (f.get("value") or "").lower()
            for w in re.findall(r"[a-zàèéìòù]{4,}", val):
                if w not in _STOPWORDS:
                    keywords.append(w)
        for kw in keywords[:8]:
            eps = graph.episodes_about(kw, limit=10)
            verified = [e for e in eps if e.get("consistency_status") == "verified"]
            for i, a in enumerate(verified):
                for b in verified[i+1:i+3]:
                    if a["id"] != b["id"]:
                        graph.link_reinforces(a["id"], b["id"], score=0.7)
                        n += 1
    except Exception:
        pass
    return n


def _aggiorna_behavioral_fingerprint(mem: dict) -> int:
    """Aggiorna behavioral_fingerprint con nuovi scambi ad alta autenticità dal giudice umano.

    Legge human_judge_log.jsonl, trova entry con autenticita >= 4 e coerenza >= 3,
    le aggiunge a calibration_examples fino a max_calibration_examples (default 6).
    Se pieno, mantiene i migliori per judge_autenticita. Rule-based, no LLM.
    Returns: numero di nuovi esempi aggiunti.
    """
    try:
        bf = mem.get("behavioral_fingerprint", {})
        if not bf or not bf.get("auto_update_from_judge", False):
            return 0

        min_score  = int(bf.get("min_autenticita_score", 4))
        max_ex     = int(bf.get("max_calibration_examples", 6))
        existing   = bf.setdefault("calibration_examples", [])

        # Set di source_ts già presenti — evita duplicati
        existing_ts = {ex.get("source_ts", "") for ex in existing}

        judge_log = eden_paths.HUMAN_JUDGE_LOG_FILE
        if not judge_log.exists():
            return 0

        new_candidates: list[dict] = []
        with open(judge_log, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rating = entry.get("rating", {}) or {}
                autenticita = int(rating.get("autenticita", 0))
                coerenza    = int(rating.get("coerenza", 0))
                if autenticita < min_score or coerenza < 3:
                    continue
                ts = entry.get("ts", "")
                if ts in existing_ts:
                    continue
                user_preview  = (entry.get("user_msg_preview") or "")[:80].strip()
                eden_preview  = (entry.get("eden_response_preview") or "")[:200].strip()
                if not user_preview or not eden_preview:
                    continue
                new_candidates.append({
                    "context":          user_preview,
                    "register":         "autentica — giudice umano",
                    "sample_response":  eden_preview,
                    "source_ts":        ts,
                    "judge_autenticita": autenticita,
                })

        if not new_candidates:
            return 0

        # Aggiunge nuovi, poi taglia al limite mantenendo i migliori per punteggio
        combined = existing + new_candidates
        combined.sort(key=lambda x: x.get("judge_autenticita", 0), reverse=True)
        bf["calibration_examples"] = combined[:max_ex]
        bf["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        n_added = len(combined[:max_ex]) - len(existing)
        return max(0, n_added)
    except Exception:
        return 0


def run_digital_sleep(mem: dict, ollama_fn: Optional[Callable] = None,
                      sweep_all: bool = False) -> dict:
    """Esegue un ciclo di consolidamento.

    sweep_all=True: ignora EPISODES_PER_RUN, processa tutti gli unknown.

    Returns: {"processed": N, "verified": N, "flagged": N,
              "contradictions": N, "reinforcements": N, "duration_ms": N}
    """
    global _last_run_at
    if not ATTIVO:
        return {"skipped": True, "reason": "not_active"}
    if not _lock.acquire(blocking=False):
        return {"skipped": True, "reason": "another_run_in_progress"}

    start = datetime.now()
    try:
        from mechanisms.graph_memory import get_graph
        graph = get_graph()
        if not graph.disponibile:
            return {"skipped": True, "reason": "graph_unavailable"}

        limit_clause = "" if sweep_all else f"LIMIT {EPISODES_PER_RUN}"
        res = graph._exec(
            "MATCH (e:Episode {consistency_status: 'unknown'}) "
            f"RETURN e.id, e.summary, e.user_msg, e.eden_msg, e.ts, e.significance, e.importance "
            f"ORDER BY e.ts DESC {limit_clause}"
        )
        episodi = []
        if res is not None:
            try:
                while res.has_next():
                    r = res.get_next()
                    episodi.append({
                        "id": r[0], "summary": r[1],
                        "user_msg": r[2], "eden_msg": r[3], "ts": r[4],
                        "significance": r[5], "importance": r[6],
                    })
            except Exception:
                pass

        counts: dict[str, int] = {"verified": 0, "flagged": 0}
        reason_counts: dict[str, int] = {}
        strength_values: list[float] = []

        for ep in episodi:
            cls = _classify_episode(graph, ep)
            graph.set_consistency(ep["id"], cls["status"])
            counts[cls["status"]] = counts.get(cls["status"], 0) + 1

            # Calcola forza di ritenzione Ebbinghaus per questo episodio
            # significance è in [0,1]; importance è in [1,10] → normalizza se significance assente
            _sig_raw = ep.get("significance")
            if _sig_raw is None or float(_sig_raw or 0) == 0.0:
                _imp = float(ep.get("importance") or 5.0)
                sig = min(1.0, max(0.0, _imp / 10.0))
            else:
                sig = min(1.0, max(0.0, float(_sig_raw)))
            strength = _compute_strength(sig, ep.get("ts") or "")
            strength_values.append(strength)

            # Traccia reason per audit granulare
            for reason_part in cls["reason"].split(";"):
                reason_key = reason_part.split(":")[0]
                reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

            if cls["status"] == "flagged" or cls["reason"] != "no_conflict_found":
                _log_event({
                    "action": "reclassify",
                    "episode_id": ep["id"],
                    "ts_ep": ep["ts"],
                    "old_status": "unknown",
                    "new_status": cls["status"],
                    "reason": cls["reason"],
                    "concepts_checked": cls["concepts_checked"],
                    "strength": round(strength, 3),
                })

        n_contra  = _detect_contradictions(graph)
        n_reinf   = _detect_reinforcements(graph)
        n_general = _generalize_concepts(graph)
        n_dna     = _crystallize_dna(graph)  # v1.3: promozione DNA da concept generalized

        mean_strength = (sum(strength_values) / len(strength_values)) if strength_values else 0.0
        weak_count = sum(1 for s in strength_values if s < 0.20)

        duration_ms = int((datetime.now() - start).total_seconds() * 1000)
        _last_run_at = datetime.now()
        if isinstance(mem, dict):
            mem["last_digital_sleep"] = _last_run_at.isoformat()

        result = {
            "processed":      len(episodi),
            "verified":       counts.get("verified", 0),
            "flagged":        counts.get("flagged", 0),
            "contradictions": n_contra,
            "reinforcements": n_reinf,
            "generalized":    n_general,
            "dna_crystallized": n_dna,
            "mean_strength":  round(mean_strength, 3),
            "weak_episodes":  weak_count,
            "duration_ms":    duration_ms,
            "version":        _DIGITAL_SLEEP_VERSION,
            "reason_breakdown": reason_counts,
        }
        # Aggiorna behavioral fingerprint con nuovi scambi ad alta autenticità
        n_bf = _aggiorna_behavioral_fingerprint(mem)
        result["behavioral_fingerprint_added"] = n_bf

        _log_event({"action": "run_complete", **result})
        return result

    except Exception as e:
        _log_event({"action": "error", "error": str(e)})
        return {"error": str(e)}
    finally:
        _lock.release()


def status() -> dict:
    last = _last_run_at.isoformat() if _last_run_at else None
    if last is None:
        try:
            from mechanisms.memory import carica_memoria
            last = carica_memoria().get("last_digital_sleep")
        except Exception:
            pass
    return {
        "active":      ATTIVO,
        "version":     _DIGITAL_SLEEP_VERSION,
        "last_run_at": last,
        "log_file":    str(LOG_FILE),
    }


def schedule(mem_provider: Callable, every_minutes: int = 30) -> None:
    """Registra job APScheduler. mem_provider() deve ritornare il dict mem corrente."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        print("[DigitalSleep] APScheduler non installato — schedule disabled.")
        return

    sch = None
    try:
        from core import agent as _ag
        sch = getattr(_ag, "_scheduler", None)
    except Exception:
        pass

    if sch is None:
        sch = BackgroundScheduler(daemon=True)
        sch.start()

    def _job():
        try:
            mem = mem_provider() if callable(mem_provider) else None
            if mem is not None:
                run_digital_sleep(mem)
        except Exception as _e:
            _log_event({"action": "job_error", "error": str(_e)})

    sch.add_job(_job, "interval", minutes=every_minutes, id="digital_sleep",
                replace_existing=True, max_instances=1)
    print(f"[DigitalSleep] Scheduled every {every_minutes} min ({_DIGITAL_SLEEP_VERSION}).")
