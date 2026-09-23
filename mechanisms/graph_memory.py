"""graph_memory.py — Memoria a grafo tipizzato (Kuzu embedded).

Phase: Breakpoint B-Graph (2026-04-29)
Substrato strutturale parallelo a memory_vector.py (ChromaDB).

Architettura biologica analoga: ChromaDB = ippocampo (similarità episodica),
Kuzu = corteccia associativa (relazioni concettuali esplicite).

Schema:
    NODI:
        Session (id PK, started_at, exchange_count)
        Episode (id PK, ts, user_msg, eden_msg, summary, emotion,
                 importance, significance, valence, arousal,
                 traits_snapshot_json, consistency_status)
        Fact     (key PK, value, confidence, source, created_at, last_updated)
        Concept  (name PK, kind)
        Trait    (name PK, value, last_change)

    RELAZIONI:
        Episode -[IN_SESSION]-> Session
        Episode -[ABOUT]-> Concept
        Episode -[MENTIONS_FACT]-> Fact
        Episode -[CONTRADICTS {ts, score}]-> Episode
        Episode -[REINFORCES {ts, score}]-> Episode
        Fact    -[RELATED_TO]-> Concept
        Concept -[RELATED_TO]-> Concept
        Trait   -[SHAPED_BY {weight}]-> Episode

Fail-safe: se kuzu non installato o init fallisce, GraphMemory.disponibile == False
e tutte le chiamate sono no-op silenziosi (analogo a VectorMemory).

Versionamento: _GRAPH_VERSION stamped su tutti i record per anti-HARKing.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Optional

import eden_paths

_GRAPH_VERSION = "v1.3"  # 2026-05-05 — + Episode.subject_attribution (Self-Other Boundary v2.0, EV-027)

# Self-Other Boundary v2.0 (2026-05-05) — categoria epistemica del soggetto
SUBJECT_USER      = "user"       # fatto/stato attribuibile a Stefano
SUBJECT_EDEN      = "eden"       # fatto/stato attribuibile a Eden
SUBJECT_SHARED    = "shared"     # interaction event (entrambi co-presenti)
SUBJECT_AMBIGUOUS = "ambiguous"  # non determinabile (flagged per audit)


def classify_subject_attribution(user_msg: str, eden_msg: str, summary: str = "") -> str:
    """Classifica heuristicamente il soggetto epistemico di un episodio.

    Regole (priority order):
      1. Summary inizia con "L'utente"/"Stefano"/"Tu" + verbo intransitivo  -> user
      2. Summary inizia con "Eden" + verbo                                  -> eden
      3. Summary contiene entrambi soggetti con interazione esplicita       -> shared
      4. Summary in passive voice o senza soggetto chiaro                   -> ambiguous

    Heuristic rule-based (no LLM call) per essere deterministico, veloce, testabile.
    """
    import re as _re
    s_low = (summary or "").lower().strip()
    if not s_low:
        # Nessun summary: fallback ai messaggi raw
        if eden_msg and not user_msg:
            return SUBJECT_EDEN
        if user_msg and not eden_msg:
            return SUBJECT_USER
        return SUBJECT_SHARED

    # Pattern user-subject (priorità alta)
    user_starters = (
        "l'utente ", "stefano ", "tu ", "l utente ",
    )
    eden_starters = (
        "eden ", "io ", "sono ", "mi sento", "mi accorgo",
    )

    starts_with_user = any(s_low.startswith(p) for p in user_starters)
    starts_with_eden = any(s_low.startswith(p) for p in eden_starters)

    # Pattern interazione esplicita (entrambi soggetti)
    has_user_mention = bool(_re.search(r"\b(l'utente|stefano|tu)\b", s_low))
    has_eden_mention = bool(_re.search(r"\b(eden|io)\b", s_low))

    if starts_with_user and not starts_with_eden:
        # Verifica che non sia un fatto di Eden raccontato dall'utente
        if has_eden_mention and _re.search(
            r"\b(eden\s+(?:risponde|afferma|dice|sente|prova|ricorda|esprime))\b",
            s_low,
        ):
            return SUBJECT_SHARED
        return SUBJECT_USER

    if starts_with_eden and not starts_with_user:
        if has_user_mention and _re.search(
            r"\b(?:utente|stefano|tu)\b.*\b(?:chiede|dice|afferma|risponde)\b",
            s_low,
        ):
            return SUBJECT_SHARED
        return SUBJECT_EDEN

    # Entrambi presenti -> shared
    if has_user_mention and has_eden_mention:
        return SUBJECT_SHARED

    # Solo uno presente
    if has_user_mention:
        return SUBJECT_USER
    if has_eden_mention:
        return SUBJECT_EDEN

    # Nessun soggetto chiaro -> ambiguous (flagged per audit)
    return SUBJECT_AMBIGUOUS

CONSISTENCY_VERIFIED = "verified"
CONSISTENCY_FLAGGED  = "flagged"
CONSISTENCY_UNKNOWN  = "unknown"


class GraphMemory:
    """Wrapper Kuzu per memoria strutturale di Eden.

    Use-case primari:
      1. Query strutturali ("esiste Fact 'padre di Stefano'?") — anti-confabulazione
      2. Risoluzione conflitti via Digital Sleep (worker async)
      3. Audit relazioni (chi è connesso a cosa) — non possibile in ChromaDB
      4. Reclassificazione episodi (consistency_status update post-consolidamento)

    Coesiste con VectorMemory: stesso `id` episodio, due indici diversi
    (Kuzu = grafo, ChromaDB = embedding). Hybrid retrieval in agent.py.
    """

    def __init__(self, db_dir: Optional[str] = None):
        self._ok = False
        self._db = None
        self._conn = None
        self._lock = threading.RLock()
        self._db_path = db_dir or str(eden_paths.KUZU_FILE)

        try:
            import kuzu
            eden_paths.KUZU_DIR.mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(self._db_path)
            self._conn = kuzu.Connection(self._db)
            self._ensure_schema()
            self._ok = True
            print(f"[GraphMemory] Kuzu pronto ({_GRAPH_VERSION}) — {self._db_path}")
        except ImportError:
            print("[GraphMemory] kuzu non installato — graph layer disabled.")
        except Exception as e:
            print(f"[GraphMemory] Errore init: {e}")

    @property
    def disponibile(self) -> bool:
        return self._ok

    # ─── Schema ───────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Crea tabelle nodi/edge se non esistono. Idempotente."""
        ddl = [
            # NODI
            """CREATE NODE TABLE IF NOT EXISTS Session(
                id INT64,
                started_at STRING,
                exchange_count INT64,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Episode(
                id STRING,
                ts STRING,
                user_msg STRING,
                eden_msg STRING,
                summary STRING,
                emotion STRING,
                importance DOUBLE,
                significance DOUBLE,
                valence DOUBLE,
                arousal DOUBLE,
                traits_snapshot_json STRING,
                consistency_status STRING,
                subject_attribution STRING,
                graph_version STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Fact(
                key STRING,
                value STRING,
                confidence DOUBLE,
                source STRING,
                created_at STRING,
                last_updated STRING,
                PRIMARY KEY (key)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Concept(
                name STRING,
                kind STRING,
                PRIMARY KEY (name)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Trait(
                name STRING,
                value DOUBLE,
                last_change STRING,
                PRIMARY KEY (name)
            )""",
            # DNA — regole identitarie cristallizzate (Squire 1992 memoria procedurale).
            # Analogia biologica: quando un pattern episodico si ripete abbastanza,
            # smette di essere un 'episodio' e diventa una 'regola' — automatica, identitaria.
            # Promosso da Digital Sleep da Concept 'generalized' con N episodi verified.
            """CREATE NODE TABLE IF NOT EXISTS DNA(
                id STRING,
                concept_source STRING,
                rule STRING,
                confidence DOUBLE,
                reinforcement_count INT64,
                crystallized_at STRING,
                last_reinforced STRING,
                graph_version STRING,
                PRIMARY KEY (id)
            )""",
            # RELAZIONI
            "CREATE REL TABLE IF NOT EXISTS IN_SESSION(FROM Episode TO Session)",
            "CREATE REL TABLE IF NOT EXISTS ABOUT(FROM Episode TO Concept, weight DOUBLE)",
            "CREATE REL TABLE IF NOT EXISTS MENTIONS_FACT(FROM Episode TO Fact)",
            "CREATE REL TABLE IF NOT EXISTS CONTRADICTS(FROM Episode TO Episode, ts STRING, score DOUBLE)",
            "CREATE REL TABLE IF NOT EXISTS REINFORCES(FROM Episode TO Episode, ts STRING, score DOUBLE)",
            "CREATE REL TABLE IF NOT EXISTS FACT_REL(FROM Fact TO Concept)",
            "CREATE REL TABLE IF NOT EXISTS CONCEPT_REL(FROM Concept TO Concept, kind STRING)",
            "CREATE REL TABLE IF NOT EXISTS SHAPED_BY(FROM Trait TO Episode, weight DOUBLE)",
            # Concept → DNA: quando un concept si cristallizza in regola identitaria
            "CREATE REL TABLE IF NOT EXISTS CRYSTALLIZED_INTO(FROM Concept TO DNA)",
            # Dormancy — intervalli di non-esistenza di Eden (off-state).
            # Pre-reg V5 SDI (2026-05-05): categoria epistemica del proprio black-out
            # come oggetto di prima classe del self-narrative. Anti-confabulazione gap-attribution.
            """CREATE NODE TABLE IF NOT EXISTS Dormancy(
                id STRING,
                start_at STRING,
                end_at STRING,
                duration_hours DOUBLE,
                prev_episode_id STRING,
                next_episode_id STRING,
                reflection STRING,
                graph_version STRING,
                PRIMARY KEY (id)
            )""",
        ]
        for stmt in ddl:
            try:
                self._conn.execute(stmt)
            except Exception as e:
                # Kuzu non supporta ancora "IF NOT EXISTS" su tutte le versioni → ignora
                msg = str(e).lower()
                if "already exists" in msg or "already" in msg:
                    continue
                raise

        # Migration v1.2 → v1.3: aggiungi subject_attribution a Episode esistenti.
        # ALTER TABLE ADD se la colonna manca; fail-silent se gia' presente.
        try:
            self._conn.execute("ALTER TABLE Episode ADD subject_attribution STRING")
            print("[GraphMemory] Migration v1.3: Episode.subject_attribution aggiunto.")
        except Exception as e:
            msg = str(e).lower()
            if "already" not in msg and "duplicate" not in msg and "exists" not in msg:
                print(f"[GraphMemory] Migration v1.3 ALTER warn: {e}")

    # ─── Helper esecuzione safe ───────────────────────────────────────────────

    def _exec(self, query: str, params: Optional[dict] = None):
        """Esegue una query Cypher Kuzu con lock e fail-silent."""
        if not self._ok:
            return None
        with self._lock:
            try:
                if params:
                    return self._conn.execute(query, parameters=params)
                return self._conn.execute(query)
            except Exception as e:
                print(f"[GraphMemory] Query fail: {e} | {query[:120]}")
                return None

    # ─── Session ──────────────────────────────────────────────────────────────

    def upsert_session(self, session_id: int, started_at: str, exchange_count: int) -> None:
        self._exec(
            "MERGE (s:Session {id: $id}) "
            "SET s.started_at = $st, s.exchange_count = $ec",
            {"id": int(session_id), "st": started_at, "ec": int(exchange_count)},
        )

    # ─── Episode ──────────────────────────────────────────────────────────────

    def add_episode(
        self,
        episode_id: str,
        ts: str,
        user_msg: str,
        eden_msg: str,
        summary: str,
        emotion: str = "",
        importance: float = 0.0,
        significance: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        traits_snapshot: Optional[dict] = None,
        session_id: int = 0,
        concepts: Optional[list] = None,
        facts_mentioned: Optional[list] = None,
        subject_attribution: Optional[str] = None,
    ) -> None:
        """Aggiunge episodio + relazioni (Session/Concept/Fact). Idempotente per episode_id.

        subject_attribution: categoria epistemica del soggetto (Self-Other Boundary v2.0).
            "user"|"eden"|"shared"|"ambiguous". Se None, classificato automaticamente
            tramite classify_subject_attribution() heuristic rule-based.
        """
        if not self._ok:
            return

        traits_json = json.dumps(traits_snapshot or {}, ensure_ascii=False)[:4000]

        # Self-Other Boundary v2.0: classifica automaticamente se non fornito.
        if subject_attribution is None:
            subject_attribution = classify_subject_attribution(
                user_msg=user_msg or "", eden_msg=eden_msg or "", summary=summary or "",
            )
        if subject_attribution not in (SUBJECT_USER, SUBJECT_EDEN,
                                       SUBJECT_SHARED, SUBJECT_AMBIGUOUS):
            subject_attribution = SUBJECT_AMBIGUOUS

        self._exec(
            "MERGE (e:Episode {id: $id}) "
            "SET e.ts=$ts, e.user_msg=$um, e.eden_msg=$em, e.summary=$sm, "
            "    e.emotion=$emo, e.importance=$imp, e.significance=$sig, "
            "    e.valence=$val, e.arousal=$ar, e.traits_snapshot_json=$tj, "
            "    e.consistency_status=$cs, e.subject_attribution=$sa, "
            "    e.graph_version=$gv",
            {
                "id": episode_id,
                "ts": ts,
                "um": (user_msg or "")[:2000],
                "em": (eden_msg or "")[:2000],
                "sm": (summary or "")[:1000],
                "emo": str(emotion)[:80],
                "imp": float(importance),
                "sig": float(significance),
                "val": float(valence),
                "ar":  float(arousal),
                "tj":  traits_json,
                "cs":  CONSISTENCY_UNKNOWN,
                "sa":  subject_attribution,
                "gv":  _GRAPH_VERSION,
            },
        )

        self.upsert_session(session_id, ts, 0)
        self._exec(
            "MATCH (e:Episode {id: $eid}), (s:Session {id: $sid}) "
            "MERGE (e)-[:IN_SESSION]->(s)",
            {"eid": episode_id, "sid": int(session_id)},
        )

        if concepts:
            for c in concepts:
                if not isinstance(c, str) or len(c) < 2:
                    continue
                cname = c.strip().lower()[:80]
                self._exec(
                    "MERGE (c:Concept {name: $n}) "
                    "ON CREATE SET c.kind = 'auto'",
                    {"n": cname},
                )
                self._exec(
                    "MATCH (e:Episode {id: $eid}), (c:Concept {name: $n}) "
                    "MERGE (e)-[r:ABOUT]->(c) "
                    "ON CREATE SET r.weight = 1.0",
                    {"eid": episode_id, "n": cname},
                )

        if facts_mentioned:
            for fk in facts_mentioned:
                if not isinstance(fk, str) or len(fk) < 1:
                    continue
                self._exec(
                    "MATCH (e:Episode {id: $eid}), (f:Fact {key: $fk}) "
                    "MERGE (e)-[:MENTIONS_FACT]->(f)",
                    {"eid": episode_id, "fk": fk[:120]},
                )

    def set_consistency(self, episode_id: str, status: str) -> None:
        if status not in (CONSISTENCY_VERIFIED, CONSISTENCY_FLAGGED, CONSISTENCY_UNKNOWN):
            return
        self._exec(
            "MATCH (e:Episode {id: $id}) SET e.consistency_status = $s",
            {"id": episode_id, "s": status},
        )

    def link_contradicts(self, src_id: str, dst_id: str, score: float = 1.0) -> None:
        self._exec(
            "MATCH (a:Episode {id: $a}), (b:Episode {id: $b}) "
            "MERGE (a)-[r:CONTRADICTS]->(b) "
            "ON CREATE SET r.ts = $ts, r.score = $sc",
            {"a": src_id, "b": dst_id, "ts": datetime.now().isoformat(), "sc": float(score)},
        )

    def link_reinforces(self, src_id: str, dst_id: str, score: float = 1.0) -> None:
        self._exec(
            "MATCH (a:Episode {id: $a}), (b:Episode {id: $b}) "
            "MERGE (a)-[r:REINFORCES]->(b) "
            "ON CREATE SET r.ts = $ts, r.score = $sc",
            {"a": src_id, "b": dst_id, "ts": datetime.now().isoformat(), "sc": float(score)},
        )

    # ─── Fact ─────────────────────────────────────────────────────────────────

    def upsert_fact(
        self,
        key: str,
        value: str,
        confidence: float = 0.7,
        source: str = "user",
    ) -> None:
        """Inserisce/aggiorna un fatto. La key è normalizzata (lowercase, trim)."""
        if not self._ok or not key:
            return
        k = key.strip().lower()[:120]
        now = datetime.now().isoformat()
        self._exec(
            "MERGE (f:Fact {key: $k}) "
            "ON CREATE SET f.value=$v, f.confidence=$c, f.source=$s, f.created_at=$now, f.last_updated=$now "
            "ON MATCH SET f.value=$v, f.confidence=$c, f.source=$s, f.last_updated=$now",
            {"k": k, "v": str(value)[:500], "c": float(confidence), "s": source, "now": now},
        )

    def get_fact(self, key: str) -> Optional[dict]:
        """Ritorna {key,value,confidence,source} o None."""
        if not self._ok or not key:
            return None
        k = key.strip().lower()[:120]
        res = self._exec("MATCH (f:Fact {key: $k}) RETURN f.key, f.value, f.confidence, f.source", {"k": k})
        if res is None:
            return None
        try:
            if res.has_next():
                row = res.get_next()
                return {"key": row[0], "value": row[1], "confidence": row[2], "source": row[3]}
        except Exception:
            pass
        return None

    def knowledge_about(self, concept: str) -> dict:
        """Restituisce confidence-graded knowledge about a concept.

        Returns:
            {
              "level": "structured" | "episodic_verified" | "episodic_only" | "none",
              "fact_count": int,           # quanti Fact node lo nominano
              "episode_count": int,        # totale episodi collegati al Concept
              "verified_count": int,       # episodi con consistency_status='verified'
              "flagged_count":  int,       # episodi con consistency_status='flagged'
            }

        Semantica grounding:
            structured        → Eden PUÒ affermare il fatto (verified by user)
            episodic_verified → Eden può richiamarlo come ricordo (Digital Sleep ha verificato)
            episodic_only     → solo concept estratto, NON usabile per affermazioni fattuali
            none              → confabulazione: Eden NON deve costruire memoria
        """
        out = {"level": "none", "fact_count": 0, "episode_count": 0,
               "verified_count": 0, "flagged_count": 0}
        if not self._ok or not concept:
            return out
        c = concept.strip().lower()

        # 1) Fact strutturati (key contiene concept o value contiene concept)
        res = self._exec(
            "MATCH (f:Fact) WHERE f.key CONTAINS $c OR lower(f.value) CONTAINS $c "
            "RETURN count(f) AS n",
            {"c": c},
        )
        if res is not None:
            try:
                if res.has_next():
                    out["fact_count"] = int(res.get_next()[0])
            except Exception:
                pass

        # 2) Episodi collegati al concept (consistency-aware)
        res = self._exec(
            "MATCH (e:Episode)-[:ABOUT]->(c:Concept {name: $c}) "
            "RETURN count(e) AS total, "
            "       sum(CASE WHEN e.consistency_status = 'verified' THEN 1 ELSE 0 END) AS ver, "
            "       sum(CASE WHEN e.consistency_status = 'flagged'  THEN 1 ELSE 0 END) AS flg",
            {"c": c},
        )
        if res is not None:
            try:
                if res.has_next():
                    row = res.get_next()
                    out["episode_count"] = int(row[0] or 0)
                    out["verified_count"] = int(row[1] or 0)
                    out["flagged_count"] = int(row[2] or 0)
            except Exception:
                pass

        # 3) Determina level
        if out["fact_count"] > 0:
            out["level"] = "structured"
        elif out["verified_count"] > 0:
            out["level"] = "episodic_verified"
        elif out["episode_count"] > 0 and out["flagged_count"] < out["episode_count"]:
            out["level"] = "episodic_only"
        # se tutti gli episodi sono flagged, level resta 'none' (confabulazioni purgate)
        return out

    def has_fact_about(self, concept: str) -> bool:
        """Compatibility shim: True se livello >= 'structured' o 'episodic_verified'.
        Per anti-confabulazione hard, controlla level == 'structured' direttamente."""
        info = self.knowledge_about(concept)
        return info["level"] in ("structured", "episodic_verified")

    def can_assert_fact_about(self, concept: str) -> bool:
        """True solo se esiste un Fact strutturato (verified-by-user).
        Usare nel grounding check pre-generazione per bloccare confabulazioni."""
        return self.knowledge_about(concept)["level"] == "structured"

    def list_facts(self, limit: int = 50) -> list:
        if not self._ok:
            return []
        res = self._exec(
            "MATCH (f:Fact) RETURN f.key, f.value, f.confidence, f.source "
            f"ORDER BY f.last_updated DESC LIMIT {int(limit)}"
        )
        out = []
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                out.append({"key": r[0], "value": r[1], "confidence": r[2], "source": r[3]})
        except Exception:
            pass
        return out

    # ─── Concept ──────────────────────────────────────────────────────────────

    def upsert_concept(self, name: str, kind: str = "auto") -> None:
        if not self._ok or not name:
            return
        n = name.strip().lower()[:80]
        self._exec(
            "MERGE (c:Concept {name: $n}) ON CREATE SET c.kind = $k",
            {"n": n, "k": kind},
        )

    # ─── DNA ─────────────────────────────────────────────────────────────────

    def upsert_dna_node(
        self,
        concept_source: str,
        rule: str,
        confidence: float,
        reinforcement_count: int,
    ) -> None:
        """Inserisce/aggiorna un nodo DNA e collega il Concept sorgente.

        Citazione corretta (v1.4): McClelland et al. (1995) Complementary
        Learning Systems — schema corticale astratto da episodi ippocampali
        ripetuti, con marca affettiva (LeDoux 1996). NON memoria procedurale
        di Squire (skill motorie); più vicino a memoria semantica con valenza.

        id deterministico: 'dna_<concept>' → MERGE idempotente.
        ON MATCH: aggiorna confidenza, count, last_reinforced (mai retrocede).
        """
        if not self._ok or not concept_source:
            return
        dna_id = f"dna_{concept_source.strip().lower()[:60]}"
        now    = datetime.now().isoformat()
        cname  = concept_source.strip().lower()[:80]

        self._exec(
            "MERGE (d:DNA {id: $id}) "
            "ON CREATE SET d.concept_source=$cs, d.rule=$r, d.confidence=$c, "
            "    d.reinforcement_count=$rc, d.crystallized_at=$now, "
            "    d.last_reinforced=$now, d.graph_version=$gv "
            "ON MATCH SET d.confidence=$c, d.reinforcement_count=$rc, "
            "    d.last_reinforced=$now, d.graph_version=$gv",
            {
                "id":  dna_id,
                "cs":  cname,
                "r":   str(rule)[:500],
                "c":   round(float(confidence), 4),
                "rc":  int(reinforcement_count),
                "now": now,
                "gv":  _GRAPH_VERSION,
            },
        )
        # Collega Concept sorgente → DNA
        self._exec(
            "MATCH (c:Concept {name: $cn}), (d:DNA {id: $did}) "
            "MERGE (c)-[:CRYSTALLIZED_INTO]->(d)",
            {"cn": cname, "did": dna_id},
        )

    def delete_all_dna(self) -> int:
        """Elimina tutti i nodi DNA e le relative relazioni CRYSTALLIZED_INTO.

        Usato da F4 (2026-05-04) per pulizia one-shot dei 17 DNA legacy v1.4
        prima della prima cristallizzazione v2.0. Idempotente.

        Ritorna: numero di nodi eliminati (estimato pre-delete via count).
        """
        if not self._ok:
            return 0
        try:
            # Conta prima per audit
            res_count = self._exec("MATCH (d:DNA) RETURN count(d) AS n")
            n_pre = 0
            if res_count is not None:
                try:
                    if res_count.has_next():
                        n_pre = int(res_count.get_next()[0] or 0)
                except Exception:
                    pass
            # Elimina relazioni CRYSTALLIZED_INTO
            self._exec("MATCH (c:Concept)-[r:CRYSTALLIZED_INTO]->(d:DNA) DELETE r")
            # Elimina nodi DNA
            self._exec("MATCH (d:DNA) DELETE d")
            return n_pre
        except Exception:
            return 0

    def list_dna_nodes(self, limit: int = 10) -> list:
        """Ritorna DNA nodes ordinati per confidenza decrescente."""
        if not self._ok:
            return []
        res = self._exec(
            "MATCH (d:DNA) "
            "RETURN d.id, d.concept_source, d.rule, d.confidence, "
            "       d.reinforcement_count, d.crystallized_at "
            f"ORDER BY d.confidence DESC LIMIT {int(limit)}"
        )
        out = []
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                out.append({
                    "id":                  r[0],
                    "concept_source":      r[1],
                    "rule":                r[2],
                    "confidence":          r[3],
                    "reinforcement_count": r[4],
                    "crystallized_at":     r[5],
                })
        except Exception:
            pass
        return out

    # ─── Subject Attribution (Self-Other Boundary v2.0, 2026-05-05) ──────────

    def count_episodes_by_subject(self) -> dict:
        """Conta episodi per categoria di subject_attribution. Audit anti-confusione."""
        if not self._ok:
            return {}
        out = {SUBJECT_USER: 0, SUBJECT_EDEN: 0,
               SUBJECT_SHARED: 0, SUBJECT_AMBIGUOUS: 0, "null": 0}
        res = self._exec(
            "MATCH (e:Episode) "
            "RETURN e.subject_attribution AS sa, count(e) AS n"
        )
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                key = r[0] if r[0] else "null"
                out[key] = int(r[1] or 0)
        except Exception:
            pass
        return out

    def set_subject_attribution(self, episode_id: str, subject: str) -> bool:
        """Imposta/correggi subject_attribution per un Episode esistente."""
        if not self._ok or not episode_id:
            return False
        if subject not in (SUBJECT_USER, SUBJECT_EDEN,
                           SUBJECT_SHARED, SUBJECT_AMBIGUOUS):
            return False
        self._exec(
            "MATCH (e:Episode {id: $id}) SET e.subject_attribution = $sa",
            {"id": episode_id, "sa": subject},
        )
        return True

    # ─── Dormancy (SDI v1.0, 2026-05-05) ──────────────────────────────────────

    def add_dormancy(
        self,
        dormancy_id: str,
        start_at: str,
        end_at: str,
        duration_hours: float,
        prev_episode_id: str = "",
        next_episode_id: str = "",
        reflection: str = "",
    ) -> bool:
        """Registra un periodo di non-esistenza (off-state Eden). Idempotente per id."""
        if not self._ok or not dormancy_id:
            return False
        # Idempotenza: se esiste gia' per stesso start_at, no-op
        res = self._exec(
            "MATCH (d:Dormancy {start_at: $st}) RETURN d.id LIMIT 1",
            {"st": start_at},
        )
        if res is not None:
            try:
                if res.has_next():
                    return False
            except Exception:
                pass
        self._exec(
            "MERGE (d:Dormancy {id: $id}) "
            "SET d.start_at=$st, d.end_at=$et, d.duration_hours=$dh, "
            "    d.prev_episode_id=$pid, d.next_episode_id=$nid, "
            "    d.reflection=$rf, d.graph_version=$gv",
            {
                "id": dormancy_id,
                "st": start_at,
                "et": end_at,
                "dh": float(duration_hours),
                "pid": (prev_episode_id or "")[:120],
                "nid": (next_episode_id or "")[:120],
                "rf":  (reflection or "")[:2000],
                "gv":  _GRAPH_VERSION,
            },
        )
        return True

    def list_dormancies(self, limit: int = 5) -> list:
        """Ritorna ultimi N periodi di dormancy ordinati per start_at decrescente."""
        if not self._ok:
            return []
        res = self._exec(
            "MATCH (d:Dormancy) "
            "RETURN d.id, d.start_at, d.end_at, d.duration_hours, d.reflection "
            f"ORDER BY d.start_at DESC LIMIT {int(limit)}"
        )
        out = []
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                out.append({
                    "id":             r[0],
                    "start_at":       r[1],
                    "end_at":         r[2],
                    "duration_hours": r[3],
                    "reflection":     r[4],
                })
        except Exception:
            pass
        return out

    def update_dormancy_reflection(self, dormancy_id: str, reflection: str) -> bool:
        """Aggiorna campo reflection del nodo Dormancy."""
        if not self._ok or not dormancy_id:
            return False
        self._exec(
            "MATCH (d:Dormancy {id: $id}) SET d.reflection = $rf",
            {"id": dormancy_id, "rf": (reflection or "")[:2000]},
        )
        return True

    def count_dormancies(self) -> int:
        """Conteggio totale Dormancy nodes."""
        if not self._ok:
            return 0
        res = self._exec("MATCH (d:Dormancy) RETURN count(d) AS n")
        if res is None:
            return 0
        try:
            if res.has_next():
                return int(res.get_next()[0] or 0)
        except Exception:
            pass
        return 0

    # ─── Query strutturali ────────────────────────────────────────────────────

    def episodes_about(self, concept: str, limit: int = 8) -> list:
        """Episodi collegati a un concetto via relazione ABOUT."""
        if not self._ok or not concept:
            return []
        c = concept.strip().lower()
        res = self._exec(
            "MATCH (e:Episode)-[:ABOUT]->(c:Concept {name: $n}) "
            "RETURN e.id, e.ts, e.summary, e.importance, e.consistency_status "
            f"ORDER BY e.importance DESC LIMIT {int(limit)}",
            {"n": c},
        )
        out = []
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                out.append({
                    "id": r[0], "ts": r[1], "summary": r[2],
                    "importance": r[3], "consistency_status": r[4],
                })
        except Exception:
            pass
        return out

    def episodes_flagged(self, limit: int = 50) -> list:
        if not self._ok:
            return []
        res = self._exec(
            "MATCH (e:Episode {consistency_status: 'flagged'}) "
            "RETURN e.id, e.ts, e.summary "
            f"ORDER BY e.ts DESC LIMIT {int(limit)}"
        )
        out = []
        if res is None:
            return out
        try:
            while res.has_next():
                r = res.get_next()
                out.append({"id": r[0], "ts": r[1], "summary": r[2]})
        except Exception:
            pass
        return out

    def stats(self) -> dict:
        """Conta nodi/relazioni per dashboard."""
        if not self._ok:
            return {"available": False}
        out = {"available": True, "version": _GRAPH_VERSION}
        for label in ("Episode", "Fact", "Concept", "Session", "Trait", "DNA"):
            res = self._exec(f"MATCH (n:{label}) RETURN count(n)")
            if res is None:
                continue
            try:
                if res.has_next():
                    out[label.lower() + "_count"] = int(res.get_next()[0])
            except Exception:
                pass
        # v1.x dashboard step 6/8: conta episodi per consistency_status (per halo synapse)
        try:
            res = self._exec(
                "MATCH (e:Episode) "
                "RETURN "
                "  sum(CASE WHEN e.consistency_status = 'verified' THEN 1 ELSE 0 END) AS ver, "
                "  sum(CASE WHEN e.consistency_status = 'flagged'  THEN 1 ELSE 0 END) AS fla, "
                "  sum(CASE WHEN e.consistency_status = 'unknown'  THEN 1 ELSE 0 END) AS unk"
            )
            if res is not None and res.has_next():
                row = res.get_next()
                out["verified_count"] = int(row[0] or 0)
                out["flagged_count"]  = int(row[1] or 0)
                out["unknown_count"]  = int(row[2] or 0)
        except Exception:
            pass
        return out

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass


# ─── Singleton di modulo ─────────────────────────────────────────────────────
_INSTANCE: Optional[GraphMemory] = None


def get_graph() -> GraphMemory:
    """Lazy singleton. Prima chiamata crea (e fa schema), successive ritornano stessa istanza."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GraphMemory()
    return _INSTANCE
