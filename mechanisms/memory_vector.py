# memory_vector.py — Memoria vettoriale ChromaDB (Phase 1.8 + Memory v2.0)
# 5 collection: episodic_memory, emotional_memory, semantic_memory, working_memory,
#               episodic_archive_memory (Memory v2.0, total recall, no importance filter)
# Modello embedding: paraphrase-multilingual-MiniLM-L12-v2 (CPU, ~120MB, multilingue)
#
# Fallback graceful: se chromadb/sentence-transformers non sono installati,
# VectorMemory.disponibile == False e tutte le chiamate sono no-op silenzioso.
#
# Memory v2.0 — Doppio layer (2026-04-21):
#   Operativo: episodic_memory (filtrato importance>=5, cap 100 in memory.json)
#   Archivio:  episodic_archive_memory (ogni scambio, no filtro, no cap)
#              + data/logs/episodic_archive.jsonl (audit trail append-only)
#   Coesistono senza interferenza. L'archivio non influenza il giudizio di
#   importance operativo — è solo un secondo canale di retrieval granulare.

import json
import os
import shutil
import threading
from datetime import datetime
from typing import Optional

EMBEDDING_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
EPISODIC_THRESHOLD = 5    # importanza minima (0-10) per salvare in episodic (operativo)
MAX_WORKING        = 20   # cap messaggi working_memory

# ─── Memory v2.0 — Archivio totale ────────────────────────────────────────
import eden_paths

_MEMORY_VERSION       = "v2.0"
_ARCHIVE_ENABLED      = True                                  # kill switch globale

# F6 — Mood-Congruent Memory (Bower 1981, PRE_REG_v4 §F6)
_MOOD_CONGRUENT_VERSION = "v1.0"
_MOOD_CONGRUENCE_WEIGHT = 0.08
_EMOTION_VALENCE = {
    "gioia": 0.8, "felicità": 0.8, "entusiasmo": 0.7, "curiosità": 0.5,
    "gratitudine": 0.7, "serenità": 0.6, "calore": 0.6, "fiducia": 0.6,
    "speranza": 0.5, "interesse": 0.4, "orgoglio": 0.5, "affetto": 0.7,
    "tristezza": -0.7, "rabbia": -0.6, "paura": -0.6, "ansia": -0.5,
    "frustrazione": -0.5, "dolore": -0.8, "delusione": -0.6,
    "preoccupazione": -0.5, "solitudine": -0.6, "rimpianto": -0.5,
    "disagio": -0.4, "sorpresa": 0.1, "neutro": 0.0, "ambivalenza": 0.0,
}

# Sub-condizione C2 corrente — auto-detect da Kuzu disponibilità.
# C2_graph: Kuzu attivo (Breakpoint B-Graph, dal 2026-04-30 sera).
# C2_chromadb: solo ChromaDB (pre-Breakpoint B-Graph).
# Stampa su ogni record archive per anti-HARKing (PRE_REG_v3 addendum 3.3).
def _detect_condition_tag() -> str:
    try:
        from mechanisms.graph_memory import get_graph as _gg
        return "C2_graph" if _gg().disponibile else "C2_chromadb"
    except Exception:
        return "C2_chromadb"
_ARCHIVE_COLLECTION   = "episodic_archive_memory"
_ARCHIVE_JSONL        = str(eden_paths.EPISODIC_ARCHIVE_FILE)
_ARCHIVE_BACKUP_DIR   = str(eden_paths.ARCHIVE_BACKUPS_DIR)
_ARCHIVE_BACKUP_MB    = 50     # soglia rotazione backup (solo copia, mai trunca)
_ARCHIVE_BACKUP_EVERY = 500    # ogni N append, controlla rotazione
_ARCHIVE_USER_TRUNC   = 500    # max char per user_msg nel documento indicizzato
_ARCHIVE_EDEN_TRUNC   = 500    # max char per assistant_msg nel documento indicizzato


class VectorMemory:
    """
    Memoria vettoriale ChromaDB per Eden.

    Le 4 collection:
      episodic_memory  — scambi con importanza >= 5, ricercati per contenuto semantico
      emotional_memory — stessi momenti indicizzati per emozione dominante
      semantic_memory  — fatti appresi su Stefano (upsert per chiave)
      working_memory   — ultimi 20 messaggi sessione corrente (nessun embedding)

    Tutto gira su CPU: nessun conflitto VRAM con Ollama.
    Prima chiamata embedding: ~3s per caricare il modello in RAM.
    Chiamate successive: <100ms.
    """

    def __init__(self, persist_dir: str = "chroma_db"):
        self._ok     = False
        self._client = None
        self._ef     = None
        self._episodic  = None
        self._emotional = None
        self._semantic  = None
        self._working   = None
        self._archive   = None                # Memory v2.0 — archivio totale
        self._archive_append_counter = 0      # conta append per trigger rotazione backup
        self._lock = threading.RLock()   # RLock: rientrante (migrate chiama altri metodi)

        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._client = chromadb.PersistentClient(path=persist_dir)
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                device="cpu"
            )

            self._episodic = self._client.get_or_create_collection(
                name="episodic_memory",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
            self._emotional = self._client.get_or_create_collection(
                name="emotional_memory",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
            self._semantic = self._client.get_or_create_collection(
                name="semantic_memory",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
            # working_memory: accesso diretto per ID (mai queryed)
            # Usa lo stesso EF per evitare download del modello ONNX default di ChromaDB
            self._working = self._client.get_or_create_collection(
                name="working_memory",
                embedding_function=self._ef
            )

            # Memory v2.0 — archivio totale (ogni scambio, no filtro importance)
            if _ARCHIVE_ENABLED:
                try:
                    self._archive = self._client.get_or_create_collection(
                        name=_ARCHIVE_COLLECTION,
                        embedding_function=self._ef,
                        metadata={
                            "hnsw:space":     "cosine",
                            "memory_version": _MEMORY_VERSION,
                            "description":    "Memory v2.0 total-recall archive — every exchange, no importance filter",
                        }
                    )
                    print(f"[VectorMemory] Archivio {_MEMORY_VERSION} pronto ({self._archive.count()} record).")
                except Exception as e:
                    print(f"[VectorMemory] Errore init archive: {e}")
                    self._archive = None

            self._ok = True
            print("[VectorMemory] ChromaDB pronto.")

        except ImportError as e:
            print(f"[VectorMemory] Dipendenze mancanti ({e}) — uso sistema legacy memory.py.")
        except Exception as e:
            print(f"[VectorMemory] Errore inizializzazione: {e}")

    @property
    def disponibile(self) -> bool:
        return self._ok

    # ─── Working memory ───────────────────────────────────────────────────────

    def add_working_memory(self, role: str, content: str) -> None:
        """Aggiunge un messaggio alla working memory e mantiene il cap a MAX_WORKING."""
        if not self._ok:
            return
        with self._lock:
            try:
                ts    = datetime.now().isoformat()
                count = self._working.count()
                doc_id = f"wm_{count:06d}_{datetime.now().strftime('%f')}"
                self._working.add(
                    documents=[content],
                    metadatas=[{"role": role, "ts": ts}],
                    ids=[doc_id]
                )
                # Rimuovi i messaggi più vecchi se supera il cap
                tutti = self._working.get(include=["metadatas"])
                if len(tutti["ids"]) > MAX_WORKING:
                    da_rimuovere = tutti["ids"][: len(tutti["ids"]) - MAX_WORKING]
                    if da_rimuovere:
                        self._working.delete(ids=da_rimuovere)
            except Exception as e:
                print(f"[VectorMemory] Errore add_working_memory: {e}")

    def get_working_memory(self) -> list:
        """Ritorna gli ultimi MAX_WORKING messaggi come lista [{"role": ..., "content": ...}]."""
        if not self._ok:
            return []
        with self._lock:
            try:
                items = self._working.get(include=["documents", "metadatas"])
                coppie = list(zip(items["documents"], items["metadatas"]))
                # Ordina per timestamp
                coppie.sort(key=lambda x: x[1].get("ts", ""))
                return [
                    {"role": meta.get("role", "user"), "content": doc}
                    for doc, meta in coppie[-MAX_WORKING:]
                ]
            except Exception as e:
                print(f"[VectorMemory] Errore get_working_memory: {e}")
                return []

    def clear_working_memory(self) -> None:
        """Svuota la working memory (chiamata ad ogni nuova sessione)."""
        if not self._ok:
            return
        with self._lock:
            try:
                tutti = self._working.get()
                if tutti["ids"]:
                    self._working.delete(ids=tutti["ids"])
                print("[VectorMemory] Working memory azzerata per nuova sessione.")
            except Exception as e:
                print(f"[VectorMemory] Errore clear_working_memory: {e}")

    # ─── Episodic + Emotional memory ──────────────────────────────────────────

    def save_episode(
        self,
        user_msg: str,
        assistant_msg: str,
        summary: str,
        emotion: str,
        importance: float,
        traits_snapshot: Optional[dict] = None,
        session_id: int = 0,
        cause: Optional[str] = None,
        appraisal_snapshot: Optional[dict] = None
    ) -> None:
        """
        Salva un episodio significativo in episodic_memory e emotional_memory.
        Scartato silenziosamente se importanza < EPISODIC_THRESHOLD.
        """
        if not self._ok or importance < EPISODIC_THRESHOLD:
            return
        with self._lock:
            try:
                ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
                doc_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

                # Documento testuale: user + eden + riassunto sintetico
                testo = (
                    f"Utente: {user_msg[:300]}\n"
                    f"Eden: {assistant_msg[:300]}\n"
                    f"Riassunto: {summary}"
                )

                metadata = {
                    "date":            ts,
                    "emotion":         str(emotion),
                    "importance":      float(importance),
                    "traits_snapshot": json.dumps(traits_snapshot or {}),
                    "session_id":      int(session_id),
                    "summary":         str(summary)[:500]
                }
                if cause:
                    metadata["cause"] = str(cause)[:240]
                if appraisal_snapshot is not None:
                    try:
                        metadata["appraisal_snapshot"] = json.dumps(appraisal_snapshot, ensure_ascii=False)[:1800]
                    except Exception:
                        metadata["appraisal_snapshot"] = "{}"

                self._episodic.add(
                    documents=[testo],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

                # Stessa data in emotional_memory — indicizzato per emozione
                self._emotional.add(
                    documents=[f"{emotion}: {summary}"],
                    metadatas=[{
                        "emotion_type": str(emotion),
                        "intensity":    float(importance),
                        "context":      str(summary)[:500],
                        "date":         ts
                    }],
                    ids=[f"em_{doc_id}"]
                )

            except Exception as e:
                print(f"[VectorMemory] Errore save_episode: {e}")

    def search_episodes(self, query: str, n: int = 5, valence: float = 0.0) -> list:
        """
        Cerca episodi simili per contenuto semantico al testo query.
        Ritorna lista di dict con chiavi: summary, date, emotion, importance, distance.
        valence: stato affettivo corrente [-1,1] per F6 Mood-Congruent Memory (Bower 1981).
        """
        if not self._ok:
            return []
        with self._lock:
            try:
                count = self._episodic.count()
                if count == 0:
                    return []
                results = self._episodic.query(
                    query_texts=[query],
                    n_results=min(n, count),
                    include=["documents", "metadatas", "distances"]
                )
                out = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    ep_emotion = meta.get("emotion", "")
                    ep_valence = _EMOTION_VALENCE.get(ep_emotion.lower().strip(), 0.0)
                    congruence = valence * ep_valence
                    adj_dist   = round(max(0.0, float(dist) - _MOOD_CONGRUENCE_WEIGHT * congruence), 3)
                    out.append({
                        "text":       doc,
                        "summary":    meta.get("summary", ""),
                        "date":       meta.get("date", ""),
                        "emotion":    ep_emotion,
                        "importance": meta.get("importance", 0),
                        "distance":   adj_dist
                    })
                out.sort(key=lambda e: e["distance"])
                return out
            except Exception as e:
                print(f"[VectorMemory] Errore search_episodes: {e}")
                return []

    def search_by_emotion(self, emotion_query: str, n: int = 3) -> list:
        """
        Cerca momenti con emozioni simili a emotion_query.
        Ritorna lista di dict con chiavi: emotion_type, intensity, context, date.
        """
        if not self._ok:
            return []
        with self._lock:
            try:
                count = self._emotional.count()
                if count == 0:
                    return []
                results = self._emotional.query(
                    query_texts=[emotion_query],
                    n_results=min(n, count),
                    include=["documents", "metadatas", "distances"]
                )
                out = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    out.append({
                        "text":         doc,
                        "emotion_type": meta.get("emotion_type", ""),
                        "intensity":    meta.get("intensity", 0),
                        "context":      meta.get("context", ""),
                        "date":         meta.get("date", ""),
                        "distance":     round(float(dist), 3)
                    })
                return out
            except Exception as e:
                print(f"[VectorMemory] Errore search_by_emotion: {e}")
                return []

    # ─── Semantic memory ──────────────────────────────────────────────────────

    def upsert_fact(self, fact_type: str, value: str, confidence: float = 0.8) -> None:
        """
        Inserisce o aggiorna un fatto su Stefano.
        Upsert: ID stabile = "fact_{fact_type}" → sovrascrive se il fatto cambia.
        """
        if not self._ok or not fact_type:
            return
        with self._lock:
            try:
                doc_id = f"fact_{fact_type}"
                self._semantic.upsert(
                    documents=[f"{fact_type}: {value}"],
                    metadatas=[{
                        "fact_type":    str(fact_type),
                        "value":        str(value)[:500],
                        "confidence":   float(confidence),
                        "last_updated": datetime.now().isoformat()
                    }],
                    ids=[doc_id]
                )
            except Exception as e:
                print(f"[VectorMemory] Errore upsert_fact: {e}")

    def get_semantic_facts(self) -> dict:
        """Restituisce tutti i fatti semantici come dizionario {fact_type: value}."""
        if not self._ok:
            return {}
        with self._lock:
            try:
                items = self._semantic.get(include=["metadatas"])
                return {
                    meta.get("fact_type", ""): meta.get("value", "")
                    for meta in items["metadatas"]
                    if meta.get("fact_type")
                }
            except Exception as e:
                print(f"[VectorMemory] Errore get_semantic_facts: {e}")
                return {}

    # ─── Facade save_exchange ─────────────────────────────────────────────────

    def save_exchange(
        self,
        user_msg: str,
        assistant_msg: str,
        summary: str = "",
        emotion: str = "",
        importance: float = 0.0,
        traits_snapshot: Optional[dict] = None,
        session_id: int = 0,
        cause: Optional[str] = None,
        appraisal_snapshot: Optional[dict] = None
    ) -> None:
        """
        Entry point principale: aggiorna working_memory e,
        se importance >= EPISODIC_THRESHOLD, salva in episodic + emotional.
        """
        if not self._ok:
            return
        self.add_working_memory("user", user_msg)
        self.add_working_memory("assistant", assistant_msg)
        if importance >= EPISODIC_THRESHOLD and summary:
            self.save_episode(
                user_msg, assistant_msg, summary, emotion,
                importance, traits_snapshot or {}, session_id,
                cause=cause, appraisal_snapshot=appraisal_snapshot
            )

    # ─── Memory v2.0 — Archivio totale ─────────────────────────────────────────
    #
    # Layer parallelo e additivo: NON sostituisce save_episode/save_exchange.
    # Scopo: eliminare le amnesie ingiuste prodotte dal filtro importance imperfetto.
    # Ogni scambio viene archiviato senza filtro in:
    #   (a) episodic_archive_memory — ChromaDB, retrieval semantico
    #   (b) data/logs/episodic_archive.jsonl — audit trail append-only, fonte di verità
    # Il canale operativo (episodic_memory filtrato) resta intatto: importante per
    # la ricerca perché il giudizio di importance è un dato osservato, non un bug
    # da rimuovere. I due canali coesistono e possono essere analizzati separatamente.

    def archive_exchange(
        self,
        user_msg: str,
        assistant_msg: str,
        summary: str = "",
        emotion: str = "",
        importance: float = 0.0,
        traits_snapshot: Optional[dict] = None,
        affective_snapshot: Optional[dict] = None,
        session_id: int = 0,
        exchange_idx: int = 0,
        cause: Optional[str] = None,
        appraisal_snapshot: Optional[dict] = None,
        significance_score: Optional[float] = None,
        significance_version: Optional[str] = None,
        llm_score: Optional[float] = None,
        cog_score: Optional[float] = None,
        episode_encoded: Optional[bool] = None,
    ) -> bool:
        """
        Memory v2.0 — archivia OGNI scambio, senza filtro importance.
        Dual-write: ChromaDB (retrieval) + JSONL (audit).

        Fail-silent: errori loggati via log_utils se disponibile, ritorna False.
        Ritorna True se almeno uno dei due write ha successo.
        """
        if not self._ok or not _ARCHIVE_ENABLED or self._archive is None:
            return False

        ok_chroma = False
        ok_jsonl  = False

        with self._lock:
            try:
                ts         = datetime.now().isoformat(timespec="seconds")
                date_human = datetime.now().strftime("%Y-%m-%d %H:%M")
                doc_id     = f"arc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

                # Documento indicizzato: user + eden (truncati) + riassunto se presente
                doc_parts = [
                    f"Utente: {str(user_msg)[:_ARCHIVE_USER_TRUNC]}",
                    f"Eden: {str(assistant_msg)[:_ARCHIVE_EDEN_TRUNC]}",
                ]
                if summary:
                    doc_parts.append(f"Riassunto: {str(summary)[:300]}")
                testo = "\n".join(doc_parts)

                metadata = {
                    "ts":             ts,
                    "date":           date_human,
                    "session_id":     int(session_id),
                    "exchange_idx":   int(exchange_idx),
                    "importance":     float(importance),
                    "emotion":        str(emotion)[:80] if emotion else "",
                    "summary":        str(summary)[:400] if summary else "",
                    "user_preview":   str(user_msg)[:200],
                    "eden_preview":   str(assistant_msg)[:200],
                    "memory_version": _MEMORY_VERSION,
                }
                if cause:
                    metadata["cause"] = str(cause)[:240]
                if traits_snapshot:
                    try:
                        metadata["traits_snapshot"] = json.dumps(traits_snapshot, ensure_ascii=False)[:500]
                    except Exception:
                        pass
                if affective_snapshot:
                    try:
                        metadata["affective_snapshot"] = json.dumps(affective_snapshot, ensure_ascii=False)[:500]
                    except Exception:
                        pass
                # Anti-HARKing: timbra ogni record archive con versione formula significance
                # e (quando disponibile) il valore calcolato + componenti appraisal.
                # Permette audit retrospettivo dell'archivio totale per Fase C/E.
                if significance_version:
                    metadata["significance_version"] = str(significance_version)
                if significance_score is not None:
                    metadata["significance_score"] = float(significance_score)
                if episode_encoded is not None:
                    metadata["episode_encoded"] = bool(episode_encoded)
                if appraisal_snapshot:
                    try:
                        metadata["appraisal_snapshot"] = json.dumps(appraisal_snapshot, ensure_ascii=False)[:600]
                    except Exception:
                        pass

                # (a) ChromaDB
                try:
                    self._archive.add(documents=[testo], metadatas=[metadata], ids=[doc_id])
                    ok_chroma = True
                except Exception as e:
                    self._log_error("archive_exchange:chroma_add", e)

                # (b) JSONL append-only (fonte di verità — più verbosa di ChromaDB meta)
                record = {
                    "id":                 doc_id,
                    "ts":                 ts,
                    "date":               date_human,
                    "session_id":         session_id,
                    "exchange_idx":       exchange_idx,
                    "user_msg":           user_msg,
                    "assistant_msg":      assistant_msg,
                    "summary":            summary,
                    "emotion":            emotion,
                    "importance":         float(importance),
                    "cause":              cause,
                    "traits_snapshot":    traits_snapshot,
                    "affective_snapshot": affective_snapshot,
                    "memory_version":     _MEMORY_VERSION,
                    "condition_tag":      _detect_condition_tag(),
                }
                # Anti-HARKing: versioning formula + componenti auditabili per Fase C.
                # llm_score/cog_score permettono calibrazione Bradley-Terry separata.
                if significance_version:
                    record["significance_version"] = str(significance_version)
                if significance_score is not None:
                    record["significance_score"] = float(significance_score)
                if llm_score is not None:
                    record["llm_score"] = float(llm_score)
                if cog_score is not None:
                    record["cog_score"] = float(cog_score)
                if episode_encoded is not None:
                    record["episode_encoded"] = bool(episode_encoded)
                if appraisal_snapshot:
                    record["appraisal_snapshot"] = appraisal_snapshot
                try:
                    self._append_archive_jsonl(record)
                    ok_jsonl = True
                except Exception as e:
                    self._log_error("archive_exchange:jsonl_append", e)

            except Exception as e:
                self._log_error("archive_exchange", e)

        return ok_chroma or ok_jsonl

    def _append_archive_jsonl(self, record: dict) -> None:
        """Append atomico al JSONL archivio. Crea dir se manca."""
        parent = os.path.dirname(_ARCHIVE_JSONL)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(_ARCHIVE_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._archive_append_counter += 1
        if self._archive_append_counter >= _ARCHIVE_BACKUP_EVERY:
            self._archive_append_counter = 0
            self._rotate_archive_backup_if_needed()

    def _rotate_archive_backup_if_needed(self) -> None:
        """
        Copia (non sposta) il JSONL in data/logs/archive_backups/ se supera la soglia.
        Il JSONL vivo NON viene mai troncato — è la fonte canonica.
        Il backup serve solo da sicurezza offline contro corruzione/perdita.
        """
        try:
            if not os.path.exists(_ARCHIVE_JSONL):
                return
            size_mb = os.path.getsize(_ARCHIVE_JSONL) / (1024 * 1024)
            if size_mb < _ARCHIVE_BACKUP_MB:
                return
            os.makedirs(_ARCHIVE_BACKUP_DIR, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(_ARCHIVE_BACKUP_DIR, f"episodic_archive_{stamp}.jsonl")
            shutil.copy2(_ARCHIVE_JSONL, backup_path)
            print(f"[VectorMemory] Backup archivio -> {backup_path} ({size_mb:.1f} MB)")
        except Exception as e:
            self._log_error("archive_rotate_backup", e)

    def search_archive(self, query: str, n: int = 3, max_distance: float = 0.45,
                        importance_boost: bool = True) -> list:
        """
        Memory v2.0 — ricerca semantica nell'archivio totale.
        Ritorna lista ordinata per rilevanza. Se importance_boost=True, gli episodi
        con importance alta salgono a parità di distance (piccolo correttivo).
        Soglia distance default 0.45 (più stretta di episodic 0.50 perché l'archivio
        contiene anche scambi a bassa salienza: serve precisione).
        """
        if not self._ok or not _ARCHIVE_ENABLED or self._archive is None:
            return []
        with self._lock:
            try:
                count = self._archive.count()
                if count == 0:
                    return []
                # Over-fetch per permettere re-ranking importance
                fetch_n = min(count, n * 2 if importance_boost else n)
                results = self._archive.query(
                    query_texts=[query],
                    n_results=fetch_n,
                    include=["documents", "metadatas", "distances"],
                )
                out = []
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    d = float(dist)
                    if d > max_distance:
                        continue
                    out.append({
                        "text":         doc,
                        "summary":      meta.get("summary", ""),
                        "date":         meta.get("date", ""),
                        "emotion":      meta.get("emotion", ""),
                        "importance":   float(meta.get("importance", 0)),
                        "user_preview": meta.get("user_preview", ""),
                        "eden_preview": meta.get("eden_preview", ""),
                        "distance":     round(d, 3),
                        "id":           meta.get("ts", ""),
                    })
                if importance_boost and len(out) > n:
                    # Score = distance - 0.03 * (importance/10)
                    # piccolo boost (max -0.03) per evitare che domini la distance
                    out.sort(key=lambda x: x["distance"] - 0.03 * (x["importance"] / 10.0))
                    out = out[:n]
                elif len(out) > n:
                    out = out[:n]
                return out
            except Exception as e:
                self._log_error("search_archive", e)
                return []

    def archive_stats(self) -> dict:
        """Sintesi archivio per UI/telemetria/endpoint."""
        if not self._ok or self._archive is None:
            return {"available": False, "enabled": _ARCHIVE_ENABLED, "memory_version": _MEMORY_VERSION}
        with self._lock:
            try:
                count = self._archive.count()
                jsonl_bytes = 0
                jsonl_lines = 0
                if os.path.exists(_ARCHIVE_JSONL):
                    jsonl_bytes = os.path.getsize(_ARCHIVE_JSONL)
                    # Conteggio righe efficiente
                    with open(_ARCHIVE_JSONL, "rb") as f:
                        jsonl_lines = sum(1 for _ in f)
                return {
                    "available":        True,
                    "enabled":          _ARCHIVE_ENABLED,
                    "memory_version":   _MEMORY_VERSION,
                    "chroma_count":     count,
                    "jsonl_path":       _ARCHIVE_JSONL,
                    "jsonl_bytes":      jsonl_bytes,
                    "jsonl_lines":      jsonl_lines,
                    "backup_dir":       _ARCHIVE_BACKUP_DIR,
                }
            except Exception as e:
                return {"available": True, "error": str(e), "memory_version": _MEMORY_VERSION}

    def _log_error(self, where: str, exc: Exception) -> None:
        """Instrada errori archivio verso log_utils se disponibile; fallback stdout."""
        msg = f"archivio v2.0 ({where}) fallito: {exc}"
        try:
            from core import log_utils
            log_utils.log_exception("VEC_MEM", f"archive.{where}", exc, severity="ERROR")
        except Exception:
            pass
        print(f"[VectorMemory] {msg}")

    # ─── Migrazione da memory.json ────────────────────────────────────────────

    def migrate_from_json(self, json_path: str = "memory.json") -> dict:
        """
        Migrazione automatica (eseguita una sola volta all'avvio se chroma_db/ non esiste).

        Legge episodic_memory e semantic_memory da memory.json e li inserisce
        nelle rispettive collection ChromaDB con embedding.

        memory.json NON viene modificato: rimane l'unica fonte per traits,
        sessions, relationship, internal_state (usati da memory.py).

        Ritorna: {"episodi_migrati": N, "fatti_migrati": M}
        """
        if not self._ok:
            return {"error": "ChromaDB non disponibile"}

        if not os.path.exists(json_path):
            return {"episodi_migrati": 0, "fatti_migrati": 0,
                    "nota": f"{json_path} non trovato — migrazione saltata"}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                dati = json.load(f)
        except Exception as e:
            return {"error": f"Impossibile leggere {json_path}: {e}"}

        episodi_migrati = 0
        fatti_migrati   = 0
        errori          = 0

        # ── Migra episodic_memory ────────────────────────────────────────────
        for i, ep in enumerate(dati.get("episodic_memory", [])):
            try:
                importance = float(ep.get("importance", 0))
                if importance < EPISODIC_THRESHOLD:
                    continue

                summary  = str(ep.get("summary", ""))
                emotion  = str(ep.get("emotion", ""))
                date_str = str(ep.get("date", datetime.now().strftime("%Y-%m-%d %H:%M")))

                # ID deterministico e idempotente basato su data + posizione
                safe_date = date_str.replace(" ", "_").replace(":", "-")
                doc_id    = f"ep_migr_{safe_date}_{i:04d}"
                em_id     = f"em_migr_{safe_date}_{i:04d}"

                # Evita duplicati: se già presente salta
                try:
                    existing = self._episodic.get(ids=[doc_id])
                    if existing["ids"]:
                        continue
                except Exception:
                    pass

                testo = f"Riassunto: {summary}"
                with self._lock:
                    self._episodic.add(
                        documents=[testo],
                        metadatas=[{
                            "date":            date_str,
                            "emotion":         emotion,
                            "importance":      importance,
                            "traits_snapshot": "{}",
                            "session_id":      0,
                            "summary":         summary[:500]
                        }],
                        ids=[doc_id]
                    )
                    self._emotional.add(
                        documents=[f"{emotion}: {summary}"],
                        metadatas=[{
                            "emotion_type": emotion,
                            "intensity":    importance,
                            "context":      summary[:500],
                            "date":         date_str
                        }],
                        ids=[em_id]
                    )
                episodi_migrati += 1

            except Exception as e:
                errori += 1
                print(f"[VectorMemory] Errore migrazione episodio {i}: {e}")

        # ── Migra semantic_memory ────────────────────────────────────────────
        for chiave, valore in dati.get("semantic_memory", {}).items():
            try:
                val_str = (
                    json.dumps(valore, ensure_ascii=False)
                    if not isinstance(valore, str)
                    else valore
                )
                self.upsert_fact(str(chiave), val_str)
                fatti_migrati += 1
            except Exception as e:
                errori += 1
                print(f"[VectorMemory] Errore migrazione fatto '{chiave}': {e}")

        risultato = {
            "episodi_migrati": episodi_migrati,
            "fatti_migrati":   fatti_migrati,
        }
        if errori:
            risultato["errori"] = errori
        return risultato
