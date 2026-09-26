"""
ricerca.py — ricerca nei ricordi: parole (FTS5) + significato (Qwen3-Embedding-0.6B su processore), unite con RRF.
Il "richiamo" (metodo predefinito di Eden) riordina il risultato con il peso dei ricordi (peso.py): emozione,
recenza, rinforzo; se la domanda chiede un ricordo per come è stato vissuto, anche l'emozione propone candidati.

Misure: docs/MISURE_2026-09.md e docs/ESAME_MEMORIA_2026-09.md (ibrido 42/55 prove nei primi 12 scambi).
Il servizio embedding è un llama-server su CPU (porta 8093) che il nucleo avvia da solo se non c'è.
"""
from __future__ import annotations

import atexit
import os
import re
import sqlite3
import subprocess
import time
from collections import defaultdict

import numpy as np
import requests

import eden_paths as P
from nucleo import peso, registro
from nucleo.testo import norm  # noqa: F401 (la usano anche altri moduli, da qui)

EMB_PORTA = 8093
EMB_URL = f"http://127.0.0.1:{EMB_PORTA}"
EMB_MODELLO = "Qwen3-Embedding-0.6B-Q8_0"
EMB_ISTRUZIONE = "Instruct: Given a question about past conversations, retrieve the passages that answer it\nQuery: "
MAX_CARATTERI_EMB = 8000
K_SCAMBI = 12
RRF_K = 60

STOP = set("""che per con come del della delle dei degli nel nella nelle sono hai ho mi ti io tu quando quale quali cosa chi
dove quanti quanto una uno gli le lo la il un in di da al ai alla anche ma se ci si mia mio miei tue tuo tua non piu poi
ricordi ricordo detto dissi dette hanno abbiamo stato stata era erano aveva quello quella quelli fatto fare fai faccio
parlato parli parlami dimmi dirmi parliamo giorno giorni volta volte prima dopo""".split())


# ---------------------------------------------------------------- servizio embedding

_processo: subprocess.Popen | None = None


def servizio_attivo() -> bool:
    try:
        return requests.get(f"{EMB_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def avvia_servizio(attesa_sec: int = 120) -> None:
    """Avvia il llama-server degli embedding su CPU (la 5060 Ti resta alla voce), se non è già attivo."""
    global _processo
    if servizio_attivo():
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    _processo = subprocess.Popen(
        [str(P.LLAMA_SERVER_EXE), "-m", str(P.EMBEDDING_MODEL_FILE), "--embeddings", "--pooling", "last",
         "-ngl", "0", "-c", "4096", "-np", "1", "--host", "127.0.0.1", "--port", str(EMB_PORTA)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    atexit.register(ferma_servizio)
    t0 = time.time()
    while time.time() - t0 < attesa_sec:
        if servizio_attivo():
            return
        if _processo.poll() is not None:
            raise RuntimeError(f"servizio embedding terminato subito (codice {_processo.returncode})")
        time.sleep(1)
    raise RuntimeError("servizio embedding non pronto")


def ferma_servizio() -> None:
    if _processo and _processo.poll() is None:
        _processo.terminate()


def embedding(testi: list[str]) -> np.ndarray:
    r = requests.post(f"{EMB_URL}/v1/embeddings", json={"input": testi, "model": "x"}, timeout=600)
    r.raise_for_status()
    v = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def indicizza_mancanti(con: sqlite3.Connection, lotto: int = 16) -> int:
    """Calcola gli embedding dei messaggi che non li hanno ancora. Ritorna quanti ne ha fatti."""
    da_fare = con.execute(
        "SELECT m.id, m.testo FROM messaggi m LEFT JOIN embedding e ON e.messaggio_id = m.id AND e.modello = ? "
        "WHERE m.ruolo IN ('user','assistant') AND e.messaggio_id IS NULL AND trim(m.testo) != '' ORDER BY m.id",
        (EMB_MODELLO,)).fetchall()
    for k in range(0, len(da_fare), lotto):
        parte = da_fare[k:k + lotto]
        v = embedding([t[:MAX_CARATTERI_EMB] for _, t in parte])
        con.executemany("INSERT OR REPLACE INTO embedding(messaggio_id, modello, vettore) VALUES (?,?,?)",
                        [(i, EMB_MODELLO, v[j].tobytes()) for j, (i, _) in enumerate(parte)])
        con.commit()
    return len(da_fare)


# ---------------------------------------------------------------- ricerca

class Ricerca:
    """Tiene in memoria gli scambi e i vettori; `aggiorna()` aggiunge solo ciò che è nuovo."""

    def __init__(self):
        self.scambi: list[dict] = []
        self.msg2sc: dict[int, int] = {}
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, 0), dtype=np.float32)
        self.peso: dict[int, dict] = {}     # peso di ogni scambio, per numero del suo primo messaggio
        self.delicati: set[int] = set()     # indici degli scambi che toccano argomenti riservati (fatti `riservato`)
        self._parole_delicate: re.Pattern | None = None

    def aggiorna(self, con: sqlite3.Connection) -> None:
        self.scambi = registro.carica_scambi(con)
        self.msg2sc = {mid: i for i, s in enumerate(self.scambi) for mid in s["ids"]}
        self.peso = peso.carica(con)
        parole = registro.parole_delicate(con)
        self._parole_delicate = re.compile(r"\b(" + "|".join(map(re.escape, sorted(parole))) + r")\b") if parole else None
        self.delicati = {i for i, s in enumerate(self.scambi) if self._tocca_delicato(f"{s['u'] or ''} {s['a'] or ''}")}
        dopo = int(self.ids.max()) if len(self.ids) else 0
        righe = con.execute("SELECT messaggio_id, vettore FROM embedding WHERE modello=? AND messaggio_id > ? "
                            "ORDER BY messaggio_id", (EMB_MODELLO, dopo)).fetchall()
        if righe:
            nuovi = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in righe])
            self.mat = nuovi if not len(self.ids) else np.vstack([self.mat, nuovi])
            self.ids = np.concatenate([self.ids, np.array([r[0] for r in righe], dtype=np.int64)])

    def _fts(self, con: sqlite3.Connection, domanda: str, limite: int, dal: str | None, al: str | None,
             n: int = 30) -> list[int]:
        parole = [w for w in re.findall(r"\w{3,}", norm(domanda)) if w not in STOP]
        if not parole:
            return []
        q = " OR ".join(f'"{w}"' for w in dict.fromkeys(parole))
        righe = con.execute("SELECT messaggi_fts.rowid FROM messaggi_fts JOIN messaggi m ON m.id = messaggi_fts.rowid "
                            "WHERE messaggi_fts MATCH ? AND substr(m.ts, 1, 10) BETWEEN ? AND ? "
                            "ORDER BY messaggi_fts.rank LIMIT ?", (q, dal or "0000-00-00", al or "9999-99-99", n * 3)).fetchall()
        return [r[0] for r in righe if r[0] in self.msg2sc and r[0] < limite][:n]

    def _emb(self, domanda: str, limite: int, dal: str | None, al: str | None, n: int = 30) -> list[int]:
        if not len(self.ids):
            return []
        q = embedding([EMB_ISTRUZIONE + domanda])[0]
        ordine = np.argsort(-(self.mat @ q))
        ammessi = (int(self.ids[j]) for j in ordine if self.ids[j] < limite and int(self.ids[j]) in self.msg2sc)
        return [m for m in ammessi if self._nelle_date(m, dal, al)][:n]

    def _nelle_date(self, mid: int, dal: str | None, al: str | None) -> bool:
        giorno = self.scambi[self.msg2sc[mid]]["ts"][:10]
        return (not dal or giorno >= dal) and (not al or giorno <= al)

    def _tocca_delicato(self, testo: str) -> bool:
        return bool(self._parole_delicate and self._parole_delicate.search(norm(testo)))

    def _canale_emotivo(self, richiesta: tuple, limite: int, dal: str | None, al: str | None, escludi: set[int],
                        n: int = 20) -> list[int]:
        """Gli scambi che hanno contato di più, per l'emozione (e il segno) che la domanda cerca."""
        _, emozione, segno = richiesta
        candidati = []
        for mid, p in self.peso.items():
            if p["intensita"] < peso.SOGLIA_CANALE or mid >= limite or mid not in self.msg2sc:
                continue
            if self.msg2sc[mid] in escludi:
                continue
            if (segno and p["segno"] * segno < 0) or not self._nelle_date(mid, dal, al):
                continue
            candidati.append((p["intensita"] * (1.5 if emozione and p["emozione"] == emozione else 1.0), mid))
        return [mid for _, mid in sorted(candidati, reverse=True)[:n]]

    def cerca(self, con: sqlite3.Connection, domanda: str, k: int = K_SCAMBI, metodo: str = "ibrido",
              prima_di_id: int | None = None, dal: str | None = None, al: str | None = None,
              adesso: str | None = None) -> list[int]:
        """Indici (in self.scambi) dei k scambi più pertinenti, dal migliore. `prima_di_id` esclude il messaggio
        corrente; `dal`/`al` (AAAA-MM-GG, inclusi) restringono ai giorni indicati. Metodi: fts · emb · ibrido (solo
        somiglianza) · richiamo (ibrido + peso dei ricordi, con `adesso` per la recenza)."""
        limite = prima_di_id if prima_di_id is not None else 1 << 62
        richiesta = peso.richiesta_affettiva(domanda)
        # ciò che Stefano ha chiesto di non nominare (fatti `riservato`) non riemerge da solo per la sua carica emotiva:
        # solo se la domanda stessa ne parla
        riservati = set() if self._tocca_delicato(domanda) else self.delicati
        liste = {"fts": lambda: [self._fts(con, domanda, limite, dal, al)],
                 "emb": lambda: [self._emb(domanda, limite, dal, al)],
                 "ibrido": lambda: [self._fts(con, domanda, limite, dal, al), self._emb(domanda, limite, dal, al)],
                 "richiamo": lambda: [self._fts(con, domanda, limite, dal, al), self._emb(domanda, limite, dal, al)] +
                                     ([self._canale_emotivo(richiesta, limite, dal, al, riservati)] if richiesta[0] else [])}[metodo]()
        punti: dict[int, float] = defaultdict(float)
        for lista in liste:
            for rango, mid in enumerate(lista):
                punti[self.msg2sc[mid]] += 1 / (RRF_K + rango)
        if metodo == "richiamo":
            adesso = adesso or registro.adesso()
            for i in punti:
                s = self.scambi[i]
                p = self.peso.get(s["ids"][0])
                if p and i in riservati:
                    p = {**p, "intensita": 0.0}
                punti[i] *= peso.moltiplicatore(p, s["ts"], adesso, richiesta)
        return sorted(punti, key=lambda i: -punti[i])[:k]
