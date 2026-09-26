"""
registro.py — lettura e scrittura di data/eden.db (unica memoria di Eden 2).

I messaggi sono letterali e non si riscrivono mai. Uno "scambio" è un messaggio di Stefano con la risposta di Eden.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import eden_paths as P
from nucleo.testo import norm

FONTE = "eden2"

SCHEMA_IDENTITA = """
CREATE TABLE IF NOT EXISTS identita (
    id            INTEGER PRIMARY KEY,
    tipo          TEXT NOT NULL,                     -- 'chi_sono' | 'ricordo' (ciò che Eden sceglie di tenere per sempre)
    testo         TEXT NOT NULL,                     -- in prima persona, scritto da Eden
    messaggio_id  INTEGER REFERENCES messaggi(id),   -- da dove nasce
    motivo        TEXT,                              -- perché l'ha scritto (o chiuso)
    valido_da     TEXT NOT NULL,
    valido_fino   TEXT,                              -- NULL = ancora vero; le voci superate si chiudono, non si cancellano
    sostituisce   INTEGER REFERENCES identita(id)
)"""


# Peso emotivo e forza dei ricordi (stile amigdala/ippocampo): lo riempiono il sonno (step 6) e il richiamo (step 5)
SCHEMA_PESO = """
CREATE TABLE IF NOT EXISTS peso (
    messaggio_id     INTEGER PRIMARY KEY REFERENCES messaggi(id),
    emozione         TEXT,                           -- es. gioia, tristezza, rabbia, tenerezza, paura
    intensita        REAL,                           -- 0..1: quanto ha contato
    segno            REAL,                           -- -1 doloroso .. +1 bello
    richiami         INTEGER NOT NULL DEFAULT 0,     -- quante volte il ricordo è tornato utile
    ultimo_richiamo  TEXT,
    fonte            TEXT                            -- chi l'ha stimato (es. 'sonno 2026-10-01')
)"""


def connetti(sola_lettura: bool = False) -> sqlite3.Connection:
    if sola_lettura:
        con = sqlite3.connect(f"file:{P.EDEN_DB_FILE.as_posix()}?mode=ro", uri=True, timeout=30)
    else:
        con = sqlite3.connect(P.EDEN_DB_FILE, timeout=30)
    con.execute("PRAGMA busy_timeout = 30000")
    return con


def prepara() -> None:
    """Tabelle nuove del nucleo (idempotente)."""
    with connetti() as con:
        con.execute(SCHEMA_IDENTITA)
        con.execute(SCHEMA_PESO)


def adesso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def carica_scambi(con: sqlite3.Connection, dopo_id: int = 0) -> list[dict]:
    """Messaggi user/assistant in ordine, riuniti in scambi (domanda + risposta)."""
    righe = con.execute("SELECT id, ts, ruolo, testo, scambio FROM messaggi "
                        "WHERE ruolo IN ('user','assistant') AND id > ? ORDER BY id", (dopo_id,)).fetchall()
    scambi, i = [], 0
    while i < len(righe):
        mid, ts, ruolo, testo, scambio = righe[i]
        nxt = righe[i + 1] if i + 1 < len(righe) else None
        if ruolo == "user" and nxt and nxt[2] == "assistant" and (nxt[1] == ts or (scambio and nxt[4] == scambio)):
            scambi.append({"ids": [mid, nxt[0]], "ts": ts, "u": testo, "a": nxt[3]})
            i += 2
        elif ruolo == "user":
            scambi.append({"ids": [mid], "ts": ts, "u": testo, "a": None})
            i += 1
        else:
            scambi.append({"ids": [mid], "ts": ts, "u": None, "a": testo})
            i += 1
    return scambi


def messaggi_dopo(con: sqlite3.Connection, dopo_id: int) -> list[tuple[int, str, str, str]]:
    """(id, ts, ruolo, testo) dei messaggi user/assistant successivi a `dopo_id`."""
    return con.execute("SELECT id, ts, ruolo, testo FROM messaggi WHERE ruolo IN ('user','assistant') AND id > ? "
                       "ORDER BY id", (dopo_id,)).fetchall()


def ultimo_id(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COALESCE(MAX(id), 0) FROM messaggi").fetchone()[0]


def salva_messaggio(ruolo: str, testo: str, scambio: str, ts: str | None = None, meta: dict | None = None) -> int:
    with connetti() as con:
        cur = con.execute("INSERT INTO messaggi(ts, ruolo, testo, sessione, fonte, scambio, meta) VALUES (?,?,?,?,?,?,?)",
                          (ts or adesso(), ruolo, testo, None, FONTE, scambio,
                           json.dumps(meta, ensure_ascii=False) if meta else None))
        return cur.lastrowid


def fatti_validi(con: sqlite3.Connection, quando: str) -> list[dict]:
    righe = con.execute("SELECT soggetto, chiave, valore, riservato FROM fatti "
                        "WHERE (valido_da IS NULL OR valido_da <= ?) AND (valido_fino IS NULL OR valido_fino > ?) "
                        "ORDER BY id", (quando, quando)).fetchall()
    return [{"soggetto": s, "chiave": k, "valore": v, "riservato": bool(r)} for s, k, v, r in righe]


SINONIMI_DELICATI: dict[str, tuple[str, ...]] = {}   # copia pubblica: i sinonimi degli argomenti riservati restano privati


def parole_delicate(con: sqlite3.Connection) -> set[str]:
    """Le parole degli argomenti che Stefano non vuole che Eden nomini per prima (fatti `riservato`): il tema
    (es. un familiare) e i suoi nomi propri o sigle, senza accenti né maiuscole."""
    parole: set[str] = set()
    for chiave, valore in con.execute("SELECT chiave, valore FROM fatti WHERE riservato = 1"):
        tema = norm(chiave.split("_")[0])
        parole.add(tema)
        parole.update(SINONIMI_DELICATI.get(tema, ()))
        parole.update(norm(w) for w in valore.replace("(", " ").replace(")", " ").split() if w[:1].isupper() and w.isalpha())
    return parole


def identita_attiva(con: sqlite3.Connection) -> list[dict]:
    try:
        righe = con.execute("SELECT tipo, testo FROM identita WHERE valido_fino IS NULL ORDER BY id").fetchall()
    except sqlite3.OperationalError:  # tabella non ancora creata (db in sola lettura mai preparato)
        return []
    return [{"tipo": t, "testo": x} for t, x in righe]


def giorni(con: sqlite3.Connection, fino_a: str, prima_di_id: int) -> list[tuple[str, int, str, str]]:
    """(giorno, messaggi di Stefano, ora del primo messaggio, ora dell'ultimo) per ogni giorno con conversazione."""
    return con.execute("SELECT substr(ts,1,10) g, SUM(ruolo='user'), MIN(substr(ts,12,5)), MAX(substr(ts,12,5)) "
                       "FROM messaggi WHERE ruolo IN ('user','assistant') AND ts <= ? AND id < ? GROUP BY g ORDER BY g",
                       (fino_a, prima_di_id)).fetchall()


def ultimi_messaggi(con: sqlite3.Connection, n: int) -> list[dict]:
    """Gli ultimi `n` messaggi; le risposte di Eden portano l'esito delle fonti che citano (`citazioni`) e il voto
    di Stefano (`voto`: 'su' · 'giu' · None; `nota`: cosa non andava o come avrebbe dovuto rispondere)."""
    righe = con.execute("SELECT id, ts, ruolo, testo, meta FROM messaggi WHERE ruolo IN ('user','assistant') "
                        "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    voti = voti_di(con, [r[0] for r in righe])
    return [{"id": i, "ts": ts, "ruolo": r, "testo": t, "citazioni": json.loads(m).get("citazioni") if m else None,
             **voti.get(i, {"voto": None, "nota": None})} for i, ts, r, t, m in reversed(righe)]


# ---------------------------------------------------------------- voti di Stefano (tabella `giudizi`)
# Servono al carattere di Eden: 'su' = risposta come la voglio, 'giu' = da non rifare, 'correzione' = come avrebbe dovuto
# rispondere. Le coppie (risposta 'giu' + correzione) diventano il materiale di preferenza per la fase LoRA dopo lo step 6.

def salva_voto(con: sqlite3.Connection, messaggio_id: int, voto: str | None, nota: str | None = None) -> None:
    """Un voto per risposta (cambiarlo sostituisce il precedente; `voto=None` lo toglie) e una nota facoltativa."""
    if voto not in (None, "su", "giu"):
        raise ValueError("voto: su, giu o niente")
    con.execute("DELETE FROM giudizi WHERE messaggio_id = ? AND fonte = ? AND tipo IN ('su','giu')", (messaggio_id, FONTE))
    if voto:
        con.execute("INSERT INTO giudizi(messaggio_id, tipo, ts, fonte) VALUES (?,?,?,?)", (messaggio_id, voto, adesso(), FONTE))
    if nota is not None:
        con.execute("DELETE FROM giudizi WHERE messaggio_id = ? AND fonte = ? AND tipo = 'correzione'", (messaggio_id, FONTE))
        if nota.strip():
            con.execute("INSERT INTO giudizi(messaggio_id, tipo, nota, ts, fonte) VALUES (?,?,?,?,?)",
                        (messaggio_id, "correzione", nota.strip(), adesso(), FONTE))
    con.commit()


def voti_di(con: sqlite3.Connection, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    segni = ",".join("?" * len(ids))
    out: dict[int, dict] = {}
    for mid, tipo, nota in con.execute(f"SELECT messaggio_id, tipo, nota FROM giudizi WHERE fonte = ? AND tipo IN "
                                       f"('su','giu','correzione') AND messaggio_id IN ({segni}) ORDER BY id", (FONTE, *ids)):
        v = out.setdefault(mid, {"voto": None, "nota": None})
        if tipo == "correzione":
            v["nota"] = nota
        else:
            v["voto"] = tipo
    return out
