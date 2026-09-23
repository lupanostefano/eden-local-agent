"""
eden_paths.py — punto unico di verità per i path del progetto.

Tutto il codice di Eden deve importare i path da qui, mai costruirli da
`os.path.dirname(__file__)` o path relativi alla CWD.

Struttura post-refactor 2026-04-26:

    eden/
    ├── eden_paths.py            ← questo file (resta in root)
    ├── core/                    ← server Flask, chat, decisione, voce, avatar
    ├── mechanisms/              ← memoria, grafo, sonno, stato interno
    ├── data/                    ← stato runtime
    │   ├── memory.json
    │   ├── chroma_db/
    │   ├── kuzu_db/
    │   ├── sessions/
    │   ├── configs/
    │   └── logs/
    ├── static/, index.html      ← UI
    ├── docs/, archive/          ← documentazione + storico
    └── avatar/, liveportrait/   ← assets multimediali
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------- Project root ----------
# `eden_paths.py` vive in root. PROJECT_ROOT è la directory che lo contiene.
PROJECT_ROOT: Path = Path(__file__).parent.resolve()


# ---------- Top-level directories ----------
CORE_DIR        = PROJECT_ROOT / "core"
MECHANISMS_DIR  = PROJECT_ROOT / "mechanisms"
DATA_DIR        = PROJECT_ROOT / "data"
UI_DIR          = PROJECT_ROOT / "ui"
DOCS_DIR        = PROJECT_ROOT / "docs"
ARCHIVE_DIR     = PROJECT_ROOT / "archive"
AVATAR_DIR      = PROJECT_ROOT / "avatar"
LIVEPORTRAIT_DIR = PROJECT_ROOT / "liveportrait"
DEBUG_DIR       = PROJECT_ROOT / "debug"


# ---------- Data subdirectories ----------
CHROMA_DIR        = DATA_DIR / "chroma_db"
KUZU_DIR          = DATA_DIR / "kuzu_db"          # Container directory per il file Kuzu
KUZU_FILE         = KUZU_DIR / "eden_graph.kuzu"  # Database Kuzu (single-file embedded)
SESSIONS_DIR      = DATA_DIR / "sessions"
CONFIGS_DIR       = DATA_DIR / "configs"
LOGS_DIR          = DATA_DIR / "logs"
ARCHIVE_BACKUPS_DIR = LOGS_DIR / "archive_backups"


# ---------- Runtime state files (data/) ----------
MEMORY_FILE             = DATA_DIR / "memory.json"
MEMORY_BACKUP_FILE      = DATA_DIR / "memory.json.bak"
AUTONOMY_LOG_FILE       = DATA_DIR / "autonomy_log.jsonl"
LIVEPORTRAIT_STATE_FILE = DATA_DIR / "liveportrait_state.json"


# ---------- Config files (data/configs/) ----------
PIONEER_CONFIG_FILE = CONFIGS_DIR / "pioneer_config.json"
MODEL_CONFIG_FILE   = CONFIGS_DIR / "model_config.json"


# ---------- Log di runtime (data/logs/) ----------
EPISODIC_ARCHIVE_FILE     = LOGS_DIR / "episodic_archive.jsonl"
HUMAN_JUDGE_LOG_FILE      = LOGS_DIR / "human_judge_log.jsonl"   # storico, sola lettura
SELF_REFLECTION_LOG_FILE  = LOGS_DIR / "self_reflection_log.jsonl"


# ---------- Assets ----------
SPEAKER_MP3_FILE = AVATAR_DIR / "voce_riferimento.MP3"
SPEAKER_WAV_FILE = AVATAR_DIR / "voce_riferimento.wav"
PORTRAIT_FILE    = AVATAR_DIR / "eden_portrait.png"


# ---------- Debug ----------
DEBUG_AUDIO_DIR = DEBUG_DIR / "audio"


# ---------- Helper ----------
def ensure_dirs() -> None:
    """Crea tutte le directory necessarie (idempotente).
    Da chiamare una volta all'avvio del processo."""
    for d in [
        CORE_DIR, MECHANISMS_DIR, DATA_DIR,
        CHROMA_DIR, KUZU_DIR, SESSIONS_DIR, CONFIGS_DIR, LOGS_DIR,
        ARCHIVE_BACKUPS_DIR, DEBUG_AUDIO_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------- Backwards-compat: stringhe ----------
# Alcuni moduli usano os.path.join e stringhe. Forniamo str() helpers.
def s(p: Path) -> str:
    return str(p)
