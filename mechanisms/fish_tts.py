"""
Fish-Speech 1.5 TTS adapter per Eden.

Architettura: Fish-Speech gira come server HTTP separato (porta 7861).
Eden chiama /v1/tts via HTTP — nessun conflitto di dipendenze con XTTS/torch.

Avvio server: tools/avvia_fishspeech.bat  (o automatico via Eden launcher)
"""

import io
import os
import threading
import time
import subprocess
import sys

import requests

_FS_BASE_URL   = "http://127.0.0.1:7861"
_FS_TIMEOUT    = 60        # secondi per generazione audio
_FS_READY      = False
_FS_LOCK       = threading.Lock()
_FS_PROC       = None      # processo server fish-speech

_SPEAKER_WAV   = None      # impostato da _init_speaker()
_SPEAKER_BYTES = None      # bytes del WAV reference, caricati una volta
# Trascrizione esatta del clip reference (da sostituire con quella del proprio campione).
# Fornirla aumenta drasticamente la somiglianza vocale (Fish-Speech usa
# il testo per allineare prosody/phonemi al campione audio).
_SPEAKER_TEXT  = "Trascrizione esatta del proprio clip di riferimento."

_FS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plugins", "fish-speech"
)
_FS_CHECKPOINT    = os.path.join(_FS_ROOT, "checkpoints", "fish-speech-1.5")
_FS_DECODER_PTH   = os.path.join(_FS_CHECKPOINT, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
_FS_DECODER_CFG   = "firefly_gan_vq"


# ── Avvio server ───────────────────────────────────────────────────────────────

def _avvia_server() -> bool:
    """Avvia il server Fish-Speech come sottoprocesso. Ritorna True se già up o avviato."""
    global _FS_PROC, _FS_READY

    if _is_server_up():
        _FS_READY = True
        return True

    print("[FishTTS] Avvio server Fish-Speech 1.5...")
    try:
        _FS_PROC = subprocess.Popen(
            [
                sys.executable, "tools/api_server.py",
                "--listen", "127.0.0.1:7861",
                "--llama-checkpoint-path",  _FS_CHECKPOINT,
                "--decoder-checkpoint-path", _FS_DECODER_PTH,
                "--decoder-config-name",    _FS_DECODER_CFG,
            ],
            cwd=_FS_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Attendi fino a 90 secondi che il server risponda
        for _ in range(45):
            time.sleep(2)
            if _is_server_up():
                _FS_READY = True
                print("[FishTTS] Server pronto.")
                return True
        print("[FishTTS] TIMEOUT: server non risponde dopo 90s.")
        return False
    except Exception as e:
        print(f"[FishTTS] Errore avvio server: {e}")
        return False


def _is_server_up() -> bool:
    try:
        r = requests.get(f"{_FS_BASE_URL}/docs", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def _ensure_server():
    """Thread-safe: garantisce che il server sia up prima di chiamare TTS."""
    global _FS_READY
    if _FS_READY:
        return True
    with _FS_LOCK:
        if _FS_READY:
            return True
        return _avvia_server()


# ── Reference speaker ─────────────────────────────────────────────────────────

def init_speaker(wav_path: str):
    """Carica il WAV del reference speaker (voce di riferimento)."""
    global _SPEAKER_WAV, _SPEAKER_BYTES
    _SPEAKER_WAV = wav_path
    if os.path.exists(wav_path):
        with open(wav_path, "rb") as f:
            _SPEAKER_BYTES = f.read()
        print(f"[FishTTS] Reference speaker caricato: {wav_path} ({len(_SPEAKER_BYTES)//1024}KB)")
    else:
        print(f"[FishTTS] ATTENZIONE: reference speaker non trovato: {wav_path}")


# ── Sintesi ───────────────────────────────────────────────────────────────────

def sintetizza(testo: str, speed: float = 1.0) -> bytes:
    """
    Sintetizza testo → WAV bytes a 44100 Hz via Fish-Speech 1.5.
    Usa il reference speaker (voce di riferimento) per voice cloning zero-shot.
    Raises RuntimeError se il server non è disponibile.
    """
    if not _ensure_server():
        raise RuntimeError("Fish-Speech server non disponibile.")

    import base64
    import ormsgpack

    # Costruisce la request msgpack (formato nativo Fish-Speech)
    payload = {
        "text": testo.strip(),
        "format": "wav",
        "streaming": False,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "temperature": 0.85,
        "references": [],
    }

    if _SPEAKER_BYTES:
        payload["references"] = [
            {
                "audio": _SPEAKER_BYTES,
                "text": _SPEAKER_TEXT,
            }
        ]

    try:
        resp = requests.post(
            f"{_FS_BASE_URL}/v1/tts",
            data=ormsgpack.packb(payload, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            headers={"Content-Type": "application/msgpack"},
            timeout=_FS_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Fish-Speech HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.content   # bytes WAV
    except requests.Timeout:
        raise RuntimeError("Fish-Speech timeout nella sintesi.")
    except Exception as e:
        raise RuntimeError(f"Fish-Speech errore: {e}")
