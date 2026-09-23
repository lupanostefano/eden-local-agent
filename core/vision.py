# vision.py — Phase 1.6/1.7 LivePortrait subprocess manager
#
# Architettura:
#   vision.py (manager)  ←→  liveportrait/idle_loop.py   (video-driven, Phase 1.7)
#                        ←→  liveportrait/run_eden.py    (procedurale, Phase 1.6 fallback)
#
# Selezione automatica del worker:
#   1. Se avatar/driving_loop.mp4 esiste → prova idle_loop.py (video-driven)
#      Se idle_loop.py esce con codice 2 (video non trovato) → fallback procedurale
#   2. Se driving_loop.mp4 mancante → usa run_eden.py direttamente
#
# Comunicazione:
#   traits → liveportrait_state.json    (scritto dal manager, letto dal worker)
#   frame  → stdout binario             (scritto dal worker, letto dal manager)
#   Protocollo stdout: [4 byte big-endian uint32 = lunghezza] [N byte JPEG]
#
# Auto-restart: se il worker termina inaspettatamente, il reader thread lo
#   riavvia automaticamente dopo 5 secondi (max 5 tentativi consecutivi).
#
# Fallback Three.js: se nessun worker può avviarsi, /api/face_stream ritorna 503
#   e il frontend Three.js rimane attivo silenziosamente.

import os
import sys
import json
import struct
import queue
import threading
import subprocess
import time
import logging
from pathlib import Path

logger = logging.getLogger('eden.vision')

BASE_DIR        = Path(__file__).parent
PORTRAIT_PATH   = BASE_DIR / 'avatar' / 'eden_portrait.png'
DRIVING_PATH    = BASE_DIR / 'avatar' / 'driving_loop.mp4'
WORKER_IDLE     = BASE_DIR / 'liveportrait' / 'idle_loop.py'
WORKER_PROC     = BASE_DIR / 'liveportrait' / 'run_eden.py'
STATE_PATH      = BASE_DIR / 'liveportrait_state.json'
WEIGHTS_CHECK   = BASE_DIR / 'liveportrait' / 'pretrained_weights' / 'liveportrait' / 'base_models' / 'appearance_feature_extractor.pth'

_DEFAULT_TRAITS     = {'curiosity': 5, 'trust': 5, 'cynicism': 5, 'warmth': 5, 'fear': 5}
_MAX_RESTARTS       = 5      # tentativi auto-restart prima di rinunciare
_RESTART_DELAY_S    = 5.0   # secondi di attesa tra un restart e il successivo
_FALLBACK_EXIT_CODE = 2      # idle_loop.py esce con 2 se driving video mancante

# ─── LivePortraitStreamer ─────────────────────────────────────────────────────

class LivePortraitStreamer:
    """
    Gestisce il subprocess LivePortrait e fornisce frame JPEG via get_frame().

    Selezione worker (in ordine di preferenza):
        1. idle_loop.py    — video-driven (richiede driving_loop.mp4)
        2. run_eden.py     — animazione procedurale (sempre disponibile)

    Auto-restart: se il worker crasha, viene riavviato automaticamente
    fino a _MAX_RESTARTS volte consecutive.

    Ciclo di vita:
        start()                → avvia il worker ottimale
        update_traits(traits)  → aggiorna stato emotivo (letto dal worker)
        get_frame(timeout)     → ultimo frame JPEG, o None se timeout
        stop()                 → ferma il worker
    """

    def __init__(self):
        self._proc:      subprocess.Popen | None = None
        self._frame_q:   queue.Queue             = queue.Queue(maxsize=6)
        self._running:   bool                    = False
        self._lock:      threading.Lock          = threading.Lock()
        self._worker:    Path | None             = None    # worker attualmente in uso
        self._restarts:  int                     = 0

    # ── Stato ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True se i prerequisiti minimi sono soddisfatti (portrait + pesi + almeno un worker)."""
        return (
            PORTRAIT_PATH.exists() and
            WEIGHTS_CHECK.exists() and
            (WORKER_IDLE.exists() or WORKER_PROC.exists())
        )

    def is_running(self) -> bool:
        return self._running and (self._proc is not None) and (self._proc.poll() is None)

    def status(self) -> dict:
        worker_name = self._worker.name if self._worker else 'nessuno'
        mode = (
            'video-driven (idle_loop)'  if self._worker == WORKER_IDLE else
            'procedurale (run_eden)'    if self._worker == WORKER_PROC else
            'offline'
        )
        return {
            'available':       self.is_available(),
            'running':         self.is_running(),
            'mode':            mode,
            'worker':          worker_name,
            'portrait_found':  PORTRAIT_PATH.exists(),
            'weights_found':   WEIGHTS_CHECK.exists(),
            'driving_found':   DRIVING_PATH.exists(),
            'restarts':        self._restarts,
            'setup_guide': (
                'Prerequisiti mancanti. '
                'Genera eden_portrait.png e scarica i pesi LivePortrait in '
                'liveportrait/pretrained_weights/. '
                'Per la modalità video-driven, aggiungi avatar/driving_loop.mp4.'
                if not self.is_available() else ''
            ),
        }

    # ── Selezione worker ───────────────────────────────────────────────────────

    def _select_worker(self) -> Path | None:
        """
        Restituisce il worker ottimale disponibile.
        Preferisce idle_loop.py (video-driven) se driving_loop.mp4 esiste.
        """
        if WORKER_IDLE.exists() and DRIVING_PATH.exists():
            return WORKER_IDLE
        if WORKER_PROC.exists():
            return WORKER_PROC
        return None

    def _fallback_worker(self, current: Path) -> Path | None:
        """
        Fallback: se idle_loop.py non riesce (driving mancante), passa a run_eden.py.
        """
        if current == WORKER_IDLE and WORKER_PROC.exists():
            logger.info('Fallback da idle_loop.py a run_eden.py.')
            return WORKER_PROC
        return None

    # ── Controllo subprocess ───────────────────────────────────────────────────

    def start(self) -> bool:
        """Avvia il worker ottimale. Ritorna True se avviato con successo."""
        with self._lock:
            if self.is_running():
                return True
            if not self.is_available():
                logger.warning(
                    'LivePortrait non disponibile — mancano: %s',
                    ', '.join(filter(None, [
                        'eden_portrait.png' if not PORTRAIT_PATH.exists() else '',
                        'pretrained_weights' if not WEIGHTS_CHECK.exists() else '',
                        'worker .py' if not WORKER_IDLE.exists() and not WORKER_PROC.exists() else '',
                    ]))
                )
                return False

            worker = self._select_worker()
            if worker is None:
                logger.error('Nessun worker trovato in liveportrait/.')
                return False

            return self._launch(worker)

    def _launch(self, worker: Path) -> bool:
        """Avvia il subprocess specificato. Chiamato con _lock acquisito."""
        self._write_state(_DEFAULT_TRAITS)

        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(worker)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BASE_DIR / 'liveportrait'),
                close_fds=True,
            )
            self._running = True
            self._worker  = worker

            threading.Thread(
                target=self._read_frames,
                name='liveportrait-reader',
                daemon=True,
            ).start()

            threading.Thread(
                target=self._log_stderr,
                name='liveportrait-stderr',
                daemon=True,
            ).start()

            logger.info('Worker avviato: %s (PID %d)', worker.name, self._proc.pid)
            return True

        except Exception as exc:
            logger.error('Impossibile avviare %s: %s', worker.name, exc)
            self._proc    = None
            self._running = False
            self._worker  = None
            return False

    def stop(self):
        """Ferma il subprocess in modo pulito."""
        with self._lock:
            self._restarts = 0   # azzera contatore restart
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                except Exception:
                    pass
                finally:
                    self._proc = None
            self._running = False
            self._worker  = None
            logger.info('LivePortrait fermato.')

    # ── API pubblica ───────────────────────────────────────────────────────────

    def update_traits(self, traits: dict, affective: dict | None = None):
        """Aggiorna tratti emotivi e stato affettivo (il worker li legge ad ogni frame)."""
        if self._running:
            self._write_state(traits, affective or {})

    def get_frame(self, timeout: float = 0.12) -> bytes | None:
        """Restituisce l'ultimo frame JPEG disponibile, o None se timeout."""
        try:
            return self._frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── I/O interno ───────────────────────────────────────────────────────────

    def _write_state(self, traits: dict, affective: dict | None = None):
        """Scrittura atomica del file di stato (traits + affective_state)."""
        try:
            payload = {**traits, **(affective or {})}
            tmp = str(STATE_PATH) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            os.replace(tmp, str(STATE_PATH))
        except Exception as exc:
            logger.debug('Errore scrittura state: %s', exc)

    def _read_frames(self):
        """
        Thread: legge frame dal subprocess e gestisce auto-restart.

        Casi di uscita dal subprocess:
          - Exit code 0: interruzione pulita (stop()) → non riavviare
          - Exit code 1: errore fatale → riavvia (entro _MAX_RESTARTS)
          - Exit code 2: driving mancante (idle_loop.py) → fallback a run_eden.py
        """
        proc           = self._proc
        current_worker = self._worker

        try:
            while proc and proc.poll() is None:
                # ── Leggi header (4 byte big-endian) ──────────────────────
                header = b''
                while len(header) < 4:
                    chunk = proc.stdout.read(4 - len(header))
                    if not chunk:
                        return   # pipe chiusa
                    header += chunk

                length = struct.unpack('>I', header)[0]
                if length == 0 or length > 15_000_000:
                    logger.error('Frame length anomala: %d byte — stop.', length)
                    return

                # ── Leggi payload JPEG ─────────────────────────────────────
                payload = b''
                while len(payload) < length:
                    chunk = proc.stdout.read(length - len(payload))
                    if not chunk:
                        return
                    payload += chunk

                # ── Pubblica frame (droppa il vecchio se la queue è piena) ─
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_q.put_nowait(payload)
                self._restarts = 0   # frame ricevuto → processo sano → azzera

        except Exception as exc:
            logger.error('Reader thread error: %s', exc)
        finally:
            exit_code = proc.poll() if proc else -1
            self._running = False
            logger.info('Worker %s terminato (exit code %s).', current_worker.name if current_worker else '?', exit_code)

        # ── Gestione post-uscita ───────────────────────────────────────────
        # 0: stop() volontario — non riavviare
        if exit_code == 0:
            return

        # 2: driving video mancante — fallback a run_eden.py
        if exit_code == _FALLBACK_EXIT_CODE and current_worker is not None:
            fallback = self._fallback_worker(current_worker)
            if fallback:
                logger.info('Avvio fallback worker: %s', fallback.name)
                with self._lock:
                    self._proc    = None
                    self._running = False
                    self._worker  = None
                    self._launch(fallback)
                return

        # Qualsiasi altro exit code (1, -1, …) → auto-restart
        if self._restarts < _MAX_RESTARTS:
            self._restarts += 1
            logger.warning(
                'Worker terminato inaspettatamente. Restart %d/%d tra %ds...',
                self._restarts, _MAX_RESTARTS, int(_RESTART_DELAY_S)
            )
            time.sleep(_RESTART_DELAY_S)
            # Riprova con lo stesso worker (o il migliore disponibile)
            next_worker = self._select_worker()
            if next_worker:
                with self._lock:
                    self._proc    = None
                    self._running = False
                    self._worker  = None
                    self._launch(next_worker)
            else:
                logger.error('Nessun worker disponibile per il restart.')
        else:
            logger.error(
                'Raggiunto il limite di %d restart. LivePortrait disabilitato.',
                _MAX_RESTARTS
            )

    def _log_stderr(self):
        """Thread: logga stderr del worker per diagnostica."""
        try:
            for raw in self._proc.stderr:
                line = raw.decode('utf-8', errors='replace').rstrip()
                logger.debug('[LP-worker] %s', line)
        except Exception:
            pass


# ─── Singleton (usato da agent.py) ───────────────────────────────────────────

_streamer: LivePortraitStreamer | None = None


def get_streamer() -> LivePortraitStreamer:
    global _streamer
    if _streamer is None:
        _streamer = LivePortraitStreamer()
    return _streamer
