# avatar_manager.py — Phase 1.7 STRATO 1: Eden's animated avatar idle loop
#
# Workflow:
#   1. start_async() → thread daemon:
#         a. Carica eden_portrait.png come frame iniziale (fallback immediato)
#         b. Esegue inference.py se il video di output manca o è obsoleto
#         c. Legge il video di output in loop infinito
#         d. Avvia mini-HTTP su porta 5001 che serve /stream come MJPEG
#   2. stop()           → chiude il server HTTP, interrompe il loop
#   3. ping()           → segnala attività (resetta il timer di inattività)
#   4. Auto-stop        → dopo INACTIVITY_MIN minuti senza client → stop()
#   5. Restart          → start_async() è no-op se già in esecuzione
#
# Fallback in agent.py: se porta 5001 non risponde entro 3s → mostra
# eden_portrait.png statica senza errori.
#
# Output inference.py:
#   avatar/output/eden_portrait--driving_loop.mp4

import os
import sys
import time
import threading
import subprocess
import logging
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

try:
    import cv2
    import numpy as np
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

logger = logging.getLogger('eden.avatar')

# ─── Percorsi e costanti ──────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
PORTRAIT      = BASE_DIR / 'avatar' / 'eden_portrait.png'
DRIVING       = BASE_DIR / 'avatar' / 'driving_loop.mp4'
OUTPUT_DIR    = BASE_DIR / 'avatar' / 'output'
OUTPUT_VIDEO  = OUTPUT_DIR / 'eden_portrait--driving_loop.mp4'
INFERENCE_PY  = BASE_DIR / 'liveportrait' / 'inference.py'

MJPEG_PORT      = 5001
FPS_SERVE       = 25          # frame/s in uscita dal MJPEG server
INACTIVITY_MIN  = 10          # minuti di silenzio prima dello stop automatico
JPEG_QUALITY    = 82          # qualità JPEG per i frame (0-100)

# ─── Stato condiviso del frame corrente ───────────────────────────────────────
# Usato sia dal loop video che dal server MJPEG (thread diversi).

_frame_lock    = threading.Lock()
_current_jpg:  bytes | None = None   # ultimo frame JPEG
_last_activity: float       = 0.0    # time.time() dell'ultimo client attivo

def _set_frame(jpg: bytes) -> None:
    global _current_jpg
    with _frame_lock:
        _current_jpg = jpg

def get_frame() -> bytes | None:
    """Ritorna l'ultimo frame JPEG disponibile (usato anche da agent.py)."""
    with _frame_lock:
        return _current_jpg

# ─── MJPEG HTTP handler ───────────────────────────────────────────────────────

class _MJPEGHandler(BaseHTTPRequestHandler):
    """Handler HTTP minimale: serve /stream come MJPEG, /frame come JPEG singolo."""

    def do_GET(self):
        global _last_activity
        if self.path == '/stream':
            _last_activity = time.time()
            self._serve_mjpeg()
        elif self.path == '/frame':
            self._serve_single_frame()
        else:
            self.send_error(404)

    def _serve_mjpeg(self):
        global _last_activity
        frame_dt = 1.0 / FPS_SERVE
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            while True:
                t0 = time.perf_counter()
                _last_activity = time.time()

                jpg = get_frame()
                if jpg:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return   # client disconnesso

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, frame_dt - elapsed))
        except Exception:
            pass

    def _serve_single_frame(self):
        jpg = get_frame()
        if jpg is None:
            self.send_error(503, 'Nessun frame disponibile.')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(jpg)))
        self.end_headers()
        self.wfile.write(jpg)

    def log_message(self, *args):
        pass   # silenzia i log HTTP (troppo verboso per un MJPEG continuo)

# ─── AvatarManager ────────────────────────────────────────────────────────────

class AvatarManager:
    """
    Gestisce il ciclo di vita dell'avatar animato di Eden.

    Uso tipico (in agent.py):
        mgr = AvatarManager()
        mgr.start_async()          # all'avvio del server
        mgr.ping()                 # ad ogni /api/chat (impedisce lo stop automatico)
        mgr.start_async()          # al prossimo messaggio (restart no-op se già attivo)
    """

    def __init__(self):
        self._running      = False
        self._lock         = threading.Lock()
        self._server:      ThreadingHTTPServer | None = None
        self._stop_event   = threading.Event()
        self._ready_event  = threading.Event()   # True quando il server è up

    # ── API pubblica ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True se tutti i prerequisiti sono presenti."""
        return (
            _CV2_OK and
            PORTRAIT.exists() and
            DRIVING.exists() and
            INFERENCE_PY.exists()
        )

    def is_running(self) -> bool:
        return self._running

    def ping(self):
        """Segnala attività: resetta il timer di inattività."""
        global _last_activity
        _last_activity = time.time()

    def start_async(self) -> None:
        """
        Avvia il manager in un thread daemon. No-op se già in esecuzione.
        Ritorna immediatamente (non blocca).
        """
        with self._lock:
            if self._running:
                return
            if not self.is_available():
                logger.warning(
                    'Avatar non disponibile — mancano: %s',
                    ', '.join(filter(None, [
                        'cv2/numpy' if not _CV2_OK else '',
                        'eden_portrait.png' if not PORTRAIT.exists() else '',
                        'driving_loop.mp4'  if not DRIVING.exists() else '',
                        'inference.py'      if not INFERENCE_PY.exists() else '',
                    ]))
                )
                return
            self._running    = True
            self._stop_event.clear()
            self._ready_event.clear()

        threading.Thread(target=self._run, daemon=True, name='avatar-manager').start()

    def stop(self) -> None:
        """Ferma il server HTTP e il loop video."""
        self._stop_event.set()
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None
        logger.info('Avatar manager fermato.')

    def status(self) -> dict:
        return {
            'available':      self.is_available(),
            'running':        self._running,
            'video_cached':   OUTPUT_VIDEO.exists(),
            'portrait_found': PORTRAIT.exists(),
            'driving_found':  DRIVING.exists(),
            'mjpeg_port':     MJPEG_PORT,
        }

    def wait_ready(self, timeout: float = 3.0) -> bool:
        """Aspetta che il server MJPEG sia in ascolto su porta 5001."""
        return self._ready_event.wait(timeout)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Thread principale: carica portrait → inference → loop video → HTTP server."""
        global _last_activity
        _last_activity = time.time()

        # ── Passo 1: portrait come frame iniziale (fallback immediato) ─────────
        self._load_portrait()

        # ── Passo 2: avvia HTTP server su porta 5001 ───────────────────────────
        try:
            self._server = ThreadingHTTPServer(('127.0.0.1', MJPEG_PORT), _MJPEGHandler)
        except OSError as exc:
            logger.error('Porta %d già in uso o non disponibile: %s', MJPEG_PORT, exc)
            self._running = False
            return

        server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='avatar-http',
        )
        server_thread.start()
        self._ready_event.set()
        logger.info('Avatar MJPEG server avviato su http://127.0.0.1:%d/stream', MJPEG_PORT)

        # ── Passo 3: inference.py se il video non è in cache ──────────────────
        if not self._is_output_current():
            logger.info('Output video mancante o obsoleto. Avvio inference.py...')
            success = self._run_inference()
            if not success:
                logger.error('Inference fallita. Il portrait statico rimane attivo.')
                self._inactivity_watcher()   # aspetta e poi si ferma
                return

        # ── Passo 4: loop video infinito ──────────────────────────────────────
        logger.info('Video output pronto: %s', OUTPUT_VIDEO)
        video_thread = threading.Thread(
            target=self._loop_video,
            daemon=True,
            name='avatar-video-loop',
        )
        video_thread.start()

        # ── Passo 5: watcher inattività ───────────────────────────────────────
        self._inactivity_watcher()

    def _load_portrait(self) -> None:
        """Carica eden_portrait.png e lo imposta come frame iniziale."""
        try:
            img = cv2.imread(str(PORTRAIT))
            if img is None:
                return
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                _set_frame(buf.tobytes())
                logger.info('Portrait statico caricato come frame iniziale.')
        except Exception as exc:
            logger.debug('Errore caricamento portrait: %s', exc)

    def _is_output_current(self) -> bool:
        """
        True se OUTPUT_VIDEO esiste ed è più recente di portrait e driving.
        Evita di ri-eseguire inference se il video è già in cache.
        """
        if not OUTPUT_VIDEO.exists():
            return False
        out_mtime = OUTPUT_VIDEO.stat().st_mtime
        return (
            out_mtime > PORTRAIT.stat().st_mtime and
            out_mtime > DRIVING.stat().st_mtime
        )

    def _run_inference(self) -> bool:
        """
        Esegue liveportrait/inference.py come subprocess.
        Blocca finché l'inferenza è completata.
        Ritorna True se il video di output è stato generato con successo.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(INFERENCE_PY),
            '--source',     str(PORTRAIT),
            '--driving',    str(DRIVING),
            '--output-dir', str(OUTPUT_DIR),
            # Relative motion: preserva l'identità del portrait (default True,
            # esplicitato per chiarezza)
            '--flag-relative-motion', 'True',
            # Pasteback: incolla il volto animato sull'immagine originale
            '--flag-pasteback', 'True',
            '--flag-do-crop', 'True',
            # Opzione expression-friendly per guidare il driving video
            '--driving-option', 'expression-friendly',
        ]

        logger.info(
            'Avvio inference.py (potrebbe richiedere 1-3 minuti)...\n  %s',
            ' '.join(cmd)
        )

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR / 'liveportrait'),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
            )

            # Logga l'output riga per riga in modo da vedere il progresso
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.info('[inference] %s', line)

            proc.wait()

            if proc.returncode != 0:
                logger.error('inference.py uscito con codice %d.', proc.returncode)
                return False

            if OUTPUT_VIDEO.exists():
                logger.info('Inference completata: %s', OUTPUT_VIDEO)
                return True

            # Cerca qualsiasi .mp4 nell'output dir (nome potrebbe differire)
            mp4_files = sorted(OUTPUT_DIR.glob('*.mp4'))
            if mp4_files:
                # Rinomina il primo .mp4 trovato nel nome atteso
                mp4_files[0].rename(OUTPUT_VIDEO)
                logger.info('Video rinominato: %s → %s', mp4_files[0].name, OUTPUT_VIDEO.name)
                return True

            logger.error('Inference completata ma nessun .mp4 trovato in %s', OUTPUT_DIR)
            return False

        except Exception as exc:
            logger.error('Errore avvio inference.py: %s', exc)
            return False

    def _loop_video(self) -> None:
        """
        Thread: legge OUTPUT_VIDEO frame per frame in loop infinito.
        Quando il video finisce, torna al frame 0.
        Aggiorna il buffer condiviso _current_jpg.
        Si ferma se _stop_event è settato.
        """
        frame_dt = 1.0 / FPS_SERVE

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(str(OUTPUT_VIDEO))
            if not cap.isOpened():
                logger.error('Impossibile aprire il video: %s', OUTPUT_VIDEO)
                time.sleep(5.0)
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS) or FPS_SERVE
            read_dt   = 1.0 / video_fps
            logger.debug('Looping video a %.1f fps (serve a %d fps)', video_fps, FPS_SERVE)

            last_t = time.perf_counter()

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break   # video finito → loop esterno riapre dall'inizio

                # Resize al formato di output e codifica JPEG
                frame_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
                enc_ok, buf   = cv2.imencode(
                    '.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if enc_ok:
                    _set_frame(buf.tobytes())

                # Rate control: rispetta il FPS del video originale
                now     = time.perf_counter()
                elapsed = now - last_t
                time.sleep(max(0.0, read_dt - elapsed))
                last_t  = time.perf_counter()

            cap.release()

        logger.info('Loop video terminato.')

    def _inactivity_watcher(self) -> None:
        """
        Blocca il thread finché:
          - stop_event è settato (stop() chiamato dall'esterno), oppure
          - INACTIVITY_MIN minuti senza client attivi → chiama stop()
        """
        inactivity_s = INACTIVITY_MIN * 60
        check_interval = 30.0   # controlla ogni 30 secondi

        while not self._stop_event.is_set():
            time.sleep(check_interval)
            if self._stop_event.is_set():
                break
            idle = time.time() - _last_activity
            if idle > inactivity_s:
                logger.info(
                    'Nessun client attivo per %.0f minuti. Avatar in pausa.',
                    idle / 60
                )
                self.stop()
                break

# ─── Singleton ────────────────────────────────────────────────────────────────

_manager: AvatarManager | None = None


def get_manager() -> AvatarManager:
    global _manager
    if _manager is None:
        _manager = AvatarManager()
    return _manager
