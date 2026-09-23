// static/avatar/user_presence.js — Rilevamento emozioni utente via MediaPipe (Phase 3+)
//
// Funzionamento:
//   1. Importa FaceLandmarker da CDN (WASM, eseguito nel browser, 0 VRAM backend)
//   2. Ascolta evento 'eden:webcam:start' dispatch da script.js quando la cam è attiva
//   3. Ogni ~150ms: rileva face blendshapes → POST /api/user_presence
//   4. Il backend aggiorna liveportrait_state.json → run_eden.py reagisce nel frame successivo
//
// Graceful degradation: se CDN offline o browser non supporta WASM, silenzio totale.

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm';

const API_BASE    = window.location.origin;
const INTERVAL_MS = 150;   // frequenza rilevamento (ms) — 6-7fps, leggero

let _landmarker   = null;
let _detecting    = false;
let _lastSendTime = 0;

// ─── Inizializzazione MediaPipe ───────────────────────────────────────────────

async function _initMediaPipe() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm'
    );
    _landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/' +
          'face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: true,
      runningMode:           'VIDEO',
      numFaces:              1,
    });
    console.log('[Eden presence] MediaPipe FaceLandmarker pronto.');
  } catch (err) {
    // CDN offline o browser incompatibile — degradazione silenziosa
    console.warn('[Eden presence] MediaPipe non disponibile:', err.message);
  }
}

// ─── Loop di rilevamento ─────────────────────────────────────────────────────

function _startDetection(video) {
  if (_detecting) return;
  if (!_landmarker) return;   // MediaPipe non caricato
  _detecting = true;
  console.log('[Eden presence] Rilevamento emozioni avviato.');

  function _tick() {
    if (!_detecting) return;

    const now = performance.now();
    if (now - _lastSendTime >= INTERVAL_MS && video.readyState >= 2) {
      _lastSendTime = now;
      _detectAndSend(video, now);
    }

    requestAnimationFrame(_tick);
  }

  requestAnimationFrame(_tick);
}

async function _detectAndSend(video, timestampMs) {
  try {
    const result = _landmarker.detectForVideo(video, timestampMs);
    if (!result?.faceBlendshapes?.[0]) return;   // nessun volto rilevato

    const bs  = result.faceBlendshapes[0].categories;
    const get = name => bs.find(c => c.categoryName === name)?.score ?? 0;

    const payload = {
      smile:        (get('mouthSmileLeft') + get('mouthSmileRight')) / 2,
      brow_raise:   get('browInnerUp'),
      brow_down:    (get('browDownLeft')   + get('browDownRight'))   / 2,
      eye_wide:     (get('eyeWideLeft')    + get('eyeWideRight'))    / 2,
      jaw_open:     get('jawOpen'),
      face_present: 1,
    };

    // Fire-and-forget: non blocca il loop di rilevamento
    fetch(`${API_BASE}/api/user_presence`, {
      method:   'POST',
      headers:  { 'Content-Type': 'application/json' },
      body:     JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {});   // silenzioso se server non risponde

  } catch {
    // errore di inferenza: silenzioso, riprova al tick successivo
  }
}

// ─── Avvio ───────────────────────────────────────────────────────────────────

// Inizializza MediaPipe subito (download asincrono, ~500ms)
_initMediaPipe();

// Quando script.js avvia la webcam, parte il loop di rilevamento
window.addEventListener('eden:webcam:start', e => {
  const video = e.detail?.video;
  if (!video) return;
  // Attendi che il video si stabilizzi, poi avvia
  video.addEventListener('loadeddata', () => _startDetection(video), { once: true });
  // Fallback se loadeddata già scattato
  if (video.readyState >= 2) _startDetection(video);
});
