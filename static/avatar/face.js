// static/avatar/face.js — Three.js avatar portrait di Eden
//
// Uso: import { initAvatar } from '/static/avatar/face.js';
//      initAvatar();  ← chiama dopo DOMContentLoaded
//
// Ascolta eventi window:
//   'eden:traits'  → { curiosity, trust, cynicism, warmth, fear }
//   'eden:state'   → { thinking?: bool, proactive?: bool, speaking?: bool }
//   'eden:chat'    → { role, valence, energy, question }

import * as THREE from 'three';

// ─── GLSL: vertex UV ──────────────────────────────────────────────────────────
const VERT = /* glsl */`
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// ─── GLSL: portrait + rim glow teal ──────────────────────────────────────────
// uReady=0 → placeholder scuro con glow pulsante mentre la texture carica.
// uReady=1 → portrait reale con rim glow sovrapposto in addizione.
const FRAG = /* glsl */`
  uniform sampler2D uTex;
  uniform float     uTime;
  uniform float     uGlow;    // 0..1  thinking / proattivo
  uniform float     uReady;   // 0 = caricamento, 1 = texture pronta
  varying vec2      vUv;

  void main() {
    // Distanza dal bordo più vicino (0 = centro, 1 = bordo)
    float edgeDist = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)) * 2.0;
    float rim      = pow(1.0 - clamp(edgeDist, 0.0, 1.0), 2.8);

    // Pulsazione lenta (respiro glow)
    float pulse  = 0.5 + 0.5 * sin(uTime * 2.19);
    float amt    = rim * (0.09 + pulse * 0.11 + uGlow * 0.58);

    vec3 teal    = vec3(0.0, 0.84, 0.76);
    vec3 tex     = texture2D(uTex, vUv).rgb;

    // Placeholder scuro mentre carica; portrait reale quando pronta
    vec3 base    = mix(vec3(0.02, 0.04, 0.06), tex, uReady);

    gl_FragColor = vec4(base + teal * amt, 1.0);
  }
`;

// ─── Stato espressivo ─────────────────────────────────────────────────────────
// TARGET / CURRENT rimangono invariati: guidano head_tilt, head_forward,
// head_back e i delta idle (micro-espressioni, saccadi).
const TARGET = {
  brow_raise: 0, brow_inner: 0, brow_lower: 0,
  eye_wide:   0, eye_soft:   0, eye_squint:   0, eye_contact: 0.5,
  smile:      0, mouth_tense: 0,
  head_tilt:  0, head_forward: 0, head_back: 0,
};
const CURRENT = { ...TARGET };

// ─── Flags di stato ───────────────────────────────────────────────────────────
let _thinking  = false;
let _proTimer  = 0;
let _speaking  = false;

let _chatTimer   = 0;
let _chatValence = 0;
let _chatEnergy  = 0;
let _chatFocus   = 0;

// Telemetria performance
let _lastRAFMs      = 0;
let _accumMs        = 0;
let _fpsSmoothed    = 60;
let _fpsReportTimer = 0;
let _displayHzHint  = 60;
let _targetFPS      = 60;

// ─── Oggetti Three.js ─────────────────────────────────────────────────────────
let renderer, scene, camera, clock;
let headGroup;
let portraitMesh;
let particles, particlePosArr, particleBasePos;

// ─── Stato blink (timer invariato, non applicato al portrait) ─────────────────
let _blinkPhase  = 'idle';
let _blinkTimer  = 0;
let _blinkNextAt = 2.2 + Math.random() * 3.5;

// ─── Animazione autonoma idle ─────────────────────────────────────────────────
const _idleDelta = {
  smile: 0, brow_raise: 0, brow_inner: 0, brow_lower: 0,
  eye_wide: 0, eye_soft: 0, mouth_tense: 0, head_tilt: 0,
};
const _idleDeltaTarget = { ...(_idleDelta) };
let _idleExprTimer  = 0;
let _idleExprNextAt = 8 + Math.random() * 14;

const _saccade       = { x: 0, y: 0 };
const _saccadeTarget = { x: 0, y: 0 };
let _saccadeTimer    = 0;
let _saccadeNextAt   = 2.5 + Math.random() * 5.5;

let _currentTraits = { curiosity: 5, trust: 5, cynicism: 5, warmth: 5, fear: 5 };

// ─── API pubblica ─────────────────────────────────────────────────────────────

export function initAvatar() {
  const container = document.getElementById('avatar-container');
  if (!container) { console.warn('[Eden portrait] #avatar-container non trovato'); return; }

  // ── Renderer ───────────────────────────────────────────────────────────────
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const cw = container.clientWidth  || 340;
  const ch = container.clientHeight || 340;
  renderer.setSize(cw, ch);
  renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;';
  container.appendChild(renderer.domElement);

  // ── Scena ──────────────────────────────────────────────────────────────────
  scene = new THREE.Scene();
  clock = new THREE.Clock();

  // ── Camera ─────────────────────────────────────────────────────────────────
  camera = new THREE.PerspectiveCamera(36, cw / ch, 0.1, 50);
  camera.position.set(0, 0.08, 4.0);
  camera.lookAt(0, 0.08, 0);

  // ── Portrait + Particelle ──────────────────────────────────────────────────
  _buildPortrait();
  _buildParticles();

  // ── Resize ─────────────────────────────────────────────────────────────────
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  ro.observe(container);

  // ── Ascolta eventi emotivi da script.js ────────────────────────────────────
  window.addEventListener('eden:traits', e => _applyTraits(e.detail));
  window.addEventListener('eden:state',  e => _applyState(e.detail));
  window.addEventListener('eden:chat',   e => _applyChat(e.detail));

  // Resilienza context loss
  renderer.domElement.addEventListener('webglcontextlost',     ev => ev.preventDefault());
  renderer.domElement.addEventListener('webglcontextrestored', () => { clock.start(); _lastRAFMs = 0; });

  _animate();
}

// ─── Costruzione geometria ────────────────────────────────────────────────────

function _buildPortrait() {
  headGroup = new THREE.Group();
  scene.add(headGroup);

  const PLANE_H = 2.2;

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTex:   { value: new THREE.Texture() },
      uTime:  { value: 0 },
      uGlow:  { value: 0 },
      uReady: { value: 0 },
    },
    vertexShader:   VERT,
    fragmentShader: FRAG,
  });

  portraitMesh = new THREE.Mesh(new THREE.PlaneGeometry(PLANE_H, PLANE_H), mat);
  headGroup.add(portraitMesh);

  new THREE.TextureLoader().load(
    '/api/portrait',
    (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      mat.uniforms.uTex.value   = tex;
      mat.uniforms.uReady.value = 1.0;
      // Adatta la larghezza del piano all'aspect ratio reale dell'immagine
      const { width: w, height: h } = tex.image;
      if (w && h) portraitMesh.scale.x = w / h;
    },
    undefined,
    () => console.warn('[Eden portrait] caricamento /api/portrait fallito'),
  );
}

function _buildParticles() {
  const N = 72;
  particlePosArr  = new Float32Array(N * 3);
  particleBasePos = new Float32Array(N * 3);

  for (let i = 0; i < N; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = 1.18 + Math.random() * 0.55;
    particleBasePos[i*3]   = r * Math.sin(phi) * Math.cos(theta);
    particleBasePos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
    particleBasePos[i*3+2] = r * Math.cos(phi);
  }
  particlePosArr.set(particleBasePos);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(particlePosArr, 3));

  particles = new THREE.Points(geo, new THREE.PointsMaterial({
    color: 0x00eedd, size: 0.028, transparent: true, opacity: 0, sizeAttenuation: true,
  }));
  scene.add(particles);
}

// ─── Blink: timer tick invariato ─────────────────────────────────────────────
// Restituisce 1 (aperto) … 0 (chiuso). Non applicato al portrait —
// tenuto per compatibilità con future versioni animate.

function _blinkMult(delta) {
  _blinkTimer += delta;
  const CLOSE = 0.076, OPEN = 0.130;

  if (_blinkPhase === 'idle') {
    if (_blinkTimer >= _blinkNextAt) { _blinkPhase = 'closing'; _blinkTimer = 0; }
    return 1.0;
  }
  if (_blinkPhase === 'closing') {
    const p = Math.min(1, _blinkTimer / CLOSE);
    if (p >= 1) { _blinkPhase = 'opening'; _blinkTimer = 0; return 0; }
    return 1 - p;
  }
  const p = Math.min(1, _blinkTimer / OPEN);
  if (p >= 1) {
    _blinkPhase = 'idle'; _blinkTimer = 0;
    _blinkNextAt = 2.5 + Math.random() * 4.5;
    return 1;
  }
  return p;
}

// ─── Particelle neurali ───────────────────────────────────────────────────────

function _updateParticles(t, delta) {
  const targetOpacity = (_thinking ? 0.62 : 0) + (_proTimer > 0 ? Math.min(0.35, _proTimer * 0.22) : 0);
  const mat = particles.material;
  mat.opacity += (targetOpacity - mat.opacity) * Math.min(1, delta * 3.5);

  if (mat.opacity < 0.008) return;

  for (let i = 0; i < 72; i++) {
    const bx = particleBasePos[i*3], by = particleBasePos[i*3+1], bz = particleBasePos[i*3+2];
    const speed = 0.24 + (i % 9) * 0.068;
    const angle = t * speed + i * 0.8727;

    const r = Math.sqrt(bx*bx + bz*bz);
    particlePosArr[i*3]   = r * Math.cos(angle);
    particlePosArr[i*3+1] = by + Math.sin(t * 0.58 + i * 0.42) * 0.038;
    particlePosArr[i*3+2] = r * Math.sin(angle);
  }
  particles.geometry.attributes.position.needsUpdate = true;
}

// ─── Animazione idle autonoma ─────────────────────────────────────────────────

function _tickIdleAnimation(delta) {
  // Micro-espressioni
  _idleExprTimer += delta;
  if (_idleExprTimer >= _idleExprNextAt) {
    _idleExprTimer  = 0;
    _idleExprNextAt = 8 + Math.random() * 22;

    const f  = _currentTraits.fear      / 10;
    const w  = _currentTraits.warmth    / 10;
    const c  = _currentTraits.curiosity / 10;
    const cy = _currentTraits.cynicism  / 10;

    _idleDeltaTarget.smile       = (Math.random() * 0.18 * (1 + w) - 0.04) * (f > 0.6 ? 0.3 : 1);
    _idleDeltaTarget.brow_raise  = Math.random() * 0.14 * c;
    _idleDeltaTarget.brow_inner  = Math.random() * 0.12 * f;
    _idleDeltaTarget.brow_lower  = Math.random() * 0.10 * cy;
    _idleDeltaTarget.eye_wide    = (Math.random() - 0.35) * 0.12 * (c + f * 0.5);
    _idleDeltaTarget.eye_soft    = Math.random() * 0.10 * w;
    _idleDeltaTarget.mouth_tense = Math.random() * 0.09 * f;
    _idleDeltaTarget.head_tilt   = (Math.random() > 0.5 ? 1 : -1) * Math.random() * 0.15 * (c * 0.7 + 0.3);
  }

  const ki = 1 - Math.pow(0.25, delta);
  for (const key of Object.keys(_idleDelta)) {
    _idleDelta[key] += (_idleDeltaTarget[key] - _idleDelta[key]) * ki;
  }

  // Saccadi oculari (guidano _saccade x/y usato per head_tilt fine)
  _saccadeTimer += delta;
  if (_saccadeTimer >= _saccadeNextAt) {
    _saccadeTimer  = 0;
    _saccadeNextAt = 2.0 + Math.random() * 6.5;
    const amp = 0.035 * (1 - _currentTraits.trust / 15);
    _saccadeTarget.x = (Math.random() - 0.5) * amp * 2;
    _saccadeTarget.y = (Math.random() - 0.5) * amp;
  }
  const ks = 1 - Math.pow(0.005, delta);
  _saccade.x += (_saccadeTarget.x - _saccade.x) * ks;
  _saccade.y += (_saccadeTarget.y - _saccade.y) * ks;
}

// ─── Applicazione tratti → TARGET ────────────────────────────────────────────

function _applyTraits(traits) {
  Object.assign(_currentTraits, traits);

  const c  = Math.min(1, (traits.curiosity ?? 5) / 10);
  const f  = Math.min(1, (traits.fear      ?? 5) / 10);
  const w  = Math.min(1, (traits.warmth    ?? 5) / 10);
  const cy = Math.min(1, (traits.cynicism  ?? 5) / 10);
  const tr = Math.min(1, (traits.trust     ?? 5) / 10);

  TARGET.brow_raise   = c * 0.6;
  TARGET.brow_inner   = f * 0.7;
  TARGET.mouth_tense  = f * 0.3;
  TARGET.brow_lower   = cy * 0.4;
  TARGET.eye_squint   = cy * 0.3;
  TARGET.head_back    = cy * 0.2;
  TARGET.eye_wide     = Math.min(1, c * 0.4 + f * 0.8);
  TARGET.eye_soft     = w * 0.4;
  TARGET.head_forward = w * 0.1;
  TARGET.eye_contact  = 0.15 + tr * 0.85;
  TARGET.head_tilt    = c * 0.2;
  TARGET.smile        = Math.min(1, w * 0.5 + tr * 0.2);
}

function _applyState({ thinking, proactive, speaking } = {}) {
  if (thinking !== undefined) _thinking = !!thinking;
  if (proactive)              _proTimer = 2.0;
  if (speaking !== undefined) _speaking = !!speaking;
}

function _applyChat({ role, valence, energy, question } = {}) {
  const v = Number.isFinite(valence) ? Math.max(-1, Math.min(1, valence)) : 0;
  const e = Number.isFinite(energy)  ? Math.max(0, Math.min(1, energy))   : 0;
  _chatValence = v;
  _chatEnergy  = e;
  _chatFocus   = question ? 1 : 0;
  _chatTimer   = role === 'proactive' ? 2.2 : 1.7;
}

function _reportFPSIfNeeded(delta) {
  _fpsReportTimer += delta;
  if (_fpsReportTimer < 1.0) return;
  _fpsReportTimer = 0;
  window.dispatchEvent(new CustomEvent('eden:avatar:metrics', {
    detail: { fps: _fpsSmoothed, target_fps: _targetFPS, display_hz_hint: _displayHzHint },
  }));
}

// ─── Loop di animazione principale ───────────────────────────────────────────

function _animate(nowMs = 0) {
  requestAnimationFrame(_animate);

  if (!_lastRAFMs) { _lastRAFMs = nowMs || performance.now(); return; }

  const refNow  = nowMs || performance.now();
  const stepRaw = Math.max(1, refNow - _lastRAFMs);
  _lastRAFMs = refNow;

  // Stima refresh display
  const instFPS = 1000 / stepRaw;
  if (instFPS > _displayHzHint) _displayHzHint = Math.min(165, instFPS);
  _displayHzHint = Math.max(50, _displayHzHint * 0.995);
  _targetFPS = _displayHzHint >= 95 ? 120 : 60;

  _accumMs += Math.min(stepRaw, 120);
  const budget = 1000 / _targetFPS;
  if (_accumMs < budget) return;

  const stepMs = _accumMs;
  _accumMs = 0;

  const delta = Math.min(stepMs / 1000, 0.1);
  const t     = refNow * 0.001;
  _fpsSmoothed = _fpsSmoothed * 0.9 + (1 / Math.max(delta, 0.0001)) * 0.1;

  if (_proTimer  > 0) _proTimer  = Math.max(0, _proTimer  - delta);
  if (_chatTimer > 0) _chatTimer = Math.max(0, _chatTimer - delta);
  const chatMix = Math.min(1, _chatTimer / 1.7);

  _tickIdleAnimation(delta);
  _blinkMult(delta); // tick timer

  // ── CURRENT state machine (head_tilt, head_forward, head_back) ─────────────
  const add = {
    smile:       Math.max(0, _chatValence) * 0.30 * chatMix + _idleDelta.smile,
    mouth_tense: (Math.max(0, -_chatValence) * 0.26 + _chatEnergy * 0.16) * chatMix + _idleDelta.mouth_tense,
    brow_raise:  (_chatEnergy * 0.25 + _chatFocus * 0.18) * chatMix + _idleDelta.brow_raise,
    brow_inner:  _idleDelta.brow_inner,
    brow_lower:  _idleDelta.brow_lower,
    eye_wide:    _chatEnergy * 0.19 * chatMix + _idleDelta.eye_wide,
    eye_soft:    _idleDelta.eye_soft,
    head_tilt:   (_chatValence >= 0 ? 1 : -1) * 0.05 * chatMix + _idleDelta.head_tilt,
    eye_contact: (_chatFocus ? 0.18 : 0.08) * chatMix,
  };

  const k = 1 - Math.pow(0.04, delta);
  const desired = {
    ...TARGET,
    smile:       Math.min(1,    Math.max(0,    TARGET.smile       + add.smile)),
    mouth_tense: Math.min(1,    Math.max(0,    TARGET.mouth_tense + add.mouth_tense)),
    brow_raise:  Math.min(1,    Math.max(0,    TARGET.brow_raise  + add.brow_raise)),
    brow_inner:  Math.min(1,    Math.max(0,    TARGET.brow_inner  + add.brow_inner)),
    brow_lower:  Math.min(1,    Math.max(0,    TARGET.brow_lower  + add.brow_lower)),
    eye_wide:    Math.min(1,    Math.max(0,    TARGET.eye_wide    + add.eye_wide)),
    eye_soft:    Math.min(1,    Math.max(0,    TARGET.eye_soft    + add.eye_soft)),
    eye_contact: Math.min(1,    Math.max(0,    TARGET.eye_contact + add.eye_contact)),
    head_tilt:   Math.max(-0.4, Math.min(0.6, TARGET.head_tilt   + add.head_tilt)),
  };
  for (const key of Object.keys(CURRENT)) {
    CURRENT[key] += (desired[key] - CURRENT[key]) * k;
  }

  // ── Respiro procedurale ────────────────────────────────────────────────────
  const fearNorm   = _currentTraits.fear   / 10;
  const warmthNorm = _currentTraits.warmth / 10;
  const breathFreq = 0.175 + fearNorm * 0.12;
  const breathAmp  = 0.007 + warmthNorm * 0.003 - fearNorm * 0.002;
  const breath     = Math.sin(t * breathFreq * Math.PI * 2);
  headGroup.scale.y = 1.0 + breath * breathAmp;

  // ── Micromovement testa ────────────────────────────────────────────────────
  const movScale = 1.0 + _currentTraits.curiosity / 25 + fearNorm * 0.3;
  const dX = (Math.sin(t * 0.31) * 0.017 + Math.sin(t * 0.19) * 0.009) * movScale;
  const dY = (Math.sin(t * 0.23) * 0.012 + Math.cos(t * 0.41) * 0.007) * movScale;
  const chatNod = Math.sin(t * 7.2) * 0.01 * _chatEnergy * chatMix;
  headGroup.rotation.x = dX + chatNod;
  headGroup.rotation.y = dY + CURRENT.head_tilt * 0.14;
  headGroup.rotation.z = -CURRENT.head_tilt * 0.17;
  headGroup.position.z = CURRENT.head_forward * 0.08 - CURRENT.head_back * 0.05;

  // ── Shader portrait ────────────────────────────────────────────────────────
  const uniforms = portraitMesh.material.uniforms;
  uniforms.uTime.value = t;
  const glow = (_thinking ? 0.32 : 0) + (_proTimer > 0 ? (_proTimer / 2.0) * 0.45 : 0);
  uniforms.uGlow.value = glow + (_speaking ? 0.10 : 0);

  // ── Particelle neurali ─────────────────────────────────────────────────────
  _updateParticles(t, delta);

  renderer.render(scene, camera);
  _reportFPSIfNeeded(delta);
}
