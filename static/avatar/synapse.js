// synapse.js — Avatar Cerebrale Eden v2.0
//
// 9 regioni anatomiche cerebrali mappate sui sottosistemi Eden.
// Nodi point-cloud, assoni CatmullRom animati, pool pulse, plasticità hebbiana,
// sharp-wave ripple replay (SIC), decay Ebbinghaus, neurotrasmettitori dinamici.
//
// Export: initAvatar, triggerChatWave, triggerProactiveBurst
// Polling: /api/research/status (4.5s), /api/status (9s),
//          /api/research/graph/status (30s), /api/inner_stream/buffer (30s)

import * as THREE from 'three';

export const AVATAR_VERSION = "v2.0";

// ─── Regioni anatomiche ───────────────────────────────────────────────────────
// pos: THREE.Vector3, r: raggio sferoide, color: hex, system: sottosistema Eden, count: nodi

const REGIONS_DEF = [
  { key: 'pfc',       pos: [0,      1.90,  0.90], r: 0.58, color: '#6fa8c8', system: 'coherence_budget',    count: 90  },
  { key: 'acc',       pos: [0,      1.15,  1.05], r: 0.34, color: '#4a8fbb', system: 'grounding_integrity', count: 50  },
  { key: 'hippo_l',   pos: [-1.10,  0.15, -0.35], r: 0.36, color: '#48c774', system: 'episodic_memory',     count: 60  },
  { key: 'hippo_r',   pos: [ 1.10,  0.15, -0.35], r: 0.36, color: '#48c774', system: 'episodic_memory',     count: 60  },
  { key: 'amyg_l',    pos: [-0.80, -0.05,  0.55], r: 0.26, color: '#ff7043', system: 'emotion',             count: 35  },
  { key: 'amyg_r',    pos: [ 0.80, -0.05,  0.55], r: 0.26, color: '#ff7043', system: 'emotion',             count: 35  },
  { key: 'basal',     pos: [0,      0.35,  0.10], r: 0.40, color: '#f5a623', system: 'dna_nodes',           count: 65  },
  { key: 'dmn',       pos: [0,      0.45, -1.65], r: 0.50, color: '#bd93f9', system: 'inner_stream',        count: 75  },
  { key: 'brainstem', pos: [0,     -1.45,  0.05], r: 0.30, color: '#50fa7b', system: 'somatic',             count: 40  },
];

// 20 assoni anatomicamente motivati
const AXON_PAIRS = [
  ['pfc','acc'],     ['pfc','hippo_l'], ['pfc','hippo_r'], ['pfc','amyg_l'],  ['pfc','amyg_r'],
  ['pfc','dmn'],     ['pfc','basal'],
  ['acc','amyg_l'],  ['acc','amyg_r'],  ['acc','brainstem'],
  ['hippo_l','hippo_r'], ['hippo_l','amyg_l'], ['hippo_r','amyg_r'],
  ['hippo_l','dmn'], ['hippo_r','dmn'],
  ['amyg_l','amyg_r'], ['amyg_l','brainstem'], ['amyg_r','brainstem'],
  ['basal','brainstem'], ['dmn','brainstem'],
];

// Frequenze di oscillazione per regione [Hz visivi]
const OSC_FREQ = {
  pfc: 10, acc: 10, hippo_l: 6, hippo_r: 6,
  amyg_l: 40, amyg_r: 40, basal: 15, dmn: 8, brainstem: 4,
};

// ─── Stato globale ────────────────────────────────────────────────────────────

function s(v) { return { cur: v, tgt: v }; }

const S = {
  valence:    s(0),   arousal:    s(0.2), certainty:  s(0.5),
  attachment: s(0.5), agency:     s(0.5), threat:     s(0),
  curiosity:  s(0.5), trust:      s(0.5), fear:       s(0.3), warmth: s(0.5),
  coherence:  s(1.0), grounding:  s(1.0), selfModel:  s(1.0),
  safeMode: false,
  thinking: false, speaking: false, proactiveT: 0, innerSpark: 0,
  wave: { active: 0, dir: 0, age: 999, strength: 0 },
  timeSec: 0,
  regionActivity: {},
  hebbWeights: {},
  lastThoughtHash: '',
  replayQueue: [],
  episodeCount: 0,
  dnaCount: 0,
};

// Timestamp ultimo accesso per Ebbinghaus decay per regione
const lastActivation = {};

// ─── Three.js refs ────────────────────────────────────────────────────────────

let renderer, scene, camera, clock, container, root;

// Nodi: un unico BufferGeometry / THREE.Points
let nodeGeo, nodeMat, nodePoints;
// Struttura dati per nodi
const nodeData = [];        // { regionKey, regionIdx, basePos, jitterPhase, idx }
const nodePositions = [];   // Float32Array view aggiornata ogni frame
let nodePosArr, nodeColorArr, nodeSizeArr, nodeAlphaArr, nodeRegionArr;

// Regioni: struttura runtime
const regions = {};         // key → { def, center: Vector3, color: Color, activity: 0..1 }

// Assoni: un unico LineSegments
let axonGeo, axonMat, axonLines;
// Struttura dati per assoni
const axonData = [];        // { keyA, keyB, ntType, curvePoints[], name, baseWeight }
let axonPosArr, axonAlphaArr;
const AXON_CURVE_SEGMENTS = 12;  // punti per assone

// Pulse pool
const PULSE_POOL = 240;
let pulseMesh, pulsePosArr;
const pulses = [];
const _tmpQuat = new THREE.Quaternion();
// Oggetti pre-allocati a livello modulo — evitano allocazioni per-frame che triggerano GC
const _pm4   = new THREE.Matrix4();
const _psc   = new THREE.Vector3();
const _ptmpA = new THREE.Vector3();
const _ptmpB = new THREE.Vector3();

// Ambient + halo
let ambientGeo, ambientMat, ambientPoints;
let halo, haloMat;

let _started = false;
let _pulseAccum = 0;
let _cameraDrift = 0;

// ─── Colori neurotrasmettitori ────────────────────────────────────────────────

const NT_COLORS = {
  dopamine:      new THREE.Color('#6fa8c8'),
  serotonin:     new THREE.Color('#a8e6cf'),
  cortisol:      new THREE.Color('#ff5252'),
  norepinephrine:new THREE.Color('#69ff47'),
  norepinephrine_dim: new THREE.Color('#69ff47'),
};
// Ambra per amigdala calm
const NT_AMYG_CALM = new THREE.Color('#ffab40');

function getNTType(keyA, keyB) {
  const amyg = new Set(['amyg_l','amyg_r']);
  const hippo = new Set(['hippo_l','hippo_r']);
  if (amyg.has(keyA) || amyg.has(keyB)) return 'amyg';
  if ((keyA === 'pfc' && (hippo.has(keyB) || keyB === 'acc')) ||
      (keyB === 'pfc' && (hippo.has(keyA) || keyA === 'acc'))) return 'dopamine';
  if ((hippo.has(keyA) && keyB === 'dmn') || (hippo.has(keyB) && keyA === 'dmn')) return 'dopamine';
  if ((keyA === 'pfc' && keyB === 'dmn') || (keyB === 'pfc' && keyA === 'dmn')) return 'dopamine';
  if (keyA === 'basal' || keyB === 'basal') return 'dopamine';
  if (keyA === 'brainstem' || keyB === 'brainstem') return 'norepinephrine';
  return 'serotonin';
}

function getNTColor(axon) {
  if (axon.ntType === 'amyg') {
    return (S.arousal.cur > 0.65 && S.valence.cur < 0) ? NT_COLORS.cortisol : NT_AMYG_CALM;
  }
  return NT_COLORS[axon.ntType] || NT_COLORS.serotonin;
}

// ─── Export pubblici ──────────────────────────────────────────────────────────

export function initAvatar() {
  if (_started) return;
  _started = true;

  container = document.getElementById('avatar-container');
  if (!container) { console.error('[synapse v2] #avatar-container non trovato'); return; }

  // Rimuovi LivePortrait layer dal DOM
  _cleanupLivePortrait();

  setupScene();
  buildRegions();
  buildNodeCloud();
  buildAxons();
  buildPulsePool();
  buildAmbientField();
  buildHalo();

  window.addEventListener('eden:traits', onTraits);
  window.addEventListener('eden:state',  onState);
  window.addEventListener('eden:chat',   onChat);

  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(onResize);
    ro.observe(container);
  } else {
    window.addEventListener('resize', onResize);
  }

  // Inizializza activity + Hebb weights
  for (const { key } of REGIONS_DEF) {
    S.regionActivity[key] = 0.5;
    lastActivation[key] = Date.now();
  }
  for (const [kA, kB] of AXON_PAIRS) {
    S.hebbWeights[`${kA}__${kB}`] = 0.5;
  }

  // Polling backend
  pollResearchStatus();
  pollStatus();
  pollGraphStatus();
  pollSIC();
  setInterval(pollResearchStatus, 4500);
  setInterval(pollStatus, 9000);
  setInterval(pollGraphStatus, 30000);
  setInterval(pollSIC, 30000);

  clock = new THREE.Clock();
  loop();
}

export function triggerChatWave(fromOutside, strength, valence) {
  S.wave.active   = 1;
  S.wave.dir      = fromOutside ? 1 : -1;
  S.wave.age      = 0;
  S.wave.strength = clamp(strength, 0, 1);
  S.valence.tgt   = clamp(S.valence.tgt * 0.7 + valence * 0.3, -1, 1);

  const n = Math.round(8 + strength * 14);
  const startKey = fromOutside ? 'pfc' : 'amyg_l';
  for (let i = 0; i < n; i++) {
    setTimeout(() => _spawnPulseFromRegion(startKey, 0.7 + strength * 0.4), i * 40);
  }
}

export function triggerProactiveBurst() {
  S.proactiveT = 4.0;
  for (let i = 0; i < 26; i++) {
    const key = REGIONS_DEF[i % REGIONS_DEF.length].key;
    setTimeout(() => _spawnPulseFromRegion(key, 0.9 + Math.random() * 0.3), i * 28);
  }
}

// ─── Setup scena ──────────────────────────────────────────────────────────────

function setupScene() {
  const w = container.clientWidth  || 360;
  const h = container.clientHeight || 380;

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h);
  renderer.setClearColor(0x000000, 0);
  Object.assign(renderer.domElement.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%', zIndex: '0',
  });
  container.insertBefore(renderer.domElement, container.firstChild);

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x04060a, 0.055);

  camera = new THREE.PerspectiveCamera(54, w / h, 0.1, 100);
  camera.position.set(0, 0, 7.0);
  camera.lookAt(0, 0, 0);

  root = new THREE.Group();
  scene.add(root);
}

function onResize() {
  if (!container || !renderer || !camera) return;
  const w = container.clientWidth  || 360;
  const h = container.clientHeight || 380;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ─── Costruzione regioni ──────────────────────────────────────────────────────

function buildRegions() {
  for (const def of REGIONS_DEF) {
    regions[def.key] = {
      def,
      center: new THREE.Vector3(...def.pos),
      color: new THREE.Color(def.color),
      activity: 0.5,
    };
  }
}

// ─── Node cloud: UN UNICO BufferGeometry / THREE.Points ──────────────────────

function buildNodeCloud() {
  const total = REGIONS_DEF.reduce((s, d) => s + d.count, 0);

  nodePosArr   = new Float32Array(total * 3);
  nodeColorArr = new Float32Array(total * 3);
  nodeSizeArr  = new Float32Array(total);
  nodeAlphaArr = new Float32Array(total);
  nodeRegionArr= new Float32Array(total);

  let idx = 0;
  for (let ri = 0; ri < REGIONS_DEF.length; ri++) {
    const def = REGIONS_DEF[ri];
    const [cx, cy, cz] = def.pos;
    const col = new THREE.Color(def.color);

    for (let k = 0; k < def.count; k++) {
      // distribuzione uniforme in sferoide
      const u = Math.random(), v = Math.random();
      const theta = u * Math.PI * 2;
      const phi   = Math.acos(2 * v - 1);
      const r     = def.r * 0.9 * Math.cbrt(Math.random());
      const bx = cx + r * Math.sin(phi) * Math.cos(theta);
      const by = cy + r * 0.88 * Math.cos(phi);
      const bz = cz + r * Math.sin(phi) * Math.sin(theta);

      nodeData.push({
        regionKey:  def.key,
        regionIdx:  ri,
        basePos:    new THREE.Vector3(bx, by, bz),
        jitterPhase: Math.random() * Math.PI * 2,
        jitterAmp:  0.015 + Math.random() * 0.022,
        idx,
      });

      nodePosArr[idx*3]   = bx;
      nodePosArr[idx*3+1] = by;
      nodePosArr[idx*3+2] = bz;
      nodeColorArr[idx*3]   = col.r;
      nodeColorArr[idx*3+1] = col.g;
      nodeColorArr[idx*3+2] = col.b;
      nodeSizeArr[idx]  = 0.06 + Math.random() * 0.05;
      nodeAlphaArr[idx] = 0.4 + Math.random() * 0.3;
      nodeRegionArr[idx]= ri;
      idx++;
    }
  }

  nodeGeo = new THREE.BufferGeometry();
  nodeGeo.setAttribute('position', new THREE.BufferAttribute(nodePosArr,   3).setUsage(THREE.DynamicDrawUsage));
  nodeGeo.setAttribute('aColor',   new THREE.BufferAttribute(nodeColorArr, 3).setUsage(THREE.DynamicDrawUsage));
  nodeGeo.setAttribute('aSize',    new THREE.BufferAttribute(nodeSizeArr,  1).setUsage(THREE.DynamicDrawUsage));
  nodeGeo.setAttribute('aAlpha',   new THREE.BufferAttribute(nodeAlphaArr, 1).setUsage(THREE.DynamicDrawUsage));
  nodeGeo.setAttribute('aRegion',  new THREE.BufferAttribute(nodeRegionArr,1));

  nodeMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime:    { value: 0 },
      uPxRatio: { value: renderer.getPixelRatio() },
      uGlitch:  { value: 0 },
      uBlur:    { value: 0 },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      attribute vec3  aColor;
      attribute float aSize;
      attribute float aAlpha;
      attribute float aRegion;
      varying vec3  vColor;
      varying float vAlpha;
      uniform float uTime;
      uniform float uPxRatio;
      uniform float uGlitch;
      void main() {
        vColor = aColor;
        vAlpha = aAlpha;
        vec3 p = position;
        if (uGlitch > 0.001) {
          float band = step(0.5, fract(p.y * 5.5 + uTime * 3.8));
          p.x += (band * 2.0 - 1.0) * 0.045 * uGlitch;
        }
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = aSize * 330.0 * uPxRatio / -mv.z;
        gl_Position  = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3  vColor;
      varying float vAlpha;
      uniform float uBlur;
      void main() {
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = length(c);
        float soft = 0.0 + uBlur * 0.28;
        float core = smoothstep(0.5, soft, d);
        float glow = smoothstep(0.5, 0.15, d);
        float a = (core * 0.82 + glow * 0.38) * vAlpha;
        if (a < 0.004) discard;
        gl_FragColor = vec4(vColor * (0.75 + glow * 0.65), a);
      }
    `,
  });

  nodePoints = new THREE.Points(nodeGeo, nodeMat);
  root.add(nodePoints);
}

// ─── Assoni: UN UNICO LineSegments ────────────────────────────────────────────

function buildAxons() {
  // Ogni assone = 12 segmenti = 13 punti (12 coppie vertex = 24 vertici per assone)
  // Ma LineSegments usa coppie: 12 segmenti = 12*2=24 vertici per assone
  const SEGS = AXON_CURVE_SEGMENTS;
  const totalVerts = AXON_PAIRS.length * (SEGS * 2);

  axonPosArr  = new Float32Array(totalVerts * 3);
  axonAlphaArr= new Float32Array(totalVerts);
  const axonColorArr = new Float32Array(totalVerts * 3);
  const axonProgressArr = new Float32Array(totalVerts); // 0..1 lungo assone

  let vIdx = 0;
  for (let ai = 0; ai < AXON_PAIRS.length; ai++) {
    const [kA, kB] = AXON_PAIRS[ai];
    const rA = regions[kA];
    const rB = regions[kB];
    const ntType = getNTType(kA, kB);
    const axonName = `${kA}__${kB}`;

    // Punti curva CatmullRom: p0, midA, midMid, midB, p1
    // Offset organico al midpoint per non-linearità
    const mid = new THREE.Vector3().lerpVectors(rA.center, rB.center, 0.5);
    const perp = new THREE.Vector3(
      (Math.random() - 0.5) * 0.5,
      (Math.random() - 0.5) * 0.5,
      (Math.random() - 0.5) * 0.5,
    );
    mid.add(perp);

    const curve = new THREE.CatmullRomCurve3([rA.center.clone(), mid, rB.center.clone()]);
    const curvePoints = curve.getPoints(SEGS + 1); // SEGS+1 punti → SEGS segmenti

    // Colore NT interpolato tra colore regione A e B
    const col = new THREE.Color().lerpColors(rA.color, rB.color, 0.5);

    // Per LineSegments: ogni segmento = 2 vertici (inizio, fine)
    const startVIdx = vIdx;
    for (let s = 0; s < SEGS; s++) {
      const pA = curvePoints[s];
      const pB = curvePoints[s + 1];
      const progA = s / SEGS;
      const progB = (s + 1) / SEGS;

      axonPosArr[vIdx*3]   = pA.x; axonPosArr[vIdx*3+1] = pA.y; axonPosArr[vIdx*3+2] = pA.z;
      axonColorArr[vIdx*3] = col.r; axonColorArr[vIdx*3+1] = col.g; axonColorArr[vIdx*3+2] = col.b;
      axonProgressArr[vIdx] = progA;
      axonAlphaArr[vIdx] = 0;
      vIdx++;

      axonPosArr[vIdx*3]   = pB.x; axonPosArr[vIdx*3+1] = pB.y; axonPosArr[vIdx*3+2] = pB.z;
      axonColorArr[vIdx*3] = col.r; axonColorArr[vIdx*3+1] = col.g; axonColorArr[vIdx*3+2] = col.b;
      axonProgressArr[vIdx] = progB;
      axonAlphaArr[vIdx] = 0;
      vIdx++;
    }

    axonData.push({
      keyA: kA, keyB: kB, ntType,
      name: axonName,
      curvePoints,
      startVIdx,
      baseWeight: 0.5,
      col: col.clone(),
    });
    S.hebbWeights[axonName] = 0.5;
  }

  axonGeo = new THREE.BufferGeometry();
  axonGeo.setAttribute('position', new THREE.BufferAttribute(axonPosArr,   3).setUsage(THREE.DynamicDrawUsage));
  axonGeo.setAttribute('aColor',   new THREE.BufferAttribute(axonColorArr, 3));
  axonGeo.setAttribute('aAlpha',   new THREE.BufferAttribute(axonAlphaArr, 1).setUsage(THREE.DynamicDrawUsage));
  axonGeo.setAttribute('aProgress',new THREE.BufferAttribute(axonProgressArr, 1));

  axonMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime:    { value: 0 },
      uOpacity: { value: 1.0 },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      attribute vec3  aColor;
      attribute float aAlpha;
      attribute float aProgress;
      varying vec3  vColor;
      varying float vAlpha;
      varying float vProgress;
      uniform float uTime;
      void main() {
        vColor    = aColor;
        vAlpha    = aAlpha;
        vProgress = aProgress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3  vColor;
      varying float vAlpha;
      varying float vProgress;
      uniform float uTime;
      uniform float uOpacity;
      void main() {
        // Pacchetti animati che scorrono lungo l'assone
        float flow = fract(vProgress * 8.0 - uTime * 1.6);
        float packet = smoothstep(0.0, 0.18, flow) * smoothstep(0.55, 0.28, flow);
        float a = vAlpha * (0.35 + packet * 0.75) * uOpacity;
        if (a < 0.003) discard;
        gl_FragColor = vec4(vColor + packet * 0.4, a);
      }
    `,
  });

  axonLines = new THREE.LineSegments(axonGeo, axonMat);
  root.add(axonLines);
}

// ─── Pulse pool ───────────────────────────────────────────────────────────────

function buildPulsePool() {
  const pGeo = new THREE.SphereGeometry(0.025, 6, 4);
  const pMat = new THREE.MeshBasicMaterial({
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });

  pulseMesh = new THREE.InstancedMesh(pGeo, pMat, PULSE_POOL);
  pulseMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const colArr = new Float32Array(PULSE_POOL * 3);
  pulseMesh.instanceColor = new THREE.InstancedBufferAttribute(colArr, 3);
  pulseMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);

  const m = new THREE.Matrix4().makeScale(0, 0, 0);
  for (let i = 0; i < PULSE_POOL; i++) {
    pulseMesh.setMatrixAt(i, m);
    pulses.push({ axonIdx: -1, t: 0, life: 1.5, intensity: 0.5, alive: false, color: new THREE.Color(1,1,1) });
  }
  pulseMesh.instanceMatrix.needsUpdate = true;
  root.add(pulseMesh);
}

function _spawnPulse(axonIdx, intensity, color) {
  if (axonIdx < 0 || axonIdx >= axonData.length) return;
  for (let i = 0; i < PULSE_POOL; i++) {
    if (!pulses[i].alive) {
      const p = pulses[i];
      p.alive = true;
      p.axonIdx = axonIdx;
      p.t = 0;
      p.life = 1.4 + Math.random() * 0.6;
      p.intensity = intensity;
      if (color) p.color.copy(color);
      else {
        const ax = axonData[axonIdx];
        p.color.copy(getNTColor(ax));
      }
      return;
    }
  }
}

function _spawnPulseFromRegion(regionKey, intensity) {
  // Reservoir sampling — zero allocazioni per frame
  let chosen = -1, count = 0;
  for (let i = 0; i < axonData.length; i++) {
    const ax = axonData[i];
    if (ax.keyA === regionKey || ax.keyB === regionKey) {
      if (((Math.random() * ++count) | 0) === 0) chosen = i;
    }
  }
  if (chosen < 0) return;
  _spawnPulse(chosen, intensity, getNTColor(axonData[chosen]));
}

// ─── Ambient field ────────────────────────────────────────────────────────────

function buildAmbientField() {
  const N = 380;
  const pos = new Float32Array(N * 3);
  const sz  = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = 7.5 * Math.cbrt(Math.random());
    const t = Math.random() * Math.PI * 2;
    const p = Math.acos(2 * Math.random() - 1);
    pos[i*3]   = r * Math.sin(p) * Math.cos(t);
    pos[i*3+1] = r * Math.cos(p);
    pos[i*3+2] = r * Math.sin(p) * Math.sin(t);
    sz[i] = 0.5 + Math.random() * 1.3;
  }

  ambientGeo = new THREE.BufferGeometry();
  ambientGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  ambientGeo.setAttribute('aSize',    new THREE.BufferAttribute(sz,  1));

  ambientMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime:      { value: 0 },
      uPxRatio:   { value: renderer.getPixelRatio() },
      uColor:     { value: new THREE.Color(0.40, 0.60, 0.80) },
      uIntensity: { value: 0.16 },
    },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      attribute float aSize;
      uniform float uTime;
      uniform float uPxRatio;
      varying float vFlick;
      void main() {
        vec3 p = position;
        p.x += sin(uTime * 0.11 + position.y * 0.7) * 0.06;
        p.y += cos(uTime * 0.09 + position.z * 0.5) * 0.05;
        vFlick = 0.55 + 0.45 * sin(uTime * 0.55 + (position.x + position.y) * 1.1);
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = aSize * 22.0 * uPxRatio / -mv.z;
        gl_Position  = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */`
      uniform vec3  uColor;
      uniform float uIntensity;
      varying float vFlick;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        float a = smoothstep(0.5, 0.0, d) * uIntensity * vFlick;
        if (a < 0.002) discard;
        gl_FragColor = vec4(uColor, a);
      }
    `,
  });

  ambientPoints = new THREE.Points(ambientGeo, ambientMat);
  scene.add(ambientPoints);
}

// ─── Halo ─────────────────────────────────────────────────────────────────────

function buildHalo() {
  const geo = new THREE.PlaneGeometry(16, 16);
  haloMat = new THREE.ShaderMaterial({
    uniforms: {
      uColor:     { value: new THREE.Color(0.10, 0.28, 0.52) },
      uIntensity: { value: 0.28 },
      uSafeMode:  { value: 0 },
      uTime:      { value: 0 },
    },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: /* glsl */`
      varying vec2 vUv;
      uniform vec3  uColor;
      uniform float uIntensity;
      uniform float uSafeMode;
      uniform float uTime;
      void main() {
        float d = distance(vUv, vec2(0.5));
        float h = smoothstep(0.52, 0.0, d) * uIntensity;
        float pulse = 0.5 + 0.5 * sin(uTime * 1.25);
        vec3 col = mix(uColor, vec3(0.95, 0.20, 0.18), uSafeMode * pulse * 0.65);
        gl_FragColor = vec4(col, h * (0.7 + uSafeMode * 0.4 * pulse));
      }
    `,
  });
  halo = new THREE.Mesh(geo, haloMat);
  halo.position.z = -3.0;
  scene.add(halo);
}

// ─── Event handlers ───────────────────────────────────────────────────────────

function onTraits(e) {
  const t = e.detail || {};
  if (typeof t.curiosity === 'number') S.curiosity.tgt = clamp01(t.curiosity / 10);
  if (typeof t.trust     === 'number') S.trust.tgt     = clamp01(t.trust     / 10);
  if (typeof t.fear      === 'number') S.fear.tgt      = clamp01(t.fear      / 10);
  if (typeof t.warmth    === 'number') S.warmth.tgt    = clamp01(t.warmth    / 10);
}

function onState(e) {
  const d = e.detail || {};
  if (typeof d.thinking  === 'boolean') S.thinking  = d.thinking;
  if (typeof d.speaking  === 'boolean') S.speaking  = d.speaking;
  if (d.proactive === true) triggerProactiveBurst();
}

function onChat(e) {
  const d = e.detail || {};
  const energy  = clamp01(d.energy  ?? 0.4);
  const valence = clamp(d.valence ?? 0, -1, 1);
  triggerChatWave(d.role === 'user', 0.5 + energy * 0.7, valence);
}

// ─── Polling backend ──────────────────────────────────────────────────────────

async function pollResearchStatus() {
  try {
    const r = await fetch('/api/research/status');
    if (!r.ok) return;
    const j = await r.json();
    const hs = j.homeostatic_state || j;
    if (typeof hs.coherence_budget     === 'number') S.coherence.tgt  = clamp01(hs.coherence_budget);
    if (typeof hs.grounding_integrity  === 'number') S.grounding.tgt  = clamp01(hs.grounding_integrity);
    if (typeof hs.self_model_stability === 'number') S.selfModel.tgt  = clamp01(hs.self_model_stability);
    if (typeof j.safe_mode === 'boolean') S.safeMode = j.safe_mode;
  } catch { /* silente */ }
}

async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const j = await r.json();
    const a = j.affective_state || {};
    if (typeof a.valence    === 'number') S.valence.tgt    = clamp(a.valence, -1, 1);
    if (typeof a.arousal    === 'number') S.arousal.tgt    = clamp01(a.arousal);
    if (typeof a.certainty  === 'number') S.certainty.tgt  = clamp01(a.certainty);
    if (typeof a.attachment === 'number') S.attachment.tgt = clamp01(a.attachment);
    if (typeof a.agency     === 'number') S.agency.tgt     = clamp01(a.agency);
    if (typeof a.threat     === 'number') S.threat.tgt     = clamp01(a.threat);
    if (j.traits) onTraits({ detail: j.traits });
  } catch { /* silente */ }
}

async function pollGraphStatus() {
  try {
    const r = await fetch('/api/research/graph/status');
    if (!r.ok) return;
    const j = await r.json();
    if (typeof j.episode_count === 'number') S.episodeCount = j.episode_count;
    if (typeof j.dna_count     === 'number') S.dnaCount     = j.dna_count;
    // Aggiorna attività regioni legate alla memoria episodica
    lastActivation['hippo_l'] = Date.now();
    lastActivation['hippo_r'] = Date.now();
  } catch { /* silente */ }
}

async function pollSIC() {
  try {
    const r = await fetch('/api/inner_stream/buffer?n=3');
    if (!r.ok) return;
    const j = await r.json();
    const thoughts = j.thoughts || j.buffer || [];
    if (!thoughts.length) return;
    const latestText = thoughts[0].text || thoughts[0].content || '';
    const newHash = _simpleHash(latestText);
    if (newHash !== S.lastThoughtHash) {
      S.lastThoughtHash = newHash;
      _triggerSICReplay(latestText);
    }
  } catch { /* silente */ }
}

// ─── SIC Sharp-Wave Ripple Replay ─────────────────────────────────────────────

function _triggerSICReplay(text) {
  const t = text.toLowerCase();
  const regionKeys = [];

  if (/ricord|episod|memor|passat|ieri/.test(t)) regionKeys.push('hippo_l', 'hippo_r');
  if (/sent|emozion|paur|amore|arrabbia|felice|triste/.test(t)) regionKeys.push('amyg_l', 'amyg_r');
  if (/pens|decid|capis|ragio|problem|analiz|valut/.test(t)) regionKeys.push('pfc');
  if (/\bio\b|\bme\b|\bmia\b|\bsono\b|esist|chi sono|identit/.test(t)) regionKeys.push('dmn');
  if (/sempre|mai|abit|tendenz|solito|pattern/.test(t)) regionKeys.push('basal');

  // Default se nessun pattern
  if (!regionKeys.length) regionKeys.push('pfc', 'dmn');

  // Deduplicazione
  const unique = [...new Set(regionKeys)];

  S.replayQueue.push({
    regionKeys: unique,
    startTime: S.timeSec,
    stagger: 0.20, // 200ms tra regioni
  });
}

function processReplayQueue() {
  const now = S.timeSec;
  for (let qi = S.replayQueue.length - 1; qi >= 0; qi--) {
    const rq = S.replayQueue[qi];
    for (let ri = 0; ri < rq.regionKeys.length; ri++) {
      const activateAt = rq.startTime + ri * rq.stagger;
      if (now >= activateAt && now < activateAt + 0.05) {
        const key = rq.regionKeys[ri];
        // Attiva la regione per 800ms
        S.regionActivity[key] = 1.0;
        lastActivation[key] = Date.now();
        // Rinforzo hebbiano sugli assoni tra regioni attivate
        for (let rj = 0; rj < rq.regionKeys.length; rj++) {
          if (rj === ri) continue;
          const name1 = `${rq.regionKeys[ri]}__${rq.regionKeys[rj]}`;
          const name2 = `${rq.regionKeys[rj]}__${rq.regionKeys[ri]}`;
          if (S.hebbWeights[name1] !== undefined) {
            S.hebbWeights[name1] = Math.min(1.0, S.hebbWeights[name1] + 0.05);
          }
          if (S.hebbWeights[name2] !== undefined) {
            S.hebbWeights[name2] = Math.min(1.0, S.hebbWeights[name2] + 0.05);
          }
        }
        // Spawn pulse da questa regione
        _spawnPulseFromRegion(key, 0.85);
      }
    }
    // Rimuovi item completato
    const endTime = rq.startTime + rq.regionKeys.length * rq.stagger + 0.1;
    if (now > endTime) {
      S.replayQueue.splice(qi, 1);
    }
  }
}

// ─── Biological Invariant 3: Ebbinghaus decay per regione ────────────────────

function computeEbbinhausDecay(regionKey, now) {
  const elapsedMs = now - (lastActivation[regionKey] || now);
  const elapsedDays = elapsedMs / (1000 * 60 * 60 * 24);
  // S(t) = S0 * exp(-t / 30gg), floor a 0.08
  return Math.max(0.08, Math.exp(-elapsedDays / 30));
}

// ─── Biological Invariant 1: Hebbian weights update ──────────────────────────

function updateHebbianWeights() {
  for (const ax of axonData) {
    const actA = S.regionActivity[ax.keyA] || 0;
    const actB = S.regionActivity[ax.keyB] || 0;
    const episodeBoost = clamp01(S.episodeCount / 500);
    const targetWeight = 0.3 + 0.7 * (actA * actB * (0.5 + 0.5 * episodeBoost));
    const current = S.hebbWeights[ax.name] || 0.5;
    // Lerp lento verso target
    S.hebbWeights[ax.name] = current + (targetWeight - current) * 0.003;
    // Ebbinghaus decay sugli assoni non aggiornati
    S.hebbWeights[ax.name] *= 0.9999;
    S.hebbWeights[ax.name] = clamp01(S.hebbWeights[ax.name]);
  }
}

// ─── Calcolo activity regioni ─────────────────────────────────────────────────

function computeRegionActivities() {
  const ra = S.regionActivity;

  // PFC: coherence_budget × (0.5 + 0.5 × curiosity)
  ra.pfc = clamp(S.coherence.cur * (0.5 + 0.5 * S.curiosity.cur), 0.1, 1.0);

  // ACC: grounding_integrity (error monitoring)
  ra.acc = clamp(S.grounding.cur, 0.1, 1.0);

  // HIPPO_L/R: episodeCount boost + certainty
  const hippoBase = 0.2 + 0.6 * clamp01(S.episodeCount / 300) + 0.2 * S.certainty.cur;
  ra.hippo_l = ra.hippo_r = clamp(hippoBase, 0.1, 1.0);

  // AMYGDALA L/R: arousal + |valence| + fear
  const amygAct = clamp(S.arousal.cur * 0.5 + Math.abs(S.valence.cur) * 0.3 + S.fear.cur * 0.2, 0.08, 1.0);
  ra.amyg_l = ra.amyg_r = amygAct;

  // BASAL: dnaCount / 16
  ra.basal = clamp(0.15 + 0.85 * S.dnaCount / 16, 0.1, 1.0);

  // DMN: inner activity proxy
  ra.dmn = clamp(0.4 + S.warmth.cur * 0.3 + S.attachment.cur * 0.3, 0.1, 1.0);

  // BRAINSTEM: somatic (threat + agency)
  ra.brainstem = clamp(S.threat.cur * 0.5 + 0.3 + S.agency.cur * 0.2, 0.1, 1.0);

  // Applica Ebbinghaus decay come floor minimo per regione
  const _now = Date.now();
  for (const key of Object.keys(ra)) {
    const decayFloor = computeEbbinhausDecay(key, _now) * 0.15;
    ra[key] = Math.max(ra[key], decayFloor);
  }
}

// ─── Update nodi ──────────────────────────────────────────────────────────────

function updateNodes(dt) {
  const t = S.timeSec;
  const arousal = effectiveArousal();
  const glitch = Math.max(0, (0.5 - S.grounding.cur) / 0.5);
  const desat  = (1 - S.grounding.cur) * 0.40;

  for (let i = 0; i < nodeData.length; i++) {
    const n = nodeData[i];
    const key = n.regionKey;
    const reg = regions[key];
    const act = S.regionActivity[key] || 0.5;
    const freq = OSC_FREQ[key] || 8;

    // Oscillazione: ampiezza proporzionale all'attività
    const amp = n.jitterAmp * (0.4 + act * 0.8) * (0.6 + arousal * 0.6);
    const ph  = n.jitterPhase;
    const dx  = amp * Math.sin(t * freq + ph);
    const dy  = amp * Math.cos(t * freq * 0.7 + ph * 1.3);
    const dz  = amp * Math.sin(t * freq * 0.5 + ph * 0.9);

    nodePosArr[i*3]   = n.basePos.x + dx;
    nodePosArr[i*3+1] = n.basePos.y + dy;
    nodePosArr[i*3+2] = n.basePos.z + dz;

    // Alpha modulata da activity + safe mode
    let alpha = 0.18 + act * 0.72;
    if (S.safeMode) alpha *= 0.4;
    nodeAlphaArr[i] = alpha;

    // Colore: base + safe mode rosso + desaturazione grounding basso
    let r = reg.color.r, g = reg.color.g, b = reg.color.b;
    if (S.safeMode) {
      r = r * 0.55 + 0.9  * 0.45;
      g = g * 0.55 + 0.15 * 0.45;
      b = b * 0.55 + 0.15 * 0.45;
    }
    if (desat > 0) {
      const luma = r * 0.30 + g * 0.59 + b * 0.11;
      r = r * (1 - desat) + luma * desat;
      g = g * (1 - desat) + luma * desat;
      b = b * (1 - desat) + luma * desat;
    }
    nodeColorArr[i*3]   = r;
    nodeColorArr[i*3+1] = g;
    nodeColorArr[i*3+2] = b;
  }

  nodeGeo.attributes.position.needsUpdate = true;
  nodeGeo.attributes.aColor.needsUpdate   = true;
  nodeGeo.attributes.aAlpha.needsUpdate   = true;
}

// ─── Update assoni ────────────────────────────────────────────────────────────

function updateAxons() {
  const SEGS = AXON_CURVE_SEGMENTS;

  for (let ai = 0; ai < axonData.length; ai++) {
    const ax  = axonData[ai];
    const actA = S.regionActivity[ax.keyA] || 0;
    const actB = S.regionActivity[ax.keyB] || 0;
    const weight = S.hebbWeights[ax.name] || 0.5;

    // Colore NT dinamico
    const ntCol = getNTColor(ax);

    // Alpha combinata: activity * hebbWeight
    const baseAlpha = weight * (actA + actB) * 0.5 * 0.55;

    const vi = ax.startVIdx;
    for (let s = 0; s < SEGS; s++) {
      // Entrambi i vertici del segmento ricevono la stessa alpha
      axonAlphaArr[(vi + s*2)    ] = baseAlpha;
      axonAlphaArr[(vi + s*2 + 1)] = baseAlpha;
    }
  }

  axonGeo.attributes.aAlpha.needsUpdate = true;
}

// ─── Update pulses ────────────────────────────────────────────────────────────

function updatePulses(dt) {
  const col = pulseMesh.instanceColor;

  for (let i = 0; i < PULSE_POOL; i++) {
    const p = pulses[i];
    if (!p.alive) continue;
    p.t += dt;
    const u = p.t / p.life;
    if (u >= 1) {
      p.alive = false;
      _pm4.makeScale(0, 0, 0);
      pulseMesh.setMatrixAt(i, _pm4);
      // Attiva regione di destinazione
      const ax = axonData[p.axonIdx];
      if (ax) {
        S.regionActivity[ax.keyB] = Math.min(1.0, (S.regionActivity[ax.keyB] || 0) + 0.1);
        lastActivation[ax.keyB] = Date.now();
      }
      continue;
    }

    const ax = axonData[p.axonIdx];
    if (!ax) { p.alive = false; continue; }

    // Posizione lungo la curva
    const eased = u * u * (3 - 2 * u);
    const pts = ax.curvePoints;
    const segF = eased * (pts.length - 1);
    const segI = Math.min(Math.floor(segF), pts.length - 2);
    const segT = segF - segI;
    _ptmpA.copy(pts[segI]);
    _ptmpB.copy(pts[segI + 1]);
    _ptmpA.lerp(_ptmpB, segT);

    const env = Math.sin(u * Math.PI);
    const s   = env * (0.55 + p.intensity * 0.65);
    _psc.set(s, s, s);
    _pm4.compose(_ptmpA, _tmpQuat, _psc);
    pulseMesh.setMatrixAt(i, _pm4);

    const ci = i * 3;
    const boost = env * env * 0.65;
    col.array[ci]   = Math.min(2, p.color.r * (1 + boost));
    col.array[ci+1] = Math.min(2, p.color.g * (1 + boost));
    col.array[ci+2] = Math.min(2, p.color.b * (1 + boost));
  }

  pulseMesh.instanceMatrix.needsUpdate = true;
  pulseMesh.instanceColor.needsUpdate  = true;
}

// ─── Spawn pulse periodici ────────────────────────────────────────────────────

function spawnPulsesByRate(dt) {
  const arousal = effectiveArousal();
  const rate = 0.6 + 7.0 * arousal;
  _pulseAccum += rate * dt;
  while (_pulseAccum >= 1) {
    _pulseAccum -= 1;
    const ai = (Math.random() * axonData.length) | 0;
    const ax = axonData[ai];
    _spawnPulse(ai, 0.4 + arousal * 0.5, getNTColor(ax));
  }
  if (S.speaking && Math.random() < dt * 4.5) {
    _spawnPulseFromRegion('dmn', 0.85);
  }
}

// ─── Update halo + ambient + camera ──────────────────────────────────────────

function updateHalo() {
  const meanAct = _meanActivity();
  haloMat.uniforms.uColor.value.setRGB(
    0.08 + S.warmth.cur * 0.15,
    0.25 + S.attachment.cur * 0.12,
    0.50 + S.coherence.cur * 0.22,
  );
  haloMat.uniforms.uIntensity.value = 0.18 + meanAct * 0.22 + (S.proactiveT > 0 ? 0.18 : 0);
  haloMat.uniforms.uSafeMode.value  = S.safeMode ? 1 : 0;
  haloMat.uniforms.uTime.value      = S.timeSec;
  if (halo) halo.lookAt(camera.position);
}

function updateAmbient() {
  ambientMat.uniforms.uTime.value      = S.timeSec;
  ambientMat.uniforms.uIntensity.value = 0.12 + S.curiosity.cur * 0.16;
  ambientMat.uniforms.uColor.value.setRGB(
    0.35 + S.arousal.cur * 0.20,
    0.55 + S.certainty.cur * 0.15,
    0.78 + S.coherence.cur * 0.12,
  );
}

function updateCamera(dt) {
  _cameraDrift += dt * (0.05 + S.agency.cur * 0.07);
  const t = _cameraDrift;

  const baseR = 7.0;
  const dolly = Math.sin(t * 0.17) * 0.40 + (1 - S.curiosity.cur) * 0.18;
  const r     = baseR + dolly;

  const yawAmp = 0.42 + S.agency.cur * 0.30;
  const pitAmp = 0.28 + S.agency.cur * 0.15;
  const yaw    = Math.sin(t * 0.35) * yawAmp + Math.sin(t * 0.12) * yawAmp * 0.38;
  const pit    = Math.cos(t * 0.27) * pitAmp + Math.cos(t * 0.10) * pitAmp * 0.30;

  camera.position.set(
    Math.sin(yaw) * Math.cos(pit) * r,
    Math.sin(pit) * r,
    Math.cos(yaw) * Math.cos(pit) * r,
  );
  camera.lookAt(0, 0, 0);

  const newFov = 54 + Math.sin(t * 0.21) * 0.70;
  if (Math.abs(newFov - camera.fov) > 0.005) {
    camera.fov = newFov;
    camera.updateProjectionMatrix();
  }

  // Shake se threat alto
  if (S.threat.cur > 0.05) {
    root.position.x = (Math.random() - 0.5) * 0.016 * S.threat.cur;
    root.position.y = (Math.random() - 0.5) * 0.016 * S.threat.cur;
  } else {
    root.position.x = Math.sin(t * 0.39) * 0.045;
    root.position.y = Math.cos(t * 0.31) * 0.038;
  }
  root.rotation.y += dt * (0.055 + S.curiosity.cur * 0.055);
  root.rotation.x  = Math.sin(t * 0.18) * (0.24 + S.warmth.cur * 0.10);
  root.rotation.z  = Math.sin(t * 0.10) * 0.10 + Math.sin(t * 0.065) * 0.045;
}

function updateGlobalUniforms() {
  nodeMat.uniforms.uTime.value   = S.timeSec;
  nodeMat.uniforms.uGlitch.value = Math.max(0, (0.5 - S.coherence.cur) / 0.5);
  nodeMat.uniforms.uBlur.value   = Math.max(0, (0.5 - S.grounding.cur) / 0.5);
  axonMat.uniforms.uTime.value   = S.timeSec;
  axonMat.uniforms.uOpacity.value= 0.80 + S.grounding.cur * 0.20;
}

// ─── Main loop ────────────────────────────────────────────────────────────────

function loop() {
  requestAnimationFrame(loop);
  const dt = Math.min(0.05, clock.getDelta());
  S.timeSec += dt;

  // Lerp tutti i valori S verso i target
  lerpAll(dt);

  // Timer transienti
  if (S.proactiveT > 0) S.proactiveT = Math.max(0, S.proactiveT - dt);
  if (S.innerSpark  > 0) S.innerSpark  = Math.max(0, S.innerSpark  - dt);
  S.wave.age += dt;
  if (S.wave.age > 1.8) S.wave.active = 0;

  // Background life: spark occasionale in dmn (inner stream)
  if (Math.random() < dt * 0.35) S.innerSpark = Math.max(S.innerSpark, 0.5);
  if (S.innerSpark > 0 && Math.random() < dt * 2.5) {
    _spawnPulseFromRegion('dmn', 0.6);
  }

  // Ricalcola attività regioni
  computeRegionActivities();

  // Replay SIC in coda
  processReplayQueue();

  // Aggiorna pesi Hebb
  updateHebbianWeights();

  // Update visivi
  updateNodes(dt);
  updateAxons();
  spawnPulsesByRate(dt);
  updatePulses(dt);
  updateHalo();
  updateAmbient();
  updateCamera(dt);
  updateGlobalUniforms();

  renderer.render(scene, camera);
}

// ─── Lerp di tutti gli stati ──────────────────────────────────────────────────

function lerpAll(dt) {
  const fast = 1 - Math.pow(1 - 0.045, dt * 60);
  const slow = 1 - Math.pow(1 - 0.012, dt * 60);
  // affective: veloce
  for (const k of ['valence','arousal','certainty','attachment','agency','threat']) {
    S[k].cur += (S[k].tgt - S[k].cur) * fast;
  }
  // traits: lenti (sono identità)
  for (const k of ['curiosity','trust','fear','warmth']) {
    S[k].cur += (S[k].tgt - S[k].cur) * slow;
  }
  // homeostatic: veloce
  for (const k of ['coherence','grounding','selfModel']) {
    S[k].cur += (S[k].tgt - S[k].cur) * fast;
  }
}

function effectiveArousal() {
  let a = S.arousal.cur;
  if (S.thinking)    a = Math.min(1, a + 0.18);
  if (S.speaking)    a = Math.min(1, a + 0.25);
  if (S.proactiveT > 0) a = Math.min(1, a + 0.28);
  if (S.threat.cur > 0) a = Math.min(1, a + S.threat.cur * 0.35);
  return a;
}

// ─── LivePortrait cleanup ─────────────────────────────────────────────────────

function _cleanupLivePortrait() {
  const fs = document.getElementById('face-stream');
  if (fs) {
    try { fs.onload = null; fs.onerror = null; fs.src = ''; fs.parentNode?.removeChild(fs); } catch {}
  }
  const fsb = document.getElementById('face-status-bar');
  if (fsb) { try { fsb.parentNode?.removeChild(fsb); } catch {} }
  try {
    const mo = new MutationObserver(muts => {
      for (const m of muts) for (const n of m.addedNodes) {
        if (n?.id === 'face-stream' || n?.id === 'face-status-bar') {
          try { n.parentNode?.removeChild(n); } catch {}
        }
      }
    });
    mo.observe(document.body, { childList: true, subtree: true });
  } catch {}
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function clamp01(v)       { return Math.max(0, Math.min(1, v)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function _meanActivity() {
  const vals = Object.values(S.regionActivity);
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0.5;
}

function _simpleHash(str) {
  let h = 0;
  for (let i = 0; i < Math.min(str.length, 80); i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  }
  return h.toString(16);
}
