// script.js â€” Logica frontend Eden

const API_BASE = window.location.origin;

// â”€â”€â”€ Riferimenti DOM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const messagesEl      = document.getElementById('messages');
const typingIndicator = document.getElementById('typing-indicator');
const inputEl         = document.getElementById('input');
const sendBtn         = document.getElementById('btn-send');
const inputForm       = document.getElementById('input-form');
const statusDot       = document.getElementById('status-dot');
const relBadge        = document.getElementById('relationship-badge');
const newSessionBtn   = document.getElementById('btn-new-session');
const fileInput       = document.getElementById('file-input');
const btnAttach       = document.getElementById('btn-attach');
const filePreview     = document.getElementById('file-preview');
const filePreviewName = document.getElementById('file-preview-name');
const fileRemoveBtn   = document.getElementById('file-remove');

const elExchange = document.getElementById('stat-exchange');
const elSessions = document.getElementById('stat-sessions');
const elHistory  = document.getElementById('stat-history');
const elMemories = document.getElementById('stat-memories');

// Autonomia
const elAutoRunning   = document.getElementById('auto-running');
const elAutoMode      = document.getElementById('auto-mode');
const elAutoInterval  = document.getElementById('auto-interval');
const elAutoDecisions = document.getElementById('auto-decisions');
const btnAutoRefresh  = document.getElementById('btn-auto-refresh');
const btnAutoTick     = document.getElementById('btn-auto-tick');

// â”€â”€â”€ Stato locale â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

let sseSource       = null;   // EventSource SSE
let pollInterval    = null;   // fallback polling handle
let sseConnected    = false;  // SSE funzionante?
let notifPermesso   = false;  // Web Notification autorizzata?

// TTS — Kokoro audio output
let _ttsEnabled      = false;
let _ttsAvailable    = false;
let _audioCtx        = null;
let _gainNode        = null;   // GainNode per controllo volume
let _currentSource   = null;   // AudioBufferSourceNode in riproduzione
let _ttsCancelToken  = 0;      // incrementato per annullare playback precedente
let _lastTraits      = { curiosity: 5, trust: 5, cynicism: 5, warmth: 5, fear: 5 };

// Stato file allegato corrente
let _allegato = null;  // { name, text } oppure { name, b64 } oppure null

// â”€â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function ora() {
  return new Date().toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' });
}

function scrollDown() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

function _clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function _analyzeChatSignal(text) {
  const raw = String(text ?? '').trim();
  if (!raw) {
    return { valence: 0, energy: 0, question: false };
  }

  const s = raw.toLowerCase();
  const pos = [
    'bene', 'ottimo', 'grazie', 'felice', 'bravo', 'bella', 'perfetto', 'amore',
    'fantastico', 'sorriso', 'calmo', 'sereno', 'fiducia'
  ];
  const neg = [
    'male', 'odio', 'paura', 'ansia', 'triste', 'rabbia', 'arrabbi', 'nervos',
    'problema', 'minaccia', 'errore', 'non va', 'stress'
  ];

  let posHit = 0;
  let negHit = 0;
  for (const w of pos) { if (s.includes(w)) posHit++; }
  for (const w of neg) { if (s.includes(w)) negHit++; }

  const punct = (raw.match(/[!?]/g) || []).length;
  const upper = (raw.match(/[A-Z]/g) || []).length;
  const letters = (raw.match(/[A-Za-zÀ-ÖØ-öø-ÿ]/g) || []).length;
  const upperRatio = letters > 0 ? upper / letters : 0;
  const question = raw.includes('?');

  const valence = Math.max(-1, Math.min(1, (posHit - negHit) / 3));
  const energy = _clamp01((punct * 0.16) + (upperRatio * 1.5) + Math.min(0.35, raw.length / 420));
  return { valence, energy, question };
}

function _dispatchAvatarChat(role, text) {
  const sig = _analyzeChatSignal(text);
  window.dispatchEvent(new CustomEvent('eden:chat', {
    detail: {
      role,
      valence: sig.valence,
      energy: sig.energy,
      question: sig.question,
      textLength: String(text ?? '').length
    }
  }));
}

// â”€â”€â”€ Messaggi chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/**
 * Aggiunge un messaggio alla chat.
 * role: 'user' | 'assistant' | 'proactive' | 'error' | 'system'
 * exchangeId: opzionale, se presente abilita pannellino rating umano (solo per 'assistant')
 */
function appendMessage(role, content, timestamp, exchangeId) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  if (role === 'system') {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;
    wrapper.appendChild(bubble);

  } else {
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent =
      role === 'user'      ? 'tu'       :
      role === 'proactive' ? 'eden -> tu' :
      role === 'error'     ? 'sistema'  : 'eden';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = timestamp || ora();

    wrapper.append(label, bubble, time);

    // Pulsante play/replay per messaggi di Eden
    if (role === 'assistant' || role === 'proactive') {
      const playBtn = document.createElement('button');
      playBtn.className = 'replay-btn';
      playBtn.title = 'Riproduci voce';
      playBtn.textContent = '▶';
      if (role === 'assistant' && exchangeId) {
        playBtn.addEventListener('click', () => _playExchangeAudio(exchangeId, content, playBtn, true));
      } else {
        playBtn.addEventListener('click', () => _playTTSDirect(content));
      }
      wrapper.appendChild(playBtn);
    }

    // Pulsante copia messaggio
    if (role !== 'error') {
      const copyBtn = document.createElement('button');
      copyBtn.className = 'msg-copy-btn';
      copyBtn.innerHTML = '⎘';
      copyBtn.title = 'Copia messaggio';
      copyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const text = (bubble.innerText || bubble.textContent || '').trim();
        const done = () => {
          copyBtn.innerHTML = '✓';
          copyBtn.style.color = 'var(--accent)';
          setTimeout(() => { copyBtn.innerHTML = '⎘'; copyBtn.style.color = ''; }, 1200);
        };
        if (navigator.clipboard && window.isSecureContext) {
          navigator.clipboard.writeText(text).then(done).catch(() => _fallbackCopy(text, done));
        } else {
          _fallbackCopy(text, done);
        }
      });
      wrapper.appendChild(copyBtn);
    }

  }
  messagesEl.appendChild(wrapper);
  scrollDown();

  if (role === 'user' || role === 'assistant' || role === 'proactive') {
    _dispatchAvatarChat(role, content);
  }

  return wrapper;
}

// â”€â”€â”€ Messaggi proattivi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/**
 * Renderizza un messaggio proattivo di Eden con stile ambra.
 * @param {object} evento â€” {text, timestamp, trigger}
 */
function appendProactiveMessage(evento) {
  const wrapper = appendMessage('proactive', evento.text, evento.timestamp || ora());
  _playTTS(evento.text);

  // Pulse sul bubble per richiamare l'attenzione
  const bubble = wrapper.querySelector('.bubble');
  bubble.classList.add('pulse');
  setTimeout(() => bubble.classList.remove('pulse'), 4000);

  // Attiva animazione "proattivo" sul volto Three.js
  window.dispatchEvent(new CustomEvent('eden:state', { detail: { proactive: true } }));

  // Web Notification se la tab Ã¨ in background
  if (document.hidden && notifPermesso) {
    mostraNotifica(evento.text);
  }
}

// â”€â”€â”€ Web Notification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async function richiediNotifiche() {
  if (!('Notification' in window)) return;
  if (Notification.permission === 'granted') {
    notifPermesso = true;
    return;
  }
  if (Notification.permission === 'default') {
    const perm = await Notification.requestPermission();
    notifPermesso = (perm === 'granted');
  }
}

function mostraNotifica(testo) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  try {
    const n = new Notification('EDEN', {
      body: testo,
      tag:  'eden-proactive',   // sostituisce notifica precedente se ancora aperta
      silent: false
    });
    // Auto-chiude dopo 8 secondi
    setTimeout(() => n.close(), 8000);
  } catch {
    // Alcune build browser bloccano Notification in HTTP â€” ignora
  }
}

// â”€â”€â”€ SSE â€” Server-Sent Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function connectSSE() {
  if (sseSource) {
    sseSource.close();
    sseSource = null;
  }

  sseSource = new EventSource(`${API_BASE}/api/stream`);

  sseSource.onopen = () => {
    sseConnected = true;
    // Se il polling era attivo, fermalo: SSE funziona
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  };

  sseSource.onmessage = e => {
    let dati;
    try { dati = JSON.parse(e.data); } catch { return; }

    if (dati.type === 'connected') {
      sseConnected = true;
    } else if (dati.type === 'proactive') {
      appendProactiveMessage(dati);
    }
  };

  sseSource.onerror = () => {
    // SSE ha perso la connessione â€” avvia il fallback
    sseConnected = false;
    sseSource.close();
    sseSource = null;
    avviaFallbackPolling();

    // Riprova la connessione SSE dopo 30 secondi
    setTimeout(connectSSE, 30000);
  };
}

// â”€â”€â”€ Fallback polling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function avviaFallbackPolling() {
  if (pollInterval) return;   // giÃ  attivo
  pollInterval = setInterval(async () => {
    if (sseConnected) {
      // SSE tornato disponibile â€” ferma il polling
      clearInterval(pollInterval);
      pollInterval = null;
      return;
    }
    try {
      const res  = await fetch(`${API_BASE}/api/pending`);
      const msgs = await res.json();
      if (Array.isArray(msgs) && msgs.length > 0) {
        msgs.forEach(appendProactiveMessage);
      }
    } catch { /* server non raggiungibile â€” riprova al prossimo tick */ }
  }, 30000);
}

// â”€â”€â”€ Typing indicator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function showTyping() {
  typingIndicator.classList.add('visible');
  scrollDown();
  window.dispatchEvent(new CustomEvent('eden:state', { detail: { thinking: true } }));
}
function hideTyping() {
  typingIndicator.classList.remove('visible');
  window.dispatchEvent(new CustomEvent('eden:state', { detail: { thinking: false } }));
}

// â”€â”€â”€ Tratti psicologici â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const TRAIT_NAMES = ['curiosity', 'trust', 'cynicism', 'warmth', 'fear'];

function updateTrait(name, value) {
  const fill  = document.querySelector(`#trait-${name} .trait-fill`);
  const label = document.querySelector(`#trait-${name} .trait-value`);
  if (!fill || !label) return;
  const prev = parseFloat(label.textContent);
  fill.style.width = `${value * 10}%`;
  if (Math.abs(value - prev) >= 0.1) {
    label.textContent = value.toFixed(1);
    label.classList.remove('changed');
    void label.offsetWidth;
    label.classList.add('changed');
    setTimeout(() => label.classList.remove('changed'), 600);
  }
}

function updateTraits(traits) {
  TRAIT_NAMES.forEach(name => {
    if (traits[name] !== undefined) updateTrait(name, traits[name]);
  });
  // Aggiorna snapshot tratti per modulazione TTS
  Object.assign(_lastTraits, traits);
  // Notifica il volto Three.js del nuovo stato emotivo
  window.dispatchEvent(new CustomEvent('eden:traits', { detail: traits }));
}

// â”€â”€â”€ Badge relazione â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const REL_LABELS = {
  ally:         'alleato',
  acquaintance: 'conoscente',
  stranger:     'sconosciuto',
  hostile:      'ostile'
};

function updateRelationship(rel) {
  relBadge.className  = rel;
  relBadge.textContent = REL_LABELS[rel] ?? rel;
}

function updateAffectiveState(af) {
  if (!af) return;
  const elV = document.getElementById('af-valence');
  const elA = document.getElementById('af-arousal');
  const elC = document.getElementById('af-certainty');
  if (elV) elV.textContent = (af.valence  ?? 0).toFixed(2);
  if (elA) elA.textContent = (af.arousal  ?? 0).toFixed(2);
  if (elC) elC.textContent = (af.certainty ?? 0).toFixed(2);
}

// â”€â”€â”€ Contatori sessione â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function updateStats(data) {
  if (elExchange && data.exchange_count !== undefined)
    elExchange.textContent = data.exchange_count;
  if (elSessions && data.session_count !== undefined)
    elSessions.textContent = data.session_count;
  if (elHistory && data.history_length !== undefined)
    elHistory.textContent = data.history_length;
  if (elMemories && data.key_memories_count !== undefined)
    elMemories.textContent = data.key_memories_count;
  if (elAutoDecisions && data.decision_log_count !== undefined)
    elAutoDecisions.textContent = data.decision_log_count;
}

function updateAutonomy(status) {
  if (!status) return;

  if (elAutoRunning) {
    const running = !!status.running;
    elAutoRunning.textContent = running ? 'ON' : 'OFF';
    elAutoRunning.classList.toggle('on', running);
    elAutoRunning.classList.toggle('off', !running);
  }

  if (elAutoMode) {
    const shadow = !!status.shadow_mode;
    elAutoMode.textContent = shadow ? 'SHADOW' : 'LIVE';
    elAutoMode.classList.toggle('shadow', shadow);
    elAutoMode.classList.toggle('live', !shadow);
  }

  if (elAutoInterval && status.interval_seconds !== undefined) {
    elAutoInterval.textContent = `${status.interval_seconds}s`;
  }
}

async function refreshAutonomyPanel() {
  try {
    const res = await fetch(`${API_BASE}/api/autonomy/status`);
    const data = await res.json();
    updateAutonomy(data);
  } catch {
    if (elAutoRunning) {
      elAutoRunning.textContent = 'ERR';
      elAutoRunning.classList.remove('on');
      elAutoRunning.classList.add('off');
    }
  }
}

async function forceAutonomyTick() {
  if (!btnAutoTick) return;
  const label = btnAutoTick.textContent;
  btnAutoTick.disabled = true;
  btnAutoTick.textContent = '...';
  try {
    await fetch(`${API_BASE}/api/autonomy/tick`, { method: 'POST' });
    await fetchStatus();
    await refreshAutonomyPanel();
  } catch {
    appendMessage('error', 'Tick autonomia non riuscito.');
  } finally {
    btnAutoTick.disabled = false;
    btnAutoTick.textContent = label;
  }
}

// â”€â”€â”€ Stato connessione â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function setOnline()  { statusDot.className = 'online'; }
function setError()   { statusDot.className = 'error'; }
function setLoading() { statusDot.className = ''; }

// â”€â”€â”€ Blocco/sblocco input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function lockInput()   { inputEl.disabled = true;  sendBtn.disabled = true; }
function unlockInput() { inputEl.disabled = false; sendBtn.disabled = false; if (_activeSection === 'chat') inputEl.focus(); }

// â”€â”€â”€ Chiamate API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async function fetchStatus() {
  try {
    const res  = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    updateTraits(data.traits);
    updateRelationship(data.relationship);
    updateStats(data);
    updateAffectiveState(data.affective_state);
    updateAutonomy(data.autonomy);
    const _mdl = document.getElementById('dhero-model');
    if (_mdl && data.model) _mdl.textContent = data.model;
    setOnline();
  } catch {
    setError();
    appendMessage('system', '- impossibile raggiungere Eden -');
  }
}

async function sendMessage(text, allegato) {
  lockInput();
  setLoading();
  showTyping();

  // Costruisce il payload includendo l'eventuale allegato
  const payload = { message: text };
  if (allegato) {
    payload.file_name = allegato.name;
    if (allegato.text !== undefined) payload.file_text = allegato.text;
    if (allegato.b64  !== undefined) payload.file_b64  = allegato.b64;
  }

  try {
    const res  = await fetch(`${API_BASE}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();
    hideTyping();

    if (!res.ok || data.errore) {
      appendMessage('error', data.errore ?? 'Errore sconosciuto.');
      setError();
      return;
    }

    const _msgWrapper = appendMessage('assistant', data.reply, undefined, data.exchange_id);
    // Auto-play via exchange pre-sintetizzato (istantaneo); fallback a sentence-by-sentence
    if (data.exchange_id) {
      const _playBtn = _msgWrapper?.querySelector('.replay-btn');
      _playExchangeAudio(data.exchange_id, data.reply, _playBtn);
    } else {
      _playTTS(data.reply);
    }
    // EV-030 — Layer 1 reflection trace (canale separato, non va in memoria)
    if (data.reflection && _msgWrapper) {
      const _reflEl = document.createElement('div');
      _reflEl.className = 'layer1-reflection';
      _reflEl.textContent = data.reflection;
      _msgWrapper.appendChild(_reflEl);
    }
    updateTraits(data.traits);
    updateRelationship(data.relationship);
    updateAffectiveState(data.affective_state);
    if (elExchange) elExchange.textContent = data.exchange_count;
    if (elHistory)  elHistory.textContent  = parseInt(elHistory.textContent || '0') + 2;
    setOnline();

  } catch {
    hideTyping();
    appendMessage('error',
      'Ollama non risponde. Assicurati che "ollama serve" sia in esecuzione.'
    );
    setError();
  } finally {
    unlockInput();
  }
}

async function startNewSession() {
  try {
    const res  = await fetch(`${API_BASE}/api/new_session`, { method: 'POST' });
    const data = await res.json();
    if (elSessions) elSessions.textContent = data.session_count;
    appendMessage('system', `- sessione ${data.session_count} -`);
  } catch {
    appendMessage('error', 'Impossibile avviare nuova sessione.');
  }
}

async function resetMemoria() {
  try {
    await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
    messagesEl.innerHTML = '';
    TRAIT_NAMES.forEach(name => updateTrait(name, 5));
    updateRelationship('stranger');
    updateStats({ exchange_count: 0, session_count: 0,
                  history_length: 0, key_memories_count: 0 });
    appendMessage('system', '- memoria cancellata -');
    await startNewSession();
    setOnline();
  } catch {
    appendMessage('error', 'Impossibile cancellare la memoria.');
    setError();
  }
}

// â”€â”€â”€ Event listeners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Textarea: auto-resize in base al contenuto
function _resizeTextarea() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
}
inputEl.addEventListener('input', _resizeTextarea);

// Enter invia, Shift+Enter va a capo
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    inputForm.dispatchEvent(new Event('submit'));
  }
});

// Gestione submit
inputForm.addEventListener('submit', e => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text && !_allegato) return;

  // Mostra il messaggio con eventuale tag file
  const label = _allegato ? `${text}\n[allegato: ${_allegato.name}]` : text;
  if (text || _allegato) appendMessage('user', label || `[allegato: ${_allegato.name}]`);

  const allegatoCorrente = _allegato;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  _rimuoviAllegato();

  sendMessage(text, allegatoCorrente);
  richiediNotifiche();
});

// â”€â”€â”€ Gestione allegati â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/** Estensioni trattate come testo puro (lette con FileReader.readAsText) */
const _EXT_TESTO = new Set([
  'txt','md','py','js','ts','jsx','tsx','json','csv','yaml','yml',
  'html','htm','css','xml','log','sh','bat','ini','toml','rs','go',
  'java','c','cpp','h','hpp','sql','r','rb','php','swift','kt',
]);

function _estensione(name) {
  return (name.split('.').pop() || '').toLowerCase();
}

function _mostraPreviewFile(name) {
  filePreviewName.textContent = name;
  filePreview.classList.remove('hidden');
  btnAttach.classList.add('has-file');
}

function _rimuoviAllegato() {
  _allegato = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  filePreviewName.textContent = '';
  btnAttach.classList.remove('has-file');
}

btnAttach.addEventListener('click', () => fileInput.click());

fileRemoveBtn.addEventListener('click', () => _rimuoviAllegato());

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  const ext = _estensione(file.name);
  const reader = new FileReader();

  if (_EXT_TESTO.has(ext)) {
    // File di testo: leggi come stringa
    reader.onload = ev => {
      _allegato = { name: file.name, text: ev.target.result };
      _mostraPreviewFile(file.name);
    };
    reader.readAsText(file, 'UTF-8');
  } else {
    // File binario (PDF, DOCX, ecc.): invia come base64 al backend
    reader.onload = ev => {
      // DataURL = "data:mime;base64,XXXXX" â€” teniamo solo la parte base64
      const b64 = ev.target.result.split(',')[1];
      _allegato = { name: file.name, b64 };
      _mostraPreviewFile(file.name);
    };
    reader.readAsDataURL(file);
  }

  reader.onerror = () => {
    appendMessage('error', `Impossibile leggere il file: ${file.name}`);
    _rimuoviAllegato();
  };
});

newSessionBtn.addEventListener('click', () => startNewSession());
if (btnAutoRefresh) btnAutoRefresh.addEventListener('click', () => refreshAutonomyPanel());
if (btnAutoTick) btnAutoTick.addEventListener('click', () => forceAutonomyTick());

// ─── Pioneer feature toggles ─────────────────────────────────────────────────
async function loadPioneerStatus() {
  try {
    const res  = await fetch(`${API_BASE}/api/pioneer/status`);
    const data = await res.json();
    ['dialogo_interno', 'user_model', 'consolidamento'].forEach(feat => {
      _updatePioneerBtn(feat, data[feat]);
    });
  } catch {}
}

function _updatePioneerBtn(feature, active) {
  const btn = document.getElementById(`pioneer-btn-${feature}`);
  if (!btn) return;
  btn.textContent = active ? 'ON' : 'OFF';
  btn.classList.toggle('on',  !!active);
  btn.classList.toggle('off', !active);
}

async function togglePioneerFeature(feature) {
  const btn = document.getElementById(`pioneer-btn-${feature}`);
  if (btn) { btn.disabled = true; btn.textContent = '...'; }
  try {
    const res  = await fetch(`${API_BASE}/api/pioneer/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature }),
    });
    const data = await res.json();
    if (data.ok) _updatePioneerBtn(feature, data.active);
    else if (btn) { btn.disabled = false; _updatePioneerBtn(feature, btn.classList.contains('on')); }
  } catch {
    if (btn) btn.disabled = false;
  } finally {
    if (btn) btn.disabled = false;
  }
}

document.querySelectorAll('.pioneer-btn[data-feature]').forEach(btn => {
  btn.classList.add('off');
  btn.addEventListener('click', () => togglePioneerFeature(btn.dataset.feature));
});
// ─── SIC — Pensieri recenti —————————————————————————————————
async function refreshSICThoughts() {
  const el = document.getElementById('sic-thoughts');
  if (!el) return;
  try {
    const res = await fetch(API_BASE + '/api/inner_stream/buffer?n=3');
    if (!res.ok) return;
    const thoughts = await res.json();
    if (!Array.isArray(thoughts) || thoughts.length === 0) {
      el.innerHTML = '<span class="sic-empty">nessun pensiero recente</span>';
      return;
    }
    el.innerHTML = thoughts.slice(0, 3).map(t => {
      const text = t.text || t.content || String(t);
      const ts   = t.ts || t.timestamp || '';
      const time = ts ? new Date(ts).toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' }) : '';
      const timeHtml = time ? '<div class="sic-thought-time">' + time + '</div>' : '';
      return '<div class="sic-thought"><div class="sic-thought-text">' + escapeHtml(text.trim()) + '</div>' + timeHtml + '</div>';
    }).join('');
  } catch {}
}

document.getElementById('btn-sic-refresh')?.addEventListener('click', refreshSICThoughts);

// â”€â”€â”€ Webcam â€” Phase 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const webcamVideo  = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamDot    = document.getElementById('webcam-dot');
const webcamLabel  = document.getElementById('webcam-label');
const webcamBtn    = document.getElementById('webcam-btn');

let _webcamStream   = null;
let _visionInterval = null;
let _visionBusy     = false;   // evita richieste sovrapposte se il modello Ã¨ lento

function _setWebcamStatus(state, label) {
  webcamDot.className    = state;   // '' | 'active' | 'sending' | 'error'
  webcamLabel.textContent = label;
}

async function initWebcam() {
  if (_webcamStream) return;
  try {
    _webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' },
      audio: false
    });
    webcamVideo.srcObject     = _webcamStream;
    webcamVideo.style.display = 'block';
    webcamBtn.classList.add('hidden');
    _setWebcamStatus('active', 'CAM');

    // Notifica user_presence.js che la webcam è pronta
    window.dispatchEvent(new CustomEvent('eden:webcam:start', { detail: { video: webcamVideo } }));

    // Primo frame dopo 1s (attende che il video si stabilizzi)
    setTimeout(() => {
      _inviaFrameVision();
      _visionInterval = setInterval(_inviaFrameVision, 2000);
    }, 1000);

  } catch {
    _setWebcamStatus('error', 'NEGATA');
    webcamBtn.textContent = 'RIPROVA';
  }
}

async function _inviaFrameVision() {
  if (!_webcamStream || _visionBusy) return;
  if (webcamVideo.readyState < 2) return;   // video non ancora pronto

  _visionBusy = true;
  _setWebcamStatus('sending', 'CAM');

  try {
    // Cattura frame su canvas nascosto â€” rispecchiato come il video
    webcamCanvas.width  = 320;
    webcamCanvas.height = 240;
    const ctx = webcamCanvas.getContext('2d');
    ctx.save();
    ctx.translate(320, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcamVideo, 0, 0, 320, 240);
    ctx.restore();

    // Base64 JPEG (qualitÃ  0.65 â€” sufficiente per il modello vision)
    const dataUrl = webcamCanvas.toDataURL('image/jpeg', 0.65);
    const base64  = dataUrl.split(',')[1];

    await fetch(`${API_BASE}/api/vision`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ image: base64 })
    });

    _setWebcamStatus('active', 'CAM');

  } catch {
    _setWebcamStatus('active', 'CAM');   // silenzioso â€” riprova al prossimo tick
  } finally {
    _visionBusy = false;
  }
}

webcamBtn.addEventListener('click', initWebcam);

// â”€â”€â”€ Microfono â€” Phase 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const micBtn = document.getElementById('btn-mic');
const btnTTS = document.getElementById('btn-tts');

let _micStream      = null;
let _mediaRecorder  = null;
let _audioChunks    = [];
let _isRecording    = false;

/** Richiede accesso al microfono. Restituisce true se concesso. */
async function _initMic() {
  if (_micStream) return true;

  // Su mobile via Tailscale: getUserMedia richiede HTTPS (secure context).
  if (!window.isSecureContext) {
    appendMessage('error',
      'Microfono non disponibile: la pagina non è servita via HTTPS. ' +
      'Riavvia Eden con il flag --https e accedi tramite https://');
    return false;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    appendMessage('error', 'Questo browser non supporta l\'accesso al microfono.');
    return false;
  }

  try {
    _micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    return true;
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Accesso al microfono negato — controlla i permessi del browser.'
      : `Microfono non disponibile: ${err.message}`;
    appendMessage('error', msg);
    return false;
  }
}

/** Avvia la registrazione. */
async function _startRecording() {
  if (!await _initMic()) {
    return;  // _initMic mostra già il messaggio di errore
  }

  _audioChunks   = [];
  _mediaRecorder = new MediaRecorder(_micStream, { mimeType: 'audio/webm' });

  _mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) _audioChunks.push(e.data);
  };

  _mediaRecorder.onstop = _inviaAudioTranscribe;

  _mediaRecorder.start(100);   // raccoglie chunk ogni 100ms
  _isRecording = true;
  micBtn.classList.add('recording');
  micBtn.textContent = 'STOP';
}

/** Ferma la registrazione (onstop chiama _inviaAudioTranscribe). */
function _stopRecording() {
  if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
    _mediaRecorder.stop();
  }
  _isRecording = false;
  micBtn.classList.remove('recording');
  micBtn.textContent = 'MIC';
}

/** Invia l'audio al backend, trascrive e manda il testo a Eden. */
async function _inviaAudioTranscribe() {
  if (_audioChunks.length === 0) return;

  const blob     = new Blob(_audioChunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append('audio', blob, 'audio.webm');

  micBtn.textContent = '...';
  micBtn.disabled    = true;

  try {
    const res  = await fetch(`${API_BASE}/api/transcribe`, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.ok && data.text) {
      appendMessage('user', data.text);
      sendMessage(data.text);
    } else if (!data.ok) {
      appendMessage('error', `Trascrizione fallita: ${data.errore ?? 'errore sconosciuto'}`);
    }
    // Se text Ã¨ vuoto (solo silenzio) non fa nulla â€” normale

  } catch {
    appendMessage('error', 'Trascrizione non disponibile - verifica che il server sia attivo.');
  } finally {
    micBtn.textContent = 'MIC';
    micBtn.disabled    = false;
  }
}

micBtn.addEventListener('click', () => {
  if (_isRecording) {
    _stopRecording();
  } else {
    _startRecording();
  }
});

// ─── TTS — Kokoro audio output (Phase 1.7b) ──────────────────────────────────

/** Controlla se Kokoro è disponibile sul server e mostra/nasconde il pulsante. */
async function _checkTTSAvailable() {
  try {
    const res  = await fetch(`${API_BASE}/api/tts_status`);
    const data = await res.json();
    _ttsAvailable = !!data.available;
  } catch {
    _ttsAvailable = false;
  }
  if (btnTTS) btnTTS.style.display = _ttsAvailable ? '' : 'none';
}

/** Abilita/disabilita TTS al click. */
function _toggleTTS() {
  if (!_ttsAvailable) return;
  _ttsEnabled = !_ttsEnabled;
  btnTTS.classList.toggle('active', _ttsEnabled);
  btnTTS.textContent = _ttsEnabled ? 'AUDIO ON' : 'AUDIO';
  // Se si disabilita mentre audio è in riproduzione, fermalo
  if (!_ttsEnabled) {
    _ttsCancelToken++;   // annulla eventuale loop in corso
    if (_currentSource) {
      try { _currentSource.stop(); } catch { /* già fermato */ }
      _currentSource = null;
    }
    window.dispatchEvent(new CustomEvent('eden:state', { detail: { speaking: false } }));
  }
}

/**
 * Calcola la velocità di parlato da applicare al TTS in base ai tratti attuali.
 * fear alto → più veloce · warmth alto → più lenta · cynicism → leggermente piatta
 * Range risultante: 0.7–1.5 (clampato)
 */
function _ttsSpeed() {
  const { fear = 5, warmth = 5, cynicism = 5 } = _lastTraits;
  const speed = 0.88
    + (fear    / 10) * 0.38   // paura: accelera fino a +0.38
    - (warmth  / 10) * 0.12   // calore: rallenta fino a -0.12
    + (cynicism/ 10) * 0.06;  // cinismo: lieve accelerazione distaccata
  return Math.max(0.70, Math.min(1.50, parseFloat(speed.toFixed(3))));
}

/**
 * Inizializza AudioContext e catena gain al primo utilizzo.
 * Su mobile l'AudioContext parte in stato 'suspended' finché non arriva
 * una gesture utente — resume() viene chiamato prima di ogni playback.
 */
function _initAudioCtx() {
  if (_audioCtx) return;
  _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  _gainNode = _audioCtx.createGain();
  _gainNode.gain.value = 1.5;
  _gainNode.connect(_audioCtx.destination);
}

/**
 * Riproduce un ArrayBuffer WAV già ricevuto dal server.
 */
async function _playAudioBuffer(arrayBuffer) {
  _initAudioCtx();
  if (_audioCtx.state === 'suspended') await _audioCtx.resume();
  const token = ++_ttsCancelToken;
  if (_currentSource) { try { _currentSource.stop(); } catch {} _currentSource = null; }
  try {
    const audioBuffer = await _audioCtx.decodeAudioData(arrayBuffer);
    if (_ttsCancelToken !== token) return;
    const source = _audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(_gainNode);
    source.start(_audioCtx.currentTime + 0.05);
    _currentSource = source;
    window.dispatchEvent(new CustomEvent('eden:state', { detail: { speaking: true } }));
    source.onended = () => {
      if (_ttsCancelToken === token) {
        _currentSource = null;
        window.dispatchEvent(new CustomEvent('eden:state', { detail: { speaking: false } }));
      }
    };
  } catch { /* decodeAudioData fallita — silenzio */ }
}

/**
 * Tenta di riprodurre l'audio pre-sintetizzato per un exchange specifico.
 * Poll ogni 700ms fino a 30 tentativi (21s); fallback a _playTTS se non disponibile.
 * @param {string} exchangeId
 * @param {string} text       — testo fallback per _playTTS
 * @param {HTMLElement} btn   — pulsante ▶ da aggiornare visivamente
 */
/**
 * userTriggered=true → click esplicito: ignora _ttsEnabled, funziona sempre.
 * userTriggered=false (default) → auto-play: rispetta _ttsEnabled.
 */
async function _playExchangeAudio(exchangeId, text, btn, userTriggered = false) {
  if (!userTriggered && (!_ttsEnabled || !_ttsAvailable)) return;
  // iOS Safari: AudioContext.resume() deve essere chiamato nel contesto sincrono
  // del gesture handler, prima di qualsiasi await. Dopo un await iOS blocca l'audio.
  _initAudioCtx();
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
  const origLabel = btn ? btn.textContent : '▶';
  if (btn) { btn.textContent = '◌'; btn.disabled = true; }
  const MAX_POLLS = 30;
  const POLL_MS   = 700;
  for (let i = 0; i < MAX_POLLS; i++) {
    try {
      const res = await fetch(`${API_BASE}/api/speak/exchange/${exchangeId}`);
      if (res.status === 200) {
        const buf = await res.arrayBuffer();
        if (btn) { btn.textContent = origLabel; btn.disabled = false; }
        await _playAudioBuffer(buf);
        return;
      }
      if (res.status === 202) {
        await new Promise(r => setTimeout(r, POLL_MS));
        continue;
      }
      break;   // 404 / 500 → fallback
    } catch { break; }
  }
  if (btn) { btn.textContent = origLabel; btn.disabled = false; }
  _playTTSDirect(text);  // fallback: forza sintesi anche senza toggle globale
}

// Su mobile: init + resume AudioContext ad ogni gesture.
// { once: false } + rimozione manuale era sbagliato: al primo touch _audioCtx era null,
// il listener si rimuoveva e non riprovava mai più.
// Ora è persistente e inizializza il contesto se serve.
['touchstart', 'click'].forEach(ev => {
  document.addEventListener(ev, () => {
    _initAudioCtx();
    if (_audioCtx && _audioCtx.state === 'suspended') _audioCtx.resume();
  }, { passive: true });
});

/**
 * Divide il testo in frasi per il playback sentence-by-sentence.
 * Tiene la punteggiatura attaccata alla frase che la precede.
 */
function _splitSentences(text) {
  // Split su . ! ? … seguito da spazio o fine stringa
  const parts = text.split(/(?<=[.!?…])\s+/);
  return parts.map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Richiede audio TTS al server frase per frase e lo riproduce in streaming.
 * La prima frase inizia a suonare appena arriva dal server, le successive
 * vengono schedulate con Web Audio API per un playback continuo e senza gap.
 * @param {string} text — testo da sintetizzare
 */
/** Variante _playTTS per click esplicito: ignora _ttsEnabled. */
async function _playTTSDirect(text) {
  if (!_ttsAvailable || !text) return;
  // iOS Safari: init + resume sincrono prima di qualsiasi await
  _initAudioCtx();
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
  return _playTTSCore(text, ++_ttsCancelToken);
}

/**
 * Sintesi TTS con richiesta unica — il backend gestisce chunking e pause.
 * Elimina gli artefatti di stitching prodotti dall'approccio sentence-by-sentence:
 * XTTS v2 produce prosodia coerente solo se riceve testo connesso, non micro-frammenti.
 */
async function _playTTSCore(text, token) {
  if (_currentSource) { try { _currentSource.stop(); } catch {} _currentSource = null; }
  _initAudioCtx();
  if (_audioCtx.state === 'suspended') await _audioCtx.resume();
  if (!text || !text.trim()) return;
  const speed = _ttsSpeed();
  try {
    const res = await fetch(`${API_BASE}/api/speak`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text: text.trim(), speed })
    });
    if (!res.ok || _ttsCancelToken !== token) return;
    const buf = await res.arrayBuffer();
    if (_ttsCancelToken !== token) return;
    const audioBuffer = await _audioCtx.decodeAudioData(buf);
    if (_ttsCancelToken !== token) return;
    const source = _audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(_gainNode);
    source.start(_audioCtx.currentTime + 0.05);
    _currentSource = source;
    window.dispatchEvent(new CustomEvent('eden:state', { detail: { speaking: true } }));
    source.onended = () => {
      if (_ttsCancelToken === token) {
        _currentSource = null;
        window.dispatchEvent(new CustomEvent('eden:state', { detail: { speaking: false } }));
      }
    };
  } catch { /* silenzio su errori di rete */ }
}

async function _playTTS(text) {
  if (!_ttsEnabled || !_ttsAvailable || !text) return;
  return _playTTSCore(text, ++_ttsCancelToken);
}

if (btnTTS) btnTTS.addEventListener('click', _toggleTTS);

// â”€â”€â”€ LivePortrait face stream â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const faceStreamEl  = document.getElementById('face-stream');
const faceStatusDot = document.getElementById('face-status-dot');
const faceStatusTxt = document.getElementById('face-status-text');
const btnLPToggle   = document.getElementById('btn-lp-toggle');

let _faceRetryTimer = null;
let _lastAvatarFPS  = null;
let _lpAvailable    = false;   // LivePortrait installato e pronto
let _lpEnabled      = true;    // abilitato di default — coerente con auto-start backend
// Avatar sinaptico (synapse.js) — niente stream antropomorfo MJPEG sul frontend.
// LivePortrait backend resta disponibile, ma non viene mostrato.
const USE_LIVEPORTRAIT_STREAM = false;

// ─── Avatar emozionale dinamico (idle_loop_dynamic.py, porta 5002) ───────────
// Usa #face-dynamic (ID diverso da #face-stream) → synapse.js non lo rimuove.

const DYNAMIC_AVATAR_URL    = 'http://localhost:5002/face_stream';
const DYNAMIC_AVATAR_HEALTH = 'http://localhost:5002/health';
let _dynamicRetryTimer = null;
let _dynamicEnabled    = false;
let _showingDynamic    = false;

function _dynEl()    { return document.getElementById('face-dynamic'); }
function _dynBtn()   { return document.getElementById('btn-dynamic-toggle'); }
function _synCanvas(){ return document.querySelector('#avatar-container canvas'); }

/**
 * Controlla se idle_loop_dynamic.py è attivo su porta 5002.
 * Se sì, pre-carica lo stream e mostra il bottone toggle.
 * Riprova ogni 20 secondi se offline.
 */
async function initDynamicAvatar() {
  if (_dynamicRetryTimer) { clearTimeout(_dynamicRetryTimer); _dynamicRetryTimer = null; }
  try {
    const res = await fetch(DYNAMIC_AVATAR_HEALTH, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    if (data.status !== 'ok') throw new Error('not ok');
    _dynamicEnabled = true;
    _loadDynamicStream();
  } catch {
    _dynamicEnabled = false;
    _dynamicRetryTimer = setTimeout(initDynamicAvatar, 20000);
  }
}

/** Punta #face-dynamic allo stream e mostra subito il bottone toggle.
 *  Non aspetta onload: il MJPEG non lo scatta se i driver sono ancora
 *  in caricamento (prima esecuzione = estrazione keypoint da tutti i clip). */
function _loadDynamicStream() {
  const el = _dynEl();
  if (!el) return;
  el.onload  = null;
  el.onerror = null;
  el.src = `${DYNAMIC_AVATAR_URL}?t=${Date.now()}`;

  // Mostra bottone subito — health check già ok, stream arriva quando pronto
  const btn = _dynBtn();
  if (btn) btn.classList.remove('hidden');

  el.onerror = () => {
    // Stream caduto: nascondi bottone, torna a sinapsi, riprova
    _dynamicEnabled = false;
    if (_showingDynamic) _switchToSynapse();
    if (btn) btn.classList.add('hidden');
    _dynamicRetryTimer = setTimeout(initDynamicAvatar, 20000);
  };
}

/** Mostra l'avatar emozionale, nasconde il canvas sinaptico. */
function _switchToDynamic() {
  const el  = _dynEl();
  const cnv = _synCanvas();
  if (!el) return;
  if (cnv) cnv.style.visibility = 'hidden';
  el.style.display = 'block';
  _showingDynamic = true;
  const btn = _dynBtn();
  if (btn) btn.textContent = 'SINAPSI';
}

/** Mostra il canvas sinaptico, nasconde l'avatar emozionale. */
function _switchToSynapse() {
  const el  = _dynEl();
  const cnv = _synCanvas();
  if (el) el.style.display = 'none';
  if (cnv) cnv.style.visibility = 'visible';
  _showingDynamic = false;
  const btn = _dynBtn();
  if (btn) btn.textContent = 'AVATAR';
}

/** Toggle manuale tra sinapsi e avatar emozionale. */
function toggleDynamicAvatar() {
  if (!_dynamicEnabled) return;
  if (_showingDynamic) _switchToSynapse();
  else _switchToDynamic();
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = _dynBtn();
  if (btn) btn.addEventListener('click', toggleDynamicAvatar);
});

function _setFaceStatus(mode, label) {
  if (faceStatusDot) { faceStatusDot.className = mode; }
  if (faceStatusTxt) { faceStatusTxt.textContent = label; }
}

window.addEventListener('eden:avatar:metrics', e => {
  const d = e.detail || {};
  const fps = Number(d.fps);
  if (!Number.isFinite(fps)) return;
  _lastAvatarFPS = Math.max(0, Math.round(fps));

  // Non sovrascrivere lo stato quando LivePortrait è attivo.
  if (!faceStatusDot || !faceStatusTxt) return;
  if (faceStatusDot.className === 'live') return;

  const tier = _lastAvatarFPS >= 95 ? '120' : '60';
  faceStatusTxt.textContent = `THREE.JS ${_lastAvatarFPS} FPS (${tier})`;
});

/** Aggiorna testo e stile del bottone toggle LP. */
function _updateLPToggleBtn() {
  if (!btnLPToggle) return;
  if (!_lpAvailable) {
    btnLPToggle.classList.add('hidden');
    return;
  }
  btnLPToggle.classList.remove('hidden');
  const isLive = faceStatusDot && faceStatusDot.className === 'live';
  if (isLive) {
    btnLPToggle.textContent = 'DISATTIVA';
    btnLPToggle.classList.add('lp-on');
  } else {
    btnLPToggle.textContent = 'ATTIVA';
    btnLPToggle.classList.remove('lp-on');
  }
}

/** Attiva o disattiva LivePortrait con effetto immediato. */
async function toggleLivePortrait() {
  if (!_lpAvailable) return;

  const isLive = faceStatusDot && faceStatusDot.className === 'live';

  if (isLive) {
    // ── Disattiva ────────────────────────────────────────────────────────
    _lpEnabled = false;

    // Ferma il timer di retry
    if (_faceRetryTimer) { clearTimeout(_faceRetryTimer); _faceRetryTimer = null; }

    // Stacca lo stream MJPEG prima di svuotare src (evita onerror spurio)
    faceStreamEl.onload  = null;
    faceStreamEl.onerror = null;
    faceStreamEl.src     = '';
    faceStreamEl.style.display = 'none';

    // Ferma il processo sul server
    try { await fetch(`${API_BASE}/api/face_stop`, { method: 'POST' }); } catch {}

    _setFaceStatus('threejs', 'THREE.JS');
    _updateLPToggleBtn();

  } else {
    // ── Riattiva ─────────────────────────────────────────────────────────
    _lpEnabled = true;
    _updateLPToggleBtn();
    await initLivePortrait();
  }
}

if (btnLPToggle) btnLPToggle.addEventListener('click', toggleLivePortrait);

/**
 * Controlla /api/face_status e connette lo stream se LivePortrait è attivo.
 * Se non disponibile, rimane silenzioso (Three.js è il fallback).
 */
async function initLivePortrait() {
  if (!USE_LIVEPORTRAIT_STREAM) {
    if (faceStreamEl) faceStreamEl.style.display = 'none';
    _setFaceStatus('threejs', 'THREE.JS');
    return;
  }

  try {
    const res  = await fetch(`${API_BASE}/api/face_status`);
    const data = await res.json();

    if (!data.available) {
      // Prerequisiti mancanti — Three.js rimane attivo, nessun messaggio
      _setFaceStatus('threejs', 'THREE.JS');
      return;
    }

    // LP installato: rendi visibile il bottone toggle
    _lpAvailable = true;
    _updateLPToggleBtn();

    // Se l'utente ha disabilitato LP, non avviare
    if (!_lpEnabled) {
      _setFaceStatus('threejs', 'THREE.JS');
      return;
    }

    if (data.running) {
      _connectFaceStream();
    } else {
      // Disponibile ma non avviato — prova ad avviare
      try {
        await fetch(`${API_BASE}/api/face_start`, { method: 'POST' });
        setTimeout(_connectFaceStream, 2000); // attendi avvio
      } catch {
        _setFaceStatus('threejs', 'THREE.JS');
      }
    }
  } catch {
    // Server non raggiungibile — Three.js rimane
    _setFaceStatus('threejs', 'THREE.JS');
  }
}

function _connectFaceStream() {
  if (!faceStreamEl) return;
  if (!_lpEnabled) return;   // rispetta la preferenza utente
  if (_faceRetryTimer) { clearTimeout(_faceRetryTimer); _faceRetryTimer = null; }

  _setFaceStatus('threejs', 'CONNESSIONE...');

  // Cache-bust per evitare reply 304
  faceStreamEl.src = `${API_BASE}/api/face_stream?t=${Date.now()}`;

  faceStreamEl.onload = () => {
    faceStreamEl.style.display = 'block';
    _setFaceStatus('live', 'LIVEPORTRAIT');
    _updateLPToggleBtn();
  };

  faceStreamEl.onerror = () => {
    faceStreamEl.style.display = 'none';
    _setFaceStatus('error', 'THREE.JS');
    // Riprova solo se l'utente non ha disabilitato LP
    if (_lpEnabled) {
      _faceRetryTimer = setTimeout(_connectFaceStream, 15000);
    }
  };
}

// â”€â”€â”€ Avvio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

(async function init() {
  appendMessage('system', '- connessione in corso -');

  // Stato iniziale dall'API (riprende la sessione in corso, senza incrementarla)
  await fetchStatus();
  await refreshAutonomyPanel();
  loadPioneerStatus();

  // Mostra numero sessione corrente senza crearne una nuova
  const sessioni = elSessions ? elSessions.textContent : '?';
  appendMessage('system', `- sessione ${sessioni} -`);

  // Connessione SSE per messaggi proattivi
  connectSSE();

  // LivePortrait stream (silenzioso se non disponibile)
  initLivePortrait();

  // Avatar emozionale dinamico su porta 5002 (override se attivo)
  initDynamicAvatar();

  // TTS: verifica disponibilità Kokoro (nasconde il pulsante se non installato)
  _checkTTSAvailable();

  // Refresh periodico pannello autonomia
  setInterval(refreshAutonomyPanel, 15000);

  // SIC — carica pensieri iniziali, poi aggiorna ogni 5 minuti
  refreshSICThoughts();
  setInterval(refreshSICThoughts, 300000);

  inputEl.focus();
})();

// ─── Archivio sessioni ────────────────────────────────────────────────────────

const archivePanel       = document.getElementById('archive-panel');
const chatSection        = document.getElementById('chat-section');
const btnArchive         = document.getElementById('btn-archive');
const btnArchiveClose    = document.getElementById('btn-archive-close');
const btnArchiveBack     = document.getElementById('btn-archive-back');
const btnSessionResume   = document.getElementById('btn-session-resume');
const archiveListView    = document.getElementById('archive-list-view');
const archiveDetailView  = document.getElementById('archive-detail-view');
const archiveList        = document.getElementById('archive-list');
const archiveDetailMsgs  = document.getElementById('archive-detail-messages');
const archiveDetailTitle = document.getElementById('archive-detail-title');

let _currentSessionMeta = null;   // metadati sessione aperta nel dettaglio

/** Formatta una data ISO in stringa leggibile (senza secondi) */
function _formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('it-IT', { day: '2-digit', month: 'short', year: 'numeric' })
      + ' ' + d.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' });
  } catch { return iso; }
}

/** Chiave giorno YYYY-MM-DD da ISO */
function _dayKey(iso) {
  if (!iso) return '';
  try { return new Date(iso).toISOString().slice(0, 10); } catch { return ''; }
}

/** Etichetta gruppo data leggibile */
function _groupLabel(dayKey) {
  if (!dayKey) return 'Senza data';
  const today     = new Date().toISOString().slice(0, 10);
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  if (dayKey === today)     return 'Oggi';
  if (dayKey === yesterday) return 'Ieri';
  const diff = (Date.now() - new Date(dayKey).getTime()) / 86400000;
  if (diff < 7) return 'Questa settimana';
  return new Date(dayKey).toLocaleDateString('it-IT', { month: 'long', year: 'numeric' });
}

/** Mostra il pannello archivio, nasconde la chat */
function apriArchivio() {
  chatSection.classList.add('hidden');
  archivePanel.classList.remove('hidden');
  archiveListView.classList.remove('hidden');
  archiveDetailView.classList.add('hidden');
  caricaListaSessioni();
}

/** Chiude il pannello archivio, torna alla chat */
function chiudiArchivio() {
  archivePanel.classList.add('hidden');
  chatSection.classList.remove('hidden');
}

/** Torna alla lista dall'interno del dettaglio */
function tornaAllaLista() {
  archiveDetailView.classList.add('hidden');
  archiveListView.classList.remove('hidden');
}

/** Carica e renderizza la lista sessioni dal backend con raggruppamento per data */
async function caricaListaSessioni() {
  archiveList.innerHTML = '<div class="archive-empty">caricamento...</div>';
  try {
    const res      = await fetch(`${API_BASE}/api/sessions`);
    const sessioni = await res.json();

    if (!Array.isArray(sessioni) || sessioni.length === 0) {
      archiveList.innerHTML = '<div class="archive-empty">nessuna sessione archiviata</div>';
      return;
    }

    archiveList.innerHTML = '';
    const currentNum = parseInt(elSessions?.textContent || '0');
    let lastGroup = null;

    sessioni.forEach(s => {
      // Intestazione gruppo data (cambia quando cambia il giorno)
      const dayKey = _dayKey(s.last_saved || s.date);
      const groupLabel = _groupLabel(dayKey);
      if (groupLabel !== lastGroup) {
        const hdr = document.createElement('div');
        hdr.className = 'archive-group-header';
        hdr.textContent = groupLabel;
        archiveList.appendChild(hdr);
        lastGroup = groupLabel;
      }

      const isCurrent = s.session_number === currentNum;
      const card = document.createElement('div');
      card.className = 'session-card' + (isCurrent ? ' current' : '');

      // Data: "27 apr 2026, 00:36" oppure con freccia se fine diversa da inizio
      const startStr = _formatDate(s.date);
      const endStr   = s.last_saved && _dayKey(s.last_saved) !== _dayKey(s.date)
        ? ' → ' + _formatDate(s.last_saved)
        : (s.last_saved && s.last_saved !== s.date
            ? ' → ' + new Date(s.last_saved).toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })
            : '');
      const scambi = Math.floor(s.message_count / 2);
      const badge  = isCurrent ? '<span class="session-current-badge">ATTIVA</span>' : '';

      card.innerHTML =
        '<span class="session-card-num">SESSIONE ' + s.session_number + badge + '</span>' +
        '<span class="session-card-count">' + scambi + ' scambi</span>' +
        '<span class="session-card-date">' + startStr + endStr + '</span>' +
        (s.preview ? '<span class="session-card-preview">“' + escapeHtml(s.preview) + '”</span>' : '');

      card.addEventListener('click', () => apriDettaglioSessione(s));
      archiveList.appendChild(card);
    });
  } catch {
    archiveList.innerHTML = '<div class="archive-empty">errore caricamento sessioni</div>';
  }
}

/** Carica e mostra i messaggi di una sessione specifica */
async function apriDettaglioSessione(meta) {
  _currentSessionMeta = meta;
  archiveListView.classList.add('hidden');
  archiveDetailView.classList.remove('hidden');
  const _scambi = Math.floor((meta.message_count || 0) / 2);
  archiveDetailTitle.textContent =
    `SESSIONE ${meta.session_number}  ·  ${_formatDate(meta.date)}  ·  ${_scambi} scambi`;
  archiveDetailMsgs.innerHTML = '<div class="archive-empty">caricamento...</div>';

  try {
    const res      = await fetch(`${API_BASE}/api/sessions/${meta.session_id}`);
    const sessione = await res.json();

    if (sessione.errore) {
      archiveDetailMsgs.innerHTML = `<div class="archive-empty">${sessione.errore}</div>`;
      return;
    }

    archiveDetailMsgs.innerHTML = '';
    // Preferisce conversation_log (completo, non compresso) — fallback a messages
    const msgDaMostrare = sessione.conversation_log || sessione.messages || [];
    msgDaMostrare.forEach(m => {
      const wrapper = document.createElement('div');
      wrapper.className = `archive-msg ${m.role}`;

      const label = document.createElement('div');
      label.className = 'message-label';
      const timeStr = m.ts
        ? ' · ' + new Date(m.ts).toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' })
        : '';
      label.textContent = (m.role === 'user' ? 'tu' : 'eden') + timeStr;

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = m.content;

      wrapper.append(label, bubble);
      archiveDetailMsgs.appendChild(wrapper);
    });

    // Scorri in cima
    archiveDetailMsgs.scrollTop = 0;
  } catch {
    archiveDetailMsgs.innerHTML = '<div class="archive-empty">errore caricamento sessione</div>';
  }
}

/** Riprende la sessione visualizzata nel dettaglio */
async function riprendi_sessione() {
  if (!_currentSessionMeta) return;

  btnSessionResume.disabled    = true;
  btnSessionResume.textContent = '...';

  try {
    const res  = await fetch(
      `${API_BASE}/api/sessions/${_currentSessionMeta.session_id}/resume`,
      { method: 'POST' }
    );
    const data = await res.json();

    if (!res.ok || data.errore) {
      chiudiArchivio();
      const msg = data.errore ?? `Impossibile riprendere la sessione (HTTP ${res.status}).`;
      appendMessage('error', msg);
      return;
    }

    // Chiudi archivio e ripopola la chat con i messaggi ripristinati
    chiudiArchivio();
    messagesEl.innerHTML = '';
    (data.messages || []).forEach(m => {
      const ts = m.ts ? (m.ts.substring(8,10)+'/'+m.ts.substring(5,7)+' '+m.ts.substring(11,16)) : null;
      appendMessage(m.role === 'user' ? 'user' : 'assistant', m.content, ts);
    });
    appendMessage('system', `— sessione ${data.session_number} ripresa —`);
    scrollDown();
    inputEl.focus();

  } catch (err) {
    chiudiArchivio();
    appendMessage('error', `Errore durante la ripresa della sessione: ${err.message ?? err}`);
  } finally {
    btnSessionResume.disabled    = false;
    btnSessionResume.textContent = 'RIPRENDI';
  }
}

// Event listeners archivio
btnArchive.addEventListener('click', apriArchivio);
btnArchiveClose.addEventListener('click', chiudiArchivio);
btnArchiveBack.addEventListener('click', tornaAllaLista);
btnSessionResume.addEventListener('click', riprendi_sessione);

// Chiudi archivio con Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !archivePanel.classList.contains('hidden')) {
    chiudiArchivio();
  }
});

// ─── Navigazione sezioni (chat / dashboard / debug / logs) ──────────────────

const mainChatEl    = document.getElementById('main-chat');
const secDashboard  = document.getElementById('section-dashboard');
const secDebug      = document.getElementById('section-debug');
const secLogs       = document.getElementById('section-logs');
const iframeDebug   = document.getElementById('iframe-debug');
const iframeLogs    = document.getElementById('iframe-logs');
let _activeSection = 'chat';

function switchSection(section) {
  if (_activeSection === section) return;
  _activeSection = section;

  mainChatEl.classList.toggle('hidden',    section !== 'chat');
  if (secDashboard) secDashboard.classList.toggle('hidden', section !== 'dashboard');
  secDebug.classList.toggle('hidden',      section !== 'debug');
  secLogs.classList.toggle('hidden',       section !== 'logs');

  // Debug: ricarica sempre per dati freschi; Train/Research: lazy (polling/SSE interno)
  if (section === 'debug') {
    iframeDebug.src = '/debug';
    _bindMobileIframeAutosize(iframeDebug);
  } else if (section === 'logs') {
    if (!iframeLogs.getAttribute('src')) iframeLogs.src = '/logs';
    _bindMobileIframeAutosize(iframeLogs);
  }

  // Aggiorna classe active nella nav
  document.querySelectorAll('.header-nav-link[data-section]').forEach(a => {
    a.classList.toggle('active', a.dataset.section === section);
  });

  // Quando si torna alla chat, rimetti il focus sull'input
  if (section === 'chat') {
    inputEl.focus();
  }

  // Mobile: su sezioni non-chat rimuovi backdrop/drawer che coprivano tutto lo schermo
  if (section !== 'chat') {
    document.documentElement.classList.add('has-page-section');
  } else {
    document.documentElement.classList.remove('has-page-section');
  }
}

document.querySelectorAll('.header-nav-link[data-section]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    switchSection(a.dataset.section);
  });
});

// ── iOS Safari mobile scroll — postMessage height bridge ────────────────────
// iOS 16+ non supporta touch-scroll dentro iframe.
// Ogni template misura body.getBoundingClientRect().bottom (altezza contenuto reale)
// e la manda via postMessage con il proprio location.pathname.
// Qui usiamo una lookup map pathname→iframe per matching affidabile (no e.source).
// Il container .page-section (overflow-y:scroll; height:calc(100vh-48px)) scorre.

const _IFRAME_PAGE_MAP = {
  '/debug':    iframeDebug,
  '/logs':     iframeLogs,
};

window.addEventListener('message', function(e) {
  if (!e.data || typeof e.data._eden_h !== 'number' || !e.data._eden_p) return;
  if (window.innerWidth > 980) return;
  const h = e.data._eden_h;
  if (h < 50) return;
  const fr = _IFRAME_PAGE_MAP[e.data._eden_p];
  if (fr) fr.style.height = (h + 32) + 'px'; // +32px margine sicurezza
});

// Reset su desktop quando si ruota da mobile a landscape > 980px
window.addEventListener('resize', function() {
  if (window.innerWidth > 980) {
    Object.values(_IFRAME_PAGE_MAP).forEach(function(fr) {
      if (fr) fr.style.height = '';
    });
  }
});

function _bindMobileIframeAutosize() {} // no-op — compatibilità chiamate in switchSection

// ─── Modalita' Presenza <-> Laboratorio ──────────────────────────────────────
// Default: 'presence' (volto + chat, niente cruscotto).
// 'lab' espone sidebar metriche, badge relazione, nav tecnica.
// Persistenza localStorage chiave 'eden_mode'. La classe e' applicata su <html>
// dallo script inline in <head> per evitare FOUC; qui aggiorniamo runtime.

const _modeToggleBtn   = document.getElementById('btn-mode-toggle');
const _labBackdrop     = document.getElementById('lab-backdrop');
const _MODE_KEY        = 'eden_mode';
const _MODE_PRESENCE   = 'presence';
const _MODE_LAB        = 'lab';

function _currentMode() {
  return document.documentElement.classList.contains('mode-lab')
    ? _MODE_LAB
    : _MODE_PRESENCE;
}

function setMode(mode) {
  if (mode !== _MODE_PRESENCE && mode !== _MODE_LAB) return;
  const html = document.documentElement;
  html.classList.remove('mode-presence', 'mode-lab');
  html.classList.add('mode-' + mode);
  try { localStorage.setItem(_MODE_KEY, mode); } catch (_) {}

  // Aggiorna aria/title sul toggle per accessibilita'
  if (_modeToggleBtn) {
    if (mode === _MODE_LAB) {
      _modeToggleBtn.setAttribute('aria-pressed', 'true');
      _modeToggleBtn.setAttribute('title', 'Torna a Presenza (Esc)');
    } else {
      _modeToggleBtn.setAttribute('aria-pressed', 'false');
      _modeToggleBtn.setAttribute('title', 'Apri Laboratorio');
    }
  }

  // Se torniamo a Presenza, riporta sempre alla CHAT (non vogliamo lasciare
  // l'utente bloccato su una sezione tecnica nascosta dal toggle).
  if (mode === _MODE_PRESENCE && _activeSection !== 'chat') {
    switchSection('chat');
  }
}

function toggleMode() {
  setMode(_currentMode() === _MODE_LAB ? _MODE_PRESENCE : _MODE_LAB);
}

if (_modeToggleBtn) {
  _modeToggleBtn.addEventListener('click', toggleMode);
}

// Click sul backdrop (mobile drawer) chiude il LAB
if (_labBackdrop) {
  _labBackdrop.addEventListener('click', () => setMode(_MODE_PRESENCE));
}

// Esc: chiudi LAB se aperto. (Esc per archivio gia' gestito altrove,
// quel handler viene prima e fa preventDefault implicito ritornando.)
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  // Non interferire quando l'utente sta scrivendo in un input/textarea
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  // Solo se LAB aperto e archivio NON aperto (l'archivio ha precedenza)
  const archiveOpen = !document.getElementById('archive-panel').classList.contains('hidden');
  if (archiveOpen) return;
  if (_currentMode() === _MODE_LAB) setMode(_MODE_PRESENCE);
});

// Init: applica gli attributi aria coerenti con lo stato corrente
setMode(_currentMode());

// ─── Copia messaggio — helpers ───────────────────────────────────────────────

function _fallbackCopy(text, done) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (e) {}
  document.body.removeChild(ta);
  if (done) done();
}

// Touch: mostra il pulsante copia su tap (mobile)
document.addEventListener('touchstart', (e) => {
  const msg = e.target.closest('.message');
  if (msg) {
    document.querySelectorAll('.message.touch-active')
      .forEach(m => m.classList.remove('touch-active'));
    msg.classList.add('touch-active');
  }
}, { passive: true });

// ─── Log Toast Overlay — Alert per ERROR/CRITICAL ─────────────────────────────
// Connessione SSE globale: riceve eventi ERROR/CRITICAL e mostra toast overlay.
// Toast stacking: max 3 simultanei. Auto-dismiss: 10s CRITICAL, 6s ERROR.
(function initLogToastListener() {
  const toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    // Crea container se non presente
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = 'position: fixed; top: 0; right: 0; z-index: 9999; pointer-events: none;';
    document.body.appendChild(container);
  }

  const logStream = new EventSource('/api/logs/stream');
  logStream.onmessage = (e) => {
    try {
      const log = JSON.parse(e.data);
      if (log.type === 'connected' || !log.severity) return;
      // Mostra toast solo per ERROR/CRITICAL
      if (log.severity === 'ERROR' || log.severity === 'CRITICAL') {
        showLogToast(log.severity, log.module, log.message);
      }
    } catch (err) {
      // Fallback: ignore parse errors
    }
  };
  logStream.onerror = () => {
    logStream.close();
    // Tentativo riconnessione dopo 5s
    setTimeout(initLogToastListener, 5000);
  };

  window._logStreamActive = true;
})();

function showLogToast(severity, module, message) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  // Max 3 toast simultanei
  const toasts = container.querySelectorAll('.log-toast');
  if (toasts.length >= 3) {
    toasts[0].remove();
  }

  const toast = document.createElement('div');
  toast.className = 'log-toast';
  const bgColor = severity === 'CRITICAL' ? '#1a0000' : '#1a1a00';
  const textColor = severity === 'CRITICAL' ? '#f92672' : '#f0c674';
  const icon = severity === 'CRITICAL' ? '🔴' : '⚠️';

  toast.innerHTML = `
    <div style="background: ${bgColor}; border: 1px solid ${textColor}; color: ${textColor}; padding: 12px 16px; border-radius: 3px; font-family: 'Share Tech Mono', monospace; font-size: 12px; max-width: 400px; box-shadow: 0 0 10px rgba(0,0,0,0.5); animation: logToastPulse 1.5s infinite;">
      <div style="font-weight: bold; margin-bottom: 6px;">${icon} ${severity} — ${module}</div>
      <div style="font-size: 11px; word-break: break-word; max-height: 80px; overflow: hidden;">${escapeHtml(message)}</div>
      <button onclick="this.parentElement.parentElement.remove()" style="margin-top: 8px; background: transparent; color: inherit; border: 1px solid currentColor; padding: 4px 8px; border-radius: 2px; cursor: pointer; float: right; font-family: inherit; font-size: 10px;">Dismiss</button>
    </div>
  `;

  container.appendChild(toast);

  // Auto-dismiss
  const dismissTime = severity === 'CRITICAL' ? 10000 : 6000;
  setTimeout(() => {
    try { toast.remove(); } catch (e) {}
  }, dismissTime);

  // Aggiungi CSS per animazione se non presente
  if (!document.getElementById('log-toast-css')) {
    const style = document.createElement('style');
    style.id = 'log-toast-css';
    style.textContent = '@keyframes logToastPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.8; } }';
    document.head.appendChild(style);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ════════════════════════════════════════════════════════════════════════════
// STEP 3/8 redesign — header command-center: drawer mobile + datetime + init
// ════════════════════════════════════════════════════════════════════════════

// ─── Datetime live in header (right zone) ──────────────────────────────────
(function startDatetimeUpdater() {
  const elDate = document.getElementById('h-dt-date');
  const elTime = document.getElementById('h-dt-time');
  if (!elDate || !elTime) return;

  const dateFmt = new Intl.DateTimeFormat('it-IT', {
    weekday: 'short', day: '2-digit', month: 'short', year: 'numeric'
  });

  function tick() {
    const now = new Date();
    // Data: solo cambia a mezzanotte ma ricalcolarla ogni secondo è trascurabile
    elDate.textContent = dateFmt.format(now).replace(/\./g, '');
    // Ora HH:MM:SS, tabular nums in CSS
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    elTime.textContent = `${hh}:${mm}:${ss}`;
  }
  tick();
  setInterval(tick, 1000);
})();

// ─── Drawer hamburger mobile ───────────────────────────────────────────────
(function setupMobileDrawer() {
  const btn      = document.getElementById('btn-mobile-menu');
  const backdrop = document.getElementById('mobile-drawer-backdrop');
  if (!btn || !backdrop) return;

  function open() {
    document.documentElement.classList.add('drawer-open');
    btn.setAttribute('aria-expanded', 'true');
  }
  function close() {
    document.documentElement.classList.remove('drawer-open');
    btn.setAttribute('aria-expanded', 'false');
  }
  function toggle() {
    if (document.documentElement.classList.contains('drawer-open')) close();
    else open();
  }

  btn.addEventListener('click', (e) => { e.preventDefault(); toggle(); });
  backdrop.addEventListener('click', close);

  // Chiude drawer quando si seleziona una nav voice (mobile)
  document.querySelectorAll('.header-nav-link[data-section]').forEach(a => {
    a.addEventListener('click', () => {
      if (document.documentElement.classList.contains('drawer-open')) close();
    });
  });

  // Esc chiude drawer
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.documentElement.classList.contains('drawer-open')) {
      close();
    }
  });

  // Resize a desktop chiude drawer automaticamente
  window.addEventListener('resize', () => {
    if (window.innerWidth > 980 && document.documentElement.classList.contains('drawer-open')) {
      close();
    }
  });
})();

// ─── Init: home default = dashboard (era chat) ─────────────────────────────
// Atteso che il DOM sia pronto e che lo script.js abbia popolato switchSection.
// Eseguito al microtask successivo per non bloccare altre init.
queueMicrotask(() => {
  try {
    if (typeof switchSection === 'function') {
      switchSection('dashboard');
    }
  } catch (e) {
    console.warn('init dashboard fallita:', e);
  }
});

// ════════════════════════════════════════════════════════════════════════════
// STEP 4/8 redesign — Dashboard renderer
//
// Strategia: la dashboard NON fa fetch propri quando i dati sono gia' in DOM
// (popolati dallo script.js esistente). Usa MutationObserver passivi sui
// sorgenti #trait-*, #stat-*, #af-*, #auto-*, #relationship-badge per
// propagare i valori. Solo per i KPI L1 (coherence/grounding/somatic) e
// stats grafo Kuzu fa fetch leggeri a /api/state e /graph/status,
// con polling 5s ma SOLO mentre la sezione dashboard e' visibile.
// ════════════════════════════════════════════════════════════════════════════

(function setupDashboard() {

  // ─── Helpers ──────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const _bootTime = Date.now();
  let _lastInputAt = Date.now();
  let _l1Poll = null;
  let _kuzuPoll = null;
  const _sparkBuf = { coh: [], grn: [], som: [] };
  const SPARK_LEN = 12;

  // ─── Aggiorna l'input timestamp ogni volta che user scrive (per IDLE hero)
  const _input = $('input');
  if (_input) {
    ['input','focus','keydown'].forEach(ev =>
      _input.addEventListener(ev, () => { _lastInputAt = Date.now(); })
    );
  }

  // ─── Radar SVG: aggiorna polygon points + label da #trait-* DOM esistente
  const TRAIT_KEYS = ['curiosity','trust','cynicism','warmth','fear'];
  // Vertici radar (pentagono): angle 0=top, poi +72°. Center (100,100), rMax=85.
  const RADAR_ANGLES = [ -Math.PI/2, -Math.PI/2 + 2*Math.PI/5, -Math.PI/2 + 4*Math.PI/5, -Math.PI/2 + 6*Math.PI/5, -Math.PI/2 + 8*Math.PI/5 ];
  // Mappa indice asse → label SVG (cur/fid/cin/pau/cal)
  // Ordine angolare: top, top-right, bottom-right, bottom-left, top-left
  // Decisione: cur=top, fid=top-right, cin=bottom-right, pau=bottom-left, cal=top-left
  const RADAR_LABELS = ['cur','fid','cin','pau','cal'];
  const TRAIT_TO_RADAR = { curiosity: 0, trust: 1, cynicism: 2, fear: 3, warmth: 4 };

  function _readTraitValue(name) {
    const el = document.querySelector(`#trait-${name} .trait-value`);
    if (!el) return null;
    const n = parseFloat(el.textContent);
    return isFinite(n) ? n : null;
  }

  function renderRadar() {
    const fill = $('dash-radar-fill');
    if (!fill) return;
    const pts = [];
    TRAIT_KEYS.forEach(name => {
      const idx = TRAIT_TO_RADAR[name];
      const v = _readTraitValue(name);
      const norm = (v === null) ? 0 : Math.max(0, Math.min(1, v / 10));
      const r = norm * 85;
      const x = 100 + r * Math.cos(RADAR_ANGLES[idx]);
      const y = 100 + r * Math.sin(RADAR_ANGLES[idx]);
      pts[idx] = `${x.toFixed(1)},${y.toFixed(1)}`;
      // Update vertex circle
      const c = $(`dash-radar-pt-${idx}`);
      if (c) { c.setAttribute('cx', x.toFixed(1)); c.setAttribute('cy', y.toFixed(1)); }
      // Update label tspan
      const lbl = $(`dash-tr-${RADAR_LABELS[idx]}-lbl`);
      if (lbl) lbl.textContent = (v === null) ? '--' : v.toFixed(1);
    });
    fill.setAttribute('points', pts.join(' '));

    // Indice medio
    const vals = TRAIT_KEYS.map(_readTraitValue).filter(v => v !== null);
    const mean = vals.length ? (vals.reduce((a,b)=>a+b,0) / vals.length) : null;
    const meanEl = $('dash-tr-mean');
    if (meanEl) meanEl.textContent = (mean === null) ? '--' : mean.toFixed(1);
  }

  // ─── Donut affettivo: valenza → arco circolare (-1..+1 → 0..2π)
  function renderAffective() {
    const elV = $('af-valence');
    const elA = $('af-arousal');
    const elC = $('af-certainty');
    if (!elV) return;

    const val = parseFloat(elV.textContent);
    const aro = parseFloat(elA?.textContent);
    const cer = parseFloat(elC?.textContent);

    const donutVal = $('dash-donut-val');
    const donutArc = $('dash-donut-arc');
    const afAro = $('dash-af-aro');
    const afCer = $('dash-af-cer');
    const afMood = $('dash-af-mood');

    if (donutVal) donutVal.textContent = isFinite(val) ? (val>=0?'+':'') + val.toFixed(2) : '--';
    if (donutArc) {
      const norm = isFinite(val) ? Math.max(-1, Math.min(1, val)) : 0;
      const circ = 2 * Math.PI * 40;  // ~251
      const frac = Math.abs(norm);
      donutArc.setAttribute('stroke-dasharray', `${(frac*circ).toFixed(1)} ${circ.toFixed(1)}`);
      // Colore: cyan se positivo, pink se negativo, gray se 0
      const c = (norm > 0.05) ? 'var(--accent)' : (norm < -0.05) ? 'var(--rel-hostile)' : 'var(--text-muted)';
      donutArc.setAttribute('stroke', c);
    }
    if (afAro) afAro.textContent = isFinite(aro) ? aro.toFixed(2) : '--';
    if (afCer) afCer.textContent = isFinite(cer) ? cer.toFixed(2) : '--';
    if (afMood) {
      let mood = '--';
      if (isFinite(val) && isFinite(aro)) {
        if (val > 0.3 && aro > 0.5)       mood = 'eccitata';
        else if (val > 0.3 && aro <= 0.5) mood = 'serena';
        else if (val < -0.3 && aro > 0.5) mood = 'agitata';
        else if (val < -0.3)              mood = 'cupa';
        else                              mood = 'neutra';
      }
      afMood.textContent = mood;
    }
  }

  // ─── Stats sessione/memoria: leggi da DOM esistente
  function renderStats() {
    const map = [
      ['stat-exchange','dash-st-exch'],
      ['stat-sessions','dash-st-sess'],
      ['stat-memories','dash-st-mem'],
      ['stat-history','dash-st-hist'],
    ];
    map.forEach(([src, dst]) => {
      const s = $(src); const d = $(dst);
      if (s && d) d.textContent = s.textContent || '--';
    });
  }

  // ─── Autonomia
  function renderAutonomy() {
    const map = [
      ['auto-running','dash-au-state'],
      ['auto-decisions','dash-au-dec'],
      ['auto-mode','dash-au-mode'],
      ['auto-interval','dash-au-int'],
    ];
    map.forEach(([src, dst]) => {
      const s = $(src); const d = $(dst);
      if (s && d) d.textContent = s.textContent || '--';
    });
    // Colora stato ON/OFF
    const elState = $('dash-au-state');
    if (elState) {
      const t = (elState.textContent || '').trim().toUpperCase();
      elState.style.color = (t === 'ON') ? 'var(--rel-ally)' : 'var(--text-muted)';
    }
  }

  // ─── Relazione
  function renderRelationship() {
    const src = $('relationship-badge');
    const dst = $('dash-rel-text');
    if (src && dst) dst.textContent = (src.textContent || '--').trim();
  }

  // ─── Hero stats: uptime, session#, exch, idle
  function _fmtDuration(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${h}h ${String(m).padStart(2,'0')}m ${String(ss).padStart(2,'0')}s`;
  }
  function _fmtIdle(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(s / 60);
    const ss = s % 60;
    return `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  }
  function renderHero() {
    const up = $('dhero-uptime');
    if (up) up.textContent = _fmtDuration(Date.now() - _bootTime);
    const idle = $('dhero-idle');
    if (idle) idle.textContent = _fmtIdle(Date.now() - _lastInputAt);
    const sess = $('dhero-session');
    const elS = $('stat-sessions');
    if (sess && elS) sess.textContent = elS.textContent || '--';
    const exch = $('dhero-exch');
    const elE = $('stat-exchange');
    if (exch && elE) exch.textContent = elE.textContent || '--';
  }

  // ─── L1 via /api/state (struttura: campi al TOP-level,
  //     vedi homeostasis.stato_sintesi: coherence_budget, grounding_integrity,
  //     self_model_stability sono chiavi dirette, NO wrapper homeostatic_state)
  async function fetchL1Snapshot() {
    try {
      const r = await fetch('/api/state', { cache: 'no-cache' });
      if (!r.ok) return;
      const j = await r.json();
      const get = (k) => {
        if (j[k] != null) return Number(j[k]);
        if (j.homeostatic_state && j.homeostatic_state[k] != null) return Number(j.homeostatic_state[k]);
        return null;
      };
      const coh = get('coherence_budget');
      const grn = get('grounding_integrity');
      const sms = get('self_model_stability');
      _setKpi('dash-fk-coh', 'dhero-l1c', coh);
      _setKpi('dash-fk-grn', 'dhero-l1g', grn);
      const heroSm = $('dhero-l1s');
      if (heroSm) heroSm.textContent = (typeof sms === 'number' && isFinite(sms)) ? sms.toFixed(2) : '--';
      _pushSpark('coh', coh, 'dash-spk-coh');
      _pushSpark('grn', grn, 'dash-spk-grn');

      // Step 6/8 fix: lega il synapse hero alle metriche L1
      _updateSynapseVars({ coh, sms });
    } catch (e) { /* fail silent */ }
  }

  // ─── Somatic via /api/state/somatic (endpoint separato)
  //     Struttura: { active, version, snapshot: { vram_norm, somatic_arousal, ... } }
  async function fetchSomaticSnapshot() {
    try {
      const r = await fetch('/api/state/somatic', { cache: 'no-cache' });
      if (!r.ok) return;
      const j = await r.json();
      const snap = j.snapshot || {};
      const aro = (typeof snap.somatic_arousal === 'number') ? snap.somatic_arousal : null;
      const vramNorm = (typeof snap.vram_norm === 'number') ? snap.vram_norm : null;
      const temp = (typeof snap.gpu_temp_c === 'number') ? snap.gpu_temp_c : null;

      _setKpi('dash-fk-som', 'dhero-arousal', aro);
      _pushSpark('som', aro, 'dash-spk-som');

      const heroVram = $('dhero-vram');
      if (heroVram && vramNorm !== null) heroVram.textContent = (vramNorm * 24).toFixed(1) + ' GB';

      // Context drawer vitals
      const bpmEl = $('ctx-vital-bpm');
      if (bpmEl && typeof aro === 'number') {
        // bpm metaforico = 60 + arousal * 30 (range 60-90)
        bpmEl.textContent = Math.round(60 + aro * 30);
      }
      const ctxVram = $('ctx-vital-vram');
      if (ctxVram && vramNorm !== null) {
        ctxVram.textContent = (vramNorm * 24).toFixed(1) + ' / 24 GB';
      }
      const ctxSom = $('ctx-vital-som');
      if (ctxSom && typeof aro === 'number') {
        ctxSom.textContent = `aro ${aro.toFixed(2)}` + (temp !== null ? ` · ${Math.round(temp)}°C` : '');
        ctxSom.style.color = (aro > 0.6) ? 'var(--amber)' : 'var(--text-primary)';
      }

      // Step 6/8 fix: lega il synapse hero a arousal somatico
      _updateSynapseVars({ aro });
    } catch (e) { /* fail silent */ }
  }

  // ─── Step 6/8 BIND: applica CSS variables al synapse hero in base a stato
  //     coherence → opacity/intensity del nucleo
  //     arousal   → velocita' rotazione anelli e halo, frequenza pulse core
  //     stability → calma drift particelle
  function _updateSynapseVars({ coh, aro, sms } = {}) {
    const root = document.documentElement;
    if (typeof coh === 'number' && isFinite(coh)) {
      // coh 0.2..1.0 → intensity 0.4..1.0
      const intensity = Math.max(0.4, Math.min(1.0, 0.4 + 0.6 * coh));
      root.style.setProperty('--syn-coh-intensity', intensity.toFixed(3));
    }
    if (typeof aro === 'number' && isFinite(aro)) {
      // aro 0..1 → speed multiplier 0.6..1.8 (più aroused = più veloce)
      const speed = Math.max(0.6, Math.min(1.8, 0.6 + aro * 1.2));
      root.style.setProperty('--syn-aro-speed', speed.toFixed(3));
      // Pulse core: 4s a riposo, 2s in alta attivazione
      const pulseDur = Math.max(2.0, Math.min(4.5, 4.5 - aro * 2.5));
      root.style.setProperty('--syn-pulse-dur', pulseDur.toFixed(2) + 's');
    }
    if (typeof sms === 'number' && isFinite(sms)) {
      // stab 0..1 → drift speed multiplier inverso (più stabile = più lento)
      const driftSpeed = Math.max(0.7, Math.min(1.6, 1.6 - sms * 0.9));
      root.style.setProperty('--syn-drift-speed', driftSpeed.toFixed(3));
    }
  }

  function _setKpi(footerId, heroId, val) {
    const f = $(footerId);
    const h = $(heroId);
    const txt = (typeof val === 'number') ? val.toFixed(2) : '--';
    if (f) f.textContent = txt;
    if (h) h.textContent = txt;
  }

  function _pushSpark(key, val, containerId) {
    if (typeof val !== 'number' || !isFinite(val)) return;
    const buf = _sparkBuf[key];
    buf.push(val);
    while (buf.length > SPARK_LEN) buf.shift();
    const el = $(containerId);
    if (!el) return;
    if (el.children.length !== buf.length) {
      el.innerHTML = '';
      buf.forEach(() => el.appendChild(document.createElement('span')));
    }
    const max = Math.max(...buf, 0.001);
    [...el.children].forEach((bar, i) => {
      const h = Math.max(2, (buf[i] / max) * 100);
      bar.style.height = h + '%';
    });
  }

  // ─── Step 6/8: Context drawer LIVE — pensieri SIC ────────────────────────
  async function fetchSIC() {
    try {
      const r = await fetch('/api/inner_stream/buffer?n=5', { cache: 'no-cache' });
      if (!r.ok) return;
      const arr = await r.json();
      const list = $('ctx-sic-list');
      const countEl = $('dash-sic-count');
      if (countEl) countEl.textContent = arr.length;
      if (!list) return;
      if (!arr.length) {
        list.innerHTML = '<span class="ctx-empty">nessun pensiero recente</span>';
        return;
      }
      list.innerHTML = arr.slice(0, 4).map(p => {
        const ts = (p.timestamp || '').slice(11, 16);  // HH:MM
        const txt = (p.text || '').trim().slice(0, 220);
        return `<div class="ctx-item">
          <div class="ctx-item-meta">${ts} &middot; ${p.mode || 'sic'}</div>
          <div class="ctx-item-text">${_escapeHtml(txt)}</div>
        </div>`;
      }).join('');
    } catch (e) { /* fail silent */ }
  }

  // ─── Step 6/8: Context drawer LIVE — richiamo (episodi flagged) ──────────
  async function fetchRecall() {
    try {
      const r = await fetch('/api/sleep/flagged?limit=3', { cache: 'no-cache' });
      if (!r.ok) return;
      const j = await r.json();
      const arr = j.flagged || [];
      const list = $('ctx-recall-list');
      const empty = $('ctx-empty');
      if (!list) return;
      if (!arr.length) {
        list.innerHTML = '';
        if (empty) empty.style.display = '';
        return;
      }
      if (empty) empty.style.display = 'none';
      list.innerHTML = arr.map(e => {
        const ts = (e.ts || '').slice(11, 16);
        const date = (e.ts || '').slice(0, 10);
        const summary = (e.summary || '(senza sintesi)').slice(0, 180);
        return `<div class="ctx-item">
          <div class="ctx-item-meta">${date} ${ts} &middot; flagged</div>
          <div class="ctx-item-text">${_escapeHtml(summary)}</div>
        </div>`;
      }).join('');
    } catch (e) { /* fail silent */ }
  }

  function _escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  // ─── Step 6/8: paint halo synapse in base a stato reale episodi.
  //     12 dot totali. La proporzione flagged/total decide quanti rosa.
  //     Il numero di "fact" decide quanti dot extra-glow cyan-hi.
  //     Distribuzione: i dot flagged li mettiamo "sparsi" sui multipli di 3
  //     per non concentrarli, fact sui multipli di 4. Resto = verified cyan.
  function _paintHalo(verifiedCount, flaggedCount, factCount) {
    const dots = document.querySelectorAll('.halo-dot');
    if (!dots.length) return;
    const total = dots.length;
    const totalEpisodes = verifiedCount + flaggedCount;
    const flaggedDots = (totalEpisodes > 0)
      ? Math.max(0, Math.min(total - 1, Math.round((flaggedCount / totalEpisodes) * total)))
      : 0;
    // Posizioni "sparse" per flagged: skip pattern (es. ogni 3 partendo dal 2)
    const flaggedIdx = new Set();
    let cnt = 0, step = 3, pos = 1;
    while (cnt < flaggedDots && pos < total) {
      flaggedIdx.add(pos);
      pos += step; cnt++;
    }
    // Fact dots: 1 ogni 50 fact (max 2 dot)
    const factDots = Math.min(2, Math.max(0, Math.floor((factCount || 0) / 50)));
    const factIdx = new Set();
    for (let i = 0; i < factDots; i++) factIdx.add(total - 1 - i * 4);

    dots.forEach((d, i) => {
      d.classList.remove('flagged','fact','dim');
      if (flaggedIdx.has(i))      d.classList.add('flagged');
      else if (factIdx.has(i))    d.classList.add('fact');
      else if (totalEpisodes === 0) d.classList.add('dim');  // tutto vuoto: dot affievoliti
    });
  }

  // Trigger pulse animation su un halo-dot casuale (segnale visivo di
  // attivita': nuovo messaggio = encoding probabile = atterraggio episodio).
  function _pulseRandomHaloDot() {
    const dots = document.querySelectorAll('.halo-dot:not(.dim)');
    if (!dots.length) return;
    const idx = Math.floor(Math.random() * dots.length);
    const d = dots[idx];
    d.classList.add('pulse');
    setTimeout(() => d.classList.remove('pulse'), 1300);
  }

  // ─── Graph Kuzu via /api/graph/status (polling solo se dashboard visibile)
  // Struttura: j.episode_count, j.fact_count, j.concept_count, j.trait_count direttamente.
  // Step 6/8: stats ora include verified_count/flagged_count per colorare l'halo.
  async function fetchKuzuSnapshot() {
    try {
      const r = await fetch('/api/graph/status', { cache: 'no-cache' });
      if (!r.ok) return;
      const j = await r.json();
      const pick = (a, b) => (typeof j[a] === 'number') ? j[a]
                          : (typeof j[b] === 'number') ? j[b] : null;
      const epCount = pick('episode_count','episodes');
      const fcCount = pick('fact_count','facts');
      const map = {
        'dash-kz-ep': epCount,
        'dash-kz-fc': fcCount,
        'dash-kz-cc': pick('concept_count','concepts'),
        'dash-kz-tr': pick('trait_count','traits'),
      };
      Object.entries(map).forEach(([id, v]) => {
        const el = $(id);
        if (el) el.textContent = (typeof v === 'number') ? v.toLocaleString('it-IT') : '--';
      });

      // Step 6/8: paint halo synapse con proporzioni reali
      const ver = (typeof j.verified_count === 'number') ? j.verified_count : 0;
      const fla = (typeof j.flagged_count  === 'number') ? j.flagged_count  : 0;
      _paintHalo(ver, fla, fcCount);
      const sleep = $('dash-kz-sleep');
      const ctxSleep = $('ctx-vital-sleep');
      const ago = j.last_sleep_ago_minutes ?? j.last_sleep_min;
      let sleepTxt = '--';
      if (typeof ago === 'number') {
        sleepTxt = ago < 1 ? 'in corso' : `${Math.round(ago)} min fa`;
      } else if (j.last_sleep_at) {
        sleepTxt = String(j.last_sleep_at).slice(11, 19);
      }
      if (sleep) sleep.textContent = sleepTxt;
      if (ctxSleep) ctxSleep.textContent = sleepTxt;
    } catch (e) { /* fail silent */ }
  }

  // ─── Chat ticker: ascolta ultimo messaggio in #messages
  function renderChatTicker() {
    const msgs = $('messages');
    if (!msgs) return;
    const last = msgs.querySelector('.message:last-child');
    if (!last) return;
    const isEden = last.classList.contains('eden');
    const bubble = last.querySelector('.bubble');
    const whoEl = $('dash-chat-who');
    const msgEl = $('dash-chat-msg');
    if (whoEl) whoEl.textContent = isEden ? 'EDEN' : 'TU';
    if (msgEl) msgEl.textContent = bubble ? (bubble.textContent || '').trim().slice(0, 200) : '';
  }

  // ─── Apri chat button
  const openChatBtn = $('dash-chat-open');
  if (openChatBtn) {
    openChatBtn.addEventListener('click', () => {
      if (typeof switchSection === 'function') switchSection('chat');
    });
  }

  // ─── MutationObserver passivi sui sorgenti dati esistenti ───────────────
  // Quando script.js esistente aggiorna i traits/stats/affective/autonomy,
  // questi observer rinfrescano la dashboard senza alcuna modifica al code path.
  function _observe(srcId, cb) {
    const el = $(srcId);
    if (!el) return;
    const mo = new MutationObserver(cb);
    mo.observe(el, { childList: true, characterData: true, subtree: true });
  }

  TRAIT_KEYS.forEach(name => {
    _observe(`trait-${name}`, renderRadar);
  });
  ['stat-exchange','stat-sessions','stat-history','stat-memories'].forEach(id => _observe(id, renderStats));
  ['af-valence','af-arousal','af-certainty'].forEach(id => _observe(id, renderAffective));
  ['auto-running','auto-mode','auto-interval','auto-decisions'].forEach(id => _observe(id, renderAutonomy));
  _observe('relationship-badge', renderRelationship);
  _observe('messages', renderChatTicker);

  // Step 6/8: observer dedicato per nuovi .message in #messages.
  // Quando appare un nuovo nodo .message, triggero un halo pulse
  // (rappresenta visivamente un episodio in encoding o richiamo).
  const msgsEl = $('messages');
  if (msgsEl) {
    const msgMo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'childList') {
          for (const node of m.addedNodes) {
            if (node.nodeType === 1 && node.classList && node.classList.contains('message')) {
              _pulseRandomHaloDot();
              break;
            }
          }
        }
      }
    });
    msgMo.observe(msgsEl, { childList: true });
  }

  // ─── Render iniziale (anche se i dati sono "--" si setta il layout)
  function renderAll() {
    renderRadar();
    renderAffective();
    renderStats();
    renderAutonomy();
    renderRelationship();
    renderHero();
    renderChatTicker();
  }
  renderAll();

  // Hero stats si aggiornano ogni secondo (uptime, idle)
  setInterval(renderHero, 1000);

  // ─── Polling L1 + Somatic + Kuzu + SIC + Recall solo se dashboard
  //     visibile o context drawer aperto.
  let _sicPoll = null, _recallPoll = null, _somPoll = null;

  function _isDashboardVisible() {
    const s = $('section-dashboard');
    return s && !s.classList.contains('hidden');
  }
  function _isContextOpen() {
    return document.documentElement.classList.contains('chat-context-open');
  }
  function _shouldPoll() {
    return _isDashboardVisible() || _isContextOpen();
  }
  function _startPolling() {
    if (!_l1Poll)     { fetchL1Snapshot();       _l1Poll     = setInterval(fetchL1Snapshot,       5000); }
    if (!_somPoll)    { fetchSomaticSnapshot();  _somPoll    = setInterval(fetchSomaticSnapshot,  5000); }
    if (!_kuzuPoll)   { fetchKuzuSnapshot();     _kuzuPoll   = setInterval(fetchKuzuSnapshot,    30000); }
    if (!_sicPoll)    { fetchSIC();              _sicPoll    = setInterval(fetchSIC,              8000); }
    if (!_recallPoll) { fetchRecall();           _recallPoll = setInterval(fetchRecall,          30000); }
  }
  function _stopPolling() {
    if (_l1Poll)     { clearInterval(_l1Poll);     _l1Poll     = null; }
    if (_somPoll)    { clearInterval(_somPoll);    _somPoll    = null; }
    if (_kuzuPoll)   { clearInterval(_kuzuPoll);   _kuzuPoll   = null; }
    if (_sicPoll)    { clearInterval(_sicPoll);    _sicPoll    = null; }
    if (_recallPoll) { clearInterval(_recallPoll); _recallPoll = null; }
  }

  // Observer sulla classe hidden della section-dashboard
  const dashEl = $('section-dashboard');
  if (dashEl) {
    const mo = new MutationObserver(() => {
      if (_shouldPoll()) _startPolling(); else _stopPolling();
    });
    mo.observe(dashEl, { attributes: true, attributeFilter: ['class'] });
  }
  // Observer su html per chat-context-open
  const htmlMo = new MutationObserver(() => {
    if (_shouldPoll()) _startPolling(); else _stopPolling();
  });
  htmlMo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

  // Init: se gia' applicabile, avvia
  if (_shouldPoll()) _startPolling();

})();

// ════════════════════════════════════════════════════════════════════════════
// STEP 5/8 redesign — Chat focused + sidebar/context drawer toggle
//
// Attiva sempre la classe html.chat-focused: layout chat full-width,
// sidebar/context come drawer slide-in.
// Edge-tab sx (#btn-open-avatar) apre la sidebar esistente.
// Edge-tab dx (#btn-open-context) apre #chat-context-drawer (popolato step 6).
// ════════════════════════════════════════════════════════════════════════════

(function setupChatFocused() {

  // Abilita il layout focused sempre — selettori esistenti restano in DOM
  document.documentElement.classList.add('chat-focused');

  const btnAvatar   = document.getElementById('btn-open-avatar');
  const btnContext  = document.getElementById('btn-open-context');
  const btnCloseCtx = document.getElementById('btn-close-context');
  const ctxDrawer   = document.getElementById('chat-context-drawer');
  const backdrop    = document.getElementById('chat-drawer-backdrop');

  function openSidebar() {
    closeContext();
    document.documentElement.classList.add('chat-sidebar-open');
    if (btnAvatar) btnAvatar.setAttribute('aria-expanded', 'true');
  }
  function closeSidebar() {
    document.documentElement.classList.remove('chat-sidebar-open');
    if (btnAvatar) btnAvatar.setAttribute('aria-expanded', 'false');
  }
  function openContext() {
    closeSidebar();
    document.documentElement.classList.add('chat-context-open');
    if (btnContext) btnContext.setAttribute('aria-expanded', 'true');
    if (ctxDrawer)  ctxDrawer.setAttribute('aria-hidden', 'false');
  }
  function closeContext() {
    document.documentElement.classList.remove('chat-context-open');
    if (btnContext) btnContext.setAttribute('aria-expanded', 'false');
    if (ctxDrawer)  ctxDrawer.setAttribute('aria-hidden', 'true');
  }
  function closeAll() { closeSidebar(); closeContext(); }

  if (btnAvatar)   btnAvatar.addEventListener('click',   openSidebar);
  if (btnContext)  btnContext.addEventListener('click',  openContext);
  if (btnCloseCtx) btnCloseCtx.addEventListener('click', closeContext);
  if (backdrop)    backdrop.addEventListener('click',    closeAll);

  // Esc chiude drawer chat-focused (priorita' sull'Esc del drawer mobile gia' esistente
  // — l'altro chiude solo se drawer-open class e' presente, quindi non c'e' conflitto)
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (document.documentElement.classList.contains('chat-sidebar-open') ||
        document.documentElement.classList.contains('chat-context-open')) {
      closeAll();
    }
  });

  // Quando la sezione attiva NON e' chat, chiudi entrambi i drawer
  // (osserviamo la classe hidden di #chat-section per tutti i cambi sezione)
  const chatSec = document.getElementById('chat-section');
  if (chatSec) {
    const mo = new MutationObserver(() => {
      if (chatSec.classList.contains('hidden') || mainChatEl?.classList.contains('hidden')) {
        closeAll();
      }
    });
    mo.observe(chatSec, { attributes: true, attributeFilter: ['class'] });
    if (mainChatEl) mo.observe(mainChatEl, { attributes: true, attributeFilter: ['class'] });
  }

  // Click su elementi che cambiano sezione dentro la sidebar (es. #btn-archive
  // mostra #archive-panel sostituendo chat-section) deve continuare a funzionare.
  // Lasciamo il drawer aperto: se #archive-panel si attiva, l'utente lo vede
  // dietro il drawer sidebar. Comportamento desiderabile: chiudi sidebar quando
  // si apre archive o quando si clicca un'action interna che cambia "view".
  const btnArchive       = document.getElementById('btn-archive');
  const btnArchiveClose  = document.getElementById('btn-archive-close');
  const btnArchiveBack   = document.getElementById('btn-archive-back');
  const btnSessionResume = document.getElementById('btn-session-resume');
  const btnNewSession    = document.getElementById('btn-new-session');
  [btnArchive, btnArchiveClose, btnArchiveBack, btnSessionResume, btnNewSession]
    .forEach(b => { if (b) b.addEventListener('click', () => setTimeout(closeAll, 50)); });

})();


