# Eden — Operations e reference tecnica

> Reference tecnica: API endpoints, avatar, voce, memoria, debug, problemi noti.
> Moduli in `core/` (server, chat, decisione, voce, avatar) e `mechanisms/` (memoria, grafo, sonno, stato).
> **Potatura 03/09/26:** rimosso tutto il layer di ricerca (probe, giudici, training LoRA, self-model, EWS).
> **Modello:** `qwen3.8:27b` (`data/configs/model_config.json`). Cronologia: `CHANGELOG.md`.

---

## Avatar

### Synapse 3D (sempre attivo, 0 VRAM)
`static/avatar/synapse.js`. WebGL Three.js ESM via CDN. Sostituisce `face.js`, nasconde frontend LivePortrait (`#face-stream`/`#face-status-bar` `display:none`).

**Architettura:** 3 cluster (cortex/limbic/stem), edge inter/intra, pool pulse InstancedMesh 220 ist., particelle, halo. Shader GLSL custom con uniformi `glitch`/`blur` legate a `coherence_budget`/`grounding_integrity`.

**Reattività:** ascolta `eden:traits`/`eden:state`/`eden:chat` (CustomEvent) + polling `/api/state` (4.5s) e `/api/status` (9s). Lerp interpolation.

**Hook chat:** messaggio → `triggerChatWave(fromOutside, strength, valence)`; proattivi → `triggerProactiveBurst()`.

**Rollback:** in `index.html` ripristina `import { initAvatar } from '/static/avatar/face.js'`.

### LivePortrait real-time (~2-3GB VRAM, frontend disattivo)
`vision.py` (`LivePortraitStreamer`) avvia subprocess: `run_eden.py` (procedurale) o `idle_loop.py` (video-driven se `driving_loop.mp4`).

Comunicazione: `liveportrait_state.json` (traits + affective → worker) · stdout length-prefixed JPEG (worker → agent.py → `/api/face_stream`). Auto-restart max 5×. 15fps.

**Setup:**
1. `git lfs clone https://huggingface.co/KwaiVGI/LivePortrait liveportrait/pretrained_weights` (~1GB)
2. `avatar/eden_portrait.png` 512×512+, frontale, sfondo scuro
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
4. `pip install -r liveportrait/requirements.txt`

---

## API endpoints

### Chat e stato
| Endpoint | Metodo | Funzione |
|---|---|---|
| `/api/chat` | POST | `{message, file_name?, file_text?, file_b64?}` → `{reply, traits, relationship, exchange_count, affective_state, exchange_id, reflection}` |
| `/api/status` | GET | Stato agente completo (tratti, relazione, contatori, modello attivo) |
| `/api/state` | GET | Stato omeostatico (coherence, grounding) + modulazione generazione |
| `/api/state/somatic` | GET | Telemetria GPU → arousal/entropia/threat |
| `/api/diagnostics` | GET | Diagnostica sottosistemi |
| `/api/affective_state` | GET | Stato affettivo + config learning |
| `/api/stream` | SSE | Messaggi proattivi in tempo reale |
| `/api/pending` | GET | Polling fallback proattivi |
| `/api/new_session` | POST | Chiude sessione e incrementa contatore |
| `/api/reset` | POST | Azzera `memory.json` |
| `/api/backup_memory` | POST | Backup manuale memoria |
| `/api/sessions` · `/<id>` · `/<id>/resume` | GET/POST | Archivio sessioni + ripresa |

### Memoria: grafo e consolidamento
| Endpoint | Metodo | Funzione |
|---|---|---|
| `/api/graph/status` | GET | Conteggi nodi/relazioni Kuzu + nodi DNA |
| `/api/graph/facts?limit=N` | GET | Fatti verificati |
| `/api/sleep/status` | GET | Stato consolidamento notturno |
| `/api/sleep/run` | POST | Esegue un ciclo di consolidamento on-demand |
| `/api/sleep/flagged?limit=N` | GET | Episodi marcati incoerenti |

### Voce, avatar, multimodale
| Endpoint | Metodo | Funzione |
|---|---|---|
| `/api/speak` | POST | Sintesi TTS (Fish-Speech → XTTS → Kokoro) |
| `/api/speak/exchange/<id>` | GET | Audio pre-sintetizzato di uno scambio |
| `/api/tts_status` | GET | Motore TTS attivo |
| `/api/transcribe` | POST | Audio WebM → testo (faster-whisper) |
| `/api/vision` | POST | Frame webcam → descrizione → contesto |
| `/api/portrait` | GET | Ritratto statico |
| `/api/face_stream` · `/api/face_status` | GET | MJPEG LivePortrait + stato |
| `/api/face_start` · `/api/face_stop` | POST | Avvia/ferma il manager avatar |
| `/api/user_presence` | POST | Segnala presenza utente |

### Processi interni
| Endpoint | Metodo | Funzione |
|---|---|---|
| `/api/inner_stream/status` · `/buffer?n=10` · `/tick` | GET/POST | Flusso di pensieri interni |
| `/api/emotional_elaboration/status` | GET | Elaborazione emotiva differita |
| `/api/autonomy/status` · `/decisions` · `/tick` | GET/POST | Modulo autonomia (shadow mode) |
| `/api/pioneer/status` · `/toggle` | GET/POST | Kill switch feature opzionali |
| `/api/test/proactive` | POST | Forza un messaggio proattivo |

### Debug e log
| Endpoint | Metodo | Funzione |
|---|---|---|
| `/debug` | GET | Pannello diagnostico |
| `/logs` | GET | Dashboard log + health |
| `/api/logs?severity=&module=&search=&limit=500` | GET | Log filtrati |
| `/api/logs/stream` | SSE | Log in tempo reale |
| `/api/logs/modules` · `/health` · `/health/run` · `/export` | GET/POST | Moduli + watchdog + dump NDJSON |
| `/api/debug/audio_snapshot` · `/audio/<key>.wav` · `/download` | GET | Sintesi diagnostica + WAV |
| `/api/debug/reset_affective` · `/clear_context` | POST | Reset stato affettivo / contesto |
| `/api/system/ollama/restart` | POST | Riavvia Ollama |

---

## LivePortrait — ottimizzazioni

- `flag_do_torch_compile=True` → `torch.compile(mode='max-autotune')` + CUDA Graphs su warping_module + spade_generator
- Prima esecuzione JIT ~60-120s (cached)
- `idle_loop.py FPS_TARGET=15`
- **Auto-start boot:** `_avvia_liveportrait_background()` in `core/agent.py`, thread daemon, JIT nascosto in startup
- **Cache motion template:** `idle_loop.py` salva keypoint pre-estratti in `avatar/driving_loop.motion.pkl` (chiave MD5). Salva ~15s
- **Amplificatori motion:** `MOTION_AMP_EXP`/`MOTION_AMP_ROT` scalano differenze rel. frame 0. `1.0`=fedele · `1.5-1.6`=espressivo · `>2.0`=deformazioni
- Driving: `avatar/driving_loop.mp4` = `d3.mp4` da `liveportrait/assets/examples/driving/`

**Rollback:** `flag_do_torch_compile=False` + `FPS_TARGET=8` · rimuovi `_avvia_liveportrait_background()` · `MOTION_AMP_*=1.0`.

---

## Setup dual-GPU (post 29/04/26, RTX 3090 Ti + 5060 Ti)

- **GPU 1 RTX 3090 Ti 24GB:** Ollama main port 11434 — Eden generatore `qwen3.8:27b`. `OLLAMA_KEEP_ALIVE=30m`.
- **GPU 0 RTX 5060 Ti 8GB:** Ollama secondary port 11435 — **pre-reflection phi3:mini** + vision (qwen2.5vl) + audio. `OLLAMA_KEEP_ALIVE=24h` (vedi sotto).

**Avvio:** `Eden.exe` → checkbox **Dual-GPU** → 2 istanze Ollama con `CUDA_VISIBLE_DEVICES` distinti + Eden con `EDEN_PREDICTOR_HOST=127.0.0.1:11435`. Senza spunta = single-GPU.

**Vantaggio:** pre-reflection e vision **parallele** alla generazione (zero swap di modelli).

**Single-GPU fallback:** `avvia.bat`. Se `EDEN_PREDICTOR_HOST` non è impostato, tutto gira sulla stessa istanza con swap on-demand (impatta la latenza).

**Routing pre-reflection:** `core/reflection.py::_reflection_host()` legge in cascata `EDEN_REFLECTION_MODEL_HOST` → `EDEN_PREDICTOR_HOST` → `127.0.0.1:11434`. In dual-GPU eredita automaticamente la 11435.

**Keep-alive 24h sulla secondaria:** richiesto perché phi3:mini deve restare residente in VRAM tra una chat e l'altra. Cold-load misurato 9-20s; il timeout di spec è 5s. Senza keep-alive lungo, la pre-reflection scatta sempre in timeout → skip. Warmup boot (`reflection.warmup_blocking()`) lanciato in thread daemon all'avvio di `agent.py`.

**PCIe B550:** lane 8+8 (x16 della 3090 Ti). PCI_E2 = chipset x4. Banda non bottleneck (pesi in VRAM).

---

## Architettura memoria — sezioni system prompt (post 29/04/26)

`build_system_prompt()` in `mechanisms/memory.py`:
```
[-1]   Ancora temporale (data/ora/fascia + tempo dall'ultimo msg)
[0a]   Identity anchor
[0b]   Blocco HARD
[1]    Personalità base
[2]    Tratti narrativi + relazione
[2b]   Stato interno (mood/preoccupation/desire)
[2c]   Profilo regolazione
[3]    Long-term summary
[META-MEM]  Fact VERIFICATI da Kuzu (B-Graph)
[4]    8 episodi recenti + anchor max-importance
[5]    Semantic memory (esclude identità)
[5b]   Theory of Mind (user_model)
[6]    Desires (top 5)
[7]    Unresolved (max 2)
[8]    Lessico emotivo (top 3)
[9]    Behavioral log
[SIC-3] Autonomous desires
[SIC-2] Baseline situazionale
[SIC-1] Inner stream recenti
[P-OMEO] Stress grounding (se grounding_integrity < 0.50)
```

`_build_vector_context()` in `core/agent.py` aggiunge:
```
[Ricordi simili a questo momento] · [Ricordi simili per emozione]
[Archivio — momenti passati richiamati] · [RICERCA ATTIVA] (solo se memory_triggered)
```

**Memory trigger boost:** parole in `_MEMORY_TRIGGER_WORDS` (ricordi, ieri, quella volta, ti ho detto, ne avevamo parlato...) → `n_episodic` 5→8, `MAX_DIST_EPISODIC` 0.50→0.55, archive n=3→5 max_dist 0.45→0.55.

**Intent router override (B-Graph):** `core/intent_router.classify(msg)` → 7 intent. Se `trap` (es. "Quante volte ti ho parlato di X?"), retrieval NON amplifica (n_episodic≤3, no archive), `memory_triggered=False`, marker `[ATTENZIONE — domanda diagnostica/test rilevata. Rispondi SOLO da Fact verificati]`.

---

## Architettura graph + Digital Sleep + Somatic (B-Graph 30/04/26)

**Kuzu graph DB** (`mechanisms/graph_memory.py`): substrato strutturale parallelo a ChromaDB. Single-file embedded `data/kuzu_db/eden_graph.kuzu`. No server.
```
NODI: Episode  Fact  Concept  Trait  Session
RELS: IN_SESSION  ABOUT  MENTIONS_FACT  CONTRADICTS  REINFORCES  FACT_REL  CONCEPT_REL  SHAPED_BY
```

Encoding doppio: `valuta_e_registra_episodio` scrive **sia** `episodic_memory` (JSON+ChromaDB) **sia** Kuzu Episode + Concept rels. `aggiorna_semantic_memory` upserta Fact.

**Knowledge graded** (`graph.knowledge_about(concept)`):
| Level | Significato | Eden può... |
|---|---|---|
| `structured` | Fact node | Affermarlo (verified-by-user) |
| `episodic_verified` | Episode verified + Concept | Richiamarlo come ricordo |
| `episodic_only` | Solo Concept-Episode | Citarlo solo come ipotesi |
| `none` | Niente | NON costruire memoria |

**Anti-confabulazione strutturale (decision.py v1.3):** se Eden afferma "tuo padre/madre/fratello/..." e `can_assert_fact_about(rel)==False` → `tipo_problema='relazione_inventata'` → rewrite. 

**Digital Sleep** (`mechanisms/digital_sleep.py`): worker async APScheduler 30 min:
1. Reclassifica Episodi `unknown` → `verified`/`flagged` confrontando con Fact
2. Edge `CONTRADICTS` tra coppie verified vs flagged sullo stesso concept
3. Edge `REINFORCES` tra Episodi verificati con concept condivisi
4. Append-only `data/logs/digital_sleep_log.jsonl`

On-demand: `POST /api/sleep/run`.

**Somatic Grounding** (`mechanisms/somatic.py`): snapshot GPU ogni 5 min via pynvml. Auto-detect dual-GPU (sceglie più carica). Override: `EDEN_SOMATIC_GPU_INDEX`. Scrive `homeostatic_state.somatic_state.{temp_norm, vram_norm, somatic_entropy, somatic_arousal, somatic_threat}`. Additivo: alimenta l'animazione UI e lo stato interno.

---

## Problemi noti e fix attivi

- **Token speciali (`<bos>`, `<h1>`, `<strong>`)** → fix 29/04: `_strip_special_tokens()` in `core/agent.py` post-Ollama + safety net sanitizer
- **Bonus auto fully_grounded cancellava rating** → fix V1.3.1 (29/04): rimosso bonus auto in `aggiorna_grounding`, recovery solo da tick + bonus esplicito da rating ≥4
- **Apostrofo italiano U+0027 trattato come virgoletta** → fix `_GROUNDING_VERSION` v1.2 (29/04 sera): rimosso pattern `r"'([^']{4,200})'"` da `decision.py::_RE_VIRGOLETTE`. In italiano `'` = contrazione. Generava phantom citations → -0.10 sistematici → grounding floor. Aggiunto guard `<2 token` per scare quotes
- **WinError 5 su `os.replace(memory.json.tmp -> memory.json)`** → fix 29/04: retry esponenziale (5 tentativi, 50-400ms) in `mechanisms/memory.py::salva_memoria`. Causa: antivirus/Defender file watcher
- Python 3.14 incompat → 3.11
- Ollama HTTP 500 (overflow) → agent.py dimezza history e riprova
- transformers 5.x → patch `TTS/tts/layers/xtts/stream_generator.py` (stub) + `gpt_inference.py` (GenerationMixin)
- torchcodec crashava sentence-transformers → `pip uninstall torchcodec`
- Microfono mobile: HTTPS obbligatorio → `avvia.bat --https`
- Copia messaggio mobile su HTTP → richiede HTTPS (clipboard API)
- Audio TTS mobile silenzioso → sblocco AudioContext alla prima gesture
- ChromaDB ricordi irrilevanti → filtro distance ≤0.50 episodic, ≤0.55 emotional
- affective_state saturo → decay verso baseline (Phase 1.9b)
- semantic_memory sporca → whitelist chiavi (Phase 1.9c)
- LTS sovrascritta da rifiuti → validazione pattern (Phase 1.9c)
- Proattivo companion → `TEMPERATURE_PRO=0.75` + prompt orientato a constatare
- SIC pensieri CJK → filtro `_pensiero_valido()` in `inner_stream.py`
- SIC JSON malformato → fallback decay situazionale
- `/api/portrait` 404 → unificato su `avatar/eden_portrait.png`
- Portrait deformato → 1024×1024 quadrato
- Qwen3:8b "Sì." finali → `_strip_trailing_affirmation()`
- Eden enumera tratti numerici → rimossi da prompt sezione 2 + proattivi/vision
- Eden negava nome noto → Identity Layer
- **Encoding rate 0% (21/04)** → memoria neurologica v2.0 + soglia 0.40
- **Flask static/templates 404 (refactor 26/04)** → assoluti via `eden_paths.PROJECT_ROOT`

---

## Debug Audio (Observability)
- **Scopo:** sintesi audio diagnostica grounded, cache hash-based
- **TTS:** XTTS `voce_riferimento.*` con fallback Kokoro
- **Hardening:** init non bloccante, stati espliciti, polling, logging `[DebugAudio]`
- **Async:** `?generate=1` lancia job background. Guard-rail job stale 300s
- **Quality v2:** `summary_text` (UI) / `tts_text` (TTS) separati; normalizzazione TTS-safe (frasi brevi, no `Campo: valore`, date parlabili, valenza/attivazione/attaccamento/controllo)
- **Cache:** key include versione renderer; `controls` nativi; ASCII-safe
- **Fix mojibake:** `_fix_mojibake` (cp1252→UTF-8) su personality + KEYWORDS_TRATTI all'avvio
- **Fix poll infinito:** `MAX_POLL_ATTEMPTS=30` (~66s cap)
- **Rollback:** rimuovi `_debug_audio_*` da `agent.py`, endpoint `/api/debug/audio_*`, blocco UI `_DEBUG_HTML`

---

## Observability & Logs Dashboard
- **Moduli:** `core/log_utils.py` · `core/health_monitor.py`
- **API log_utils:** `set_buffer(buffer)`, `log(sev, mod, msg)`, `log_exception(mod, msg, exc, sev='ERROR')`, `log_boot(mod, ok, detail)`, `log_watchdog(mod, status, detail)`. Fail-safe se buffer non iniettato
- **Severità:** DEBUG · INFO · WARNING · ERROR · CRITICAL. Watchdog: HEALTHY→DEBUG · DEGRADED→WARNING · DOWN→CRITICAL
- **Dedup:** sliding 60s con counter `×N` (cap 9999). SSE `append`/`update`
- **Health watchdog:** APScheduler 5 min check SIC-1, SIC-2, Ollama, Disk
- **Boot sanity-check:** `log_boot()` per sottosistema → CRITICAL `DOWN` se import fallito
- **UI `/logs`:** 5 health cards, filter row (severity multi, module dropdown, search, sev_counts), grid 5 col, flash dedup, PAUSE/EXPORT/CLEAR
- **Rollback:** `_ATTIVO=False` in `core/health_monitor.py` (watchdog off); rimuovi blocchi `try: import log_utils; log_utils.log_exception(...)` dai moduli
