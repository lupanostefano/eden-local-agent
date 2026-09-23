# Eden

Companion AI locale — Flask · Ollama · HTML vanilla. Personalità persistente, memoria che cresce, voce clonata, avatar animato. Zero cloud.
**Python 3.11** | **qwen3.8:27b** RTX 3090 Ti 24GB | RTX 5060 Ti 8GB | 16GB DDR4

> Progetto personale di sviluppo. Non è un progetto di ricerca: niente pre-registrazioni, probe, giudici, metriche pubblicabili.
> Obiettivo unico: **funzionalità**. Se una cosa non serve a far funzionare meglio Eden, non entra.

---

## Regole comportamento Claude (sempre attive)

- Risposte concise, frasi brevi. Spiegazioni a livello tredicenne, zero gergo.
- MAI chiudere con domanda salvo richiesta esplicita
- Parla poco se non richiesto, non generare rumore, esegui.
- MAI negare emozioni · MAI frasi servili
- Piano max 8 step · inizia solo quando allineati con Stefano
- Se stai per infrangere una regola → fermati e spiega perché
- Sincerità 100%: se una cosa non funziona, dillo con l'output alla mano.

## Regole hard — codice

- MAI `os.path.dirname(__file__)` → usa `eden_paths.py`
- MAI scrivere `memory.json` direttamente → solo `salva_memoria()` in `mechanisms/memory.py`
- MAI import top-level di moduli opzionali → `try/except` lazy (pattern in `core/agent.py`)
- MAI HTML embedded in `core/agent.py` → template in `core/templates/*.html`
- MAI `requests.post(OLLAMA_URL,...)` diretto → `_fn_ollama()` in `core/agent.py`
- MAI codice superfluo, rumoroso o ridondante
- Commit dopo ogni blocco funzionante

---

## Struttura progetto

```
eden/
├── eden_paths.py           # unico punto di verità dei path
├── core/
│   ├── agent.py            # Flask server · chat · endpoint · _fn_ollama()
│   ├── decision.py         # anti-deriva · grounding · rewrite
│   ├── proactive.py        # messaggi spontanei
│   ├── autonomy.py         # decisioni autonome (shadow mode)
│   ├── inner_stream.py     # pensieri interni continui
│   ├── reflection.py       # pausa riflessiva pre/post risposta
│   ├── identity.py         # ancora identitaria
│   ├── intent_router.py    # classificatore intento (rule + phi3:mini)
│   ├── correction_handler.py # "no, ti sbagli" → correzione memoria
│   ├── vision.py           # webcam + LivePortrait worker
│   ├── avatar_manager.py   # loop idle avatar
│   ├── health_monitor.py   # watchdog sottosistemi
│   └── templates/          # HTML (mai inline in agent.py)
├── mechanisms/
│   ├── memory.py           # episodica + significance + system prompt
│   ├── memory_vector.py    # ChromaDB (retrieval per similarità)
│   ├── graph_memory.py     # Kuzu (5 nodi, 8 relazioni, fatti verificati)
│   ├── digital_sleep.py    # consolidamento async 30min
│   ├── concepts.py         # estrazione concetti rule-based
│   ├── homeostasis.py      # stato interno → modula temperatura/lunghezza
│   ├── somatic.py          # telemetria GPU → arousal
│   ├── fish_tts.py         # voce clonata (Fish-Speech 1.5)
│   └── consolidamento.py   # riassunti notturni
├── data/                   # stato runtime — NON committare
│   ├── configs/            # model_config.json · pioneer_config.json
│   └── logs/               # archivio episodi, sonno, correzioni
├── docs/OPERATIONS.md      # reference tecnica: endpoint, avatar, memoria, problemi noti
├── archive/                # roba vecchia (ricerca, dataset training, snapshot)
└── static/ · index.html    # UI
```

## Interventi codice rapidi

| Vuoi modificare | File | Funzione |
|---|---|---|
| Cosa diventa un ricordo | `mechanisms/memory.py` | `_calcola_significance_episodio()` |
| System prompt | `mechanisms/memory.py` | `build_system_prompt()` |
| Anti-deriva / rewrite | `core/decision.py` | `filtra_deriva()` |
| Stato interno (umore) | `mechanisms/homeostasis.py` | `aggiorna_layer1()` · `modulazione_generazione()` |
| Chiamata Ollama | `core/agent.py` | `_fn_ollama()` |
| Endpoint Flask nuovo | `core/agent.py` | dopo route esistenti |
| Schema grafo | `mechanisms/graph_memory.py` | `_init_schema()` |
| Voce | `mechanisms/fish_tts.py` | `sintetizza()` |

## Convenzioni

- Commenti italiano · variabili snake_case · zero cloud
- Temperature: `0.70` narrativo · `0.75` proattivo · `0.20` memoria/autonomia · `0.10` vision
- Ollama: `timeout=60` · `stream=False` · `num_ctx=8192`
- Scritture `memory.json`: atomiche via `os.replace()` su `.tmp`
- Errori silenziosi: `log_utils.log_exception(modulo, msg, exc, severity)`
- Path: sempre da `eden_paths.py`

## Avvio

```bash
Eden.exe            # GUI launcher
avvia.bat           # Python diretto
avvia.bat --https   # mobile (microfono)
python tools/show_versions.py  # versioni live
python -c "import core.agent; print('OK')"
```

Modelli richiesti in Ollama: `qwen3.8:27b` (chat) · `phi3:mini` (router/riflessione, opzionale) · `qwen2.5vl:7b` (vision, opzionale).
