"""
mechanisms/ — memoria e stato interno di Eden

Contenuto:
    memory.py          encoding episodi + significance + system prompt
    memory_vector.py   retrieval per similarità (ChromaDB)
    graph_memory.py    fatti verificati e relazioni (Kuzu)
    digital_sleep.py   consolidamento asincrono
    concepts.py        estrazione concetti rule-based
    consolidamento.py  riassunti notturni
    homeostasis.py     stato interno che modula la generazione
    somatic.py         telemetria GPU → arousal
    fish_tts.py        voce clonata (Fish-Speech 1.5)

Garanzia di sys.path: assicura che il project root sia importabile
anche da chi importa solo questo package (per accedere a decision,
identity, log_utils, eden_paths che vivono in root).
"""
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
