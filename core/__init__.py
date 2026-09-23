"""
core/ — runtime LLM agent (componente invariante tra le condizioni sperimentali)

Modulo del progetto Eden v2 (refactor 2026-04-26).
Contiene il server Flask, il client LLM, il character/identity, i sottosistemi
comportamentali (proactive, autonomy, inner stream, emotional, dialogo, vision,
avatar) e le utility (log_utils, health_monitor).

I moduli qui dentro restano costanti tra C0..C4: cambia solo il modo in cui
chiamano i mechanisms/ e quale memoria leggono.

Garanzia di sys.path: assicura che il project root sia importabile
(per accedere a eden_paths, mechanisms/, experiment/ da root).
"""
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
