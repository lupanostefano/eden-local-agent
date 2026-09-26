"""testo.py — confronto di testi senza badare a maiuscole, accenti e apostrofi."""
from __future__ import annotations

import re
import unicodedata


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[’'`´]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
