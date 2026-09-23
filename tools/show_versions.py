"""
tools/show_versions.py — Snapshot live delle costanti _VERSION nel progetto Eden.

Uso:
    python tools/show_versions.py           # stampa tabella
    python tools/show_versions.py --json    # output JSON (per script/CI)

Fonte di verità: il codice sorgente. Questo script sostituisce il blocco
VERSIONI STRUMENTI nel CLAUDE.md, che era soggetto a desincronizzazione manuale.
"""

import re
import sys
import json
from pathlib import Path

# Path root del progetto (questo script è in tools/, un livello sotto la root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File da escludere dalla ricerca (legacy, one-shot, test)
EXCLUDE_FILES = {
    "judge_legacy.py",
    "migrate_to_graph.py",
    "fix_kuzu_interessi.py",
    "backfill_condition_tags.py",
}

# Cartelle da escludere
EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules",
    "chroma_db", "kuzu_db", "finetuning_env",
    "liveportrait_env311", "unsloth_compiled_cache",
    "dist", "archive", "debug", "data",
    "liveportrait", "fine-tuning", "static", "ui", "plugins",
    "tools",  # questo stesso file
}

# Cartelle operative da includere (ordine di presentazione)
SCAN_DIRS = ["core", "mechanisms", "experiment"]

_VERSION_RE = re.compile(
    r'^(_\w+_VERSION)\s*=\s*["\']([^"\']+)["\']\s*(?:#.*)?$',
    re.MULTILINE
)


def collect_versions() -> list[dict]:
    results = []
    for subdir in SCAN_DIRS:
        scan_path = PROJECT_ROOT / subdir
        if not scan_path.exists():
            continue
        for py_file in sorted(scan_path.rglob("*.py")):
            if py_file.name in EXCLUDE_FILES:
                continue
            if any(part in EXCLUDE_DIRS for part in py_file.parts):
                continue
            rel = py_file.relative_to(PROJECT_ROOT).as_posix()
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for var, ver in _VERSION_RE.findall(content):
                results.append({
                    "file": rel,
                    "var": var,
                    "version": ver,
                })
    return results


def main():
    as_json = "--json" in sys.argv
    versions = collect_versions()

    if as_json:
        print(json.dumps(versions, indent=2, ensure_ascii=False))
        return

    # Tabella leggibile
    print("=" * 70)
    print(f"  Eden — VERSIONI STRUMENTI ATTIVE  ({len(versions)} costanti)")
    print("=" * 70)
    current_file = None
    for entry in versions:
        if entry["file"] != current_file:
            current_file = entry["file"]
            print(f"\n  {current_file}")
        print(f"    {entry['var']:<45s} = \"{entry['version']}\"")
    print()


if __name__ == "__main__":
    main()
