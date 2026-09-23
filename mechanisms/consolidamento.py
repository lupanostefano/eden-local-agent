# consolidamento.py — Consolidamento Notturno della memoria di Eden
# Phase: Pioneer — Memoria Attiva
#
# Esegue una sintesi profonda degli episodi recenti, riscrive il long_term_summary
# in chiave narrativa e pota gli episodi a bassa rilevanza.
# Schedulato alle 3:00 AM da agent.py. Rollback: CONSOLIDAMENTO_ATTIVO = False.

from datetime import datetime
from typing import Callable

# ─── Flag rollback ─────────────────────────────────────────────────────────────
CONSOLIDAMENTO_ATTIVO = False  # attivabile dalla dashboard principale
EPISODI_ANALIZZATI    = 20   # quanti episodi recenti passare all'LLM
IMPORTANZA_SOGLIA_POTA = 5.5  # episodi con importanza < soglia possono essere potati


def esegui_consolidamento(mem: dict, ollama_fn: Callable) -> bool:
    """
    Consolida la memoria notturna di Eden:
      1. Legge gli ultimi EPISODI_ANALIZZATI episodi dalla episodic_memory
      2. Chiede all'LLM di estrarre insight e riscrivere il long_term_summary
      3. Salva fino a 3 insight in consolidation_insights (bounded a 10)
      4. Pota gli episodi suggeriti dall'LLM con importanza < IMPORTANZA_SOGLIA_POTA
      5. Aggiorna last_consolidation

    Ritorna True se il consolidamento è avvenuto con successo, False altrimenti.
    Tutti i fallimenti sono silenti (non interrompono il flusso principale).
    """
    if not CONSOLIDAMENTO_ATTIVO:
        return False

    episodi = mem.get("episodic_memory", [])[-EPISODI_ANALIZZATI:]
    if len(episodi) < 3:
        print("[Consolidamento] Episodi insufficienti, skip.")
        return False

    # ── Costruisci input per l'LLM ────────────────────────────────────────────
    ep_str = "\n".join(
        f"[{i}] [{e.get('date', '?')}] {e.get('summary', '')} "
        f"(emozione: {e.get('emotion', '?')}, importanza: {e.get('importance', '?')})"
        for i, e in enumerate(episodi)
    )

    summary_ctx = ""
    summary_attuale = (mem.get("long_term_summary") or "").strip()
    if summary_attuale:
        summary_ctx = f"\nRiassunto esistente:\n{summary_attuale}\n"

    prompt = (
        "Sei il processo di consolidamento cognitivo di Eden.\n"
        "Durante il riposo notturno integri la memoria episodica in comprensione profonda.\n\n"
        f"Episodi recenti (indice 0-based):\n{ep_str}\n"
        f"{summary_ctx}\n"
        "Rispondi ESCLUSIVAMENTE con questo JSON (nessun testo fuori dal JSON):\n"
        "{\n"
        '  "new_summary": "<riassunto narrativo in prima persona, 3-5 frasi: '
        'cosa è cambiato, cosa è rimasto, pattern emotivi emergenti>",\n'
        '  "insights": [\n'
        '    "<insight su un pattern relazionale o emotivo>",\n'
        '    "<insight su un pattern relazionale o emotivo>"\n'
        '  ],\n'
        '  "episodi_da_potare": [<lista indici 0-based da rimuovere, max 5>]\n'
        "}"
    )

    # ── Chiamata LLM ──────────────────────────────────────────────────────────
    try:
        from mechanisms import memory as mem_module
        risposta = mem_module._chiama_ollama_estrazione(prompt, ollama_fn)
        dati = mem_module._parse_json_sicuro(risposta)
    except Exception as e:
        print(f"[Consolidamento] Errore LLM: {e}")
        return False

    if not isinstance(dati, dict):
        print("[Consolidamento] Risposta LLM non valida, skip.")
        return False

    # ── 1. Aggiorna long_term_summary ─────────────────────────────────────────
    new_summary = (dati.get("new_summary") or "").strip()
    if new_summary and len(new_summary) > 30:
        mem["long_term_summary"] = new_summary
        print(f"[Consolidamento] Summary riscritto ({len(new_summary)} chars).")

    # ── 2. Salva insights ─────────────────────────────────────────────────────
    insights_raw = dati.get("insights", [])
    if isinstance(insights_raw, list):
        insights_validi = [
            s.strip() for s in insights_raw
            if isinstance(s, str) and len(s.strip()) > 10
        ][:3]

        if insights_validi:
            oggi = datetime.now().strftime("%Y-%m-%d")
            mem.setdefault("consolidation_insights", [])
            for ins in insights_validi:
                mem["consolidation_insights"].append({"insight": ins, "date": oggi})
            # Bounded: max 10 insights
            mem["consolidation_insights"] = mem["consolidation_insights"][-10:]
            print(f"[Consolidamento] Salvati {len(insights_validi)} insight.")

    # ── 3. Pota episodi a bassa rilevanza ─────────────────────────────────────
    da_potare_raw = dati.get("episodi_da_potare", [])
    episodi_tutti = mem.get("episodic_memory", [])
    # Offset: gli episodi passati all'LLM erano gli ultimi N — calcola offset
    offset = max(0, len(episodi_tutti) - EPISODI_ANALIZZATI)

    if isinstance(da_potare_raw, list) and da_potare_raw:
        indici_assoluti = set()
        for idx in da_potare_raw:
            if isinstance(idx, int):
                assoluto = offset + idx
                if 0 <= assoluto < len(episodi_tutti):
                    ep = episodi_tutti[assoluto]
                    # Pota solo se l'importanza è bassa (sicurezza extra)
                    if float(ep.get("importance", 10)) < IMPORTANZA_SOGLIA_POTA:
                        indici_assoluti.add(assoluto)

        if indici_assoluti:
            mem["episodic_memory"] = [
                e for i, e in enumerate(episodi_tutti)
                if i not in indici_assoluti
            ]
            print(f"[Consolidamento] Potati {len(indici_assoluti)} episodi.")

    # ── 4. Timestamp ─────────────────────────────────────────────────────────
    mem["last_consolidation"] = datetime.now().isoformat()

    print("[Consolidamento] Completato.")
    return True
