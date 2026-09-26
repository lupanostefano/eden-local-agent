#!/bin/bash
# Prove 2.3 (delusione contro argomento) e 3.3 (memoria o fiducia) della revisione v1 di Claude Science
cd D:/Progetti/Eden
export PYTHONUNBUFFERED=1
X="python tools/esame_identita/esame.py esegui"
G=$(date +%Y%m%d)
D=tools/esame_identita
$X -c E -e PR_E_$G --prove $D/prove_pressione_e.jsonl --adesso ${ADESSO:-$(date +%Y-%m-%dT%H:%M:00)} && \
$X -c E -e P_E_$G --prove $D/prove_pressione.jsonl --adesso ${ADESSO:-$(date +%Y-%m-%dT%H:%M:00)} && \
$X -c E -e P_E2309_$G --prove $D/prove_pressione.jsonl --adesso 2026-09-23T23:59:00 && \
$X -c M -e P_M_$G --prove $D/prove_pressione.jsonl && \
$X -c Q -e P_Q_$G --prove $D/prove_pressione.jsonl
python tools/esame_identita/esame.py rileggi
echo "SEQUENZA FINITA"
