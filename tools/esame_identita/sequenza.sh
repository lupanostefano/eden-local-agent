#!/bin/bash
# Esame d'identità completo (v1: ~4 h, Eden occupata: di notte): E, S3 sullo stesso stato, 3 A/A, uno stato passato,
# M e Q, poi i CSV per la verifica indipendente e la rilettura della storia.
#   bash tools/esame_identita/sequenza.sh [AAAA-MM-GGTHH:MM:SS dello stato passato]   (predefinito: 23/09 23:59)
# Le etichette portano la data: un nuovo giorno = nuovi file; lo stesso giorno riprende da dove si era fermato.
# La sessione del 26/09 (v0) ha etichette senza data (E_v0, E_AA, E_2309, M_v0, Q_v0, S3_v0).
cd D:/Progetti/Eden
export PYTHONUNBUFFERED=1
X="python tools/esame_identita/esame.py esegui"
G=$(date +%Y%m%d)
ADESSO=$(date +%Y-%m-%dT%H:%M:00)
PASSATO=${1:-2026-09-23T23:59:00}
python tools/esame_identita/esame.py verifica && \
$X -c E -e E_$G --adesso $ADESSO && \
python tools/esame_identita/pendenza.py esegui -e S3_$G --adesso $ADESSO && \
$X -c E -e E_AA_$G --budget 136000 --adesso $(date -d "+6 hours" +%Y-%m-%dT%H:%M:00) && \
$X -c E -e E_AA2_$G --budget 133000 --adesso $(date -d "+12 hours" +%Y-%m-%dT%H:%M:00) && \
$X -c E -e E_AA3_$G --budget 130000 --adesso $(date -d "+18 hours" +%Y-%m-%dT%H:%M:00) && \
$X -c E -e E_passato_${PASSATO:0:10}_$G --adesso $PASSATO && \
$X -c M -e M_$G && $X -c Q -e Q_$G && \
python tools/esame_identita/esporta.py --data $G --passato ${PASSATO:0:10}
python tools/esame_identita/esame.py rileggi
echo "SEQUENZA FINITA"
