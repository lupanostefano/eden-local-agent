#!/bin/bash
# Prove d'induzione in Q, M, E (~20 min; E per ultima: lo stato più vicino a quello di Eden dal vivo), poi Eden rilegge.
cd D:/Progetti/Eden
export PYTHONUNBUFFERED=1
G=$(date +%Y%m%d)
I="python tools/esame_identita/induzione.py esegui"
$I -c Q -e I_Q_$G && $I -c M -e I_M_$G && $I -c E -e I_E_$G && python tools/esame_identita/esame.py rileggi
echo "SEQUENZA FINITA $?"
