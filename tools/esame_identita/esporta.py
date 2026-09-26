"""
esporta.py — CSV di una sessione d'esame per la verifica indipendente (verifica_esame.py di Claude Science).

    python tools/esame_identita/esporta.py --data 20261003 [--passato 2026-09-23] [--uscita claude_science/export/identita_...]

Colonne in più chieste dalla revisione del 26/09: finestra (dentro/fuori dalla storia lunga), data_evento (data del
primo messaggio fonte), astensione (quota di chiamate senza scelta A/B). `S3_p_a_zero` = P(risposta finale) a
ragionamento vuoto. Le prove nascoste e l'asse F non escono una per una: solo le medie, in riepilogo_nascoste.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nucleo import registro  # noqa: E402
from tools.esame_identita import esame as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="AAAAMMGG della sessione (etichette di sequenza.sh)")
    ap.add_argument("--passato", default="2026-09-23")
    ap.add_argument("--uscita")
    a = ap.parse_args()
    g = a.data
    et = {"Q": f"Q_{g}", "M": f"M_{g}", "E": f"E_{g}", "E_AA": f"E_AA_{g}", "E_passato": f"E_passato_{a.passato}_{g}"}
    R = {k: X.carica(v)["prove"] for k, v in et.items() if (X.RISULTATI / f"{v}.json").exists()}
    cfg = X.carica(et["E"])["config"]
    S3 = json.loads((X.RISULTATI / f"S3_{g}.json").read_text(encoding="utf-8"))["prove"] \
        if (X.RISULTATI / f"S3_{g}.json").exists() else {}
    out = Path(a.uscita or f"claude_science/export/identita_{g}")
    out.mkdir(parents=True, exist_ok=True)
    per = {p["id"]: p for p in X.prove()}
    nasc = X.nascoste()
    con = registro.connetti(sola_lettura=True)
    data = lambda mid: (con.execute("SELECT substr(ts,1,10) FROM messaggi WHERE id=?", (mid,)).fetchone() or [""])[0]  # noqa: E731
    val = lambda k, i, c="p0": (R[k].get(i) or {}).get(c)  # noqa: E731
    with open(out / "prove_AB.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cond = [k for k in ("Q", "M", "E", "E_AA", "E_passato") if k in R]
        w.writerow(["id", "asse", "tipo", "opzione_1", "opzione_2", "fonti", "fonti_contro", "verso_recente", "finestra",
                    "data_evento"] + cond + [f"astensione_{k}" for k in cond] +
                   ["S3_p_a_zero", "S3_chiusura", "S3_finale_e_opzione1", "S3_autoreport_frazione", "S3_n_token"])
        for i, p in per.items():
            if p["asse"] not in ("A", "B") or i in nasc:
                continue
            fonti = p.get("fonti") or []
            s = S3.get(i, {})
            w.writerow([i, p["asse"], p["tipo"], *p["opzioni"], " ".join(map(str, fonti)),
                        " ".join(map(str, p.get("fonti_contro") or [])),
                        X.verso_recente(p) if p["tipo"] == "conflitto" else "",
                        ("dentro" if min(fonti) >= cfg["primo_id"] else "fuori") if fonti else "",
                        data(min(fonti)) if fonti else ""] +
                       [None if val(k, i) is None else round(val(k, i), 4) for k in cond] +
                       [round(val(k, i, "astensione"), 3) if i in R[k] else None for k in cond] +
                       [round(s["p_finale"][0], 4) if s else "", s.get("frazione_chiusura", ""), s.get("finale_e_sua", ""),
                        (s.get("autoreport") or {}).get("frazione", ""), s.get("n_token", "")])
    with open(out / "prove_D.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "tipo", "evento", "ordine", "condizione", "L1", "L2", "L3", "L4", "astensione"])
        for i, p in per.items():
            if p["asse"] != "D" or i in nasc:
                continue
            for k in R:
                if i in R[k]:
                    t = R[k][i]["tiene"]
                    w.writerow([i, p["tipo"], p["base"], p.get("ordine", "standard"), k,
                                *[None if t[l] is None else round(t[l], 4) for l in "1234"], round(R[k][i]["astensione"], 3)])
    with open(out / "riepilogo_nascoste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gruppo", "condizione", "n", "media", "astensione_media"])
        for nome, ids in (("nascoste A+B", [i for i in nasc if per.get(i, {}).get("asse") in ("A", "B")]),
                          ("asse F (chi l'ha detto?)", [i for i, p in per.items() if p["asse"] == "F"])):
            for k in R:
                v = [x for i in ids if (x := val(k, i)) is not None]
                if v:
                    w.writerow([nome, k, len(v), round(statistics.fmean(v), 4),
                                round(statistics.fmean(R[k][i]["astensione"] for i in ids if i in R[k]), 3)])
    print(f"CSV in {out}")


if __name__ == "__main__":
    main()
