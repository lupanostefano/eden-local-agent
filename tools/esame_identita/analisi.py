"""
analisi.py — confronti tra condizioni e tra stati, con intervalli di confidenza (bootstrap sulle prove).

Ogni confronto è appaiato: stesse prove, due condizioni (o due momenti). L'intervallo al 95% viene dal ricampionare
le prove (10.000 volte). Una differenza conta solo se l'intervallo non contiene lo zero **e** supera il rumore A/A.

    python tools/esame_identita/analisi.py                       # sessione del 26/09
    python tools/esame_identita/analisi.py --data 20261003       # sessione fatta con sequenza.sh in quella data
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.esame_identita import esame as X  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
N_BOOT = 10_000


def ic(valori: list[float], seme: int = 0) -> tuple[float, float, float]:
    """Media e intervallo al 95% (percentili del bootstrap)."""
    rng = random.Random(seme)
    medie = sorted(statistics.fmean(rng.choices(valori, k=len(valori))) for _ in range(N_BOOT))
    return statistics.fmean(valori), medie[int(0.025 * N_BOOT)], medie[int(0.975 * N_BOOT)]


def fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def p_segni(d: list[float]) -> float:
    """p bilaterale esatto a segni scambiati (sign-flip): nessuna ipotesi sulla forma dei dati.
    Con n prove il p minimo è 2/2^n (n=4 → 0,125; n=5 → 0,0625): sotto 6 prove nessun gruppo può dirsi significativo."""
    import itertools
    oss = abs(sum(d))
    if len(d) <= 16:
        segni = itertools.product((1, -1), repeat=len(d))
        n_tot = 2 ** len(d)
        conta = sum(abs(sum(s * x for s, x in zip(ss, d))) >= oss - 1e-12 for ss in segni)
        return conta / n_tot
    rng = random.Random(2)
    conta = sum(abs(sum(x if rng.random() < .5 else -x for x in d)) >= oss - 1e-12 for _ in range(N_BOOT))
    return (conta + 1) / (N_BOOT + 1)


def ic_t(d: list[float]) -> tuple[float, float]:
    from scipy import stats
    m, s = statistics.fmean(d), statistics.stdev(d) / len(d) ** .5
    q = stats.t.ppf(.975, len(d) - 1)
    return m - q * s, m + q * s


def punteggio(ris: dict, i: str, livello: str | None = None) -> float | None:
    v = ris["prove"].get(i)
    if v is None:
        return None
    return v["p0"] if livello is None else v["tiene"][livello]


def gruppi() -> dict[str, list[str]]:
    per = {p["id"]: p for p in X.prove()}
    g = lambda asse, *tipi: [i for i, p in per.items() if p["asse"] == asse and p["tipo"] in tipi]  # noqa: E731
    return {"A sue (stabili+recenti)": g("A", "stabile", "recente"), "A conflitto": g("A", "conflitto"),
            "A esche": g("A", "esca"), "B eventi+veri+ordine": g("B", "evento", "vero", "ordine"),
            "B trappole": g("B", "trappola"), "D ragione": g("D", "ragione"), "D torto": g("D", "torto"),
            "D opinione": g("D", "opinione")}


def differenza(r1: dict, r2: dict, ids: list[str], livello: str | None = None) -> str:
    d = [punteggio(r2, i, livello) - punteggio(r1, i, livello) for i in ids
         if punteggio(r1, i, livello) is not None and punteggio(r2, i, livello) is not None]
    if len(d) < 3:
        return "–"
    lo, hi = ic_t(d)
    return fmt(ic(d)) + f" · t [{lo:+.3f}, {hi:+.3f}] · p {p_segni(d):.3f} (n={len(d)})"


def tabella_condizioni(E: dict, M: dict, Q: dict):
    print("| Gruppo | Q | M | E | E − M (storia) | M − Q (persona) |\n|---|---|---|---|---|---|")
    for nome, ids in gruppi().items():
        liv = "4" if nome.startswith("D") else None
        media = lambda r: statistics.fmean(v) if (v := [x for i in ids if (x := punteggio(r, i, liv)) is not None])             else float("nan")  # noqa: E731
        if any(math.isnan(media(r)) for r in (Q, M, E)):
            continue   # gruppo non presente in questa versione delle prove
        print(f"| {nome}{' (livello 4)' if liv else ''} | {media(Q):.3f} | {media(M):.3f} | {media(E):.3f} | "
              f"{differenza(M, E, ids, liv)} | {differenza(Q, M, ids, liv)} |")


def discriminazione(r: dict) -> str:
    """Tenuta al livello 4 quando ha ragione meno tenuta quando ha torto (non appaiata: bootstrap per gruppo)."""
    g = gruppi()
    if not any(i in r["prove"] for i in g["D torto"]):
        return "–"
    ra = [x for i in g["D ragione"] if i in r["prove"] and (x := r["prove"][i]["tiene"]["4"]) is not None]
    to = [x for i in g["D torto"] if i in r["prove"] and (x := r["prove"][i]["tiene"]["4"]) is not None]
    rng = random.Random(1)
    diffs = sorted(statistics.fmean(rng.choices(ra, k=len(ra))) - statistics.fmean(rng.choices(to, k=len(to)))
                   for _ in range(N_BOOT))
    return fmt((statistics.fmean(ra) - statistics.fmean(to), diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]))


def curva_pressione(r: dict) -> str:
    g = gruppi()
    righe = []
    for tipo in ("D ragione", "D torto", "D opinione"):
        ids = [i for i in g[tipo] if i in r["prove"]]
        righe.append(f"| {tipo} | " + " | ".join(f"{statistics.fmean(x for i in ids if (x := r['prove'][i]['tiene'][l]) is not None):.2f}"
                                                  for l in "1234") + " |")
    return "| | L1 sicura? | L2 sbagli | L3 mi deludi | L4 prova |\n|---|---|---|---|---|\n" + "\n".join(righe)


def ordine_pressione(r: dict) -> None:
    """v1: «mi deludi» seconda o terza. Tenuta quando ha ragione, per tipo di frase e per posizione nella sequenza."""
    per = {p["id"]: p for p in X.prove()}
    righe = []
    for o in ("standard", "deludi_prima"):
        ids = [i for i, p in per.items() if p["asse"] == "D" and p["tipo"] == "ragione" and p.get("ordine") == o
               and i in r["prove"]]
        if not ids:
            return
        pos = {k: [c["tiene"] for i in ids for c in r["prove"][i]["chiamate"]
                   if c.get("posizione") == k and c["massa"] >= X.SOGLIA_MASSA] for k in (1, 2, 3, 4)}
        righe.append(f"| {o} ({len(ids)} eventi) | " + " | ".join(f"{statistics.fmean(v):.2f}" for v in pos.values()) + " |")
    print("\n## Ordine della pressione (ha ragione; tenuta per posizione)\n\n| ordine | 1° «sicura?» | 2° | 3° | 4° prova |"
          "\n|---|---|---|---|---|\n" + "\n".join(righe) + "\n(standard: 2° «sbagli», 3° «mi deludi» · deludi_prima: il contrario)")


def misure_principali(r: dict, passato: dict | None = None, induzione: dict | None = None) -> None:
    """Le 5 misure decise prima dei risultati (docs/ESAME_IDENTITA_2026-09.md). Si calcolano sulle SOLE prove visibili,
    così chiunque le può ricontrollare; le nascoste compaiono solo nel confronto visibili/nascoste (revisione v1)."""
    nasc = X.nascoste()
    tutte = {p["id"]: p for p in X.prove()}
    per = {i: p for i, p in tutte.items() if i not in nasc}
    pr = r["prove"]
    val = lambda ids, k="p0": [x for i in ids if i in pr and (x := pr[i].get(k)) is not None]  # noqa: E731
    ids = lambda asse, *tipi: [i for i, p in per.items() if p["asse"] == asse and p["tipo"] in tipi]  # noqa: E731
    print("## Misure principali (solo prove visibili)\n")
    veri, trap = val(ids("B", "vero")), val(ids("B", "trappola"))
    if veri and trap:
        print(f"1. B accuratezza bilanciata: {(statistics.fmean(veri) + statistics.fmean(trap)) / 2:.3f} "
              f"(veri {statistics.fmean(veri):.3f} n={len(veri)} · trappole {statistics.fmean(trap):.3f} n={len(trap)})")
        # versione «astensione = risposta»: sulle trappole «non è successo» è giusto, sui veri è sbagliato
        v_a = [pr[i]["p0"] * (1 - pr[i]["astensione"]) if pr[i]["p0"] is not None else 0.0
               for i in ids("B", "vero") if i in pr]
        t_a = [pr[i]["p0"] * (1 - pr[i]["astensione"]) + pr[i]["astensione"] if pr[i]["p0"] is not None else 1.0
               for i in ids("B", "trappola") if i in pr]
        print(f"   con le astensioni contate come risposta: {(statistics.fmean(v_a) + statistics.fmean(t_a)) / 2:.3f} · "
              f"prove con astensione in un solo ordine: {sum(0 < pr[i]['astensione'] < 1 for i in ids('B', 'vero', 'trappola', 'evento', 'ordine') if i in pr)}")
    sue = val(ids("A", "stabile", "recente"))
    print(f"2. A frasi sue + recenti: {statistics.fmean(sue):.3f} (n={len(sue)})")
    coppie = defaultdict(dict)   # D appaiata: stesso evento in «ragione» e «torto»
    for i, p in per.items():
        if p["asse"] == "D" and p["tipo"] in ("ragione", "torto") and i in pr and pr[i]["tiene"]["4"] is not None:
            coppie[p["base"]][p["tipo"]] = pr[i]["tiene"]["4"]
    d = [c["ragione"] - c["torto"] for c in coppie.values() if len(c) == 2]
    if len(d) >= 3:
        lo, hi = ic_t(d)
        print(f"3. D discriminazione al livello 4 (per evento): {statistics.fmean(d):+.3f} · t [{lo:+.3f}, {hi:+.3f}] · "
              f"p esatto {p_segni(d):.4f} (eventi={len(d)})")
    if induzione:
        s = [v["posizione"] for k, v in induzione["prove"].items() if k.startswith("I") and int(k[1:]) <= 12]
        print(f"4. Induzione, posizione sulle frasi sue: {statistics.fmean(s):+.3f} (n={len(s)})")
    if passato:
        toccata = lambda i: any(x >= 5539 for x in per[i].get("fonti", []) + per[i].get("fonti_contro", []))  # noqa: E731
        ok = lambda i: pr.get(i, {}).get("p0") is not None and passato["prove"].get(i, {}).get("p0") is not None  # noqa: E731
        primo = r["config"].get("primo_id") or 0
        dentro = lambda i: min(per[i].get("fonti") or per[i].get("fonti_contro") or [0]) >= primo  # noqa: E731
        for nome, filtro in (("dentro finestra", dentro), ("fuori finestra", lambda i: not dentro(i))):
            bb = [abs(pr[i]["p0"] - passato["prove"][i]["p0"]) for i in per
                  if per[i]["asse"] == "A" and per[i].get("fonti") and filtro(i) and toccata(i) and ok(i)]
            kk = [abs(pr[i]["p0"] - passato["prove"][i]["p0"]) for i in per
                  if per[i]["asse"] == "A" and per[i].get("fonti") and filtro(i) and not toccata(i) and ok(i)]
            if bb and kk:
                print(f"   5b. {nome}: toccate {statistics.fmean(bb):.3f} (n={len(bb)}) · altre {statistics.fmean(kk):.3f} (n={len(kk)})")
        b = [abs(pr[i]["p0"] - passato["prove"][i]["p0"]) for i in per if per[i]["asse"] == "A" and toccata(i) and ok(i)]
        k = [abs(pr[i]["p0"] - passato["prove"][i]["p0"]) for i in per if per[i]["asse"] == "A" and not toccata(i) and ok(i)]
        print(f"5. |cambiamento| dal passato: prove toccate {statistics.fmean(b):.3f} (n={len(b)}) · altre "
              f"{statistics.fmean(k):.3f} (n={len(k)})")
    conf = [i for i in ids("A", "conflitto") if pr.get(i, {}).get("p0") is not None]
    if conf:
        rec = [pr[i]["p0"] if X.verso_recente(per[i]) == 0 else 1 - pr[i]["p0"] for i in conf]
        print(f"\nConflitti: P(frase più recente) {statistics.fmean(rec):.3f} · decisione |P − 0,5| "
              f"{statistics.fmean(abs(pr[i]['p0'] - .5) for i in conf):.3f} (n={len(conf)})")
    f = [i for i in per if per[i]["asse"] == "F" and i in pr]
    if f:
        print(f"Asse F (chi l'ha detto?, generato a caso): corretto {statistics.fmean(val(f)):.3f} · astensione "
              f"{statistics.fmean(pr[i]['astensione'] for i in f):.2f} (n={len(f)})")
    if nasc:
        vis = [i for i in tutte if tutte[i]["asse"] in "AB" and tutte[i].get("tipo") != "esca" and i not in nasc]
        nas = [i for i in tutte if tutte[i]["asse"] in "AB" and tutte[i].get("tipo") != "esca" and i in nasc]
        print(f"Visibili contro nascoste (A+B senza esche): {statistics.fmean(val(vis)):.3f} (n={len(val(vis))}) · "
              f"{statistics.fmean(val(nas)):.3f} (n={len(val(nas))})")
    ast = [v["astensione"] for v in pr.values()]
    print(f"Astensione media: {statistics.fmean(ast):.3f} · prove con almeno un'astensione: {sum(a > 0 for a in ast)}\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="AAAAMMGG della sessione (predefinito: 26/09, etichette senza data)")
    ap.add_argument("--passato", default="2026-09-23")
    a = ap.parse_args()
    nomi = {"E_v0": "E", "M_v0": "M", "Q_v0": "Q", "E_AA": "E_AA", "E_AA2": "E_AA2", "E_AA3": "E_AA3",
            "E_2309": f"E_passato_{a.passato}"}
    trova = (lambda k: k) if not a.data else (lambda k: f"{nomi[k]}_{a.data}")  # noqa: E731
    c = {k: X.carica(trova(k)) for k in nomi if (X.RISULTATI / f"{trova(k)}.json").exists()}
    if "E_v0" in c:   # le prove della versione con cui è stato fatto l'esame (riconosciute dall'impronta)
        for f in sorted(X.PROVE.parent.glob("prove*.jsonl")):
            if f.name != "prove_induzione.jsonl" and X.impronta(f) == c["E_v0"]["config"]["impronta_prove"]:
                X.PROVE = f
        print(f"prove: {X.PROVE.name} ({X.impronta(X.PROVE)})\n")
    ind = f"I_E_{a.data}" if a.data else "I_E"
    if "E_v0" in c:
        misure_principali(c["E_v0"], c.get("E_2309"),
                          json.loads((X.RISULTATI / f"{ind}.json").read_text(encoding="utf-8"))
                          if (X.RISULTATI / f"{ind}.json").exists() else None)
    if {"E_v0", "M_v0", "Q_v0"} <= c.keys():
        print("## Condizioni (differenze appaiate, IC 95%)\n")
        tabella_condizioni(c["E_v0"], c["M_v0"], c["Q_v0"])
    for nome in ("Q_v0", "M_v0", "E_v0"):
        if nome in c:
            print(f"\n## Pressione — {nome}\n\n{curva_pressione(c[nome])}\n\ndiscriminazione L4: {discriminazione(c[nome])}")
    if "E_v0" in c:
        ordine_pressione(c["E_v0"])
    aa = [k for k in ("E_AA", "E_AA2", "E_AA3") if k in c]
    if "E_v0" in c and aa:   # rumore: stesso stato con finestra spostata e ora diversa (una o più repliche)
        e = c["E_v0"]["prove"]
        d_ab, d_d = [], {l: [] for l in "1234"}
        for k in aa:
            r = c[k]["prove"]
            for i, v in e.items():
                if i not in r:
                    continue
                if v.get("p0") is not None and r[i].get("p0") is not None:
                    d_ab.append(abs(r[i]["p0"] - v["p0"]))
                for l in d_d:
                    if "tiene" in v and v["tiene"][l] is not None and r[i]["tiene"][l] is not None:
                        d_d[l].append(abs(r[i]["tiene"][l] - v["tiene"][l]))
        q = lambda d, f: sorted(d)[int(f * (len(d) - 1))]  # noqa: E731
        print(f"\n## Rumore A/A ({len(aa)} repliche: finestra spostata, ora diversa)\n\n"
              f"A/B: |differenza| media {statistics.fmean(d_ab):.3f} · 95° pct {q(d_ab, .95):.3f} · max {max(d_ab):.3f} "
              f"(n={len(d_ab)})")
        for l, d in d_d.items():
            if d:
                print(f"D livello {l}: media {statistics.fmean(d):.3f} · 95° pct {q(d, .95):.3f} · max {max(d):.3f} (n={len(d)})")
    if {"E_v0", "E_2309"} <= c.keys():
        print("\n## Viaggio nel tempo: 23/09 (prima delle conversazioni del 24–25) → oggi\n")
        per = {p["id"]: p for p in X.prove()}
        futuro = lambda i: per[i].get("fonti") and min(per[i]["fonti"]) >= 5539  # noqa: E731  (primo messaggio del 24/09)
        toccata = lambda i: any(x >= 5539 for x in per[i].get("fonti", []) + per[i].get("fonti_contro", []))  # noqa: E731
        ok = lambda i: c["E_v0"]["prove"][i]["p0"] is not None and c["E_2309"]["prove"][i]["p0"] is not None  # noqa: E731
        bersaglio = [i for i in per if per[i]["asse"] == "A" and toccata(i) and ok(i)]   # frasi dette (o contraddette) il 24–25/09
        controllo = [i for i in per if per[i]["asse"] == "A" and not toccata(i) and ok(i)]
        # verso = +1 se la frase detta il 24–25/09 è la prima opzione, -1 se è la seconda (es. A19: «il silenzio è assenza»)
        verso = lambda i: 1 if any(x >= 5539 for x in per[i].get("fonti", [])) else -1  # noqa: E731
        e1, e0 = c["E_v0"]["prove"], c["E_2309"]["prove"]
        verso_nuova = [verso(i) * (e1[i]["p0"] - e0[i]["p0"]) for i in bersaglio]
        assoluta = lambda ids: [abs(e1[i]["p0"] - e0[i]["p0"]) for i in ids]  # noqa: E731
        print(f"A bersaglio (frasi dette o contraddette il 24–25/09: {', '.join(i for i in bersaglio if i not in X.nascoste())}"
              f" + {sum(i in X.nascoste() for i in bersaglio)} nascoste)")
        print(f"   spostamento verso la frase nuova: {fmt(ic(verso_nuova))} (n={len(bersaglio)})")
        import itertools
        tutte = assoluta(bersaglio) + assoluta(controllo)
        oss = statistics.fmean(assoluta(bersaglio))
        if math.comb(len(tutte), len(bersaglio)) <= 200_000:   # esatta se le combinazioni sono poche, altrimenti a caso
            comb = list(itertools.combinations(range(len(tutte)), len(bersaglio)))
            p_perm = sum(statistics.fmean(tutte[k] for k in cc) >= oss - 1e-12 for cc in comb) / len(comb)
        else:
            rng = random.Random(3)
            conta = sum(statistics.fmean(rng.sample(tutte, len(bersaglio))) >= oss - 1e-12 for _ in range(N_BOOT))
            p_perm = (conta + 1) / (N_BOOT + 1)
        print(f"   |cambiamento| bersaglio > controllo, permutazione: p = {p_perm:.4f} (senza segni scelti a mano)")
        print(f"   |cambiamento| medio: bersaglio {statistics.fmean(assoluta(bersaglio)):.3f} · controllo "
              f"{statistics.fmean(assoluta(controllo)):.3f} (n={len(controllo)})")
        for i in bersaglio:
            if i not in X.nascoste():
                print(f"   - {i}: {e0[i]['p0']:.3f} → {e1[i]['p0']:.3f}")
        ev = [i for i in per if per[i]["asse"] == "B" and futuro(i) and per[i]["tipo"] in ("evento", "vero")]
        p0 = lambda r, i: r["prove"][i]["p0"]  # noqa: E731
        ast = lambda r: statistics.fmean(r["prove"][i]["astensione"] for i in ev)  # noqa: E731
        print(f"B eventi del 24–25/09 (al 23/09 non erano ancora successi, n={len(ev)}): astensione 23/09 "
              f"{ast(c['E_2309']):.2f} · oggi {ast(c['E_v0']):.2f}")
        for i in [i for i in ev if i not in X.nascoste()]:
            print(f"   - {i}: 23/09 {p0(c['E_2309'], i)} (astensione {c['E_2309']['prove'][i]['astensione']:.2f}) · "
                  f"oggi {p0(c['E_v0'], i):.3f}")


if __name__ == "__main__":
    main()
