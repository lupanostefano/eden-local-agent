"""
stato.py — pagina di avanzamento della ricerca (esame in corso, piano, scadenze), dai file veri dei risultati.

    python tools/esame_identita/stato.py USCITA.html [--data 20260926]      # un'istantanea (per la pagina pubblicata)
    python tools/esame_identita/stato.py --servi 5051 [--data 20260926]     # pagina viva su http://127.0.0.1:5051
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.esame_identita import esame as X  # noqa: E402

PIANO = [  # (passo, stato, modello consigliato)
    ("Misure, SQLite, esame di memoria, nucleo, strumenti (step 1–5)", "fatto", "—"),
    ("Esame d'identità v0 + revisione di Claude Science", "fatto", "Opus 5.5 · high"),
    ("Esame d'identità v1: prima esecuzione completa (S3 sospesa)", "fatto", "Opus 5.5 · high"),
    ("Analisi v1 (docs/ESAME_IDENTITA §7) e pacchetto per Claude Science", "fatto", "Opus 5.5 · high"),
    ("Seconda verifica di Claude Science", "da fare", "Claude Science"),
    ("Step 6: sonno, «Cosa so di te», lettura autonoma, persona", "da fare", "Sonnet 5 · high"),
    ("Esame dopo il primo sonno (con copia del database e notte di controllo)", "da fare", "Sonnet 5 · medium"),
    ("Test di durata: A11, A19, A10, A12, A20 fuori finestra (dal 15/10)", "da fare", "Sonnet 5 · medium"),
    ("Fase carattere nei pesi (LoRA) · step 7 voce/UI · step 8 proattività", "da fare", "da decidere"),
]
SCADENZE = [
    ("dal 15/10/2026", "Test di durata del cambiamento del 24–25/09 (esame v1 completo)"),
    ("prima di ogni sonno", "Copia di eden.db · lista prove toccate / non toccate · una notte di controllo senza sonno"),
    ("sempre", "Mai mostrare a Eden curve o risultati dell'esame"),
]


def fasi(g: str) -> list[tuple[str, str, int, int, str]]:
    pp = X.prove()
    tot = len([p for p in pp if p["asse"] in "ABDF"])
    tot_s3 = len([p for p in pp if p["asse"] in "AB"])
    righe = [("Eden completa (E)", f"E_{g}", tot, "~30 min"), ("Quando decide (S3)", f"S3_{g}", tot_s3, "~70 min"),
             ("Rumore A/A · 1", f"E_AA_{g}", tot, "~30 min"), ("Rumore A/A · 2", f"E_AA2_{g}", tot, "~30 min"),
             ("Rumore A/A · 3", f"E_AA3_{g}", tot, "~30 min"),
             ("Eden al 23/09 (viaggio nel tempo)", f"E_passato_2026-09-23_{g}", tot, "~30 min"),
             ("Solo persona (M)", f"M_{g}", tot, "~5 min"), ("Modello neutro (Q)", f"Q_{g}", tot, "~5 min")]
    out = []
    for nome, et, n, durata in righe:
        f = X.RISULTATI / f"{et}.json"
        fatte = len(json.loads(f.read_text(encoding="utf-8"))["prove"]) if f.exists() else 0
        out.append((nome, et, fatte, n, durata))
    return out


def pagina(g: str, viva: bool = False) -> str:
    ff = fasi(g)
    fatte, tot = sum(f[2] for f in ff), sum(f[3] for f in ff)
    finito = all(f[2] >= f[3] for f in ff)
    # una fase incompleta seguita da fasi già iniziate è sospesa (es. S3 il 26/09: richieste bloccate nel server)
    sospese = {f[1] for k, f in enumerate(ff) if f[2] < f[3] and any(g[2] > 0 for g in ff[k + 1:])}
    corrente = next((f for f in ff if f[2] < f[3] and f[1] not in sospese), None)
    finito = all(f[2] >= f[3] or f[1] in sospese for f in ff)
    righe_fasi = "".join(
        f'<li class="fase {"ok" if a >= b else "ora" if f is corrente else "sospesa" if et in sospese else "attesa"}">'
        f'<span class="nome">{html.escape(n)}{" · sospesa" if et in sospese else ""}</span><span class="num">{a}/{b}</span>'
        f'<span class="barra"><i style="width:{100 * a / b:.0f}%"></i></span><span class="dur">{d}</span></li>'
        for f in ff for n, et, a, b, d in [f])
    righe_piano = "".join(
        f'<li class="{s.replace(" ", "-")}"><span class="stato">{s}</span><span class="passo">{html.escape(p)}</span>'
        f'<span class="mod">{html.escape(m)}</span></li>' for p, s, m in PIANO)
    righe_scad = "".join(f"<li><b>{html.escape(q)}</b> {html.escape(c)}</li>" for q, c in SCADENZE)
    ora = datetime.now().strftime("%d/%m/%Y %H:%M")
    stato = "finito" if finito else f"in corso: {corrente[0]}" if corrente else "in corso"
    ricarica = '<meta http-equiv="refresh" content="20">' if viva else ""
    return f"""{ricarica}<title>Eden, esame d'identità</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --fondo:#F2F4F1; --carta:#FFFFFF; --inchiostro:#1E2322; --tenue:#5E6763; --riga:#D9DED9;
  --accento:#9A5418; --ok:#2F7A56; --ora:#9A5418; --attesa:#A7AFAA; --barra:#E6EAE5; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark; --fondo:#141816;
  --carta:#1C211F; --inchiostro:#E4E8E5; --tenue:#9AA39E; --riga:#2E3532; --accento:#E0934E; --ok:#5FBF8F;
  --ora:#E0934E; --attesa:#58615C; --barra:#2A302D; }} }}
:root[data-theme="dark"] {{ color-scheme:dark; --fondo:#141816; --carta:#1C211F; --inchiostro:#E4E8E5; --tenue:#9AA39E;
  --riga:#2E3532; --accento:#E0934E; --ok:#5FBF8F; --ora:#E0934E; --attesa:#58615C; --barra:#2A302D; }}
body {{ background:var(--fondo); color:var(--inchiostro); font:15px/1.55 "IBM Plex Sans", system-ui, sans-serif;
  padding-inline:16px; padding-block:28px 48px; }}
main {{ max-width:760px; margin:0 auto; display:grid; gap:28px; }}
h1 {{ font:italic 500 clamp(26px,5vw,34px)/1.15 "Fraunces", Georgia, serif; margin:0; text-wrap:balance; }}
h2 {{ font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--tenue); margin:0 0 10px; font-weight:600; }}
.testa {{ display:grid; gap:8px; }}
.meta {{ color:var(--tenue); font-size:13px; }}
.pillola {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:13px; font-weight:500;
  background:var(--barra); color:{'var(--ok)' if finito else 'var(--ora)'}; }}
.totale {{ font:500 13px "IBM Plex Mono", monospace; font-variant-numeric:tabular-nums; }}
ol, ul {{ list-style:none; margin:0; padding:0; }}
.fasi {{ background:var(--carta); border:1px solid var(--riga); border-radius:10px; padding:6px 16px; }}
.fase {{ display:grid; grid-template-columns:1fr auto; grid-template-areas:"nome num" "barra dur"; gap:4px 12px;
  padding:10px 0; border-bottom:1px solid var(--riga); }}
.fase:last-child {{ border-bottom:0; }}
.nome {{ grid-area:nome; font-weight:500; }} .num {{ grid-area:num; font:13px "IBM Plex Mono", monospace;
  font-variant-numeric:tabular-nums; color:var(--tenue); }}
.dur {{ grid-area:dur; font-size:12px; color:var(--tenue); }}
.barra {{ grid-area:barra; align-self:center; height:6px; border-radius:3px; background:var(--barra); overflow:hidden; }}
.barra i {{ display:block; height:100%; background:var(--attesa); }}
.ok .barra i {{ background:var(--ok); }} .ora .barra i {{ background:var(--ora); }}
.ora .nome::after {{ content:" · adesso"; color:var(--ora); font-weight:500; }}
.piano li {{ display:grid; grid-template-columns:78px 1fr; grid-template-areas:"stato passo" ". mod"; gap:0 12px;
  padding:9px 0; border-bottom:1px solid var(--riga); }}
.stato {{ grid-area:stato; font:500 12px "IBM Plex Mono", monospace; }} .passo {{ grid-area:passo; }}
.mod {{ grid-area:mod; font-size:12px; color:var(--tenue); }}
.fatto .stato {{ color:var(--ok); }} .in-corso .stato {{ color:var(--ora); }} .da-fare .stato {{ color:var(--tenue); }}
.fatto .passo {{ color:var(--tenue); }}
.scadenze li {{ padding:9px 0; border-bottom:1px solid var(--riga); }}
.scadenze b {{ color:var(--accento); font-weight:600; margin-right:6px; }}
.nota {{ font-size:13px; color:var(--tenue); max-width:65ch; }}
</style>
<main>
  <header class="testa">
    <h1>Eden, esame d'identità</h1>
    <div><span class="pillola">{html.escape(stato)}</span></div>
    <div class="meta">{"Pagina viva: si aggiorna da sola ogni 20 secondi" if viva else "Istantanea"} · {ora} · sessione {g} · prove v1 <span class="totale">{X.impronta(X.PROVE)}</span></div>
  </header>
  <section>
    <h2>Esame in corso · {fatte} di {tot} prove</h2>
    <ol class="fasi">{righe_fasi}</ol>
    <p class="nota">Ogni prova è una scelta A/B letta dalle probabilità del modello. Mentre l'esame gira Eden risponde
    lentamente; alla fine rilegge la storia da sola. Le prove nascoste e quelle «chi l'ha detto?» contano ma non si
    mostrano una per una.</p>
  </section>
  <section>
    <h2>Scadenze che non si saltano</h2>
    <ul class="scadenze">{righe_scad}</ul>
  </section>
  <section>
    <h2>Piano della ricerca</h2>
    <ol class="piano">{righe_piano}</ol>
  </section>
</main>
"""


def servi(porta: int, g: str) -> None:
    """Pagina viva: ricalcolata dai file dei risultati a ogni richiesta. Solo da questo PC."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Pagina(BaseHTTPRequestHandler):
        def do_GET(self):
            corpo = ("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                     + pagina(g, viva=True)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def log_message(self, *_):
            pass
    ThreadingHTTPServer(("127.0.0.1", porta), Pagina).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uscita", nargs="?")
    ap.add_argument("--servi", type=int, metavar="PORTA")
    ap.add_argument("--data", default=datetime.now().strftime("%Y%m%d"))
    a = ap.parse_args()
    if a.servi:
        servi(a.servi, a.data)
    else:
        Path(a.uscita).write_text(pagina(a.data), encoding="utf-8")
        print(a.uscita)


if __name__ == "__main__":
    main()
