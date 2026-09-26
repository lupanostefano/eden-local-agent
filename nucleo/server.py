"""
server.py — Eden 2: pagina di chat e /api/chat in streaming (SSE). Di solito solo da questo PC (127.0.0.1).

    python -m nucleo.server [--porta 5050] [--rete]

Con `--rete` ascolta anche dalla rete, ma risponde solo a questo PC e ai dispositivi della tua rete Tailscale
(indirizzi 100.64.0.0/10): il telefono apre http://<indirizzo Tailscale del PC>:5050. Senza HTTPS: la pagina scrive e
legge, non usa il microfono. Chiunque altro (la rete di casa, internet) riceve 403.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import queue
import threading

from flask import Flask, Response, jsonify, render_template, request

import eden_paths as P
from nucleo import modello, registro
from nucleo.eden import Eden

log = logging.getLogger("nucleo")
app = Flask(__name__, template_folder=str(P.NUCLEO_TEMPLATES_DIR))
app.config["TEMPLATES_AUTO_RELOAD"] = True  # la pagina si aggiorna senza riavviare (il riavvio costa 3 minuti di rilettura)
eden = Eden()


RETI_AMMESSE = [ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("100.64.0.0/10"),  # questo PC · Tailscale
                ipaddress.ip_network("::1/128")]


@app.before_request
def solo_pc_e_tailscale():
    try:
        ip = ipaddress.ip_address(request.remote_addr or "")
    except ValueError:
        return jsonify(errore="Non autorizzato."), 403
    if not any(ip in rete for rete in RETI_AMMESSE):
        return jsonify(errore="Non autorizzato."), 403


@app.get("/")
def pagina():
    return render_template("chat.html")


@app.get("/api/stato")
def stato():
    # occupata = sta generando (o rileggendo): la pagina che ha perso il collegamento aspetta e poi ricarica i messaggi
    return jsonify(stato=eden.stato, errore=eden.errore, modello=modello.stato(), occupata=eden._turno.locked())


@app.post("/api/rileggi")
def rileggi():
    """Rilegge la storia (≈3 minuti) quando qualcos'altro ha usato il modello (esami, prove): la cache è persa e il
    prossimo messaggio di Stefano la rifarebbe da capo. Solo da questo PC."""
    if not ipaddress.ip_address(request.remote_addr or "0.0.0.0").is_loopback:
        return jsonify(errore="Solo da questo PC."), 403
    return jsonify(avviata=eden.rileggi_in_pausa())


@app.get("/api/messaggi")
def messaggi():
    n = min(request.args.get("n", 60, type=int), 500)
    with registro.connetti(sola_lettura=True) as con:
        return jsonify(registro.ultimi_messaggi(con, n))


@app.get("/api/ricordo/<int:mid>")
def ricordo(mid: int):
    """Lo scambio a cui appartiene il messaggio `mid`: serve alla pagina per aprire le fonti citate."""
    r = eden.ricerca
    i = r.msg2sc.get(mid)
    if i is None:
        return jsonify(errore="Non è uno scambio dei nostri ricordi."), 404
    s = r.scambi[i]
    return jsonify(id=s["ids"][0], ts=s["ts"], u=s["u"], a=s["a"])


@app.post("/api/voto")
def voto():
    """Il voto di Stefano a una risposta di Eden: {id, voto: 'su'|'giu'|null, nota?}."""
    d = request.get_json(silent=True) or {}
    try:
        with registro.connetti() as con:
            if not con.execute("SELECT 1 FROM messaggi WHERE id = ? AND ruolo = 'assistant'", (d.get("id"),)).fetchone():
                return jsonify(errore="Non è una risposta di Eden."), 404
            registro.salva_voto(con, d["id"], d.get("voto"), d.get("nota"))
    except (ValueError, KeyError) as e:
        return jsonify(errore=str(e)), 400
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    testo = ((request.get_json(silent=True) or {}).get("messaggio") or "").strip()
    if not testo:
        return jsonify(errore="Messaggio vuoto."), 400
    coda: queue.Queue = queue.Queue()

    def lavora():  # thread a parte: se la pagina si chiude a metà, la risposta arriva in fondo e si salva lo stesso
        try:
            for evento in eden.rispondi(testo):
                coda.put(evento)
        except Exception as e:
            log.exception("risposta fallita")
            coda.put({"tipo": "errore", "testo": f"Errore: {e}"})
        coda.put(None)

    threading.Thread(target=lavora, daemon=True).start()

    def flusso():
        while (evento := coda.get()) is not None:
            yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"

    return Response(flusso(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


def main():
    ap = argparse.ArgumentParser(description="Eden 2")
    ap.add_argument("--porta", type=int, default=5050)
    ap.add_argument("--rete", action="store_true", help="ascolta anche da Tailscale (il telefono)")
    a = ap.parse_args()
    P.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(P.NUCLEO_LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    threading.Thread(target=eden.sveglia, daemon=True, name="risveglio").start()
    log.info("Eden 2 su http://127.0.0.1:%d", a.porta)
    app.run(host="0.0.0.0" if a.rete else "127.0.0.1", port=a.porta, threaded=True)


if __name__ == "__main__":
    main()
