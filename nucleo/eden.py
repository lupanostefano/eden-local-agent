"""
eden.py — il cuore: un messaggio di Stefano → una risposta di Eden.

    salva il messaggio → cerca nei ricordi → turno (il modello risponde in streaming, usa gli strumenti che vuole,
    le fonti che cita vengono verificate) → salva la risposta → rinforza i ricordi che sono tornati utili
Il messaggio di Stefano è salvato PRIMA della chiamata e la risposta subito dopo: niente si perde, anche se la
pagina si chiude a metà. Una risposta alla volta: il server llama.cpp ha un solo slot.

La storia lunga si ricostruisce (≈3 minuti di rilettura) solo quando serve: all'avvio, quando il contesto si
riempie o quando un dato fisso scade (compleanno). Se può aspettare, lo fa in una pausa, non mentre Stefano scrive.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Iterator

from nucleo import citazioni, contesto, diario, modello, passato, peso, registro, ricerca, strumenti, turno

log = logging.getLogger("nucleo")

CONTESTO_MASSIMO = 204_800        # c del preset qwen3.8-27b
RISERVA = 8_000 + 12_000 + 12_000  # risposta con ragionamento + risultati della ricerca + strumenti
SOGLIA_RICOSTRUZIONE = 165_000    # oltre, si ricostruisce alla prossima pausa
PAUSA_SEC = 10 * 60               # "pausa" = nessun messaggio da 10 minuti
K_PASSATO, K_NORMALE = 12, 6      # scambi dalla ricerca: domanda sul passato (come nell'esame) / messaggio normale
PENSA_SEMPRE = True               # scelta di Stefano (24/09): ragiona sempre prima di parlare, niente risposte vuote


class Eden:
    def __init__(self):
        self._turno = threading.Lock()        # una generazione (o ricostruzione) per volta
        self._indice = threading.Lock()
        self.ricerca = ricerca.Ricerca()
        self.ctx: contesto.Contesto | None = None
        self.stato = "spenta"                 # spenta · si_sveglia · pronta · rilegge · errore
        self.errore = ""
        self.ultima_attivita = time.time()
        self.ultimo_totale = 0                # token dell'ultima richiesta (storia + turni + ricerca)

    # ------------------------------------------------------------ ciclo di vita

    def sveglia(self) -> None:
        """Avvio: servizio embedding, indice, storia lunga letta dal modello. Da chiamare in un thread."""
        try:
            self.stato = "si_sveglia"
            registro.prepara()
            ricerca.avvia_servizio()
            self._indicizza()
            with registro.connetti(sola_lettura=True) as con:  # subito, così le fonti citate si aprono anche prima del primo messaggio
                self.ricerca.aggiorna(con)
            modello.attendi_server()
            with self._turno:
                self._ricostruisci()
            self.stato = "pronta"
            threading.Thread(target=self._guardiano, daemon=True, name="guardiano").start()
        except Exception as e:
            self.stato, self.errore = "errore", str(e)
            log.exception("risveglio fallito")

    def _ricostruisci(self, prima_di_id: int | None = None) -> None:
        """Nuova storia lunga (i messaggi prima di `prima_di_id`) e prima lettura del modello: la cache del server si
        riempie qui. La lettura di prova ha la stessa forma di una richiesta vera (storia + ultimo messaggio),
        così il messaggio successivo riusa tutta la storia."""
        precedente, self.stato = self.stato, "rilegge"
        t0 = time.time()
        with registro.connetti() as con:
            self.ctx = contesto.costruisci(con, registro.adesso(), modello.conta_token, prima_di_id=prima_di_id,
                                           con_id=True)
            msgs = self.ctx.messaggi(con, self.ctx.ultimo_id + 1, "Rispondi solo con la parola: pronta.")
        for tipo, dato in modello.genera(msgs, False, max_tokens=1, strumenti=strumenti.DEFINIZIONI):
            if tipo == "fine":
                self.ultimo_totale = (dato["prompt_token"] or 0) + (dato["cache_token"] or 0)
        log.info("storia ricostruita: dal messaggio #%s, %s token, letta in %.0f s",
                 self.ctx.primo_id, self.ctx.token_storia, time.time() - t0)
        diario.scrivi_evento(registro.adesso(), "Rilegge la storia",
                             f"{self.ctx.token_storia:,}".replace(",", ".") + f" token, in {time.time() - t0:.0f} s")
        self.stato = precedente

    def _guardiano(self) -> None:
        """Ricostruisce nelle pause quando il contesto è quasi pieno o un dato fisso è scaduto."""
        while True:
            time.sleep(60)
            if time.time() - self.ultima_attivita < PAUSA_SEC or not self._da_ricostruire(registro.adesso()):
                continue
            if self._turno.acquire(blocking=False):
                try:
                    self._ricostruisci()
                except Exception:
                    log.exception("ricostruzione in pausa fallita")
                finally:
                    self._turno.release()

    def rileggi_in_pausa(self) -> bool:
        """Rilegge la storia in un thread, se Eden non sta rispondendo. False = occupata o non ancora sveglia."""
        if self.stato != "pronta" or not self._turno.acquire(blocking=False):
            return False

        def lavora():
            try:
                self._ricostruisci()
            except Exception:
                log.exception("rilettura richiesta fallita")
            finally:
                self._turno.release()
        threading.Thread(target=lavora, daemon=True, name="rilettura").start()
        return True

    def _da_ricostruire(self, adesso: str) -> bool:
        return self.ctx is None or self.ctx.scaduto(adesso) or self.ultimo_totale > SOGLIA_RICOSTRUZIONE

    def _indicizza(self) -> None:
        with self._indice:
            try:
                with registro.connetti() as con:
                    n = ricerca.indicizza_mancanti(con)
                    marcati = peso.riempi_mancanti(con)
                if n or marcati:
                    log.info("indicizzati %d messaggi, pesati %d scambi", n, marcati)
            except Exception:
                log.exception("indicizzazione fallita (si riprova al prossimo messaggio)")

    # ------------------------------------------------------------ risposta

    def rispondi(self, testo: str) -> Iterator[dict]:
        """Eventi per la pagina: inizio · rilegge · pensa · testo · fine · errore."""
        if self.stato in ("spenta", "si_sveglia", "errore"):
            yield {"tipo": "errore", "testo": "Mi sto ancora svegliando." if self.stato != "errore" else self.errore}
            return
        if not self._turno.acquire(blocking=False):
            if self.stato != "rilegge":
                yield {"tipo": "errore", "testo": "Sto ancora rispondendo al messaggio di prima."}
                return
            yield {"tipo": "rilegge"}  # ricostruzione in pausa già partita: si aspetta che finisca
            self._turno.acquire()
        try:
            yield from self._rispondi(testo)
        finally:
            self.ultima_attivita = time.time()
            self._turno.release()
            threading.Thread(target=self._indicizza, daemon=True).start()

    def _rispondi(self, testo: str) -> Iterator[dict]:
        ts = registro.adesso()
        scambio = f"e2_{ts}_{uuid.uuid4().hex[:6]}"
        id_utente = registro.salva_messaggio("user", testo, scambio, ts=ts)
        self.ultima_attivita = time.time()
        # contesto troppo pieno per questo messaggio o dato scaduto: si rilegge adesso (non può aspettare la pausa)
        if self.ctx is None or self.ctx.scaduto(ts) or self.ultimo_totale + RISERVA > CONTESTO_MASSIMO:
            yield {"tipo": "rilegge"}
            self._ricostruisci(id_utente)  # la storia si ferma prima di questo messaggio, che arriva come ultimo
        elif modello.stato() != "loaded":  # un altro programma ha usato il server: la cache è persa
            yield {"tipo": "rilegge"}
        sul_passato = passato.sul_passato(testo)
        pensa = PENSA_SEMPRE or sul_passato
        if not ricerca.servizio_attivo():  # il servizio degli embedding può cadere (es. dopo un riavvio del server)
            ricerca.avvia_servizio()
        con = registro.connetti()
        try:
            self.ricerca.aggiorna(con)
            indici = self.ricerca.cerca(con, testo, k=K_PASSATO if sul_passato else K_NORMALE, metodo="richiamo",
                                        prima_di_id=id_utente, adesso=ts)
            t = strumenti.Turno(con, self.ricerca, prima_di_id=id_utente, adesso=ts, primo_id=self.ctx.primo_id)
            for i in indici:
                t.visti.update(self.ricerca.scambi[i]["ids"])
            risultati = contesto.blocco_ricerca(self.ricerca.scambi, indici, con_id=True) if indici else ""
            msgs = self.ctx.messaggi(con, id_utente, contesto.messaggio_finale(
                testo, ts, risultati, storia_da=contesto.data_it(self.ctx.primo_ts, ora=False)))
            for m in msgs[1:]:  # le fonti citate nelle sue risposte di poco fa sono ancora sotto i suoi occhi
                if m["role"] == "assistant":
                    t.visti.update(citazioni.numeri(m["content"]))
            yield {"tipo": "inizio", "pensa": pensa, "id": id_utente}
            r = turno.Risultato()
            try:
                yield from turno.rispondi(msgs, t, pensa, r)
            except Exception as e:
                log.exception("generazione interrotta")
                if r.testo.strip():  # quello che è già arrivato si salva, segnato come interrotto
                    registro.salva_messaggio("assistant", r.testo.strip(), scambio,
                                             meta={"interrotta": True, "pensiero": r.pensiero or None})
                diario.scrivi_evento(ts, "Errore", f"Il modello non ha risposto a: {testo[:200]} ({e})")
                yield {"tipo": "errore", "testo": f"Il modello non risponde: {e}"}
                return
            if not r.testo:  # nemmeno il giro di ripiego ha risposto: meglio dirlo che salvare un silenzio
                log.error("risposta vuota anche dopo il ripiego (giri %d)", r.giri)
                diario.scrivi_turno(ts, testo, "(nessuna risposta)", r, t, interrotta=True)
                yield {"tipo": "errore", "testo": "Non sono riuscita a risponderti. Riprova a scrivermi."}
                return
            id_eden = registro.salva_messaggio("assistant", r.testo, scambio, meta={
                "pensiero": r.pensiero or None, "ricerca": [self.ricerca.scambi[i]["ids"][0] for i in indici],
                "strumenti": t.chiamate or None, "citazioni": [c.per_registro() for c in r.citazioni] or None,
                "bozza": r.bozza, **r.statistiche()})
            diario.scrivi_turno(ts, testo, r.testo, r, t, id_eden)
            for scambio_citato in {c.scambio for c in r.citazioni if c.stato == "ok"}:  # è tornato utile: il ricordo si rinforza
                peso.rinforza(con, scambio_citato, registro.adesso())
        finally:
            con.close()
        self.ultimo_totale = r.contesto_iniziale  # i risultati degli strumenti non restano: il turno dopo parte da qui
        log.info("risposta #%d: pensa=%s, %.1f s (primo testo %s s), giri %d, strumenti %d, contesto %s token (riletti %s)",
                 id_eden, pensa, r.sec, r.primo_testo_sec, r.giri, len(t.chiamate), r.contesto, r.prompt_token)
        if r.citazioni:
            yield {"tipo": "citazioni", "citazioni": [c.per_registro() for c in r.citazioni]}
        yield {"tipo": "fine", "id": id_eden, "sec": round(r.sec, 2)}
