"""
modello.py — il server llama.cpp (router su :8080, condiviso con OpenCode; vedi docs/MISURE_2026-09.md).

Una sola chiamata per messaggio, in streaming. Il ragionamento va chiesto in modo esplicito
(`enable_thinking`): senza, il modello ragiona di nascosto a ogni risposta.
Livello `medium`: con `xhigh` (il predefinito del template) il template scrive una riga in cima al system prompt,
e accendere il ragionamento farebbe rileggere tutta la storia (misurato: 177 s invece di 2 s).
"""
from __future__ import annotations

import json
import time
from typing import Iterator

import requests

URL = "http://127.0.0.1:8080"
MODELLO = "qwen3.8-27b"          # ufficiale: pari all'uncensored nell'esame, ma non mescola l'inglese
TIMEOUT_LETTURA = 1800            # la prima lettura della storia lunga richiede ~3 minuti (di più se carica il modello)

# Valori consigliati da Qwen per Qwen3: con ragionamento 0.6/0.95, senza 0.7/0.8
PARAMETRI = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8000},
             False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1500}}


def stato() -> str:
    """'loaded' · 'loading' · 'unloaded' · 'irraggiungibile'."""
    try:
        dati = requests.get(f"{URL}/v1/models", timeout=5).json()["data"]
    except (requests.RequestException, ValueError, KeyError):
        return "irraggiungibile"
    return next((m["status"]["value"] for m in dati if m["id"] == MODELLO), "irraggiungibile")


def conta_token(testo: str) -> int:
    r = requests.post(f"{URL}/tokenize", json={"content": testo, "model": MODELLO}, timeout=600)
    r.raise_for_status()
    return len(r.json()["tokens"])


def attendi_server(attesa_sec: int = 600) -> None:
    """Aspetta che il router risponda. Se un altro modello è in caricamento non si chiede il nostro:
    il supervisore di avvio del server lo ucciderebbe (vedi D:\\llama.cpp\\run-server.ps1)."""
    t0 = time.time()
    while time.time() - t0 < attesa_sec:
        try:
            dati = requests.get(f"{URL}/v1/models", timeout=5).json()["data"]
            if not any(m["status"]["value"] == "loading" for m in dati):
                return
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(5)
    raise TimeoutError("server llama.cpp non pronto")


def genera(messaggi: list[dict], pensiero: bool, max_tokens: int | None = None,
           strumenti: list[dict] | None = None, senza_strumenti: bool = False) -> Iterator[tuple[str, object]]:
    """Eventi: ('pensiero', testo) · ('testo', testo) · ('chiamata', {id, nome, argomenti}) · ('fine', statistiche).
    `strumenti` va mandato sempre uguale (sta in cima al prompt: cambiarlo fa rileggere tutto); `senza_strumenti`
    lascia le definizioni ma vieta di chiamarli (ultimo giro)."""
    corpo = {"model": MODELLO, "messages": messaggi, "stream": True, "cache_prompt": True,
             "chat_template_kwargs": {"enable_thinking": pensiero, "reasoning_effort": "medium"}, **PARAMETRI[pensiero]}
    if max_tokens is not None:
        corpo["max_tokens"] = max_tokens
    if strumenti:
        corpo["tools"] = strumenti
        corpo["tool_choice"] = "none" if senza_strumenti else "auto"
    t0, primo, stat, chiamate = time.time(), None, {}, {}
    with requests.post(f"{URL}/v1/chat/completions", json=corpo, stream=True, timeout=(10, TIMEOUT_LETTURA)) as r:
        r.raise_for_status()
        for grezza in r.iter_lines():  # byte: il server non dichiara il charset e requests userebbe latin-1
            riga = grezza.decode("utf-8")
            if not riga.startswith("data: "):
                continue
            dato = riga[6:]
            if dato == "[DONE]":
                break
            j = json.loads(dato)
            if j.get("timings"):
                stat = j["timings"]
            for scelta in j.get("choices") or []:
                delta = scelta.get("delta") or {}
                if delta.get("reasoning_content"):
                    yield "pensiero", delta["reasoning_content"]
                if delta.get("content"):
                    primo = primo or time.time()
                    yield "testo", delta["content"]
                for c in delta.get("tool_calls") or []:  # arrivano a pezzi: nome e argomenti si compongono per indice
                    x = chiamate.setdefault(c.get("index", 0), {"id": None, "nome": "", "argomenti": ""})
                    x["id"] = c.get("id") or x["id"]
                    x["nome"] += (c.get("function") or {}).get("name") or ""
                    x["argomenti"] += (c.get("function") or {}).get("arguments") or ""
    for i in sorted(chiamate):
        yield "chiamata", chiamate[i]
    yield "fine", {"sec": round(time.time() - t0, 2), "primo_testo_sec": round(primo - t0, 2) if primo else None,
                   "prompt_token": stat.get("prompt_n"), "cache_token": stat.get("cache_n"),
                   "token_scritti": stat.get("predicted_n")}
