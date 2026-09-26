"""
web.py — Eden guarda fuori: cerca sul web e legge una pagina. Solo lettura.

Ricerca: gli RSS di Bing (web e notizie), senza account né chiavi. DuckDuckGo era la scelta iniziale, ma risponde con un
test anti-robot (CAPTCHA) e non lo si aggira: il motore è una sola funzione (`cerca`), cambiarlo non tocca il resto.
Lettura: solo GET. Rete pubblica soltanto: mai indirizzi locali, privati, Tailscale (100.64/10), link-local. Il controllo
è fatto due volte: sul nome prima di connettersi e sull'indirizzo a cui la connessione arriva davvero (contro i nomi
che cambiano indirizzo tra un controllo e l'altro), a ogni salto di reindirizzamento. Niente cookie, form, POST, proxy.
Tutto il web pubblico è aperto tranne gli argomenti che Stefano ha escluso (per adulti, armi): filtro su ricerca, indirizzi,
risultati e pagine (vedi `argomento_escluso`).
Il testo che torna è materiale esterno: la pagina non può dare ordini (lo dicono l'intestazione del risultato e la persona).
"""
from __future__ import annotations

import ipaddress
import re
import socket
import unicodedata
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

UA = "Eden/2 (compagna personale locale; sola lettura)"
TIMEOUT = (5, 15)
MAX_BYTE = 1_500_000            # oltre, la pagina è troncata
MAX_CARATTERI_PAGINA = 6000
MAX_SALTI = 4                   # reindirizzamenti seguiti
K_RISULTATI = 6
MAX_QUERY = 200
PORTE = {80, 443}
TAG_TESTO = ["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"]
TAG_RUMORE = ["script", "style", "noscript", "template", "svg", "iframe", "form", "nav", "footer", "header", "aside",
              "button", "select", "figure"]


class ReteVietata(ValueError):
    """Indirizzo non pubblico o non consentito: il testo è quello che legge Eden."""


# ---- contenuti che Stefano ha escluso: per adulti e armi (tutto il resto del web è aperto) ----
# Si controllano ricerca, indirizzo, ogni risultato e ogni pagina. Le parole sono volutamente larghe («armi» blocca anche
# una notizia sulle armi nucleari): meglio un no di troppo che un sì di troppo; si allarga o si stringe qui.
_ADULTI = (r"porn\w*|xxx|hentai|onlyfans|escort\w*|nsfw|camgirl\w*|erotic\w*|xvideos|xnxx|redtube|youporn|xhamster|bdsm|fetish\w*|"
           r"sex\s?(?:cam|chat|tape|video|shop)s?|video\s+hot|adult\s+(?:video|content)")
_ARMI = (r"armi|arma|pistol\w+|fucil\w+|munizion\w+|esplosiv\w+|arms?|firearms?|guns?|rifles?|ammo|ammunition|silenziator\w+|"
         r"kalashnikov|ak-?47|ar-?15|granat\w+|gunsmith\w*|ghost\s+gun")
_VIETATO = {"per adulti": re.compile(rf"\b(?:{_ADULTI})\b"), "sulle armi": re.compile(rf"\b(?:{_ARMI})\b")}
MIN_PAROLE_PAGINA = 3       # nel corpo di una pagina servono almeno tante parole vietate; nel titolo o nell'indirizzo ne basta una


def _piano(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def argomento_escluso(testo: str, minimo: int = 1) -> str | None:
    """«per adulti» / «sulle armi» se il testo ne parla (almeno `minimo` parole), altrimenti None."""
    t = _piano(testo)
    return next((nome for nome, rx in _VIETATO.items() if len(rx.findall(t)) >= minimo), None)


def _escluso(testo: str, minimo: int = 1) -> None:
    if (nome := argomento_escluso(testo, minimo)):
        raise ReteVietata(f"Argomento che Stefano ha escluso ({nome}): non lo cerco né lo apro.")


def _pubblico(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if a.version == 6 and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_global and not a.is_multicast


def verifica_url(url: str) -> str:
    """URL consentito (http/https, porta 80/443, senza credenziali, host che risolve solo a indirizzi pubblici) o ReteVietata."""
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ReteVietata("Indirizzo non valido: serve un http:// o https:// completo.")
    if u.username or u.password:
        raise ReteVietata("Indirizzi con nome utente o password non ammessi.")
    try:
        porta = u.port or (443 if u.scheme == "https" else 80)
        ips = {i[4][0] for i in socket.getaddrinfo(u.hostname, porta, type=socket.SOCK_STREAM)}
    except (ValueError, OSError):
        raise ReteVietata(f"Non trovo il sito {u.hostname}.") from None
    if porta not in PORTE:
        raise ReteVietata("Solo le porte web normali (80 e 443).")
    if not ips or not all(_pubblico(i) for i in ips):
        raise ReteVietata("Questo indirizzo non è un sito pubblico: non lo apro.")
    return u.geturl()


def _controlla_arrivo(sock: socket.socket) -> None:
    if not _pubblico(sock.getpeername()[0]):
        sock.close()
        raise ReteVietata("Questo indirizzo non è un sito pubblico: non lo apro.")


class _Conn(HTTPConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _controlla_arrivo(sock)
        return sock


class _ConnS(HTTPSConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _controlla_arrivo(sock)
        return sock


class _Pool(HTTPConnectionPool):
    ConnectionCls = _Conn


class _PoolS(HTTPSConnectionPool):
    ConnectionCls = _ConnS


class _Adattatore(HTTPAdapter):
    def init_poolmanager(self, *a, **k):
        super().init_poolmanager(*a, **k)
        self.poolmanager.pool_classes_by_scheme = {"http": _Pool, "https": _PoolS}


def _sessione() -> requests.Session:
    """Nuova a ogni richiesta (niente cookie che passano da un sito all'altro), senza proxy né netrc dell'ambiente."""
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = UA
    s.mount("http://", _Adattatore())
    s.mount("https://", _Adattatore())
    return s


def _prendi(url: str) -> tuple[int, str | None, str, bytes]:
    """(stato, Location, tipo di contenuto, corpo troncato a MAX_BYTE)."""
    with _sessione() as s, s.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False) as r:
        tipo = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if r.is_redirect:
            return r.status_code, r.headers.get("location"), tipo, b""
        corpo = bytearray()
        for pezzo in r.iter_content(65536):
            corpo += pezzo
            if len(corpo) >= MAX_BYTE:
                break
        return r.status_code, None, tipo, bytes(corpo[:MAX_BYTE])


def _pulisci(t: str) -> str:
    """Il testo del web non può imitare una fonte (`[[#n]]`, `[[@n]]`): le parentesi doppie si spezzano."""
    return re.sub(r"\s+", " ", t.replace("[[", "[ [").replace("]]", "] ]")).strip()


def estrai(html: bytes) -> tuple[str, str]:
    """(titolo, testo leggibile): l'articolo se c'è, niente menu né script."""
    s = BeautifulSoup(html, "lxml")
    titolo = _pulisci(s.title.get_text(" ")) if s.title else ""
    for t in s(TAG_RUMORE):
        t.decompose()
    testo = ""
    for radice in (s.find("article"), s.find("main"), s.body, s):
        if radice is None:
            continue
        righe = []
        for b in radice.find_all(TAG_TESTO):
            if b.find(TAG_TESTO):
                continue                    # solo i blocchi più interni: niente testo doppio
            r = _pulisci(b.get_text(" "))
            if (b.name in ("h1", "h2", "h3", "h4") and len(r) > 2 or len(r) >= 40) and (not righe or righe[-1] != r):
                righe.append(r)
        testo = "\n".join(righe)
        if len(testo) >= 300:
            break
    return titolo, testo


def leggi(url: str) -> tuple[str, str, str]:
    """(indirizzo finale, titolo, testo) di una pagina, seguendo al massimo MAX_SALTI reindirizzamenti (ognuno verificato)."""
    for _ in range(MAX_SALTI + 1):
        url = verifica_url(url)
        _escluso(unquote(url))
        try:
            stato, dove, tipo, corpo = _prendi(url)
        except requests.RequestException as e:
            raise ReteVietata(f"Non riesco a leggere la pagina ({type(e).__name__}).") from None
        if dove:
            url = urljoin(url, dove)
            continue
        if stato >= 400:
            raise ReteVietata(f"La pagina risponde con errore {stato}.")
        if tipo not in ("text/html", "application/xhtml+xml", "text/plain"):
            raise ReteVietata(f"Non è una pagina di testo (tipo: {tipo or 'sconosciuto'}).")
        if tipo == "text/plain":
            titolo, testo = "", "\n".join(filter(None, (_pulisci(r) for r in corpo.decode("utf-8", "replace").splitlines())))
        else:
            titolo, testo = estrai(corpo)
        if not testo.strip():
            raise ReteVietata("La pagina non ha testo che riesca a leggere (forse serve JavaScript).")
        _escluso(titolo)
        _escluso(testo[:MAX_CARATTERI_PAGINA], MIN_PAROLE_PAGINA)
        return url, titolo, testo
    raise ReteVietata("Troppi reindirizzamenti.")


def taglia(testo: str, massimo: int = MAX_CARATTERI_PAGINA) -> tuple[str, bool]:
    """Il testo tagliato a fine riga (o parola) entro `massimo` caratteri; secondo valore: se è stato tagliato."""
    if len(testo) <= massimo:
        return testo, False
    t = testo[:massimo]
    for sep in ("\n", " "):
        if sep in t[massimo // 2:]:
            t = t[:t.rindex(sep)]
            break
    return t, True


def dominio(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def _senza_reindirizzo(url: str) -> str:
    """I link di Bing Notizie passano da un reindirizzo (apiclick.aspx?url=…): si prende l'indirizzo vero."""
    u = urlparse(url)
    if dominio(url).endswith("bing.com") and "apiclick" in u.path:
        vero = parse_qs(u.query).get("url")
        if vero:
            return vero[0]
    return url


_ITALIANO = re.compile(r"\b(?:di|del|della|dei|il|lo|la|le|gli|che|per|come|cosa|perch\w+|quando|dove|chi|non|un|una|è|sono)\b|[àèéìòù]")
K_SCIENZA = 3                   # su K_RISULTATI: articoli scientifici accanto al web, per le domande in inglese


def _bing(domanda: str, notizie: bool) -> list[dict]:
    try:
        r = requests.get("https://www.bing.com/" + ("news/search" if notizie else "search"), timeout=TIMEOUT,
                         params={"q": domanda[:MAX_QUERY], "format": "rss", "setlang": "it", "cc": "it", "adlt": "strict"},
                         headers={"User-Agent": UA})
        r.raise_for_status()
    except requests.RequestException as e:
        raise ReteVietata(f"Il motore di ricerca non risponde ({type(e).__name__}).") from None
    out = []
    for i in BeautifulSoup(r.content, "xml").find_all("item"):
        titolo, link = i.find("title"), i.find("link")
        if not titolo or not link or not link.text.strip().startswith("http"):
            continue
        estratto, data = i.find("description"), i.find("pubDate")
        out.append({"titolo": _pulisci(titolo.text), "url": _senza_reindirizzo(link.text.strip()),
                    "estratto": _pulisci(BeautifulSoup(estratto.text, "lxml").get_text(" "))[:300] if estratto else "",
                    "data": data.text.strip() if data else ""})
    return out


def _scienza(domanda: str) -> list[dict]:
    """Articoli di OpenAlex (archivio scientifico aperto, senza chiavi). Bing con domande tecniche in inglese risponde fuori
    tema (misurato: «drift diffusion model» → giochi di drifting): per queste serve una fonte che capisca il tema. Se non
    risponde, niente: il web resta."""
    try:
        r = requests.get("https://api.openalex.org/works", timeout=TIMEOUT, headers={"User-Agent": UA},
                         params={"search": domanda[:MAX_QUERY], "per-page": K_SCIENZA + 2,
                                 "select": "title,doi,publication_year,abstract_inverted_index,primary_location"})
        r.raise_for_status()
        lavori = r.json().get("results", [])
    except (requests.RequestException, ValueError):
        return []
    out = []
    for w in lavori:
        url = w.get("doi") or ((w.get("primary_location") or {}).get("landing_page_url") or "")
        if not w.get("title") or not url.startswith("http"):
            continue
        parole = sorted((p, t) for t, ps in (w.get("abstract_inverted_index") or {}).items() for p in ps)
        out.append({"titolo": _pulisci(w["title"]), "url": url, "estratto": _pulisci(" ".join(t for _, t in parole))[:300],
                    "data": str(w.get("publication_year") or "")})
    return out


def cerca(domanda: str, notizie: bool = False) -> tuple[list[dict], int]:
    """(risultati come {titolo, url, estratto, data}, quanti scartati dal filtro sugli argomenti esclusi). Bing in RSS; per
    le domande in inglese (non notizie) anche articoli scientifici da OpenAlex, messi per primi.
    Lancia ReteVietata se la domanda è su un argomento escluso o il motore non risponde."""
    _escluso(domanda)
    trovati = _bing(domanda, notizie)
    if not notizie and not _ITALIANO.search(domanda.lower()):
        trovati = _scienza(domanda)[:K_SCIENZA] + trovati
    out, scartati = [], 0
    for x in trovati:
        if argomento_escluso(f"{x['titolo']} {unquote(x['url'])} {x['estratto']}"):
            scartati += 1
        elif len(out) < K_RISULTATI:
            out.append(x)
    return out, scartati
