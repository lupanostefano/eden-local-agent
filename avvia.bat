@echo off
REM avvia.bat — Avvio rapido Eden
REM
REM Uso normale:  avvia.bat
REM Mobile/HTTPS: avvia.bat --https
REM   (richiede pyOpenSSL; sul telefono: https://<IP-Tailscale>:5000)
REM   Al primo accesso mobile accetta l'eccezione sicurezza nel browser.

cd /d "%~dp0"

REM Raccoglie argomenti extra (es. --https)
set ARGS=%*

REM Refactor 2026-04-26: agent.py vive in core/. Build .exe legacy archiviato in archive/old_launchers/.

echo Avvio Eden (Python)...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python 3.11 non trovato.
    pause & exit /b 1
)

py -3.11 -m pip install -r requirements.txt -q
py -3.11 core\agent.py %ARGS%
