@echo off
REM avvia_eden2.bat — Eden 2 (nucleo nuovo). Apre http://127.0.0.1:5050 e risponde anche dal telefono via Tailscale.
REM Serve il server llama.cpp (task pianificato "llama.cpp-server") con qwen3.8-27b.
REM Al primo avvio rilegge la storia (circa 3 minuti). Dal telefono: http://<indirizzo Tailscale del PC>:5050
cd /d "%~dp0"
start "" http://127.0.0.1:5050
py -3.11 -m nucleo.server --rete %*
