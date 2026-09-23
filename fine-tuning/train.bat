@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo Attivazione ambiente finetuning_env...
call finetuning_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERRORE: attivazione venv fallita. Crea prima il venv finetuning_env nella root del progetto.
    pause
    exit /b 1
)

echo Avvio training...
python fine-tuning/train_eden.py

echo.
echo Training completato.
pause
