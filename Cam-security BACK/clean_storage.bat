@echo off
chcp 65001 >nul
echo ========================================================
echo   Limpeza de Storage e Histórico - Cam-Security
echo ========================================================

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe clean_storage.py %*
) else (
    python clean_storage.py %*
)

pause
