@echo off
chcp 65001 >nul
echo ========================================================
echo   Instalador de Dependências - Cam-Security Backend
echo ========================================================

:: Verifica se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python não foi encontrado no PATH. Instale o Python 3.10+ e tente novamente.
    pause
    exit /b 1
)

:: Cria ambiente virtual se não existir
if not exist ".venv" (
    echo [1/3] Criando ambiente virtual (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao criar ambiente virtual.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Ambiente virtual (.venv) já existe.
)

:: Ativa o ambiente virtual
echo [2/3] Ativando ambiente virtual...
call .venv\Scripts\activate.bat

:: Atualiza pip e instala dependências
echo [3/3] Atualizando pip e instalando dependências do requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo   [SUCESSO] Todas as dependências foram instaladas!
    echo ========================================================
) else (
    echo.
    echo [AVISO] Ocorreram erros durante a instalação. Verifique os logs acima.
)

pause
