@echo off
setlocal enabledelayedexpansion

REM Determine the application directory
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%\.venv"
set "PYTHON_EXEC=python"

where %PYTHON_EXEC% >nul 2>&1
if errorlevel 1 (
    set "PYTHON_EXEC=py -3"
    %PYTHON_EXEC% --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3 was not found on PATH. Install Python 3 and try again.
        pause
        exit /b 1
    )
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating local virtual environment...
    %PYTHON_EXEC% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)
python -m pip install -r "%APP_DIR%requirements.txt"
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

python "%APP_DIR%app.py"
endlocal
