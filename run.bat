@echo off
setlocal enabledelayedexpansion

:: === CONFIG ===
set VENV_DIR=.venv
set SCRIPT_NAME=main.py
set WINDOW_TITLE=Local Kokoro TTS Launcher
:: ==============

echo.
echo [Setup] Checking Python and uv...

:: Check for Python
where python >nul 2>nul || (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.11+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check for uv
where uv >nul 2>nul || (
    echo 'uv' is not installed.
    echo Run: pip install uv
    echo Or visit: https://github.com/astral-sh/uv
    start https://github.com/astral-sh/uv
    pause
    exit /b 1
)

:: Create venv if missing
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    uv venv %VENV_DIR% --python python3.11
)

:: Ensure pip exists inside venv
echo Ensuring pip is available in venv...
%VENV_DIR%\Scripts\python.exe -m ensurepip --upgrade

:: Activate venv
echo Activating environment...
call %VENV_DIR%\Scripts\activate

:: Install non-Torch dependencies
echo Installing dependencies...
uv pip install -r requirements.txt

:: Install correct PyTorch build.
:: On CPU-only systems, skip reinstall if torch is already present.
:: Force a clean reinstall with:  run.bat reinstall   (close the app first!)
%VENV_DIR%\Scripts\python.exe -c "import importlib.util, os, shutil, sys; has_torch = importlib.util.find_spec('torch') is not None; has_gpu = os.path.exists(r'C:\Windows\System32\nvcuda.dll') or shutil.which('nvidia-smi') is not None; cuda_ready = has_torch and ((not has_gpu) or __import__('torch').cuda.is_available()); sys.exit(0 if cuda_ready else 1)" >nul 2>nul
if %errorlevel% equ 0 if /i not "%~1"=="reinstall" (
    echo PyTorch already installed and configured correctly - skipping reinstall.
    goto :gpu_ok
)

echo Installing PyTorch based on CUDA support...
%VENV_DIR%\Scripts\python.exe install_torch_uv.py
if %errorlevel% neq 0 (
    echo Failed to install PyTorch.
    echo Please check your CUDA installation and try again.
    pause
    exit /b 1
)
:: Check for GPU support
echo Checking for GPU support...
%VENV_DIR%\Scripts\python.exe -c "import torch; print('GPU support:', torch.cuda.is_available())"
if %errorlevel% neq 0 (
    echo Failed to check GPU support.
    echo Please ensure PyTorch is installed correctly.
    pause
    exit /b 1
)
:gpu_ok
:: Check for ffmpeg
where ffmpeg >nul 2>nul || (
    echo Warning: 'ffmpeg' is not found in your system PATH.
    echo Audio playback or export may not work properly.
    echo Download it from: https://ffmpeg.org/download.html
    echo.
)

:: Launch the app
echo Launching the app...
start "%WINDOW_TITLE%" cmd /k "%VENV_DIR%\Scripts\python.exe %SCRIPT_NAME%"
