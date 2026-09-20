@echo off
REM LoRA Studio - one-shot setup for Windows
setlocal

echo.
echo === Creating virtual environment ===
python -m venv venv
if errorlevel 1 goto :err

call venv\Scripts\activate.bat

echo.
echo === Installing GUI dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :err

echo.
echo === Installing torch (CUDA 12.4 build) ===
echo If you have a different CUDA version, cancel and install torch manually.
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo.
echo === Environment check ===
python check_env.py

echo.
echo Done. Launch with:  venv\Scripts\activate  then  python run_gui.py
pause
exit /b 0

:err
echo.
echo Setup failed. Scroll up for the error.
pause
exit /b 1
