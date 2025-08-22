@echo off
SETLOCAL

REM Set the working directory to the script’s location
cd /d "%~dp0"

REM Set up the virtual environment
python -m venv game_analyzer_env

REM Activate the virtual environment
call game_analyzer_env\Scripts\activate.bat

REM Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Installation complete. You can now run the app using run.bat
pause
