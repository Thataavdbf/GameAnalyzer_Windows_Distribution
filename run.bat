
@echo off
echo Starting Game Analyzer...

REM Check if virtual environment exists
if not exist "game_analyzer_env\Scripts\python.exe" (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Run the application
game_analyzer_env\Scripts\python.exe src\main_gui.py

pause
