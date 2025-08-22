@echo off
title Game Analyzer
echo ========================================
echo          Game Analyzer v1.0
echo    Video Game Footage Analysis Tool
echo ========================================
echo.

REM Check if virtual environment exists
if exist "game_analyzer_env\Scripts\python.exe" (
    echo Starting Game Analyzer...
    echo.
    game_analyzer_env\Scripts\python.exe src\main_gui.py
) else (
    echo Virtual environment not found.
    echo Please run install.bat first to set up the application.
    echo.
    pause
)

if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    echo Please check that all requirements are installed.
    echo.
    pause
)
