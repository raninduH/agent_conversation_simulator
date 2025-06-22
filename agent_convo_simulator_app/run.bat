@echo off
echo Starting Multi-Agent Conversation Simulator...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    pause
    exit /b 1
)

REM Change to the application directory
cd /d "%~dp0"

REM Check if required packages are installed
echo Checking dependencies...
python -c "import tkinter, json, os, datetime, threading, typing" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Some required packages are missing
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Try to import the custom modules to check if everything is properly set up
python -c "from data_manager import DataManager; from conversation_engine import ConversationSimulatorEngine" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some LangGraph dependencies might be missing
    echo The application will start but you may need to install additional packages
    echo Run: pip install -r requirements.txt
    echo.
)

REM Start the application
echo Starting application...
python main.py

if errorlevel 1 (
    echo.
    echo Application exited with error. Check the console output above.
    pause
)
