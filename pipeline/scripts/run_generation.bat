@echo off
cd /d "%~dp0"
python run_generation.py --verify %*
pause
