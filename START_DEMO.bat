@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Follow the setup steps in README.md first.
  pause
  exit /b 1
)
if not exist ".env" (
  echo .env file not found. Copy .env.example to .env and add your Gemini API key.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
