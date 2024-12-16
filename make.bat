@echo off
python3 -m venv .venv --system-site-packages
echo set PYTHONPATH=%%cd%% >> .venv\Scripts\activate.bat
call .venv\Scripts\activate
python -m pip install pip-tools
pip-compile requirements/requirements.in
pip-sync requirements/requirements.txt
echo .venv/ >> .gitignore