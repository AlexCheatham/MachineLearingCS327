Venv is just the virtual environment, all installs are to your active venv, setup on Windows by
python -m venv .venv
# https://code.visualstudio.com/docs/python/environments

.gitignore tells source control/git to ignore our virtual environment venv

requirements.txt holds all the requirements for code quality, make sure to install it once you have everything
pip install -r requirements.txt