# Mandatory Set-up:
## File downloads
1. Navigate to https://drive.google.com/drive/folders/1TzzlKlM2wT8nvdzCBMrqZJGEs-V1wzsh.
2. Download the Bytefight Client 2026, Client Terminal, Player showcase folders.
3. Unzip them in the root of this repo
4. Place the unzipped player showcase folder inside of the _player_files_ folder.

## Environment Set-up
1. Install Python (if you do not have it)
    - Note: ensure python is 3.10 <= version < 3.14
    - I used `brew upgrade python`
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. Confirm where your Python interpreter is located with `which python`
5. If `which pip` doesn't show a path, install pip installer and confirm installation with the following commands:
    ```
    python -m pip install --upgrade pip
    python -m pip --version
    ```
6. `python -m pip install -r player_files/requirements.txt`

Game is now set-up, now you have two options to play/test: GUI and terminal


### GUI:
1. Copy virtual environment python path from `which python`
1. In order for Mac to run this app, run `xattr -dr com.apple.quarantine "Bytefight Client 2026.app"`.
2. Open _Bytefight Client 2026.app_
3. Go to Config tab
4. Set Python Path to the virtual environment path for python3 and save changes
5. When adding your own player/bot file, add the directory within the player_files/bots directory, and ensure that your
player subdirectory contains an \__init__.py, controller.py, and player_board.py

### Terminal:
1. Ensure virtual environment is activated
2. Follow player_files/docs/instructions/installation.md
unfinished