"""Entry point used for development (`python run.py`) and by PyInstaller."""
import multiprocessing

from rsstatchem.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
