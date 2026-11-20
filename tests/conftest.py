import os
import site
from pathlib import Path

_DLL_HANDLES = []

# Ensure Windows loads package DLLs (such as faiss.dll) correctly
if hasattr(os, "add_dll_directory"):
    try:
        for sp in site.getsitepackages():
            faiss_dir = Path(sp) / "faiss"
            if faiss_dir.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(faiss_dir)))
    except Exception:
        pass

try:
    import faiss
except Exception:
    pass



import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A fresh RADAR database seeded from data/units.csv only (fast, isolated)."""
    from src import config, radar
    monkeypatch.setattr(radar, "DB_PATH", tmp_path / "radar.db")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)  # skip the 3.5k-row xlsx seed
    monkeypatch.setattr(config, "AUTO_REBUILD_INDEX", False)
    monkeypatch.setattr(config, "PRICE_CHANGE_HOLD_PCT", 25.0)
    radar.init_db()
    return radar
