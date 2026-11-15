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

