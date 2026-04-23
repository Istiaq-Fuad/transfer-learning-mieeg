with open("data/loader.py", "r") as f:
    text = f.read()

# remove my broken patch block entirely
bad_patch = """import pooch
_orig_retrieve = pooch.retrieve

def _patched_pooch_retrieve(url, known_hash, fname=None, path=None, **kwargs):
    if path is not None and fname is not None:
        p = Path(path) / fname
        if p.exists() and p.stat().st_size > 1000000:
            return str(p)
    return _orig_retrieve(url, known_hash, fname=fname, path=path, **kwargs)

pooch.retrieve = _patched_pooch_retrieve

import mne"""
text = text.replace(bad_patch, "import mne")

# insert it at the TOP of the file
import_patch = """
import pooch
from pathlib import Path
_orig_retrieve = pooch.retrieve

def _patched_pooch_retrieve(url, known_hash, fname=None, path=None, **kwargs):
    if path is not None and fname is not None:
        p = Path(path) / fname
        if p.exists() and p.stat().st_size > 1000000:
            return str(p)
    return _orig_retrieve(url, known_hash, fname=fname, path=path, **kwargs)

pooch.retrieve = _patched_pooch_retrieve
"""

# inject right after typing imports
text = text.replace("import logging", "import logging\n" + import_patch)

with open("data/loader.py", "w") as f:
    f.write(text)
