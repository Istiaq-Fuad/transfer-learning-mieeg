import pooch
from pathlib import Path

_orig_retrieve = pooch.retrieve

def patched_retrieve(url, known_hash, fname=None, path=None, **kwargs):
    if path is not None and fname is not None:
        p = Path(path) / fname
        if p.exists() and p.stat().st_size > 1000000:
            # File exists and is >1MB, just return it instead of checking hash/downloading
            return str(p)
    return _orig_retrieve(url, known_hash, fname=fname, path=path, **kwargs)

pooch.retrieve = patched_retrieve
