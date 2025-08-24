# utils/weights.py
from pathlib import Path
import urllib.request, hashlib, time

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_weights(dst: Path, url: str, sha256: str | None = None, retries: int = 3) -> Path:
    """学習済みモデルを自動ダウンロードして保存する"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        last = None
        for i in range(retries):
            try:
                urllib.request.urlretrieve(url, dst)
                break
            except Exception as e:
                last = e
                time.sleep(1+i)
        else:
            raise RuntimeError(f"Download failed: {last}")
    if sha256:
        got = _sha256(dst).lower()
        if got != sha256.lower():
            dst.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch (expected {sha256}, got {got})")
    return dst
