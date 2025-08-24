from pathlib import Path
import hashlib, time, os
import requests

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_weights(dst: Path, url: str, sha256: str | None = None, retries: int = 3, timeout: int = 60) -> Path:
    """学習済みモデルをダウンロードして dst に保存。既にあれば再DLしない。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        last = None
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        for i in range(retries):
            try:
                with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                break
            except Exception as e:
                last = e
                time.sleep(1 + i)
        else:
            raise RuntimeError(f"Download failed from {url}: {type(last).__name__}: {last}")

    if sha256:
        got = _sha256(dst).lower()
        if got != sha256.lower():
            dst.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch: expected={sha256}, got={got}")
    return dst
