"""Download the trained model weights into models/ and verify the checksum.

    python scripts/download_model.py

The weights (87 MB) are not stored in git. They are attached to the GitHub
release for this project instead.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/jaswanreddych/classifyai-api/releases/download/"
    "v1.0.0/model_weights.weights.h5"
)
SHA256 = "bc2f758b5dcb1689a625677b280a9501f2f09f82aaebe354b7684b6eaf713a5d"
TARGET = Path(__file__).resolve().parent.parent / "models" / "model_weights.weights.h5"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if TARGET.exists() and sha256_of(TARGET) == SHA256:
        print(f"Model already present and verified: {TARGET}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix(".part")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, partial)  # noqa: S310 (fixed https URL above)

    if sha256_of(partial) != SHA256:
        partial.unlink()
        print("Checksum mismatch, download discarded.", file=sys.stderr)
        return 1

    partial.replace(TARGET)
    print(f"Saved and verified: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
