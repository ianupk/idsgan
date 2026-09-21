"""
Download the NSL-KDD dataset into the data/ directory.

Usage
-----
    python download_data.py

Files downloaded
----------------
    data/KDDTrain+.txt  ~19 MB  Training split (125,973 records)
    data/KDDTest+.txt   ~ 3 MB  Test split     ( 22,544 records)

Source repository
-----------------
    https://github.com/jmnwong/NSL-KDD-Dataset
"""

import os
import requests

from src import config

# ---------------------------------------------------------------------------
# Remote URLs
# ---------------------------------------------------------------------------
_BASE = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
URLS: dict[str, str] = {
    "KDDTrain+.txt": f"{_BASE}/KDDTrain+.txt",
    "KDDTest+.txt":  f"{_BASE}/KDDTest+.txt",
}


def download_file(url: str, dest_path: str, chunk_size: int = 8_192) -> None:
    """Stream *url* to *dest_path*, skipping if the file already exists."""
    if os.path.exists(dest_path):
        print(f"  Already exists – skipping: {dest_path}")
        return

    print(f"  Downloading {os.path.basename(dest_path)} ...", end=" ", flush=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            downloaded += len(chunk)

    size_mb = downloaded / 1_048_576
    print(f"done ({size_mb:.1f} MB)")


def main() -> None:
    print("=" * 50)
    print("  NSL-KDD Dataset Downloader")
    print("=" * 50)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for filename, url in URLS.items():
        dest = os.path.join(config.DATA_DIR, filename)
        download_file(url, dest)
    print("\nAll files are ready in:", config.DATA_DIR)


if __name__ == "__main__":
    main()
