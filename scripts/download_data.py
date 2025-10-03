"""Utilities for downloading datasets required by PlantGuard."""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

BASE_URL = "https://example.com"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PlantGuard datasets")
    parser.add_argument("--vision", action="store_true", help="Download PlantVillage subset")
    parser.add_argument("--audio", action="store_true", help="Download synthetic audio samples")
    parser.add_argument("--text", action="store_true", help="Download FAQ corpus")
    args = parser.parse_args()

    if args.vision:
        download_file(f"{BASE_URL}/plantvillage.zip", Path("data/raw/plantvillage.zip"))
    if args.audio:
        download_file(f"{BASE_URL}/audio.zip", Path("data/raw/audio.zip"))
    if args.text:
        download_file(f"{BASE_URL}/faq.json", Path("data/raw/faq.json"))


if __name__ == "__main__":
    main()
