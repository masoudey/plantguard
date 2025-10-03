"""Sanity checks for the React + Tailwind frontend scaffold."""
from __future__ import annotations

import json
from pathlib import Path


def test_package_json_scripts_exist():
    package_path = Path("plantguard/frontend/package.json")
    assert package_path.exists(), "React package.json missing"
    package = json.loads(package_path.read_text())
    assert package["scripts"].get("dev") == "vite"
    assert package["dependencies"].get("react") is not None


def test_tailwind_setup_files():
    assert Path("plantguard/frontend/tailwind.config.js").exists()
    assert Path("plantguard/frontend/postcss.config.js").exists()
    css = Path("plantguard/frontend/src/styles.css").read_text()
    assert "@tailwind base;" in css
