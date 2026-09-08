"""Build the small source dataset archive to attach to Kaggle."""

import hashlib
import json
import zipfile
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    destination = root / "dist" / "aerorl-kaggle-source.zip"
    destination.parent.mkdir(exist_ok=True)
    paths = []
    for directory, suffixes in [
        ("training", {".py"}),
        ("shared", {".json", ".md"}),
        ("docs", {".md"}),
        ("notebooks", {".py", ".md", ".ipynb", ".in", ".lock"}),
    ]:
        paths.extend(
            p for p in (root / directory).rglob("*") if p.suffix in suffixes and p.is_file()
        )
    paths.extend(root / p for p in ["pyproject.toml", "uv.lock", "README.md"])
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            entry = zipfile.ZipInfo(
                path.relative_to(root).as_posix(), date_time=(2026, 9, 6, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    report = {
        "file": destination.name,
        "files": len(paths),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
