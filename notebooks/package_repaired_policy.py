"""Package the completed local imitation-repair experiment and its measured evidence."""

import hashlib
import json
import zipfile
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs" / "imitation-repair"
    report = json.loads((run / "report.json").read_text(encoding="utf-8-sig"))
    source = run / "train_imitation.executed.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["script_sha256"]
    report["dagger_rounds"] = len(report["history"]) - 1
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    broad_path = root / "runs" / "imitation-broad" / "report.json"
    broad = json.loads(broad_path.read_text())
    stages = report["validation"] + broad["stages"]
    assert [row["stage"] for row in stages] == list(range(7)), "Complete all stage tests first"
    checkpoint = Path(report["best_checkpoint"])
    manifest = {
        "method": report["method"],
        "source_fingerprint": report["fingerprint"],
        "deterministic": True,
        "ppo_resume_compatible": False,
        "successes": sum(row["successes"] for row in stages),
        "cases": sum(row["cases"] for row in stages),
        "stages": [{key: row[key] for key in ("stage", "successes", "cases")} for row in stages],
        "sha256": {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in ("model.zip", "actor.onnx")
        },
    }
    (run / "policy-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    destination = root / "dist" / "aerorl-repaired-policy.zip"
    destination.parent.mkdir(exist_ok=True)
    files = [
        (checkpoint / name, name) for name in ("model.zip", "actor.onnx", "export-report.json")
    ]
    files += [
        (run / name, name)
        for name in (
            "report.json",
            "policy-manifest.json",
            "train_imitation.executed.py",
            "single-gate-replay.html",
            "ten-gate-replay.html",
            "demonstrations.npz",
        )
    ]
    files += [
        (broad_path, "broader-evaluation.json"),
        (root / "notebooks" / "POLICY_REPAIR.md", "README.md"),
    ]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
    print(
        json.dumps(
            {
                "file": str(destination),
                "successes": manifest["successes"],
                "cases": manifest["cases"],
                "bytes": destination.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
