"""Publish recorded curriculum attempts in per-run/per-lesson local view bundles."""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS=dict(hover="Hover retention",gentle="Gentle gates",wide_spacing="Wide spacing",
            turns_60="Turns up to 60°",sharp_120="Sharp turns up to 120°",
            rotated_gates="Independent gate headings",reversals="150–180° reversals",
            mixed="Mixed three-gate routes",ten_gate_mixed="Mixed ten-gate routes")

def build():
    root=Path(__file__).resolve().parents[1]
    target=root/"web/public/curriculum-montage"; target.mkdir(exist_ok=True)
    runs=[]
    for directory in sorted((root/"runs").glob("agile-curriculum-*")):
        if not (directory/"trajectories.jsonl").exists(): continue
        config=json.loads((directory/"configuration.json").read_text())
        groups=defaultdict(list)
        with (directory/"trajectories.jsonl").open(encoding="utf-8") as handle:
            for index,line in enumerate(handle):
                if not line.endswith("\n"): break  # Ignore an in-progress final write.
                row=json.loads(line)
                lesson="hover" if row["level"]<0 else config["profiles"][row["level"]]["name"]
                source=row["source"]
                phase="ppo" if source=="critic_preparation" else source if source in ("ppo","demonstration","dagger") else "baseline" if source=="original-baseline" else "evaluation"
                groups[lesson].append(dict(id=f"{directory.name}-{index}",source="curriculum",run=directory.name,
                    lesson=lesson,phase=phase,stage=row["stage"],outcome=row["outcome"],duration=row["duration"],
                    course=row["course"],samples=row["samples"],
                    provenance=f"Recorded {source}; seed {row['seed']}; PPO category includes critic preparation"))
        lessons=[]
        for lesson, traces in groups.items():
            name=f"{directory.name}-{lesson}.json"
            destination=target/name
            temporary=destination.with_suffix(".partial")
            temporary.write_text(json.dumps(traces,separators=(",",":")),encoding="utf-8")
            temporary.replace(destination)
            lessons.append(dict(id=lesson,label=LABELS[lesson],count=len(traces),url=f"/curriculum-montage/{name}"))
        report=json.loads((directory/"report.json").read_text()) if (directory/"report.json").exists() else None
        runs.append(dict(id=directory.name,lessons=lessons,status="Finished" if report else "In progress",
                         validation=[] if not report else [{k:r[k] for k in ("name","successes","cases")} for r in report["validation"]],
                         qualified=False if not report else report["release_ready"]))
    index=dict(generated_at=datetime.now(timezone.utc).isoformat(),runs=runs)
    temporary=target/"index.partial"; temporary.write_text(json.dumps(index,indent=2),encoding="utf-8"); temporary.replace(target/"index.json")
    print(json.dumps(dict(runs=len(runs),attempts=sum(l["count"] for r in runs for l in r["lessons"]))))

if __name__=="__main__": build()
