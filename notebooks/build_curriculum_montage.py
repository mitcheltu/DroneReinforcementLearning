"""Publish recorded curriculum attempts in per-run/per-lesson local view bundles."""
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS=dict(hover="Hover retention",gentle="Gentle gates",wide_spacing="Wide spacing",
            turns_60="Turns up to 60°",sharp_120="Sharp turns up to 120°",
            rotated_gates="Independent gate headings",reversals="150–180° reversals",
            mixed="Mixed three-gate routes",ten_gate_mixed="Mixed ten-gate routes")
LABELS.update(climbs="Climbs",descents="Descents",alternating_heights="Alternating heights",
              descending_turns="Descending turns",maneuver_ten="Ten-gate 3D maneuvers")
LABELS.update(lateral_small="Sideways · small",lateral_large="Sideways · wide",
              lateral_extreme="Sideways · extreme",backtracking="Backtracking",
              independent_headings="Independent headings",steep_zigzag="Steep zigzags",
              near_coincident="Closely spaced gates",broad_ten="Wide ten-gate routes")

def replace_snapshot(temporary,destination):
    # Windows file scanners/readers can briefly deny rename while examining JSON.
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.1)

def build():
    root=Path(__file__).resolve().parents[1]
    target=root/"web/public/curriculum-montage"; target.mkdir(exist_ok=True)
    runs=[]
    for directory in sorted((root/"runs").glob("agile-curriculum-*")):
        if not (directory/"trajectories.jsonl").exists(): continue
        config=json.loads((directory/"configuration.json").read_text())
        groups=defaultdict(list)
        # Publish every lesson path on the first snapshot. A production Next.js
        # server discovers public paths at startup, before later lessons run.
        groups["hover"]=[]
        for profile in config["profiles"]:
            groups[profile["name"]]=[]
        with (directory/"trajectories.jsonl").open(encoding="utf-8") as handle:
            for index,line in enumerate(handle):
                if not line.endswith("\n"): break  # Ignore an in-progress final write.
                row=json.loads(line)
                lesson="hover" if row["level"]<0 else config["profiles"][row["level"]]["name"]
                source=row["source"]
                phase="demonstration" if source in ("retention-anchor","retention-demonstration") else "dagger" if source=="retention-dagger" else "ppo" if source=="critic_preparation" else source if source in ("ppo","demonstration","dagger") else "baseline" if source in ("original-baseline","retention-baseline") else "evaluation"
                groups[lesson].append(dict(id=f"{directory.name}-{index}",source="curriculum",run=directory.name,
                    lesson=lesson,phase=phase,stage=row["stage"],outcome=row["outcome"],duration=row["duration"],
                    course=row["course"],samples=row["samples"],
                    provenance=f"Recorded {source}; seed {row['seed']}; {config.get('collision_mode','solid gates')}; PPO category includes critic preparation"))
        lessons=[]
        for lesson, traces in groups.items():
            name=f"{directory.name}-{lesson}.json"
            destination=target/name
            temporary=destination.with_suffix(".partial")
            payload=json.dumps(traces,separators=(",",":"))
            if not destination.exists() or destination.read_text(encoding="utf-8")!=payload:
                temporary.write_text(payload,encoding="utf-8")
                replace_snapshot(temporary,destination)
            lessons.append(dict(id=lesson,label=LABELS.get(lesson,lesson.replace('_',' ')),count=len(traces),url=f"/curriculum-montage/{name}"))
        lessons.sort(key=lambda lesson: lesson["count"]==0)
        report=json.loads((directory/"report.json").read_text()) if (directory/"report.json").exists() else None
        runs.append(dict(id=directory.name,lessons=lessons,status="Finished" if report else "In progress",
                         validation=[] if not report else [{k:r[k] for k in ("name","successes","cases")} for r in report["validation"]],
                         qualified=False if not report else report["release_ready"]))
    legacy=root/'web/public/montage.json'
    if legacy.exists():
        groups=defaultdict(list)
        for row in json.loads(legacy.read_text())['traces']:
            row['phase']={'ppo':'ppo','demonstrations':'demonstration','validation':'evaluation'}.get(row['source'],'evaluation')
            groups[row['stage']].append(row)
        lessons=[]
        labels=['Hover','One gate','Varied one gate','Three gates','Three-gate turns','Ten gates','Varied ten gates']
        for stage,traces in sorted(groups.items()):
            name=f'original-stage-{stage}.json'
            temporary=(target/name).with_suffix('.partial')
            temporary.write_text(json.dumps(traces,separators=(',',':')),encoding='utf-8')
            replace_snapshot(temporary,target/name)
            lessons.append(dict(id=f'original-{stage}',label=f'Original · {labels[stage]}',count=len(traces),url=f'/curriculum-montage/{name}'))
        runs.insert(0,dict(id='original-recordings',lessons=lessons,status='Finished',validation=[],qualified=False))
    index=dict(generated_at=datetime.now(timezone.utc).isoformat(),runs=runs)
    temporary=target/"index.partial"; temporary.write_text(json.dumps(index,indent=2),encoding="utf-8"); replace_snapshot(temporary,target/"index.json")
    print(json.dumps(dict(runs=len(runs),attempts=sum(l["count"] for r in runs for l in r["lessons"]))))

if __name__=="__main__": build()
