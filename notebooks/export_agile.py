"""Export a selected agile candidate without replacing the deployed browser model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from training.learning.export import Actor


def run(directory, checkpoint=None):
    directory=Path(directory).resolve()
    report=json.loads((directory/"report.json").read_text())
    selected=directory/checkpoint if checkpoint else Path(report["selected_model"])
    model=PPO.load(selected,device="cpu")
    torch.set_num_threads(1)
    actor=Actor(model.policy).eval()
    width=model.observation_space.shape[0]
    target=directory/("experimental-latest.onnx" if checkpoint else "candidate.onnx")
    torch.onnx.export(actor,torch.zeros(1,width),str(target),opset_version=17,
                      input_names=["observation"],output_names=["action"],
                      dynamic_axes={"observation":{0:"batch"},"action":{0:"batch"}},dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    # Real expanded-course observations, not only the original narrow generator.
    samples=[]
    for path in sorted((directory/"final-validation").glob("*.npz")):
        with np.load(path,allow_pickle=False) as trace:
            values=trace["observations"]
            samples.extend(values[::max(1,len(values)//128)])
    values=np.asarray(samples,dtype=np.float32)
    if values.ndim!=2 or values.shape[1]!=width:
        raise ValueError("Missing compatible held-out observations")
    session=ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
    actual=session.run(None,{"observation":values})[0]
    with torch.no_grad(): expected=actor(torch.from_numpy(values)).numpy()
    error=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(actual).all() or error>1e-5:
        raise AssertionError(f"ONNX parity failed: {error}; candidate must not be deployed")
    manifest=dict(input_dimension=width,output_dimension=4,max_absolute_error=error,cases=len(values),
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  checkpoint=str(selected), curriculum_screen_passed=False if checkpoint else report["release_ready"],
                  browser_deployed=False,observation_contract={40:"obs-bodyrate-v1",46:"agile-previous-gate-v2",109:"agile-other-gates-v1"}[width],
                  timeout_s={0:5,1:60,3:180,10:600},mastered_level=report["mastered_level"])
    (directory/("experimental-export-report.json" if checkpoint else "export-report.json")).write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--checkpoint", help="Explicit experimental checkpoint; never claims the selected-model release result")
    run(**vars(parser.parse_args()))
