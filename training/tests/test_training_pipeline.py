import json

import nbformat
import numpy as np
import pytest

from training.envs.settings import ROOT
from training.learning.export import export
from training.learning.train import run
from training.recording.visualize import visualize


def test_notebook_structure_and_python_syntax():
    notebook = nbformat.read(ROOT / "notebooks" / "AeroRL_Kaggle_Training.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook-cell", "exec")


def test_train_resume_export_and_recording(tmp_path):
    first = run(tmp_path, steps=256, smoke=True)
    first_state = json.loads((first / "campaign.json").read_text())
    assert first_state["steps"] == 256
    second = run(tmp_path, steps=128, smoke=True, resume=first)
    assert json.loads((second / "campaign.json").read_text())["steps"] == 384
    traces = list((tmp_path / "training-traces").glob("*.npz"))
    assert traces
    with np.load(traces[0], allow_pickle=False) as trace:
        assert len(trace["actions"]) == len(trace["raw_actions"]) == len(trace["rewards"])
        np.testing.assert_allclose(np.clip(trace["raw_actions"], -1, 1), trace["actions"][:, 2:6])
        assert len(trace["commands"]) + 1 == len(trace["states"])
        assert np.all(np.diff(trace["states"][:, 0]) > 0)
    html = visualize(traces[0], tmp_path / "flight.html")
    assert "Plotly.addFrames" in html.read_text(encoding="utf-8")
    assert export(second).exists()
    assert json.loads((second / "export-report.json").read_text())["max_absolute_error"] <= 1e-5
    with pytest.raises(ValueError, match="identical budget"):
        run(tmp_path, steps=64, smoke=True, resume=second, budget=999)
