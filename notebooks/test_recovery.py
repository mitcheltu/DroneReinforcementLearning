"""Recovery safety checks; kept outside the original checkpoint fingerprint."""

import zipfile

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from notebooks.recover_training import model_file, regressed
from training.envs.drone import DroneEnv


def test_fixed_stages_preserve_equal_transition_counts_across_resets():
    env = DummyVecEnv([lambda stage=stage: DroneEnv(stage, evaluation=True) for stage in (0, 1)])
    env.reset()
    counts = [0, 0]
    resets = 0
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[:, 0] = -1  # No thrust: force unequal episode lengths and auto-resets.
    for _ in range(100):
        _, _, dones, infos = env.step(actions)
        for info in infos:
            counts[info["stage"]] += 1
        resets += sum(dones)
    env.close()
    assert counts == [100, 100]
    assert resets > 0


def test_guard_rejects_forgetting_even_when_total_success_improves():
    assert regressed({"successes": [63, 30]}, {"successes": [64, 0]})
    assert not regressed({"successes": [64, 30]}, {"successes": [64, 0]})


def test_extracted_kaggle_model_is_rebuilt_without_touching_input(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    for name in ("data", "policy.pth", "policy.optimizer.pth"):
        (directory / name).write_bytes(name.encode())
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with model_file(tmp_path) as path:
        assert path.is_file()
        with zipfile.ZipFile(path) as archive:
            assert {name: archive.read(name) for name in archive.namelist()} == before
    assert not path.exists()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert not (tmp_path / "model.zip").exists()


def test_standard_zip_and_incomplete_archive(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        with model_file(tmp_path):
            pass
    with zipfile.ZipFile(tmp_path / "model.zip", "w") as archive:
        archive.writestr("data", "{}")
    with model_file(tmp_path) as path:
        assert path == tmp_path / "model.zip"
