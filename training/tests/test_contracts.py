from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from training.contracts import ContractError, ContractRegistry, load_config_bundle
from training.contracts.loader import parse_json, validate_config_bundle, verify_manifest_configs

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "shared"
CASES = json.loads((SHARED / "fixtures/contracts/cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry() -> ContractRegistry:
    return ContractRegistry(SHARED / "schemas")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_shared_acceptance_corpus(registry: ContractRegistry, case: dict) -> None:
    original = copy.deepcopy(case["data"])
    if case["accepted"]:
        registry.validate(case["schema"], original)
    else:
        with pytest.raises(ContractError):
            registry.validate(case["schema"], original)
    assert original == case["data"], "Validation must not mutate caller data"


def test_hashes_preserve_original_bytes(registry: ContractRegistry) -> None:
    raw = (SHARED / "vehicle.v1.json").read_bytes()
    loaded = registry.load_bytes("vehicle", raw)
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
    whitespace_variant = raw + b"\n "
    variant = registry.load_bytes("vehicle", whitespace_variant)
    assert variant.data == loaded.data
    assert variant.sha256 != loaded.sha256
    with pytest.raises(ContractError, match="SHA-256"):
        registry.load_bytes("vehicle", whitespace_variant, loaded.sha256)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e999}',
        b'{"a":9007199254740992}',
        b'{"a":1,}',
        b'{/*comment*/"a":1}',
        b"\xff",
        b"\xef\xbb\xbf{}",
    ],
)
def test_rejects_nonportable_json(raw: bytes) -> None:
    with pytest.raises(ContractError):
        parse_json(raw)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_python_objects(registry: ContractRegistry, value: float) -> None:
    state = next(case["data"] for case in CASES if case["id"] == "state-valid")
    state = copy.deepcopy(state)
    state["position_m"][0] = value
    with pytest.raises(ContractError):
        registry.validate("state", state)


def test_quaternion_roundoff_normalized_without_mutation(registry: ContractRegistry) -> None:
    state = copy.deepcopy(next(case["data"] for case in CASES if case["id"] == "state-valid"))
    state["quaternion_wxyz"] = [1.0000005, 0, 0, 0]
    result = registry.validate("state", state)
    assert result["quaternion_wxyz"] == [1, 0, 0, 0]
    assert state["quaternion_wxyz"][0] == 1.0000005


def test_observation_contract_and_control_capacity() -> None:
    bundle = load_config_bundle(SHARED)
    observation = bundle["observation"].data
    assert [f["index"] for f in observation["fields"]] == list(range(40))
    assert len({f["name"] for f in observation["fields"]}) == 40
    assert observation["fields"][21]["name"] == "ground_clearance"
    assert observation["fields"][30]["name"] == "current_gate_present"
    assert observation["fields"][39]["name"] == "time_remaining_fraction"
    vehicle = bundle["vehicle"].data
    assert vehicle["action"]["max_collective_n"] > vehicle["mass_kg"] * vehicle["gravity_mps2"]


def test_typescript_parity() -> None:
    node = os.environ.get("AERORL_NODE") or shutil.which("node")
    assert node, "Install Node.js to run required cross-language checks"
    tsx = ROOT / "web/node_modules/tsx/dist/cli.mjs"
    assert tsx.exists(), "Run npm ci in web/ before the cross-language tests"
    command = [
        node,
        str(tsx),
        str(ROOT / "web/scripts/check-contracts.ts"),
        "--shared",
        str(SHARED),
        "--fixtures",
        str(SHARED / "fixtures/contracts/cases.json"),
    ]
    js = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    py = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.check_contracts",
            "--shared",
            str(SHARED),
            "--fixtures",
            str(SHARED / "fixtures/contracts/cases.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(js.stdout) == json.loads(py.stdout)


def test_cross_config_compatibility() -> None:
    bundle = load_config_bundle(SHARED)
    bundle["vehicle"].data["rotor_max_thrust_n"] = 5
    with pytest.raises(ContractError, match="scaling"):
        validate_config_bundle(bundle)


def test_manifest_configuration_binding() -> None:
    bundle = load_config_bundle(SHARED)
    manifest = {
        "vehicle_sha256": bundle["vehicle"].sha256,
        "observation_sha256": bundle["observation"].sha256,
        "rules_sha256": bundle["course-rules"].sha256,
        "training_config_sha256": bundle["training"].sha256,
    }
    verify_manifest_configs(manifest, bundle)
    manifest["vehicle_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="vehicle_sha256"):
        verify_manifest_configs(manifest, bundle)
