"""Load the shared contract schemas offline, validate JSON, and hash original bytes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONFIG_NAMES = ("vehicle", "observation", "course-rules", "training")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1


class ContractError(ValueError):
    """Stable error code plus a useful, local diagnostic."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LoadedArtifact:
    data: Any
    sha256: str
    byte_length: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_key", f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError("nonfinite", f"Invalid JSON number: {value}")


def _check_numbers(value: Any, path: str = "$", depth: int = 0) -> None:
    if depth > 64:
        raise ContractError("depth", "JSON nesting exceeds 64 levels")
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ContractError("nonfinite", f"Nonfinite number at {path}")
        if value == int(value) and abs(value) > MAX_SAFE_INTEGER:
            raise ContractError("unsafe_integer", f"Use a decimal string for the integer at {path}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_numbers(item, f"{path}[{index}]", depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_numbers(item, f"{path}.{key}", depth + 1)


def parse_json(raw: bytes) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise ContractError("size", "JSON artifact exceeds 16 MiB")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("json", f"Invalid UTF-8 JSON: {exc}") from exc
    _check_numbers(value)
    return value


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ContractError(code, message)


def _state_semantics(state: dict[str, Any]) -> None:
    quaternion = state["quaternion_wxyz"]
    norm = math.sqrt(sum(component * component for component in quaternion))
    _require(abs(norm - 1) <= 1e-6, "quaternion", "Quaternion norm must be within 1e-6 of one")
    state["quaternion_wxyz"] = [component / norm for component in quaternion]


def _semantics(name: str, data: Any) -> None:
    if name == "state":
        _state_semantics(data)
    elif name == "course":
        _state_semantics(data["initial_state"])
        gates = data["gates"]
        _require(
            len({gate["id"] for gate in gates}) == len(gates), "duplicate_id", "Gate IDs repeat"
        )
        _require(
            [gate["label"] for gate in gates] == list(range(1, len(gates) + 1)),
            "invalid_label",
            "Gate labels must follow array order, starting at one",
        )
        generator = data["generator"]
        if generator is not None:
            for field in ("seed", "reset_seed"):
                _require(int(generator[field]) <= 2**64 - 1, "seed", "Seed exceeds uint64")
            if data["mode"] == "curriculum":
                expected = [0, 1, 1, 3, 3, 10, 10][generator["stage"]]
                _require(len(gates) == expected, "gate_count", "Stage and gate count disagree")
    elif name == "vehicle":
        _require(data["physics_hz"] == 2 * data["policy_hz"], "timing", "Expected two ticks/action")
        _require(
            abs(data["action"]["max_collective_n"] - 4 * data["rotor_max_thrust_n"]) <= 1e-9,
            "collective",
            "Collective limit differs from rotor capacity",
        )
        b = data["arm_length_m"] / math.sqrt(2)
        expected_positions = [[b, b, 0], [-b, b, 0], [-b, -b, 0], [b, -b, 0]]
        for index, rotor in enumerate(data["rotors"]):
            _require(rotor["index"] == index, "rotors", "Rotor indices are not ordered")
            _require(
                rotor["reaction_sign"] == [1, -1, 1, -1][index]
                and all(
                    abs(a - b_) <= 1e-12
                    for a, b_ in zip(rotor["position_m"], expected_positions[index], strict=True)
                ),
                "rotors",
                "Rotor geometry/signs differ from the X-frame convention",
            )
    elif name == "course-rules":
        for minimum, maximum in zip(data["workspace_min_m"], data["workspace_max_m"], strict=True):
            _require(minimum < maximum, "bounds", "Workspace minimum must precede maximum")
        for field in ("horizontal_spacing_m", "gate_center_height_m"):
            _require(0 < data[field][0] <= data[field][1], "bounds", f"Invalid {field}")
    elif name == "training":
        ppo = data["ppo"]
        _require(0 < ppo["gamma"] <= 1, "gamma", "Gamma must be in (0,1]")
        for n_envs in ppo["environment_candidates"]:
            _require(
                n_envs in (1, 2, 4, 8) and ppo["n_steps"] * n_envs % ppo["batch_size"] == 0,
                "rollout",
                "Every rollout must divide into complete minibatches",
            )
        curriculum = data["curriculum"]
        _require(
            math.isclose(
                curriculum["frontier_probability"] + curriculum["previous_probability"], 1
            ),
            "probability",
            "Curriculum probabilities must sum to one",
        )
        _require(
            [stage["index"] for stage in curriculum["stages"]] == list(range(7)),
            "stages",
            "Curriculum stages must be ordered 0 through 6",
        )
    elif name == "replay":
        _require(
            data["transition_end"] >= data["transition_start"], "order", "Reversed transitions"
        )
        _require(
            data["policy_revision_end"] >= data["policy_revision_start"],
            "order",
            "Reversed policy revisions",
        )
        total = 0
        for filename, entry in data["files"].items():
            total += entry["byte_length"]
            if entry["dtype"] != "json":
                width = {"float64": 8, "float32": 4, "uint32": 4}[entry["dtype"]]
                _require(
                    math.prod(entry["shape"]) * width == entry["byte_length"],
                    "byte_length",
                    f"Inconsistent array size for {filename}",
                )
        _require(total <= 64 * 1024 * 1024, "size", "Replay exceeds uncompressed size budget")
    elif name == "worker-message":
        if data["type"] == "LOAD_COURSE":
            _semantics("course", data["course"])
        elif data["type"] == "SNAPSHOT":
            _state_semantics(data["state"])
            _state_semantics(data["previous_state"])


class ContractRegistry:
    """All schema references resolve from a supplied local directory; never the network."""

    def __init__(self, schema_directory: Path) -> None:
        self.schemas = {
            path.name.removesuffix(".schema.json"): parse_json(path.read_bytes())
            for path in sorted(schema_directory.glob("*.schema.json"))
        }
        if not self.schemas:
            raise ContractError("schemas", f"No schemas found in {schema_directory}")
        registry: Registry[Any] = Registry()
        for schema in self.schemas.values():
            Draft202012Validator.check_schema(schema)
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
        self.validators = {
            name: Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
            for name, schema in self.schemas.items()
        }

    def validate(self, name: str, data: Any) -> Any:
        if name not in self.validators:
            raise ContractError("schema", f"Unknown schema: {name}")
        _check_numbers(data)
        errors = list(self.validators[name].iter_errors(data))
        if errors:
            error = errors[0]
            location = "/" + "/".join(str(part) for part in error.absolute_path)
            raise ContractError("schema", f"{name}{location}: {error.message}")
        validated = copy.deepcopy(data)
        if name == "config":
            for kind in CONFIG_NAMES:
                if self.validators[kind].is_valid(validated):
                    _semantics(kind, validated)
                    break
        else:
            _semantics(name, validated)
        return validated

    def load_bytes(
        self, name: str, raw: bytes, expected_sha256: str | None = None
    ) -> LoadedArtifact:
        _require(len(raw) <= MAX_JSON_BYTES, "size", "JSON artifact exceeds 16 MiB")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ContractError("hash", "Artifact SHA-256 does not match expected bytes")
        return LoadedArtifact(self.validate(name, parse_json(raw)), digest, len(raw))

    def load(self, name: str, path: Path, expected_sha256: str | None = None) -> LoadedArtifact:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ContractError("size", "JSON artifact exceeds 16 MiB")
        return self.load_bytes(name, path.read_bytes(), expected_sha256)


def load_config_bundle(shared_directory: Path) -> dict[str, LoadedArtifact]:
    registry = ContractRegistry(shared_directory / "schemas")
    bundle = {
        name: registry.load(name, shared_directory / f"{name}.v1.json") for name in CONFIG_NAMES
    }
    validate_config_bundle(bundle)
    return bundle


def validate_config_bundle(bundle: dict[str, LoadedArtifact]) -> None:
    """Reject independently valid configurations that do not describe the same simulator."""
    vehicle = bundle["vehicle"].data
    observation = bundle["observation"].data
    rules = bundle["course-rules"].data
    _require(
        all(
            field["scale"] == vehicle["rotor_max_thrust_n"]
            for field in observation["fields"][26:30]
        ),
        "compatibility",
        "Motor observation scaling differs from vehicle capacity",
    )
    clearance = vehicle["collision_radius_m"] + vehicle["gate_pass_margin_m"]
    _require(
        min(rules["gate"]["width_m"], rules["gate"]["height_m"]) / 2 > clearance,
        "compatibility",
        "Drone cannot fit through the gate's pass aperture",
    )
    for timeout in rules["timeouts_s"].values():
        ticks = timeout * vehicle["physics_hz"]
        _require(ticks > 0 and ticks % 2 == 0, "compatibility", "Timeout must end on a policy tick")


def verify_manifest_configs(manifest: dict[str, Any], bundle: dict[str, LoadedArtifact]) -> None:
    """Verify configuration references; this is not a model-quality or actor-file check."""
    for field, name in {
        "vehicle_sha256": "vehicle",
        "observation_sha256": "observation",
        "rules_sha256": "course-rules",
        "training_config_sha256": "training",
    }.items():
        _require(manifest.get(field) == bundle[name].sha256, "compatibility", f"Mismatch: {field}")
