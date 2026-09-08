"""Usage: python -m training.scripts.check_contracts --shared shared"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.contracts import ContractError, ContractRegistry, load_config_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate shared contracts and report byte hashes")
    parser.add_argument(
        "--shared", type=Path, default=Path(__file__).resolve().parents[2] / "shared"
    )
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    try:
        bundle = load_config_bundle(args.shared)
        result: dict[str, object] = {
            name: {"sha256": value.sha256, "byte_length": value.byte_length, "data": value.data}
            for name, value in bundle.items()
        }
        if args.fixtures:
            registry = ContractRegistry(args.shared / "schemas")
            cases = json.loads(args.fixtures.read_text(encoding="utf-8"))
            outcomes = []
            for case in cases:
                try:
                    registry.validate(case["schema"], case["data"])
                    outcomes.append({"id": case["id"], "accepted": True})
                except ContractError:
                    outcomes.append({"id": case["id"], "accepted": False})
            result["fixtures"] = outcomes
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ContractError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
