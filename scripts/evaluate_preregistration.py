"""Score the damage-blindness runs against the plan written before they existed.

    python scripts/evaluate_preregistration.py

Reads every per-cell-line artifact the sweep wrote, refuses any that was run
under a different version of the plan, and applies the plan's criteria and
verdict rule without any further choice. The verdict is whatever the plan says
it is -- including "not replicated", which is a result and gets reported as one.
"""

from __future__ import annotations

import json
from pathlib import Path

from tribovision import preregistration, provenance

PLAN = Path("preregistration/damage_blindness_v1.json")
ARTIFACTS = Path("runs/damage")
OUT = Path("runs/damage/preregistered_outcome.json")


def main() -> None:
    plan = preregistration.load_plan(PLAN)
    paths = sorted(ARTIFACTS.glob("blindness_combined_*.json"))
    summary, sources = preregistration.merge_artifacts(paths, PLAN)
    outcome = preregistration.evaluate(plan, summary)
    outcome["plan"] = preregistration.plan_fingerprint(PLAN)
    outcome["artifacts"] = sources
    outcome["environment"] = provenance.environment()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Plan {plan['id']}  verdict: {outcome['verdict'].upper()}\n")
    replication = outcome["criteria"]["replication"]
    control = outcome["criteria"]["negative_control"]
    primary = plan["primary_endpoint"]["primary_segmenter"]
    for group, lines in (
        ("confirmatory", plan["confirmatory_cell_lines"]),
        ("exploratory", plan["exploratory_cell_lines"]),
    ):
        print(f"  {group}")
        for line in lines:
            interval = replication[line]["interval"]
            shown = (
                "no result"
                if interval is None
                else f"gap {interval['difference']:+.3f} "
                f"[{interval['ci_low']:+.3f}, {interval['ci_high']:+.3f}]"
            )
            print(
                f"    {line:7s} {outcome[group][line]:8s} {shown}   "
                f"control {control[line][primary]['status']}"
            )
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
