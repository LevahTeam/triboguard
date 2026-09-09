"""Chart where mechanism inference is safe, where it abstains, and where it misleads.

The map's purpose is not to show the method working. It is to find the designs
where the method reports a single mechanism and is wrong, because those are the
only failures that cost a researcher anything they cannot get back.

Each cell calibrates on its own design and is evaluated on held-out experiments
of that same design. Pooling calibration across designs was measured at 53%
coverage against a 90% target, so a map built that way would chart the pooling
bug instead of the method.
"""

from __future__ import annotations

import json
from pathlib import Path

from triboguard import boundary

# Well counts spanning what a student laboratory runs (3) to what a screening
# facility runs (48). Effects run from "smaller than the classifier's own
# threshold" upward, so the map includes the regime where there is genuinely
# nothing to find.
WELLS = (3, 6, 12, 24, 48)
EFFECTS = (0.10, 0.25, 0.40, 0.70, 1.20)
TIMES = (0.0, 12.0, 24.0, 36.0, 48.0)


def main() -> None:
    print(f"sweeping {len(WELLS)} well counts x {len(EFFECTS)} effect sizes", flush=True)
    report = boundary.sweep(
        wells=WELLS,
        effects=EFFECTS,
        times=TIMES,
        calibration_experiments=60,
        test_experiments=80,
        confidence=0.9,
        resamples=100,
        seed=1,
        progress=True,
    )
    out = Path("runs/triboguard/failure_boundary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    from tribovision import provenance

    report["environment"] = provenance.environment()
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n  regime map (rows: wells, columns: effect size)")
    header = "  wells " + "".join(f"{effect:>16.2f}" for effect in EFFECTS)
    print(header)
    for well_count in WELLS:
        row = f"  {well_count:5d} "
        for effect in EFFECTS:
            cell = next(
                c for c in report["cells"] if c["wells"] == well_count and c["effect"] == effect
            )
            row += f"{cell['regime']:>16s}"
        print(row)
    nulls = report["summary"]["null_behaviour"]
    if nulls:
        print("\n  when there is nothing to find (truth is 'no effect' everywhere)")
        print(
            f"    {'wells':>5s}  {'commits on':>10s}  {'wrong then':>10s}  {'spurious claims':>15s}"
        )
        for row in sorted(nulls, key=lambda r: r["wells"]):
            wrong = row["wrong_when_it_commits"]
            shown = "n/a" if wrong is None else f"{wrong:.2f}"
            print(
                f"    {row['wells']:5d}  {row['commits_on']:10.2f}  {shown:>10s}  "
                f"{row['spurious_mechanism_rate']:14.1%}"
            )
        print("    NOTE: the smallest effect is a real 10% rate change scored as 'no effect'")
        print("    under a 15% rule, so a precise experiment is penalised for resolving it.")

    summary = {k: v for k, v in report["summary"].items() if k != "null_behaviour"}
    print(f"\n  {json.dumps(summary, indent=2)}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
