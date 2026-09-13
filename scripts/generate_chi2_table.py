"""Regenerate the browser's chi-squared table from scipy.

    python scripts/generate_chi2_table.py            # write web/lib/chi2.ts
    python scripts/generate_chi2_table.py --check    # confirm it still matches scipy

The website computes the same intervals as the Python package, and the only
hard part in a browser is the chi-squared quantile. Shipping a table keeps the
two exactly equal; approximating it in TypeScript would make them nearly equal,
which is the kind of difference that shows up as an unexplained discrepancy in a
figure months later.

``--check`` compares numbers, not text. The first check diffed the generated
file as text and failed on GitHub's Linux runner the first time it ran: scipy
there differs from macOS in about the 15th significant digit. That is rounding
noise between machines, not drift, and a check that can only pass on the
machine that wrote the file checks nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from scipy import stats

DEGREES = 400
OUT = Path("web/lib/chi2.ts")
LEVELS = (("C95", 0.025, 0.975), ("C90", 0.05, 0.95))

#: How far a committed value may sit from a fresh one and still be the same
#: value. Machines disagree near 1e-15; any real drift is orders of magnitude
#: larger. The browser's parity test holds itself to the same bar.
RELATIVE_TOLERANCE = 1e-12


def tables() -> dict[str, list[float]]:
    """Every quantile the browser ships, keyed by the constant it is exported as."""
    values: dict[str, list[float]] = {}
    for name, lo, hi in LEVELS:
        for suffix, quantile in (("LO", lo), ("HI", hi)):
            values[f"{name}_{suffix}"] = [
                float(stats.chi2.ppf(quantile, degree)) for degree in range(1, DEGREES + 1)
            ]
    return values


def _typescript_array(name: str, values: list[str]) -> list[str]:
    """Render the same stable wrapping that the TypeScript formatter uses."""
    lines = [f"export const {name}: readonly number[] = ["]
    for start in range(0, len(values), 4):
        lines.append("  " + ", ".join(values[start : start + 4]) + ",")
    lines.append("];")
    return lines


def render(values: dict[str, list[float]]) -> str:
    """The TypeScript module for a set of tables."""
    lines = [
        "// Chi-squared quantiles for 1..400 degrees of freedom, generated from scipy",
        "// by scripts/generate_chi2_table.py. Tabulated rather than approximated so the",
        "// browser agrees with the Python package exactly instead of nearly.",
        "//",
        "// DO NOT EDIT BY HAND. Regenerate and re-run the parity test.",
        "",
    ]
    for name, numbers in values.items():
        lines.extend(_typescript_array(name, [f"{value:.15g}" for value in numbers]))
    return "\n".join(lines) + "\n"


def parse(text: str) -> dict[str, list[float]]:
    """The arrays a generated module exports, read back as numbers."""
    found: dict[str, list[float]] = {}
    pattern = r"export const (\w+): readonly number\[\] = \[(.*?)\];"
    for name, body in re.findall(pattern, text, re.S):
        found[name] = [float(token) for token in body.split(",") if token.strip()]
    return found


def differences(
    committed: dict[str, list[float]],
    fresh: dict[str, list[float]],
    tolerance: float = RELATIVE_TOLERANCE,
) -> list[str]:
    """Every place the committed tables disagree with fresh ones beyond the tolerance."""
    if sorted(committed) != sorted(fresh):
        return [f"arrays differ: committed {sorted(committed)}, fresh {sorted(fresh)}"]
    problems: list[str] = []
    for name, numbers in fresh.items():
        if len(committed[name]) != len(numbers):
            problems.append(
                f"{name}: {len(committed[name])} values committed, {len(numbers)} fresh"
            )
            continue
        for index, (old, new) in enumerate(zip(committed[name], numbers, strict=True)):
            if abs(old - new) > tolerance * max(abs(old), abs(new)):
                problems.append(f"{name}[{index}]: committed {old!r}, fresh {new!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    fresh = tables()
    if "--check" in arguments:
        if not OUT.is_file():
            print(f"{OUT} is missing; run without --check to generate it.", file=sys.stderr)
            return 1
        problems = differences(parse(OUT.read_text(encoding="utf-8")), fresh)
        if problems:
            print(f"{OUT} has drifted from scipy:", file=sys.stderr)
            for problem in problems[:10]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        count = sum(len(numbers) for numbers in fresh.values())
        print(f"{OUT} matches scipy: {count} values within a relative {RELATIVE_TOLERANCE:g}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(fresh), encoding="utf-8")
    print(f"WROTE {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
