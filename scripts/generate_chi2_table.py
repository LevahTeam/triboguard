"""Regenerate the browser's chi-squared table from scipy.

The website computes the same intervals as the Python package, and the only
hard part in a browser is the chi-squared quantile. Shipping a table keeps the
two exactly equal; approximating it in TypeScript would make them nearly equal,
which is the kind of difference that shows up as an unexplained discrepancy in a
figure months later.
"""

from __future__ import annotations

from pathlib import Path

from scipy import stats

DEGREES = 400
OUT = Path("web/lib/chi2.ts")
LEVELS = (("C95", 0.025, 0.975), ("C90", 0.05, 0.95))


def _typescript_array(name: str, values: list[str]) -> list[str]:
    """Render the same stable wrapping that the TypeScript formatter uses."""
    lines = [f"export const {name}: readonly number[] = ["]
    for start in range(0, len(values), 4):
        lines.append("  " + ", ".join(values[start : start + 4]) + ",")
    lines.append("];")
    return lines


def main() -> None:
    lines = [
        "// Chi-squared quantiles for 1..400 degrees of freedom, generated from scipy",
        "// by scripts/generate_chi2_table.py. Tabulated rather than approximated so the",
        "// browser agrees with the Python package exactly instead of nearly.",
        "//",
        "// DO NOT EDIT BY HAND. Regenerate and re-run the parity test.",
        "",
    ]
    for name, lo, hi in LEVELS:
        for suffix, quantile in (("LO", lo), ("HI", hi)):
            values = [
                f"{stats.chi2.ppf(quantile, degree):.15g}" for degree in range(1, DEGREES + 1)
            ]
            lines.extend(_typescript_array(f"{name}_{suffix}", values))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
