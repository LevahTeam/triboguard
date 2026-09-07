"""Why a shape-index correlation is low: bad segmentation, or nothing to track?

A rank correlation between true and recovered shape index confounds two things.
One is how accurately the segmenter measures q on a single image. The other is
how much q actually varies between the images being ranked. When the second is
small the correlation collapses even for a near-perfect segmenter, so a low
number is not by itself evidence that the method failed.

This separates them. The measurement error comes from each method's limits of
agreement, which are already recorded: LoA spans 3.92 standard deviations, so
sd_error = (upper - lower) / 3.92. The biological variation comes from the
polygon annotations alone, with no segmenter involved. Their ratio is a
signal-to-noise ratio, and the classical attenuation formula turns it into the
correlation you would expect to see:

    rho_expected = 1 / sqrt(1 + (sd_error / sd_biological)^2)

That formula is derived for Pearson correlation of independent errors, and
neither condition holds exactly here -- the error grows with q rather than being
independent of it. It is used as a *direction* check, not a fit: the question is
whether the lines that fail are the lines with nothing to track.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from tribovision import provenance  # noqa: E402

#: Limits of agreement span mean +/- 1.96 sd, so the full width is 3.92 sd.
LOA_SPAN_IN_SD = 3.92


def _sd_error(entry: dict[str, Any]) -> float | None:
    loa = entry.get("limits_of_agreement")
    if not loa or len(loa) != 2:
        return None
    return (float(loa[1]) - float(loa[0])) / LOA_SPAN_IN_SD


def _expected_rho(sd_error: float, sd_biological: float) -> float:
    return 1.0 / math.sqrt(1.0 + (sd_error / sd_biological) ** 2)


def main() -> None:
    spread = json.loads(Path("runs/mechanics/q_dynamic_range.json").read_text())["per_line"]
    origin = json.loads(Path("runs/mechanics/segmenter_agreement.json").read_text())
    across_path = Path("runs/mechanics/across_cell_lines.json")
    if not across_path.is_file():
        print("runs/mechanics/across_cell_lines.json not written yet", file=sys.stderr)
        raise SystemExit(1)
    across = json.loads(across_path.read_text())

    # A172 uses the original artifact's key names; the other lines use the
    # cross-line runner's. Same three methods either way.
    sources: dict[str, dict[str, dict[str, Any]]] = {
        "A172": {
            "ground truth (raster)": origin["truth_raster"],
            "Cellpose": origin["cellpose"],
            "three-class": origin["three_class_304"],
        }
    }
    for key in ("mcf7", "shsy5y", "skbr3"):
        row = across.get(key)
        if not row:
            continue
        sources[row["cell_line"]] = {
            "ground truth (raster)": row["ground_truth_raster"],
            "Cellpose": row["cellpose"],
            "three-class": row["three_class"],
        }

    rows: list[dict[str, Any]] = []
    for line, methods in sources.items():
        sd_biological = float(spread[line]["sd_between_images"])
        for method, entry in methods.items():
            sd_error = _sd_error(entry)
            observed = entry.get("spearman_rho")
            if sd_error is None or observed is None:
                continue
            rows.append(
                {
                    "cell_line": line,
                    "method": method,
                    "sd_biological": sd_biological,
                    "sd_error": sd_error,
                    "signal_to_noise": sd_biological / sd_error,
                    "observed_rho": float(observed),
                    "expected_rho": _expected_rho(sd_error, sd_biological),
                }
            )

    rows.sort(key=lambda r: r["signal_to_noise"])

    # Which of the three candidates actually predicts the correlation? The first
    # prediction this project made here was that biological spread alone would,
    # and SkBr3 falsified it. Reporting all three rank correlations keeps the
    # failed candidates visible instead of quietly dropping them.
    from scipy import stats

    observed = [row["observed_rho"] for row in rows]
    predicts = {
        name: {
            "spearman_rho": float(stats.spearmanr(values, observed).statistic),
            "p_value": float(stats.spearmanr(values, observed).pvalue),
        }
        for name, values in (
            ("signal_to_noise", [row["signal_to_noise"] for row in rows]),
            ("biological_spread_alone", [row["sd_biological"] for row in rows]),
            ("measurement_error_alone", [row["sd_error"] for row in rows]),
        )
    }
    report = {
        "rows": rows,
        "what_predicts_the_correlation": predicts,
        "note": (
            "expected_rho uses the classical attenuation formula, which assumes "
            "Pearson correlation and errors independent of the true value. Neither "
            "holds exactly here, so it is read as a direction check rather than a fit."
        ),
        "environment": provenance.environment(),
    }
    out = Path("runs/mechanics/attenuation.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"{'line':8s} {'method':22s} {'bio sd':>7s} {'err sd':>7s} "
        f"{'SNR':>6s} {'rho':>6s} {'exp':>6s}"
    )
    for r in rows:
        print(
            f"{r['cell_line']:8s} {r['method']:22s} {r['sd_biological']:7.4f} "
            f"{r['sd_error']:7.4f} {r['signal_to_noise']:6.2f} "
            f"{r['observed_rho']:6.3f} {r['expected_rho']:6.3f}"
        )
    print()
    for name, entry in predicts.items():
        print(
            f"Spearman({name:24s}, observed rho) = "
            f"{entry['spearman_rho']:+.3f}  p={entry['p_value']:.4f}"
        )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
