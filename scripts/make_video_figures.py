"""Draw the three figures shown in the video and the README.

    python scripts/make_video_figures.py               # all three, into media/
    python scripts/make_video_figures.py --chart-only  # the chart alone; needs only results/

Frames 1 and 2 are one real MCF7 image from the pre-registered test, damaged and
segmented by exactly the sweep's recipe. A picture is the easiest thing in a
project to stage, so the re-run is checked against the detection rates the sweep
recorded for that image, and nothing is written if the two disagree. The chart
is read straight from the published artifacts.

The frames need matplotlib and the optional Cellpose extra; the chart needs only
matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("results")
OUT = Path("media")

#: The image in frames 1 and 2. MCF7 is a confirmatory line, and this field is
#: mid-sized (309 cells), which is what reads at video resolution. Its rates sit
#: inside the line's range rather than at an extreme: at this severity another
#: MCF7 image finds every healthy cell and none of 73 damaged ones.
LINE, IMAGE_ID, SEVERITY, COVERAGE = "MCF7", "526733", 0.6, 0.5

#: Every line in the sweep, in the order the chart shows them.
LINES = (
    ("MCF7", "breast cancer", True),
    ("SHSY5Y", "neuroblastoma", True),
    ("SkBr3", "breast cancer", True),
    ("A172", "glioblastoma", False),
)
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8)

# The demo's palette, so the frames and the page read as one piece of work.
GROUND, INK, MUTED, GRID = "#eceeeb", "#0f1213", "#5a6360", "#d3d8d5"
FOUND, MISSED, DAMAGED = (18, 163, 109), (224, 54, 79), (240, 160, 32)
FONTS = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def detection_curve(
    summary: dict[str, Any],
    line: str,
    severities: tuple[float, ...],
    segmenter: str = "cellpose",
    coverage: float = COVERAGE,
) -> tuple[list[float], list[float], list[int]]:
    """Mean healthy and damaged detection rates at each severity, and the image
    count behind each. A severity missing from the artifact is a KeyError, not a
    gap in the line: a chart that quietly skips a point has changed its data."""
    entries = [summary[f"{line}/{s:.1f}/{segmenter}/cov{coverage:.1f}"] for s in severities]
    return (
        [entry["mean_intact_rate"] for entry in entries],
        [entry["mean_damaged_rate"] for entry in entries],
        [entry["images"] for entry in entries],
    )


def representative_window(
    centres: dict[int, tuple[float, float]],
    first: set[int],
    second: set[int],
    shape: tuple[int, int],
    size: int,
    step: int = 10,
) -> tuple[int, int]:
    """Top-left corner of the square window holding the most of the scarcer kind.

    The zoomed crop is chosen by this rule rather than by eye, so it cannot be
    picked for looking dramatic: it is the window where both kinds of cell are
    best represented. Ties keep the first window found, scanning top to bottom.
    """
    best = (-1, 0, 0)
    for y in range(0, shape[0] - size + 1, step):
        for x in range(0, shape[1] - size + 1, step):
            inside = {
                cell
                for cell, (cy, cx) in centres.items()
                if y <= cy < y + size and x <= cx < x + size
            }
            score = min(len(inside & first), len(inside & second))
            if score > best[0]:
                best = (score, y, x)
    return best[1], best[2]


def agrees_with_record(
    intact_found: int,
    intact: int,
    damaged_found: int,
    damaged: int,
    row: dict[str, Any],
    coverage: float = COVERAGE,
) -> bool:
    """Whether a re-run reproduces the rates the sweep recorded for the same image."""
    key = f"{coverage:.1f}"
    return (
        abs(intact_found / intact - row[f"intact@{key}"]) <= 1e-9
        and abs(damaged_found / damaged - row[f"damaged@{key}"]) <= 1e-9
    )


def _outline(image: np.ndarray, labels: np.ndarray, colour_of: Any, width: int) -> np.ndarray:
    """Draw each labelled cell's boundary onto a greyscale image, coloured per cell."""
    from scipy import ndimage

    rgb = np.stack([image] * 3, axis=-1).copy()
    edge = (
        ndimage.maximum_filter(labels, size=width) != ndimage.minimum_filter(labels, size=width)
    ) & (labels > 0)
    for cell in np.unique(labels[edge]):
        colour = colour_of(int(cell))
        if colour is not None:
            rgb[edge & (labels == cell)] = colour
    return rgb


def draw_chart(out: Path) -> None:
    """Frame 3: detection rate against damage severity, one panel per line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = FONTS
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=GROUND)
    counts: set[int] = set()
    for k, (line, kind, confirmatory) in enumerate(LINES):
        payload = json.loads(
            (RESULTS / f"damage__blindness_combined_{line}.json").read_text(encoding="utf-8")
        )
        healthy, hurt, images = detection_curve(payload["summary"], line, SEVERITIES)
        counts.update(images)
        ax = fig.add_axes((0.04 + k * 0.237, 0.2, 0.2, 0.52), facecolor=GROUND)
        ax.axvline(SEVERITY, color=INK, linewidth=1, linestyle=":", zorder=1)
        for rates, colour in ((healthy, FOUND), (hurt, MISSED)):
            ax.plot(
                SEVERITIES,
                [100 * rate for rate in rates],
                "-o",
                color=_hex(colour),
                linewidth=3.5,
                markersize=9,
                zorder=3,
            )
        ax.set_ylim(-4, 104)
        ax.set_xlim(-0.05, 0.85)
        ax.set_xticks(SEVERITIES)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"] if k == 0 else [])
        ax.tick_params(colors=INK, labelsize=14, length=0)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.set_title(line, fontsize=24, fontweight="bold", color=INK, loc="left", pad=34)
        role = "pre-registered test" if confirmatory else "exploratory (seen first)"
        ax.text(
            0,
            1.035,
            f"{kind} · {role}",
            transform=ax.transAxes,
            fontsize=14,
            color=MUTED,
            style="normal" if confirmatory else "italic",
        )
        # Right of the line and between the curves: clear air on every panel.
        ax.text(
            SEVERITY + 0.02, 30, "pre-registered\nseverity", fontsize=11.5, color=MUTED, va="center"
        )
        ax.set_xlabel("damage severity", fontsize=15, color=INK, labelpad=8)
    (count,) = counts  # the subtitle states one number, so there must be one
    fig.text(
        0.04,
        0.9,
        "More damage, fewer cells seen: on every cell line",
        fontsize=36,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.85,
        "Share of cells Cellpose finds, healthy vs damaged, as the damage gets "
        f"stronger. Each point is the mean over {count} real images.",
        fontsize=18,
        color=MUTED,
    )
    fig.text(
        0.04, 0.085, "●  healthy cells found", fontsize=22, fontweight="bold", color=_hex(FOUND)
    )
    fig.text(
        0.25, 0.085, "●  damaged cells found", fontsize=22, fontweight="bold", color=_hex(MISSED)
    )
    fig.text(
        0.04,
        0.03,
        "At severity 0 nothing is damaged, so both groups should match: that is "
        "the built-in control. Damage is simulated on real LIVECell images.",
        fontsize=14.5,
        color=MUTED,
    )
    fig.savefig(out, facecolor=GROUND)
    plt.close(fig)


def draw_frames(out_dir: Path) -> None:
    """Frames 1 and 2: one real image, re-run and checked against the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from scipy import ndimage

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_damage_blindness as sweep

    from tribovision import damage, external
    from tribovision.manifest import load_manifest

    records = load_manifest(sweep.MANIFESTS[LINE])[: sweep.IMAGES]
    position, record = next((i, r) for i, r in enumerate(records) if str(r.image_id) == IMAGE_ID)
    truth = sweep._labels_for(record, {})

    # The sweep's recipe for this image position, step for step.
    rng = np.random.default_rng(1000 + position)
    damaged, intact = damage.split_cells(truth, fraction=0.5, rng=rng)
    with Image.open(record.image_path) as handle:
        grey = np.asarray(handle.convert("L"), dtype=np.float32)
    degraded = damage.apply_damage(
        grey, truth, damaged, damage.Damage.at(SEVERITY), rng=rng
    ).astype(np.uint8)
    predicted = external.cellpose_instances(degraded, diameter=sweep.CELLPOSE_DIAMETER)
    shares = damage.detection_coverage(predicted, truth)
    found = {cell for cell, share in shares.items() if share >= COVERAGE}
    intact_found, damaged_found = len(found & intact), len(found & damaged)

    payload = json.loads(
        (RESULTS / f"damage__blindness_combined_{LINE}.json").read_text(encoding="utf-8")
    )
    row = next(
        r
        for r in payload["per_image"]
        if r["segmenter"] == "cellpose"
        and abs(r["severity"] - SEVERITY) < 1e-9
        and str(r["image_id"]) == IMAGE_ID
    )
    if not agrees_with_record(intact_found, len(intact), damaged_found, len(damaged), row):
        raise SystemExit(
            f"re-run found {intact_found}/{len(intact)} healthy and {damaged_found}/"
            f"{len(damaged)} damaged, which is not what the sweep recorded; not drawing it"
        )

    # Display only: one contrast stretch, shared by both pictures so neither is flattered.
    lo, hi = np.percentile(grey, [0.5, 99.5])

    def stretch(values: np.ndarray) -> np.ndarray:
        scaled = (values.astype(np.float32) - lo) / (hi - lo) * 255
        return np.clip(scaled, 0, 255).astype(np.uint8)

    shown_original, shown_degraded = stretch(grey), stretch(degraded)

    cells = [int(c) for c in np.unique(truth) if c]
    centres = dict(zip(cells, ndimage.center_of_mass(truth > 0, truth, cells), strict=True))
    size, zoom = 190, 4
    y0, x0 = representative_window(centres, intact & found, damaged, truth.shape, size)

    def enlarge(values: np.ndarray) -> np.ndarray:
        crop = Image.fromarray(values[y0 : y0 + size, x0 : x0 + size])
        return np.asarray(crop.resize((size * zoom, size * zoom), Image.LANCZOS))

    crop_labels = np.kron(truth[y0 : y0 + size, x0 : x0 + size], np.ones((zoom, zoom), truth.dtype))
    zoomed = enlarge(shown_degraded)
    panels = [
        (enlarge(shown_original), "1   Real cells, as imaged"),
        (
            _outline(zoomed, crop_labels, lambda c: DAMAGED if c in damaged else None, 5),
            "2   Half damaged by coin flip (outlined)",
        ),
        (
            _outline(zoomed, crop_labels, lambda c: FOUND if c in found else MISSED, 5),
            "3   What the AI (Cellpose) found",
        ),
    ]
    healthy_share, damaged_share = intact_found / len(intact), damaged_found / len(damaged)

    plt.rcParams["font.family"] = FONTS
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=GROUND)
    fig.text(
        0.035,
        0.925,
        "The AI finds the healthy cells and misses the damaged ones",
        fontsize=36,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.875,
        f"One real {LINE} breast-cancer image (LIVECell)  ·  same lighting, "
        "same focus  ·  only the damage differs",
        fontsize=19,
        color=MUTED,
    )
    for k, (image, title) in enumerate(panels):
        ax = fig.add_axes((0.035 + k * 0.315, 0.2, 0.3, 0.6))
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(1.2)
        ax.set_title(title, fontsize=20, fontweight="bold", color=INK, loc="left", pad=10)
    fig.text(
        0.035,
        0.12,
        f"Healthy cells found:  {intact_found} of {len(intact)}  ({healthy_share:.0%})",
        fontsize=26,
        fontweight="bold",
        color=_hex(FOUND),
    )
    fig.text(
        0.52,
        0.12,
        f"Damaged cells found:  {damaged_found} of {len(damaged)}  ({damaged_share:.0%})",
        fontsize=26,
        fontweight="bold",
        color=_hex(MISSED),
    )
    fig.text(
        0.035,
        0.045,
        "Zoomed into one region; counts are for the whole image. Green = found, "
        "red = missed. Cell outlines are the dataset's human annotations.\nDamage is simulated "
        f"(fading, blur, vacuoles at severity {SEVERITY}) on a real photo. Pre-registered "
        "test: the gap replicated on all 3 unseen cell lines.",
        fontsize=14.5,
        color=MUTED,
        linespacing=1.5,
    )
    fig.savefig(out_dir / "1_blind_spot_zoom.png", facecolor=GROUND)
    plt.close(fig)

    field = _outline(shown_degraded, truth, lambda c: FOUND if c in found else MISSED, 3)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=GROUND)
    ax = fig.add_axes((0.02, 0.05, 0.64, 0.9))
    ax.imshow(field)
    ax.add_patch(
        Rectangle((x0, y0), size, size, fill=False, edgecolor=INK, linewidth=2.5, linestyle="--")
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
    fig.text(0.69, 0.84, "The whole image", fontsize=34, fontweight="bold", color=INK)
    fig.text(
        0.69, 0.79, f"{len(cells)} cells, outlined by human annotators", fontsize=19, color=MUTED
    )
    fig.text(0.69, 0.64, f"{healthy_share:.0%}", fontsize=80, fontweight="bold", color=_hex(FOUND))
    fig.text(
        0.69,
        0.595,
        f"of healthy cells found  ({intact_found}/{len(intact)})",
        fontsize=19,
        color=INK,
    )
    fig.text(0.69, 0.44, f"{damaged_share:.0%}", fontsize=80, fontweight="bold", color=_hex(MISSED))
    fig.text(
        0.69,
        0.395,
        f"of damaged cells found  ({damaged_found}/{len(damaged)})",
        fontsize=19,
        color=INK,
    )
    fig.text(
        0.69,
        0.17,
        "A drug that damages cells would look like\nit did nothing, because the "
        "counter cannot\nsee the cells it hurt.\n\nDashed box: the zoomed region.",
        fontsize=17,
        color=MUTED,
        linespacing=1.5,
    )
    fig.savefig(out_dir / "2_blind_spot_whole_image.png", facecolor=GROUND)
    plt.close(fig)
    print(
        f"re-run agrees with the sweep: {intact_found}/{len(intact)} healthy, "
        f"{damaged_found}/{len(damaged)} damaged"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chart-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.chart_only:
        import warnings

        warnings.filterwarnings("ignore")
        draw_frames(args.out)
    draw_chart(args.out / "3_damage_vs_detection.png")
    print(f"WROTE {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
