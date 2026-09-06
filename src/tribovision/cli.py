"""Command-line interface for preparing, training, predicting, and analysing.

Every stage of the research pipeline is reachable from here, including the two
that were previously unreachable: running a trained checkpoint on a new image,
and comparing that checkpoint against the classical baseline on held-out data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tribovision.livecell import CELL_TYPES, PreparationError


def _add_prepare(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-livecell",
        help="Download and prepare a leakage-free LIVECell subset.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/livecell"))
    parser.add_argument("--profile", choices=("demo", "full"), default="demo")
    parser.add_argument("--cell-types", nargs="+", choices=CELL_TYPES, default=["A172"])
    parser.add_argument(
        "--max-images",
        type=int,
        default=60,
        help="Maximum images per official split and cell type (default: 60).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-by",
        choices=("well", "acquisition"),
        default="well",
        help="Unit kept whole when re-splitting train/validation (default: well).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument(
        "--micrometers-per-pixel",
        type=float,
        default=None,
        help="Microscope calibration. Without it, morphology stays in pixels.",
    )
    parser.add_argument("--images-zip", type=Path)
    parser.add_argument("--confirm-full-download", action="store_true")


def _add_verify(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "verify",
        help="Check manifests for split leakage, hash drift, and annotation mismatches.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/livecell"))
    parser.add_argument("--group-by", choices=("well", "acquisition"), default="well")
    parser.add_argument("--skip-hashes", action="store_true")


def _add_baseline(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "baseline", help="Run the transparent non-learned segmentation baseline."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/livecell/manifests/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/classical_baseline"))
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--background-radius", type=float, default=7.0)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument(
        "--instance-method",
        choices=("connected_components", "watershed_split"),
        default="connected_components",
    )
    parser.add_argument("--micrometers-per-pixel", type=float, default=None)


def _add_train(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Train and evaluate the U-Net segmenter.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/livecell"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--group-by", choices=("well", "acquisition"), default="well")
    parser.add_argument("--no-augment", action="store_true")


def _add_predict(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "predict", help="Segment new microscope images with a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predictions"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--instance-method",
        choices=("connected_components", "watershed_split"),
        default="watershed_split",
    )
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--micrometers-per-pixel", type=float, default=None)
    parser.add_argument("--no-overlays", action="store_true")


def _add_compare(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "compare", help="Score the trained model against the classical baseline on held-out data."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/best_model.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/livecell/manifests/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)


def _add_instance_benchmark(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "instance-benchmark",
        help="Score instance separation against the classical rule, Cellpose, and the ceiling.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/livecell/manifests/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/instance_benchmark"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument(
        "--cellpose-diameter",
        type=float,
        default=15.0,
        help="Expected cell diameter in pixels; choose it on the training split.",
    )
    parser.add_argument("--no-cellpose", action="store_true")
    parser.add_argument(
        "--three-class-checkpoint",
        type=Path,
        default=Path("runs/instance_model/best_model.pt"),
        help="Three-class instance model to include in the comparison.",
    )
    parser.add_argument("--interior-threshold", type=float, default=0.7)


def _add_mechanics(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "mechanics",
        help="Measure the tissue shape index q = P/sqrt(A) and the jamming state.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/livecell"))
    parser.add_argument(
        "--cell-types",
        nargs="+",
        choices=CELL_TYPES,
        default=list(CELL_TYPES),
        help="Cell types to measure (default: all available).",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mechanics"))
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument(
        "--time-course",
        action="store_true",
        help="Follow each well's mechanical state over the time-lapse.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Pool train, val and test annotations for more wells.",
    )


def _add_treatment(subparsers: Any) -> None:
    template = subparsers.add_parser(
        "treatment-template",
        help="Write an empty treatment manifest with every required column.",
    )
    template.add_argument("--output", type=Path, default=Path("data/treatment/manifest.csv"))
    template.add_argument(
        "--plan",
        action="store_true",
        help="Emit every row the planned experiment needs, not just a header.",
    )
    template.add_argument("--days", nargs="+", default=["day-1", "day-2", "day-3"])
    template.add_argument(
        "--concentrations", nargs="+", type=float, default=[0.0, 12.5, 25.0, 50.0, 100.0, 200.0]
    )
    template.add_argument(
        "--exposure-hours", nargs="+", type=float, default=[0.0, 6.0, 12.0, 24.0, 48.0]
    )
    template.add_argument("--wells-per-condition", type=int, default=3)
    template.add_argument("--fields", type=int, default=2)
    template.add_argument("--cell-line", default="RAW264.7")

    analyse = subparsers.add_parser(
        "analyze-treatment",
        help="Segment treatment images and run the dose-response and viability analysis.",
    )
    analyse.add_argument("--manifest", type=Path, required=True)
    analyse.add_argument("--output-dir", type=Path, default=Path("runs/treatment"))
    analyse.add_argument("--checkpoint", type=Path, default=None)
    analyse.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    analyse.add_argument("--threshold", type=float, default=0.5)
    analyse.add_argument(
        "--instance-method",
        choices=("connected_components", "watershed_split"),
        default="watershed_split",
    )
    analyse.add_argument("--min-area", type=int, default=20)
    analyse.add_argument(
        "--primary-feature",
        default=None,
        help="Pre-specified confirmatory endpoint (default: a shape feature).",
    )
    analyse.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Permutations for the day-stratified dose-response test.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tribovision",
        description="Prepare LIVECell, train and audit a segmenter, and analyse treatment images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare(subparsers)
    _add_verify(subparsers)
    _add_baseline(subparsers)
    _add_train(subparsers)
    _add_predict(subparsers)
    _add_compare(subparsers)
    _add_instance_benchmark(subparsers)
    _add_mechanics(subparsers)
    _add_treatment(subparsers)
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _run(args: argparse.Namespace) -> int:
    if args.command == "prepare-livecell":
        from tribovision.livecell import download_full, prepare_demo

        if args.profile == "full":
            paths = download_full(args.data_dir, confirmed=args.confirm_full_download)
            _print({key: str(value) for key, value in paths.items()})
            return 0
        result = prepare_demo(
            args.data_dir,
            cell_types=args.cell_types,
            max_images=args.max_images,
            seed=args.seed,
            images_zip=args.images_zip,
            group_by=args.group_by,
            val_fraction=args.val_fraction,
            micrometers_per_pixel=args.micrometers_per_pixel,
        )
        _print(
            {
                "prepared_images": result["counts"],
                "wells_by_split": result["summary"]["wells_by_split"],
                "split_leakage_free": result["split_report"]["clean"],
            }
        )
        return 0

    if args.command == "verify":
        from tribovision.manifest import check_group_leakage, load_manifest

        manifests = Path(args.data_dir) / "manifests"
        splits = {
            split: load_manifest(manifests / f"{split}.jsonl", verify_hashes=not args.skip_hashes)
            for split in ("train", "val", "test")
        }
        report = check_group_leakage(splits, group_by=args.group_by)
        report["images_per_split"] = {name: len(rows) for name, rows in splits.items()}
        report["wells_per_split"] = {
            name: sorted({record.well for record in rows}) for name, rows in splits.items()
        }
        report["hashes_verified"] = not args.skip_hashes
        _print(report)
        return 0 if report["clean"] else 1

    if args.command == "baseline":
        from tribovision.baseline import run_baseline

        report = run_baseline(
            args.manifest,
            args.output_dir,
            max_images=args.max_images,
            background_radius=args.background_radius,
            min_area=args.min_area,
            instance_method=args.instance_method,
            micrometers_per_pixel=args.micrometers_per_pixel,
        )
        _print({key: value for key, value in report.items() if key != "environment"})
        return 0

    if args.command == "train":
        from tribovision.training import TrainConfig, train

        result = train(
            TrainConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                image_size=args.image_size,
                base_channels=args.base_channels,
                depth=args.depth,
                seed=args.seed,
                workers=args.workers,
                patience=args.patience,
                device=args.device,
                group_by=args.group_by,
                augment=not args.no_augment,
            )
        )
        _print({key: value for key, value in result.items() if key not in ("history",)})
        return 0

    if args.command == "predict":
        from tribovision.predict import run_prediction

        report = run_prediction(
            args.checkpoint,
            args.inputs,
            args.output_dir,
            device=args.device,
            threshold=args.threshold,
            image_size=args.image_size,
            instance_method=args.instance_method,
            min_area=args.min_area,
            micrometers_per_pixel=args.micrometers_per_pixel,
            save_overlays=not args.no_overlays,
        )
        _print({key: value for key, value in report.items() if key != "environment"})
        return 0

    if args.command == "compare":
        from tribovision.benchmark import compare

        report = compare(
            args.checkpoint,
            args.manifest,
            args.output_dir,
            device=args.device,
            threshold=args.threshold,
            max_images=args.max_images,
        )
        _print(
            {key: value for key, value in report.items() if key not in ("per_image", "environment")}
        )
        return 0 if report["passed"] else 2

    if args.command == "instance-benchmark":
        from tribovision.instance_benchmark import run as run_instance_benchmark

        checkpoint = Path(args.checkpoint)
        report = run_instance_benchmark(
            args.manifest,
            args.output_dir,
            checkpoint=checkpoint if checkpoint.is_file() else None,
            device=args.device,
            max_images=args.max_images,
            min_area=args.min_area,
            cellpose_diameter=args.cellpose_diameter,
            include_cellpose=not args.no_cellpose,
            three_class_checkpoint=args.three_class_checkpoint,
            interior_threshold=args.interior_threshold,
        )
        _print(
            {key: value for key, value in report.items() if key not in ("per_image", "environment")}
        )
        return 0

    if args.command == "mechanics":
        import json as _json

        from tribovision import mechanics

        results = {}
        for cell_type in args.cell_types:
            path = Path(args.data_dir) / "annotations" / cell_type.casefold() / f"{args.split}.json"
            if not path.is_file():
                continue
            measured = mechanics.measure_annotation_file(
                path, cell_type=cell_type, min_area_pixels=args.min_area
            )
            measured["density"] = mechanics.density_relationship(measured["images"])
            measured["interpretation"] = mechanics.interpret(measured["summary"])
            if args.time_course:
                wanted = ("train", "val", "test") if args.all_splits else (args.split,)
                sources = [
                    Path(args.data_dir) / "annotations" / cell_type.casefold() / f"{name}.json"
                    for name in wanted
                ]
                measured["time_course"] = mechanics.time_course(
                    [source for source in sources if source.is_file()],
                    cell_type=cell_type,
                    min_area_pixels=args.min_area,
                )
            results[cell_type] = measured
        if not results:
            raise ValueError(
                f"No {args.split} annotations found under {args.data_dir}. "
                "Run prepare-livecell, or download the annotation files first."
            )
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "split": args.split,
            "jamming_threshold": mechanics.JAMMING_THRESHOLD,
            "circle_shape_index": mechanics.CIRCLE_SHAPE_INDEX,
            "hexagon_shape_index": mechanics.HEXAGON_SHAPE_INDEX,
            "cell_types": results,
            "environment": __import__(
                "tribovision.provenance", fromlist=["environment"]
            ).environment(),
        }
        (output_dir / "mechanics.json").write_text(
            _json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _print(
            {
                name: {
                    **value["summary"],
                    **(
                        {"time_course": value["time_course"]["verdict"]}
                        if "time_course" in value
                        else {}
                    ),
                }
                for name, value in results.items()
            }
        )
        return 0

    if args.command == "treatment-template":
        from tribovision.treatment import plan_experiment, write_template

        if args.plan:
            summary = plan_experiment(
                args.output,
                days=tuple(args.days),
                concentrations=tuple(args.concentrations),
                exposure_hours=tuple(args.exposure_hours),
                wells_per_condition=args.wells_per_condition,
                fields=args.fields,
                cell_line=args.cell_line,
            )
            _print(summary)
            return 0 if all(summary["requirements_met"].values()) else 1
        path = write_template(args.output)
        _print(
            {
                "template": str(path),
                "next_step": "fill one row per image, then run analyze-treatment",
                "hint": "add --plan to emit every row the experiment needs",
            }
        )
        return 0

    if args.command == "analyze-treatment":
        from tribovision.treatment import run_treatment_analysis

        report = run_treatment_analysis(
            args.manifest,
            args.output_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            instance_method=args.instance_method,
            min_area=args.min_area,
            permutations=args.permutations,
            **({"primary_feature": args.primary_feature} if args.primary_feature else {}),
        )
        _print(
            {
                key: value
                for key, value in report.items()
                if key not in ("well_level_rows", "environment")
            }
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    from tribovision.data import DatasetError
    from tribovision.manifest import ManifestError
    from tribovision.predict import PredictionError
    from tribovision.training import ConfigError, DivergenceError
    from tribovision.treatment import TreatmentDataError

    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (
        PreparationError,
        ManifestError,
        DatasetError,
        ConfigError,
        DivergenceError,
        PredictionError,
        TreatmentDataError,
    ) as exc:
        print(f"tribovision {args.command}: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"tribovision {args.command}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
