"""CLI surface: every command reachable, every error translated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_synthetic_experiment

from tribovision.cli import build_parser, main

SMALL = ["--image-size", "32", "--base-channels", "4", "--depth", "2", "--device", "cpu"]


def run(argv: list[str], capsys) -> tuple[int, dict]:
    code = main(argv)
    captured = capsys.readouterr()
    payload: dict = {}
    if captured.out.strip():
        try:
            payload = json.loads(captured.out)
        except json.JSONDecodeError:
            payload = {"raw": captured.out}
    return code, payload


def test_every_command_is_registered() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions and set(actions[0].choices) == {
        "prepare-livecell",
        "verify",
        "baseline",
        "train",
        "predict",
        "compare",
        "treatment-template",
        "analyze-treatment",
    }


def test_a_predict_command_exists_at_all() -> None:
    """There was no way to run a trained checkpoint on a new image."""
    assert (
        "predict"
        in build_parser().parse_args(["predict", "--checkpoint", "x.pt", "img.tif"]).command
    )


def test_verify_reports_clean_splits(tiny_training_data: Path, capsys) -> None:
    code, payload = run(["verify", "--data-dir", str(tiny_training_data)], capsys)
    assert code == 0
    assert payload["clean"] is True
    assert payload["hashes_verified"] is True


def test_verify_fails_loudly_on_leaking_splits(leaking_training_data: Path, capsys) -> None:
    code, payload = run(["verify", "--data-dir", str(leaking_training_data)], capsys)
    assert code == 1
    assert payload["clean"] is False
    assert "train|val" in payload["overlaps"]


def test_train_then_predict_then_compare(tiny_training_data: Path, tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    code, _ = run(
        [
            "train",
            "--data-dir",
            str(tiny_training_data),
            "--output-dir",
            str(run_dir),
            "--epochs",
            "20",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.005",
            "--no-augment",
            *SMALL,
        ],
        capsys,
    )
    assert code == 0
    checkpoint = run_dir / "best_model.pt"
    assert checkpoint.is_file()

    image = next((tiny_training_data / "images" / "A172").glob("*.tif"))
    code, payload = run(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            str(image),
            "--output-dir",
            str(tmp_path / "pred"),
            "--device",
            "cpu",
        ],
        capsys,
    )
    assert code == 0 and payload["images"]

    code, payload = run(
        [
            "compare",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(tiny_training_data / "manifests" / "test.jsonl"),
            "--output-dir",
            str(tmp_path / "cmp"),
            "--device",
            "cpu",
        ],
        capsys,
    )
    # 0 when the model wins, 2 when it does not; either way it must not crash.
    assert code in (0, 2)
    assert "verdict" in payload


def test_baseline_command_writes_a_report(tiny_training_data: Path, tmp_path: Path, capsys) -> None:
    code, payload = run(
        [
            "baseline",
            "--manifest",
            str(tiny_training_data / "manifests" / "test.jsonl"),
            "--output-dir",
            str(tmp_path / "base"),
            "--max-images",
            "1",
        ],
        capsys,
    )
    assert code == 0
    assert payload["overall"]["images"] == 1


def test_treatment_template_and_analysis(tmp_path: Path, capsys) -> None:
    code, payload = run(["treatment-template", "--output", str(tmp_path / "t.csv")], capsys)
    assert code == 0 and Path(payload["template"]).is_file()

    manifest = build_synthetic_experiment(tmp_path / "study", seed=4)
    code, payload = run(
        [
            "analyze-treatment",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--instance-method",
            "connected_components",
            "--min-area",
            "5",
        ],
        capsys,
    )
    assert code == 0
    assert payload["wells"] == 20
    assert "dose_response" in payload


@pytest.mark.parametrize(
    "argv,fragment",
    [
        (["verify", "--data-dir", "/nonexistent/path"], "does not exist"),
        (["baseline", "--manifest", "/nonexistent/m.jsonl"], "does not exist"),
        (["predict", "--checkpoint", "/nonexistent.pt", "/nonexistent.tif"], "does not exist"),
        (["analyze-treatment", "--manifest", "/nonexistent.csv"], "does not exist"),
    ],
)
def test_missing_inputs_produce_a_readable_error_and_exit_one(
    argv: list[str], fragment: str, capsys
) -> None:
    assert main(argv) == 1
    assert fragment in capsys.readouterr().err


def test_invalid_training_options_exit_one_without_a_traceback(
    tiny_training_data: Path, tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "train",
            "--data-dir",
            str(tiny_training_data),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "0",
            *SMALL,
        ]
    )
    assert code == 1
    assert "epochs must be an integer >= 1" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_training_on_leaking_data_is_refused_from_the_cli(
    leaking_training_data: Path, tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "train",
            "--data-dir",
            str(leaking_training_data),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            *SMALL,
        ]
    )
    assert code == 1
    assert "Split leakage detected" in capsys.readouterr().err


def test_prepare_rejects_an_unknown_cell_type_at_the_parser(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["prepare-livecell", "--cell-types", "NOT_A_CELL"])


def test_full_profile_requires_confirmation(tmp_path: Path, capsys) -> None:
    code = main(["prepare-livecell", "--profile", "full", "--data-dir", str(tmp_path)])
    assert code == 1
    assert "--confirm-full-download" in capsys.readouterr().err


def test_compare_exits_zero_when_the_model_actually_wins(tmp_path: Path, capsys) -> None:
    """The success exit path — untested while every fixture made the gate unwinnable."""
    from conftest import build_crowded_dataset

    data = build_crowded_dataset(tmp_path / "crowded")
    code, _ = run(
        [
            "train",
            "--data-dir",
            str(data),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "60",
            "--batch-size",
            "2",
            "--image-size",
            "64",
            "--base-channels",
            "8",
            "--depth",
            "2",
            "--learning-rate",
            "0.003",
            "--device",
            "cpu",
        ],
        capsys,
    )
    assert code == 0

    code, payload = run(
        [
            "compare",
            "--checkpoint",
            str(tmp_path / "run" / "best_model.pt"),
            "--manifest",
            str(data / "manifests" / "test.jsonl"),
            "--output-dir",
            str(tmp_path / "cmp"),
            "--device",
            "cpu",
        ],
        capsys,
    )
    assert code == 0
    assert payload["passed"] is True
