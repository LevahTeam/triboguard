"""Manifest validation, containment, and provenance enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import rewrite_manifest

from tribovision.manifest import (
    ManifestError,
    acquisition_group,
    check_group_leakage,
    load_manifest,
    parse_file_name,
    require_no_group_leakage,
    safe_output_name,
    well_group,
)


def test_loads_a_valid_manifest(tiny_training_data: Path) -> None:
    records = load_manifest(tiny_training_data / "manifests" / "train.jsonl")
    assert len(records) == 2
    assert all(record.split == "train" for record in records)
    assert all(record.image_path.is_file() for record in records)


def test_rejects_image_whose_hash_changed(tiny_training_data: Path) -> None:
    record = json.loads(
        (tiny_training_data / "manifests" / "train.jsonl").read_text().splitlines()[0]
    )
    image = tiny_training_data / record["image_path"]
    image.write_bytes(image.read_bytes() + b"\x00")

    with pytest.raises(ManifestError, match="no longer matches its recorded SHA-256"):
        load_manifest(tiny_training_data / "manifests" / "train.jsonl")


def test_hash_check_can_be_skipped_but_is_on_by_default(tiny_training_data: Path) -> None:
    path = tiny_training_data / "manifests" / "train.jsonl"
    rewrite_manifest(
        tiny_training_data,
        "train",
        lambda rows: [{**row, "image_sha256": "0" * 64} for row in rows],
    )
    assert load_manifest(path, verify_hashes=False)
    with pytest.raises(ManifestError, match="SHA-256"):
        load_manifest(path)


def test_rejects_record_whose_split_contradicts_the_file(tiny_training_data: Path) -> None:
    path = rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, "split": "test"} for row in rows]
    )
    with pytest.raises(ManifestError, match="declares split 'test'"):
        load_manifest(path)


def test_rejects_annotation_that_belongs_to_another_image(tiny_training_data: Path) -> None:
    def mutate(rows: list[dict]) -> list[dict]:
        rows[0]["annotation_ids"] = rows[1]["annotation_ids"]
        return rows

    path = rewrite_manifest(tiny_training_data, "train", mutate)
    with pytest.raises(ManifestError, match="belongs to image"):
        load_manifest(path)


def test_rejects_missing_annotation_id(tiny_training_data: Path) -> None:
    path = rewrite_manifest(
        tiny_training_data,
        "train",
        lambda rows: [{**row, "annotation_ids": [999999]} for row in rows],
    )
    with pytest.raises(ManifestError, match="is missing from"):
        load_manifest(path)


def test_rejects_duplicate_annotation_ids_within_a_record(tiny_training_data: Path) -> None:
    def mutate(rows: list[dict]) -> list[dict]:
        rows[0]["annotation_ids"] = [rows[0]["annotation_ids"][0]] * 2
        return rows

    path = rewrite_manifest(tiny_training_data, "train", mutate)
    with pytest.raises(ManifestError, match="more than once"):
        load_manifest(path)


def test_rejects_duplicate_image_ids(tiny_training_data: Path) -> None:
    def mutate(rows: list[dict]) -> list[dict]:
        rows[1]["image_id"] = rows[0]["image_id"]
        return rows

    path = rewrite_manifest(tiny_training_data, "train", mutate)
    with pytest.raises(ManifestError, match="twice"):
        load_manifest(path)


def test_rejects_duplicate_annotation_ids_in_the_source_file(tiny_training_data: Path) -> None:
    source = tiny_training_data / "annotations" / "source.json"
    payload = json.loads(source.read_text())
    payload["annotations"].append(dict(payload["annotations"][0]))
    source.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="Duplicate annotation id"):
        load_manifest(tiny_training_data / "manifests" / "train.jsonl")


@pytest.mark.parametrize("field", ["image_path", "annotation_path"])
@pytest.mark.parametrize("value", ["../../outside.json", "/etc/passwd"])
def test_rejects_path_escape(tiny_training_data: Path, field: str, value: str) -> None:
    path = rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, field: value} for row in rows]
    )
    with pytest.raises(ManifestError, match="escapes the data directory|must be relative"):
        load_manifest(path)


@pytest.mark.parametrize("field", ["width", "height"])
def test_rejects_non_positive_dimensions(tiny_training_data: Path, field: str) -> None:
    path = rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, field: 0} for row in rows]
    )
    with pytest.raises(ManifestError, match="must be positive"):
        load_manifest(path)


def test_rejects_missing_required_field(tiny_training_data: Path) -> None:
    def mutate(rows: list[dict]) -> list[dict]:
        rows[0].pop("cell_type")
        return rows

    path = rewrite_manifest(tiny_training_data, "train", mutate)
    with pytest.raises(ManifestError, match="missing required field 'cell_type'"):
        load_manifest(path)


def test_rejects_empty_and_malformed_manifests(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir(parents=True)
    empty = tmp_path / "manifests" / "train.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="contains no records"):
        load_manifest(empty)

    broken = tmp_path / "manifests" / "val.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="Invalid JSON"):
        load_manifest(broken)

    listy = tmp_path / "manifests" / "test.jsonl"
    listy.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="must be a JSON object"):
        load_manifest(listy)


def test_rejects_negative_calibration(tiny_training_data: Path) -> None:
    path = rewrite_manifest(
        tiny_training_data,
        "train",
        lambda rows: [{**row, "micrometers_per_pixel": -1.0} for row in rows],
    )
    with pytest.raises(ManifestError, match="must be positive"):
        load_manifest(path)


@pytest.mark.parametrize("value", ["../evil", "a/b", "..", "", "x" * 200, "semi;colon"])
def test_output_names_cannot_traverse(value: str) -> None:
    with pytest.raises(ManifestError):
        safe_output_name(value)


def test_output_names_accept_ordinary_ids() -> None:
    assert safe_output_name(255101) == "255101"
    assert safe_output_name("A172_1") == "A172_1"


def test_file_name_parsing_and_group_keys() -> None:
    parts = parse_file_name("A172_Phase_C7_2_02d12h00m_3.tif")
    assert parts is not None and parts["well"] == "C7" and parts["timestamp"] == "02d12h00m"
    assert parse_file_name("not-a-livecell-name.png") is None

    same_view = {"source_file_name": "A172_Phase_C7_2_02d12h00m_3.tif", "cell_type": "A172"}
    other_crop = {"source_file_name": "A172_Phase_C7_2_02d12h00m_9.tif", "cell_type": "A172"}
    other_time = {"source_file_name": "A172_Phase_C7_2_03d12h00m_3.tif", "cell_type": "A172"}
    # Crops of one field of view are the same observation; different times are not.
    assert acquisition_group(same_view) == acquisition_group(other_crop)
    assert acquisition_group(same_view) != acquisition_group(other_time)
    assert well_group(same_view) == well_group(other_time)


def test_group_leakage_is_detected_even_without_shared_filenames(
    leaking_training_data: Path,
) -> None:
    splits = {
        split: load_manifest(leaking_training_data / "manifests" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    # No file name is shared, yet the same well appears in both.
    train_names = {record.source_file_name for record in splits["train"]}
    val_names = {record.source_file_name for record in splits["val"]}
    assert not train_names & val_names

    report = check_group_leakage(splits, group_by="well")
    assert not report["clean"]
    assert "train|val" in report["overlaps"]
    with pytest.raises(ManifestError, match="Split leakage detected"):
        require_no_group_leakage(splits, group_by="well")


def test_clean_splits_pass_the_leakage_check(tiny_training_data: Path) -> None:
    splits = {
        split: load_manifest(tiny_training_data / "manifests" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    assert require_no_group_leakage(splits, group_by="well")["clean"]
    assert require_no_group_leakage(splits, group_by="acquisition")["clean"]


def test_unknown_group_level_is_rejected(tiny_training_data: Path) -> None:
    splits = {"train": load_manifest(tiny_training_data / "manifests" / "train.jsonl")}
    with pytest.raises(ManifestError, match="group_by must be one of"):
        check_group_leakage(splits, group_by="plate")


@pytest.mark.parametrize("value", ["bad\x00name.tif", "\x00", "a\x00/b.json"])
def test_a_nul_byte_in_a_path_is_a_manifest_error_not_a_raw_oserror(
    tiny_training_data: Path, value: str
) -> None:
    """Every bad manifest value should surface through one exception type."""
    path = rewrite_manifest(
        tiny_training_data, "train", lambda rows: [{**row, "image_path": value} for row in rows]
    )
    with pytest.raises(ManifestError):
        load_manifest(path)
