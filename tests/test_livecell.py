"""Dataset preparation: selection, extraction safety, downloads, and splits."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import make_annotation, make_image_record
from PIL import Image

from tribovision.livecell import (
    MAX_COMPRESSION_RATIO,
    MAX_MEMBER_BYTES,
    PreparationError,
    _archive_member_lookup,
    _download,
    _existing_images,
    _extract_local_members,
    _safe_image_destination,
    _select_images,
    _write_manifests,
    check_zip_member,
    download_full,
    prepare_demo,
)
from tribovision.manifest import load_manifest, require_no_group_leakage

WIDTH, HEIGHT = 40, 24


def _tif_bytes(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    buffer = io.BytesIO()
    pixels = np.full((height, width), 40, dtype=np.uint8)
    pixels[4:14, 6:20] = 210
    Image.fromarray(pixels).save(buffer, format="TIFF")
    return buffer.getvalue()


# --------------------------------------------------------------------- selection


def test_subset_selection_is_deterministic_and_seeded(coco_payload: dict) -> None:
    first = _select_images(coco_payload, ("A172", "BV2"), max_images=10, seed=41)
    repeat = _select_images(coco_payload, ("A172", "BV2"), max_images=10, seed=41)
    changed = _select_images(coco_payload, ("A172", "BV2"), max_images=10, seed=42)
    assert [item["id"] for item in first] == [item["id"] for item in repeat]
    assert [item["id"] for item in first] != [item["id"] for item in changed]


def test_subset_selection_rejects_unknown_cell_type(coco_payload: dict) -> None:
    with pytest.raises(PreparationError, match="No images matched"):
        _select_images(coco_payload, ("NOT_A_CELL",), max_images=3, seed=1)


@pytest.mark.parametrize(
    "mutation",
    [
        {"width": 0},
        {"height": -5},
        {"file_name": 12345},
    ],
)
def test_malformed_coco_image_records_are_rejected(mutation: dict) -> None:
    payload = {"images": [{**make_image_record(1), **mutation}], "annotations": []}
    with pytest.raises(PreparationError):
        _select_images(payload, ("A172",), max_images=1, seed=0)


# ------------------------------------------------------------------ zip safety


def test_safe_destination_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="Unsafe image filename"):
        _safe_image_destination(tmp_path, "A172", "../escape.tif")


def test_duplicate_archive_basenames_are_dropped() -> None:
    lookup = _archive_member_lookup(["a/x.tif", "b/x.tif", "c/y.tif", "dir/"])
    assert "x.tif" not in lookup
    assert lookup["y.tif"] == "c/y.tif"


def test_local_zip_extraction_ignores_member_directories(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    image = make_image_record(1, width=WIDTH, height=HEIGHT)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"images/livecell_train_val_images/{image['file_name']}", _tif_bytes())
    extracted = _extract_local_members(archive_path, [image], tmp_path / "images")
    destination = next(iter(extracted.values()))
    assert destination.is_file()
    assert destination.parent.name == "A172"


def test_an_oversized_member_is_refused() -> None:
    info = zipfile.ZipInfo("huge.tif")
    info.file_size = MAX_MEMBER_BYTES + 1
    info.compress_size = MAX_MEMBER_BYTES
    with pytest.raises(PreparationError, match="per-file limit"):
        check_zip_member(info)


def test_a_decompression_bomb_is_refused() -> None:
    info = zipfile.ZipInfo("bomb.tif")
    info.compress_size = 1024
    info.file_size = int(1024 * (MAX_COMPRESSION_RATIO + 10))
    with pytest.raises(PreparationError, match="decompression-bomb limit"):
        check_zip_member(info)


def test_a_directory_entry_is_refused() -> None:
    with pytest.raises(PreparationError, match="directory entry"):
        check_zip_member(zipfile.ZipInfo("some/dir/"))


def test_an_ordinary_member_passes_the_checks() -> None:
    info = zipfile.ZipInfo("fine.tif")
    info.file_size = 500_000
    info.compress_size = 400_000
    check_zip_member(info)


def test_extraction_refuses_a_real_decompression_bomb(tmp_path: Path) -> None:
    archive_path = tmp_path / "bomb.zip"
    image = make_image_record(1)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(image["file_name"], b"\x00" * (2 * 1024 * 1024))
    with pytest.raises(PreparationError, match="decompression-bomb limit"):
        _extract_local_members(archive_path, [image], tmp_path / "images")


def test_a_corrupt_archive_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"this is not a zip file")
    with pytest.raises(PreparationError, match="not a valid ZIP"):
        _extract_local_members(broken, [make_image_record(1)], tmp_path / "images")


def test_a_missing_member_is_reported(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("something_else.tif", _tif_bytes())
    with pytest.raises(PreparationError, match="missing or ambiguous"):
        _extract_local_members(archive_path, [make_image_record(1)], tmp_path / "images")


def test_a_member_with_the_wrong_dimensions_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    image = make_image_record(1)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(image["file_name"], _tif_bytes(width=999, height=7))
    with pytest.raises(PreparationError, match="corrupt or has unexpected dimensions"):
        _extract_local_members(archive_path, [image], tmp_path / "images")
    assert not list((tmp_path / "images").rglob("*.tif"))


def test_already_present_images_are_reused_without_touching_the_archive(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image = make_image_record(1)
    (image_root / "A172").mkdir(parents=True)
    (image_root / "A172" / image["file_name"]).write_bytes(_tif_bytes())
    found, missing = _existing_images([image], image_root)
    assert not missing and len(found) == 1


# -------------------------------------------------------------------- downloads


def test_download_rejects_declared_oversize_before_writing(tmp_path: Path, monkeypatch) -> None:
    class Response:
        headers = {"content-length": str(10**9)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):  # pragma: no cover - never reached
            return b""

    monkeypatch.setattr("tribovision.livecell.urllib.request.urlopen", lambda *a, **k: Response())
    with pytest.raises(PreparationError, match="safety limit"):
        _download("https://example.invalid/x", tmp_path / "x.json", expected_max_bytes=1024)
    assert not (tmp_path / "x.json").exists()


def test_download_stream_limit_leaves_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    class Response:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            return b"x" * 4096

    monkeypatch.setattr("tribovision.livecell.urllib.request.urlopen", lambda *a, **k: Response())
    with pytest.raises(PreparationError, match="exceeded"):
        _download("https://example.invalid/x", tmp_path / "x.json", expected_max_bytes=1024)
    assert list(tmp_path.iterdir()) == []


def test_download_passes_an_explicit_timeout(tmp_path: Path, monkeypatch) -> None:
    """A request with no timeout can hang a preparation run forever."""
    seen: dict[str, object] = {}

    class Response:
        headers = {"content-length": "2"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            if seen.get("done"):
                return b""
            seen["done"] = True
            return b"{}"

    def fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("tribovision.livecell.urllib.request.urlopen", fake_urlopen)
    _download("https://example.invalid/x", tmp_path / "x.json", validator=lambda _p: True)
    assert isinstance(seen["timeout"], int) and seen["timeout"] > 0


def test_download_reuses_a_validated_existing_file_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "cached.json"
    destination.write_text('{"images": [], "annotations": []}', encoding="utf-8")
    assert _download("https://example.invalid/never", destination, validator=lambda _p: True)


def test_full_download_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="--confirm-full-download"):
        download_full(tmp_path, confirmed=False)


# ----------------------------------------------------------------- preparation


def _fake_source(tmp_path: Path, coco_payload: dict) -> tuple[Path, Path]:
    """Build local annotation files and an images ZIP so preparation needs no network."""
    root = tmp_path / "data"
    a172 = [image for image in coco_payload["images"] if image["file_name"].startswith("A172")]
    # Wells A1..A3 in the official train/val pool, A4 held out as the official test.
    by_split = {
        "train": [i for i in a172 if "_A1_" in i["file_name"] or "_A2_" in i["file_name"]],
        "val": [i for i in a172 if "_A3_" in i["file_name"]],
        "test": [i for i in a172 if "_A4_" in i["file_name"]],
    }
    for split, images in by_split.items():
        ids = {image["id"] for image in images}
        payload = {
            "images": images,
            "annotations": [make_annotation(1000 + image["id"], image["id"]) for image in images],
            "categories": coco_payload["categories"],
        }
        assert ids
        path = root / "annotations" / "a172" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image in a172:
            archive.writestr(
                f"images/{image['file_name']}", _tif_bytes(image["width"], image["height"])
            )
    return root, archive_path


def test_preparation_produces_well_disjoint_splits(tmp_path: Path, coco_payload: dict) -> None:
    root, archive = _fake_source(tmp_path, coco_payload)
    result = prepare_demo(
        root, cell_types=["A172"], max_images=10, seed=42, images_zip=archive, group_by="well"
    )
    wells = result["summary"]["wells_by_split"]
    assert not set(wells["train"]) & set(wells["val"])
    assert wells["test"] == ["A4"]
    assert result["split_report"]["clean"]

    splits = {
        split: load_manifest(root / "manifests" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    assert require_no_group_leakage(splits, group_by="well")["clean"]
    for split, records in splits.items():
        assert all(record.split == split for record in records)
        assert all(record.image_sha256 for record in records)


def test_preparation_records_the_original_official_split(
    tmp_path: Path, coco_payload: dict
) -> None:
    root, archive = _fake_source(tmp_path, coco_payload)
    prepare_demo(root, cell_types=["A172"], max_images=10, seed=42, images_zip=archive)
    rows = [
        json.loads(line) for line in (root / "manifests" / "train.jsonl").read_text().splitlines()
    ]
    assert all(row["official_split"] in {"train", "val"} for row in rows)
    assert all(row["acquisition_group"] for row in rows)


def test_preparation_is_reproducible(tmp_path: Path, coco_payload: dict) -> None:
    root, archive = _fake_source(tmp_path, coco_payload)
    first = prepare_demo(root, cell_types=["A172"], max_images=10, seed=42, images_zip=archive)
    manifest_a = (root / "manifests" / "train.jsonl").read_text()
    second = prepare_demo(root, cell_types=["A172"], max_images=10, seed=42, images_zip=archive)
    assert first["counts"] == second["counts"]
    assert manifest_a == (root / "manifests" / "train.jsonl").read_text()


def test_calibration_is_recorded_only_when_supplied(tmp_path: Path, coco_payload: dict) -> None:
    root, archive = _fake_source(tmp_path, coco_payload)
    prepare_demo(root, cell_types=["A172"], max_images=10, seed=1, images_zip=archive)
    summary = json.loads((root / "manifests" / "summary.json").read_text())
    assert summary["micrometers_per_pixel"] is None
    assert "must not be quoted in micrometres" in summary["calibration_note"]

    prepare_demo(
        root,
        cell_types=["A172"],
        max_images=10,
        seed=1,
        images_zip=archive,
        micrometers_per_pixel=0.62,
    )
    summary = json.loads((root / "manifests" / "summary.json").read_text())
    assert summary["micrometers_per_pixel"] == 0.62


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max_images": 0}, "max_images"),
        ({"group_by": "plate"}, "group_by"),
        ({"val_fraction": 0.0}, "val_fraction"),
        ({"val_fraction": 1.5}, "val_fraction"),
        ({"micrometers_per_pixel": -1.0}, "micrometers_per_pixel"),
        ({"cell_types": ["NOPE"]}, "Unknown cell type"),
    ],
)
def test_invalid_preparation_arguments_are_rejected(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(PreparationError, match=message):
        prepare_demo(tmp_path / "data", **{"cell_types": ["A172"], **kwargs})


def test_writing_leaking_manifests_is_refused(tmp_path: Path, coco_payload: dict) -> None:
    """A direct call that would place one well in two splits must not write files."""
    root = tmp_path / "data"
    images = [make_image_record(index, well="A1", location=index, crop=1) for index in (1, 2, 3, 4)]
    coco = {
        "images": images,
        "annotations": [make_annotation(100 + i["id"], i["id"]) for i in images],
    }
    annotation_path = root / "annotations" / "a172" / "train.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")
    image_paths = {}
    for image in images:
        path = root / "images" / "A172" / image["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_tif_bytes())
        image_paths[("train", image["file_name"].casefold())] = path
    assignment = {
        ("train", 1): "train",
        ("train", 2): "val",  # same well as image 1 -> leakage
        ("train", 3): "test",
        ("train", 4): "train",
    }
    with pytest.raises(PreparationError, match="Refusing to write leaking manifests"):
        _write_manifests(
            root,
            [("train", annotation_path, "https://example.invalid", coco, images)],
            image_paths,
            assignment,
            42,
            "well",
            None,
            {"train": 10, "val": 10, "test": 10},
        )


def test_per_split_caps_hold_the_test_set_fixed_while_training_grows(
    tmp_path: Path, coco_payload: dict
) -> None:
    """A scaling experiment that also changes its held-out set answers nothing."""
    root, archive = _fake_source(tmp_path, coco_payload)
    small = prepare_demo(
        root,
        cell_types=["A172"],
        max_images={"train": 2, "val": 2, "test": 2},
        seed=42,
        images_zip=archive,
    )
    test_manifest = (root / "manifests" / "test.jsonl").read_text()
    large = prepare_demo(
        root,
        cell_types=["A172"],
        max_images={"train": 20, "val": 20, "test": 2},
        seed=42,
        images_zip=archive,
    )
    # The training pool grew; the held-out split is byte-identical.
    assert large["counts"]["train"] > small["counts"]["train"]
    assert (root / "manifests" / "test.jsonl").read_text() == test_manifest
    assert large["summary"]["selection_caps"]["test"] == 2


def test_an_invalid_per_split_cap_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="at least 1 for every split"):
        prepare_demo(tmp_path / "d", cell_types=["A172"], max_images={"train": 0})


def test_a_transient_remote_failure_is_retried_and_resumes(
    tmp_path: Path, coco_payload: dict, monkeypatch
) -> None:
    """Each image is written atomically, so a retry must not restart from zero."""
    root, archive = _fake_source(tmp_path, coco_payload)
    attempts: list[int] = []
    real = _extract_local_members

    def flaky(url: str, selected: list, image_root: Path) -> dict:
        attempts.append(len(selected))
        # Extract half, then fail, the way a read timeout mid-batch behaves.
        real(archive, selected[: max(1, len(selected) // 2)], image_root)
        if len(attempts) < 3:
            raise PreparationError("simulated read timeout")
        return real(archive, selected, image_root)

    monkeypatch.setattr("tribovision.livecell._extract_remote_members", flaky)
    result = prepare_demo(root, cell_types=["A172"], max_images=10, seed=42)
    assert len(attempts) == 3
    # Each attempt asked for strictly fewer images than the last: it resumed.
    assert attempts == sorted(attempts, reverse=True)
    assert attempts[-1] < attempts[0]
    assert result["counts"]["train"] > 0


def test_a_remote_failure_that_makes_no_progress_gives_up_with_advice(
    tmp_path: Path, coco_payload: dict, monkeypatch
) -> None:
    root, _ = _fake_source(tmp_path, coco_payload)

    def always_fails(url: str, selected: list, image_root: Path) -> dict:
        raise PreparationError("simulated outage")

    monkeypatch.setattr("tribovision.livecell._extract_remote_members", always_fails)
    with pytest.raises(PreparationError, match="re-running resumes|--images-zip"):
        prepare_demo(root, cell_types=["A172"], max_images=10, seed=42)


def test_a_second_cell_type_can_be_prepared_without_clobbering_the_first(
    tmp_path: Path, coco_payload: dict
) -> None:
    """A transfer test needs two manifests side by side, sharing one image tree."""
    root, archive = _fake_source(tmp_path, coco_payload)
    prepare_demo(root, cell_types=["A172"], max_images=10, seed=42, images_zip=archive)
    original = (root / "manifests" / "test.jsonl").read_text()

    prepare_demo(
        root,
        cell_types=["A172"],
        max_images=6,
        seed=7,
        images_zip=archive,
        manifests_dirname="manifests_other",
    )
    assert (root / "manifests_other" / "test.jsonl").is_file()
    assert (root / "manifests" / "test.jsonl").read_text() == original
    # Both manifests resolve images out of the same shared tree.
    from tribovision.manifest import load_manifest

    for name in ("manifests", "manifests_other"):
        for record in load_manifest(root / name / "test.jsonl"):
            assert record.image_path.is_file()
