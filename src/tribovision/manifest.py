"""One validated manifest schema, path resolver, and provenance enforcer.

Both the neural dataset and the classical baseline used to parse manifests with
their own ad-hoc code, and only one of them contained the dataset directory. A
single implementation removes that asymmetry: every consumer gets the same field
validation, the same containment guarantee, and the same provenance checks.

Provenance is enforced, not merely recorded. Loading verifies that the image on
disk still hashes to the value written at preparation time, that the record's
declared split matches the manifest it lives in, that referenced COCO
annotations really belong to the referenced image, and that no annotation or
image is claimed twice within or across splits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tribovision.provenance import sha256_file

SPLITS = ("train", "val", "test")

# LIVECell file names look like ``A172_Phase_C7_2_02d12h00m_3.tif``:
# cell type, imaging mode, well, field/location, timestamp, crop index.
_FILE_NAME_PATTERN = re.compile(
    r"^(?P<cell_type>[^_]+)_(?P<mode>[^_]+)_(?P<well>[A-H]\d{1,2})_(?P<location>\d+)_"
    r"(?P<timestamp>\d+d\d+h\d+m)_(?P<crop>\d+)\.(?P<extension>tif|tiff|png)$",
    re.IGNORECASE,
)

REQUIRED_FIELDS = (
    "image_id",
    "image_path",
    "annotation_path",
    "annotation_ids",
    "width",
    "height",
    "cell_type",
    "well",
    "split",
    "source_file_name",
)


class ManifestError(RuntimeError):
    """Raised when a manifest is malformed, unsafe, or fails a provenance check."""


def parse_file_name(file_name: str) -> dict[str, str] | None:
    """Split a LIVECell file name into its acquisition components."""
    match = _FILE_NAME_PATTERN.match(PurePosixPath(file_name).name)
    if match is None:
        return None
    parts = match.groupdict()
    parts["well"] = parts["well"].upper()
    return parts


def acquisition_group(record: ManifestRecord | dict[str, Any]) -> str:
    """Return the acquisition-group key used to prevent split leakage.

    Two crops taken from the same well, field of view, and timestamp are the same
    physical observation. Placing one in training and the other in validation
    inflates validation scores without any real generalisation, so the group key
    deliberately ignores the crop index.
    """
    file_name = (
        record.source_file_name
        if isinstance(record, ManifestRecord)
        else str(record.get("source_file_name", ""))
    )
    cell_type = (
        record.cell_type if isinstance(record, ManifestRecord) else str(record.get("cell_type", ""))
    )
    parts = parse_file_name(file_name)
    if parts is None:
        # Unrecognised naming falls back to the whole file name, which is the
        # conservative choice: it never merges two distinct observations.
        return f"{cell_type}|unparsed|{PurePosixPath(file_name).name}"
    return f"{parts['cell_type']}|{parts['well']}|{parts['location']}|{parts['timestamp']}"


def well_group(record: ManifestRecord | dict[str, Any]) -> str:
    """Return the well-level group key (the strictest separation available here)."""
    file_name = (
        record.source_file_name
        if isinstance(record, ManifestRecord)
        else str(record.get("source_file_name", ""))
    )
    cell_type = (
        record.cell_type if isinstance(record, ManifestRecord) else str(record.get("cell_type", ""))
    )
    parts = parse_file_name(file_name)
    well = (
        parts["well"]
        if parts
        else str(
            record.well if isinstance(record, ManifestRecord) else record.get("well", "unknown")
        )
    )
    return f"{parts['cell_type'] if parts else cell_type}|{well}"


GROUP_KEYS = {"well": well_group, "acquisition": acquisition_group}


@dataclass(frozen=True)
class ManifestRecord:
    """One validated manifest row with a resolved, contained filesystem path."""

    image_id: int
    image_path: Path
    annotation_path: Path
    annotation_ids: tuple[int, ...]
    width: int
    height: int
    cell_type: str
    well: str
    split: str
    source_file_name: str
    image_sha256: str | None = None
    official_split: str | None = None
    micrometers_per_pixel: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def relative_image_path(self) -> str:
        return str(self.raw.get("image_path", self.image_path.name))


def resolve_contained_path(root: Path, value: object, *, field_name: str) -> Path:
    """Resolve a manifest-supplied relative path, refusing anything outside *root*."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field {field_name!r} must be a nonempty relative path.")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ManifestError(f"Manifest field {field_name!r} must be relative, got {value!r}.")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ManifestError(f"Manifest field {field_name!r} escapes the data directory: {value!r}.")
    return resolved


def safe_output_name(value: object, *, field_name: str = "image_id") -> str:
    """Return a filename component that cannot traverse or escape a directory.

    Manifest values reach output filenames (``<image_id>_mask.png``). Without this
    guard a hostile manifest could write PNGs anywhere the process can write.
    """
    text = str(value)
    if not text or text in {".", ".."}:
        raise ManifestError(f"Manifest field {field_name!r} is not usable as a filename: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", text):
        raise ManifestError(
            f"Manifest field {field_name!r} must match [A-Za-z0-9._-]{{1,128}}, got {value!r}."
        )
    return text


def _require_int(record: dict[str, Any], key: str, *, positive: bool = False) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ManifestError(f"Manifest field {key!r} must be an integer, got {value!r}.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Manifest field {key!r} must be an integer, got {value!r}.") from exc
    if positive and number <= 0:
        raise ManifestError(f"Manifest field {key!r} must be positive, got {number}.")
    return number


def _require_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field {key!r} must be a nonempty string, got {value!r}.")
    return value


def parse_record(
    raw: dict[str, Any], *, root: Path, expected_split: str | None, line_number: int
) -> ManifestRecord:
    for name in REQUIRED_FIELDS:
        if name not in raw:
            raise ManifestError(f"Manifest line {line_number} is missing required field {name!r}.")

    annotation_ids = raw.get("annotation_ids")
    if not isinstance(annotation_ids, list):
        raise ManifestError(f"Manifest line {line_number}: 'annotation_ids' must be a list.")
    parsed_ids: list[int] = []
    for item in annotation_ids:
        if isinstance(item, bool) or not isinstance(item, int | str | float):
            raise ManifestError(
                f"Manifest line {line_number}: annotation id {item!r} is not an integer."
            )
        try:
            parsed_ids.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"Manifest line {line_number}: annotation id {item!r} is not an integer."
            ) from exc
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ManifestError(
            f"Manifest line {line_number} lists the same annotation id more than once."
        )

    split = _require_str(raw, "split")
    if split not in SPLITS:
        raise ManifestError(
            f"Manifest line {line_number}: split {split!r} must be one of {SPLITS}."
        )
    if expected_split is not None and split != expected_split:
        raise ManifestError(
            f"Manifest line {line_number} declares split {split!r} but lives in the "
            f"{expected_split!r} manifest."
        )

    calibration = raw.get("micrometers_per_pixel")
    if calibration is not None:
        try:
            calibration = float(calibration)
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"Manifest line {line_number}: 'micrometers_per_pixel' must be a number."
            ) from exc
        if not calibration > 0:
            raise ManifestError(
                f"Manifest line {line_number}: 'micrometers_per_pixel' must be positive."
            )

    return ManifestRecord(
        image_id=_require_int(raw, "image_id"),
        image_path=resolve_contained_path(root, raw.get("image_path"), field_name="image_path"),
        annotation_path=resolve_contained_path(
            root, raw.get("annotation_path"), field_name="annotation_path"
        ),
        annotation_ids=tuple(parsed_ids),
        width=_require_int(raw, "width", positive=True),
        height=_require_int(raw, "height", positive=True),
        cell_type=_require_str(raw, "cell_type"),
        well=_require_str(raw, "well"),
        split=split,
        source_file_name=_require_str(raw, "source_file_name"),
        image_sha256=raw.get("image_sha256") if isinstance(raw.get("image_sha256"), str) else None,
        official_split=(
            raw.get("official_split") if isinstance(raw.get("official_split"), str) else None
        ),
        micrometers_per_pixel=calibration,
        raw=raw,
    )


def load_manifest(
    manifest_path: Path,
    *,
    root: Path | None = None,
    expect_split: bool = True,
    verify_hashes: bool = True,
) -> list[ManifestRecord]:
    """Load, validate, and provenance-check one manifest file.

    ``root`` defaults to the dataset directory two levels above the manifest
    (``data/livecell/manifests/train.jsonl`` -> ``data/livecell``).
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest does not exist: {manifest_path}")
    dataset_root = (root or manifest_path.parent.parent).resolve()
    expected_split = manifest_path.stem if expect_split and manifest_path.stem in SPLITS else None

    records: list[ManifestRecord] = []
    seen_image_ids: set[int] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"Invalid JSON in {manifest_path.name} at line {line_number}."
            ) from exc
        if not isinstance(raw, dict):
            raise ManifestError(f"Manifest record at line {line_number} must be a JSON object.")
        record = parse_record(
            raw, root=dataset_root, expected_split=expected_split, line_number=line_number
        )
        if record.image_id in seen_image_ids:
            raise ManifestError(
                f"Manifest {manifest_path.name} lists image_id {record.image_id} twice."
            )
        seen_image_ids.add(record.image_id)
        records.append(record)

    if not records:
        raise ManifestError(f"Manifest {manifest_path.name} contains no records.")
    _verify_annotation_ownership(records)
    if verify_hashes:
        verify_image_hashes(records)
    return records


def _verify_annotation_ownership(records: Iterable[ManifestRecord]) -> None:
    """Confirm every referenced annotation exists and belongs to the stated image."""
    cache: dict[Path, dict[int, int]] = {}
    seen_annotations: dict[int, int] = {}
    for record in records:
        if record.annotation_path not in cache:
            try:
                payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ManifestError(
                    f"Annotation file does not exist: {record.annotation_path}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"Invalid COCO annotation JSON: {record.annotation_path}"
                ) from exc
            annotations = payload.get("annotations")
            if not isinstance(annotations, list):
                raise ManifestError(f"COCO annotations must be a list: {record.annotation_path}")
            owners: dict[int, int] = {}
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    raise ManifestError(
                        f"COCO annotation must be an object in {record.annotation_path}."
                    )
                try:
                    annotation_id = int(annotation["id"])
                    owner = int(annotation["image_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ManifestError(
                        f"COCO annotation has an invalid id/image_id in {record.annotation_path}."
                    ) from exc
                if annotation_id in owners:
                    raise ManifestError(
                        f"Duplicate annotation id {annotation_id} in {record.annotation_path}."
                    )
                owners[annotation_id] = owner
            cache[record.annotation_path] = owners
        owners = cache[record.annotation_path]
        for annotation_id in record.annotation_ids:
            declared_owner = owners.get(annotation_id)
            if declared_owner is None:
                raise ManifestError(
                    f"Annotation id {annotation_id} is missing from {record.annotation_path.name}."
                )
            if declared_owner != record.image_id:
                raise ManifestError(
                    f"Annotation id {annotation_id} belongs to image {declared_owner}, "
                    f"not to "
                    f"image {record.image_id}."
                )
            previous = seen_annotations.get(annotation_id)
            if previous is not None and previous != record.image_id:
                raise ManifestError(
                    f"Annotation id {annotation_id} is claimed by images {previous} "
                    f"and {record.image_id}."
                )
            seen_annotations[annotation_id] = record.image_id


def verify_image_hashes(records: Iterable[ManifestRecord]) -> int:
    """Re-hash every image that recorded a SHA-256 and reject silent edits."""
    checked = 0
    for record in records:
        if not record.image_sha256:
            continue
        if not record.image_path.is_file():
            raise ManifestError(f"Image referenced by the manifest is missing: {record.image_path}")
        actual = sha256_file(record.image_path)
        if actual != record.image_sha256:
            raise ManifestError(
                f"Image {record.image_path.name} no longer matches its recorded SHA-256. "
                f"Expected {record.image_sha256[:16]}…, found {actual[:16]}…."
            )
        checked += 1
    return checked


def check_group_leakage(
    splits: dict[str, list[ManifestRecord]], *, group_by: str = "well"
) -> dict[str, Any]:
    """Report — and by contract refuse — shared acquisition groups between splits.

    Filename-level duplicate detection is not enough: LIVECell stores several
    crops of one field of view under different names, so two splits can share the
    same physical observation while sharing no file name at all.
    """
    if group_by not in GROUP_KEYS:
        raise ManifestError(f"group_by must be one of {sorted(GROUP_KEYS)}, got {group_by!r}.")
    key = GROUP_KEYS[group_by]
    groups = {name: {key(record) for record in records} for name, records in splits.items()}
    images = {
        name: {record.source_file_name for record in records} for name, records in splits.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_groups = sorted(groups[left] & groups[right])
            shared_images = sorted(images[left] & images[right])
            if shared_groups or shared_images:
                overlaps[f"{left}|{right}"] = sorted(set(shared_groups) | set(shared_images))
    return {
        "group_by": group_by,
        "groups_per_split": {name: len(value) for name, value in groups.items()},
        "overlaps": overlaps,
        "clean": not overlaps,
    }


def require_no_group_leakage(
    splits: dict[str, list[ManifestRecord]], *, group_by: str = "well"
) -> dict[str, Any]:
    report = check_group_leakage(splits, group_by=group_by)
    if not report["clean"]:
        details = "; ".join(
            f"{pair}: {len(items)} shared {group_by} group(s) e.g. {items[:3]}"
            for pair, items in report["overlaps"].items()
        )
        raise ManifestError(
            f"Split leakage detected at the {group_by} level — {details}. "
            "Re-prepare the dataset with group-aware splitting."
        )
    return report
