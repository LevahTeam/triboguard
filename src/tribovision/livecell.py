"""Download and prepare a small, reproducible subset of the LIVECell dataset.

The demo path uses official per-cell-type annotations and extracts only selected
members of the official image ZIP.

The official ``train`` and ``val`` files are *not* used as the model-development
split, because they share wells. They are pooled and re-partitioned by
acquisition group so that no field of view crosses the train/validation
boundary; the official ``test`` split is left exactly as the LIVECell authors
defined it. Both the assigned split and the original official split are recorded
for every image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from tribovision.manifest import ManifestError, acquisition_group, well_group
from tribovision.splits import assign_group_splits, summarise

S3_ROOT = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"
IMAGES_URL = f"{S3_ROOT}/images.zip"
CELL_TYPES = ("A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3")
SINGLE_CELL_ANNOTATION_URLS = {
    cell_type: {
        split: (f"{S3_ROOT}/annotations/LIVECell_single_cells/{cell_type.casefold()}/{split}.json")
        for split in ("train", "val", "test")
    }
    for cell_type in CELL_TYPES
}
FULL_ANNOTATION_URLS = {
    split: f"{S3_ROOT}/annotations/LIVECell/livecell_coco_{split}.json"
    for split in ("train", "val", "test")
}
_WELL_PATTERN = re.compile(r"^[^_]+_Phase_([A-H]\d{1,2})_", re.IGNORECASE)

DOWNLOAD_TIMEOUT_SECONDS = 120
#: Selective ZIP extraction is resumable, so a transient timeout is retried.
REMOTE_ATTEMPTS = 5
#: A single LIVECell frame is under a megabyte; 64 MB is generous but bounded.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
#: A decompression bomb inflates far more than a TIFF ever does.
MAX_COMPRESSION_RATIO = 200.0


class PreparationError(RuntimeError):
    """Raised when LIVECell cannot be prepared safely or consistently."""


@dataclass(frozen=True)
class ManifestRecord:
    image_id: int
    image_path: str
    annotation_path: str
    annotation_ids: list[int]
    width: int
    height: int
    cell_type: str
    well: str
    split: str
    official_split: str
    acquisition_group: str
    source_file_name: str
    source_annotation_url: str
    source_doi: str
    data_license: str
    selection_seed: int
    image_sha256: str
    micrometers_per_pixel: float | None


def _download(
    url: str,
    destination: Path,
    *,
    expected_max_bytes: int | None = None,
    validator: Callable[[Path], bool] | None = None,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Stream *url* atomically and reuse an existing file only after validation."""
    if (
        destination.is_file()
        and destination.stat().st_size > 0
        and (validator is None or validator(destination))
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    downloaded = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            try:
                declared_size = int(response.headers.get("content-length", "0"))
            except (TypeError, ValueError):
                declared_size = 0
            if expected_max_bytes and declared_size > expected_max_bytes:
                raise PreparationError(
                    f"Refusing {declared_size:,}-byte download; safety limit is "
                    f"{expected_max_bytes:,} bytes."
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                temp_path = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if expected_max_bytes and downloaded > expected_max_bytes:
                        raise PreparationError(
                            f"Download exceeded the {expected_max_bytes:,}-byte safety limit."
                        )
                    handle.write(chunk)
        if downloaded == 0:
            raise PreparationError(f"Received an empty response while downloading {url}.")
        if declared_size and downloaded != declared_size:
            raise PreparationError(
                f"Incomplete download from {url}: expected {declared_size:,} bytes, "
                f"received {downloaded:,}."
            )
        if validator is not None and not validator(temp_path):
            raise PreparationError(f"Downloaded file failed validation: {url}")
        temp_path.replace(destination)
    except PreparationError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise PreparationError(f"Could not download {url}: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return destination


def _valid_coco_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("images"), list)
        and isinstance(payload.get("annotations"), list)
    )


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid annotation file: {path}") from exc
    if not isinstance(payload, dict):
        raise PreparationError(f"COCO annotation root must be an object: {path}")
    if not isinstance(payload.get("images"), list) or not isinstance(
        payload.get("annotations"), list
    ):
        raise PreparationError(f"COCO annotation file lacks image or annotation lists: {path}")
    return payload


def _valid_zip(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


def _cell_type(file_name: str) -> str:
    stem = PurePosixPath(file_name).name
    return stem.split("_", 1)[0]


def _well(file_name: str) -> str:
    match = _WELL_PATTERN.match(PurePosixPath(file_name).name)
    return match.group(1).upper() if match else "unknown"


def _stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _select_images(
    coco: dict[str, Any], cell_types: Iterable[str], max_images: int, seed: int
) -> list[dict[str, Any]]:
    requested = {name.casefold() for name in cell_types}
    images: list[dict[str, Any]] = []
    for image in coco.get("images", []):
        if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
            raise PreparationError("COCO image records must contain a string file_name.")
        try:
            int(image["id"])
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError(
                f"Invalid COCO image metadata for {image.get('file_name')!r}."
            ) from exc
        if width <= 0 or height <= 0:
            raise PreparationError(f"Invalid COCO image dimensions for {image.get('file_name')!r}.")
        if _cell_type(image["file_name"]).casefold() in requested:
            images.append(image)
    if not images:
        available = sorted({_cell_type(str(i["file_name"])) for i in coco.get("images", [])})
        raise PreparationError(
            f"No images matched {sorted(requested)}. Available types: {available or 'none'}."
        )
    images.sort(key=lambda item: _stable_number(str(item["file_name"]), seed))
    return images[:max_images]


def _archive_member_lookup(names: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    duplicates: set[str] = set()
    for name in names:
        if name.endswith("/"):
            continue
        basename = PurePosixPath(name).name.casefold()
        if basename in lookup:
            duplicates.add(basename)
        else:
            lookup[basename] = name
    for basename in duplicates:
        lookup.pop(basename, None)
    return lookup


def check_zip_member(info: zipfile.ZipInfo) -> None:
    """Refuse oversized or suspiciously compressible archive members."""
    if info.is_dir():
        raise PreparationError(f"Refusing to extract directory entry {info.filename!r}.")
    if info.file_size > MAX_MEMBER_BYTES:
        raise PreparationError(
            f"Archive member {info.filename!r} declares {info.file_size:,} bytes, above the "
            f"{MAX_MEMBER_BYTES:,}-byte per-file limit."
        )
    if info.compress_size > 0:
        ratio = info.file_size / info.compress_size
        if ratio > MAX_COMPRESSION_RATIO:
            raise PreparationError(
                f"Archive member {info.filename!r} expands {ratio:.0f}x, above the "
                f"{MAX_COMPRESSION_RATIO:.0f}x decompression-bomb limit."
            )


def _safe_image_destination(root: Path, cell_type: str, basename: str) -> Path:
    safe_name = PurePosixPath(basename).name
    if safe_name != basename or safe_name in {"", ".", ".."}:
        raise PreparationError(f"Unsafe image filename in dataset: {basename!r}")
    destination = root / cell_type / safe_name
    resolved_root = root.resolve()
    if resolved_root not in destination.resolve().parents:
        raise PreparationError(f"Image destination escapes data directory: {destination}")
    return destination


def _image_is_valid(path: Path, expected_width: int, expected_height: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (expected_width, expected_height)
    except (OSError, ValueError):
        return False


def _copy_image_member(source: Any, destination: Path, image: dict[str, Any]) -> None:
    try:
        expected_width = int(image["width"])
        expected_height = int(image["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError(f"Invalid dimensions for image {image.get('file_name')!r}.") from exc
    if expected_width <= 0 or expected_height <= 0:
        raise PreparationError(f"Invalid dimensions for image {image.get('file_name')!r}.")
    if _image_is_valid(destination, expected_width, expected_height):
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as output:
            temp_path = Path(output.name)
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_MEMBER_BYTES:
                    raise PreparationError(
                        f"Extracted image {image.get('file_name')!r} exceeded the "
                        f"{MAX_MEMBER_BYTES:,}-byte limit while streaming."
                    )
                output.write(chunk)
        if not _image_is_valid(temp_path, expected_width, expected_height):
            raise PreparationError(
                f"Extracted image is corrupt or has unexpected dimensions: "
                f"{image.get('file_name')!r}."
            )
        temp_path.replace(destination)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _extract_members(
    archive: Any,
    infos: dict[str, zipfile.ZipInfo],
    lookup: dict[str, str],
    selected: list[dict[str, Any]],
    image_root: Path,
    source_label: str,
) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    for image in selected:
        source_name = PurePosixPath(str(image["file_name"])).name
        key = source_name.casefold()
        member = lookup.get(key)
        if member is None:
            raise PreparationError(f"{source_name!r} is missing or ambiguous in {source_label}.")
        info = infos.get(member)
        if info is not None:
            check_zip_member(info)
        destination = _safe_image_destination(
            image_root, _cell_type(str(image["file_name"])), source_name
        )
        with archive.open(member) as source:
            _copy_image_member(source, destination, image)
        extracted[key] = destination
    return extracted


def _existing_images(
    selected: list[dict[str, Any]], image_root: Path
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Split *selected* into images already verified on disk and those still needed."""
    found: dict[str, Path] = {}
    missing: list[dict[str, Any]] = []
    for image in selected:
        file_name = str(image["file_name"])
        source_name = PurePosixPath(file_name).name
        destination = _safe_image_destination(image_root, _cell_type(file_name), source_name)
        try:
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError(f"Invalid dimensions for image {file_name!r}.") from exc
        if _image_is_valid(destination, width, height):
            found[source_name.casefold()] = destination
        else:
            missing.append(image)
    return found, missing


def _extract_remote_members(
    url: str, selected: list[dict[str, Any]], image_root: Path
) -> dict[str, Path]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise PreparationError(
            "Install project dependencies to enable selective ZIP extraction."
        ) from exc

    try:
        with RemoteZip(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            lookup = _archive_member_lookup(infos)
            return _extract_members(
                archive, infos, lookup, selected, image_root, "the official image archive"
            )
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"Could not read remote image archive {url}: {exc}") from exc


def _extract_local_members(
    archive_path: Path, selected: list[dict[str, Any]], image_root: Path
) -> dict[str, Path]:
    if not _valid_zip(archive_path):
        raise PreparationError(
            f"Image archive does not exist or is not a valid ZIP: {archive_path}"
        )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            lookup = _archive_member_lookup(infos)
            return _extract_members(archive, infos, lookup, selected, image_root, str(archive_path))
    except PreparationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PreparationError(f"Could not read image archive {archive_path}: {exc}") from exc


def _write_manifests(
    root: Path,
    selections: list[tuple[str, Path, str, dict[str, Any], list[dict[str, Any]]]],
    image_paths: dict[tuple[str, str], Path],
    assignment: dict[tuple[str, int], str],
    seed: int,
    group_by: str,
    micrometers_per_pixel: float | None,
    caps: dict[str, int],
) -> dict[str, Any]:
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    records_by_split: dict[str, list[ManifestRecord]] = defaultdict(list)
    cell_counts: Counter[str] = Counter()
    annotation_hashes: dict[str, str] = {}

    for official_split, coco_path, source_url, coco, selected in selections:
        selected_ids = {int(image["id"]) for image in selected}
        annotations_by_image: dict[int, list[int]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id in selected_ids:
                annotations_by_image[image_id].append(int(annotation["id"]))
        annotation_hashes[os.path.relpath(coco_path, root)] = _sha256(coco_path)
        for image in selected:
            image_id = int(image["id"])
            file_name = str(image["file_name"])
            cell_type = _cell_type(file_name)
            source_name = PurePosixPath(file_name).name.casefold()
            path = image_paths[(official_split, source_name)]
            split = assignment[(official_split, image_id)]
            cell_counts[cell_type] += 1
            payload = {"source_file_name": file_name, "cell_type": cell_type}
            records_by_split[split].append(
                ManifestRecord(
                    image_id=image_id,
                    image_path=os.path.relpath(path, root),
                    annotation_path=os.path.relpath(coco_path, root),
                    annotation_ids=sorted(annotations_by_image[image_id]),
                    width=int(image["width"]),
                    height=int(image["height"]),
                    cell_type=cell_type,
                    well=_well(file_name),
                    split=split,
                    official_split=official_split,
                    acquisition_group=acquisition_group(payload),
                    source_file_name=file_name,
                    source_annotation_url=source_url,
                    source_doi="10.1038/s41592-021-01249-6",
                    data_license="CC BY-NC 4.0",
                    selection_seed=seed,
                    image_sha256=_sha256(path),
                    micrometers_per_pixel=micrometers_per_pixel,
                )
            )

    key = acquisition_group if group_by == "acquisition" else well_group
    groups = {
        split: {
            key({"source_file_name": r.source_file_name, "cell_type": r.cell_type, "well": r.well})
            for r in records
        }
        for split, records in records_by_split.items()
    }
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(groups[left] & groups[right])
            if shared:
                raise PreparationError(
                    f"Refusing to write leaking manifests: {left} and {right} share "
                    f"{len(shared)} {group_by} group(s), e.g. {shared[:3]}."
                )

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        records = sorted(records_by_split.get(split, []), key=lambda item: item.image_id)
        if not records:
            raise PreparationError(
                f"The {split} split is empty. Increase --max-images or add cell types."
            )
        counts[split] = len(records)
        content = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
        _atomic_write_text(manifests_dir / f"{split}.jsonl", content)

    summary = {
        "dataset": "LIVECell",
        "license": "CC BY-NC 4.0",
        "source": "https://sartorius-research.github.io/LIVECell/",
        "annotation_files_sha256": annotation_hashes,
        "split_policy": {
            "test": "official LIVECell test split, untouched",
            "train_val": (
                "official train+val pooled, then re-partitioned so that no "
                f"{group_by} group crosses the boundary"
            ),
            "group_by": group_by,
            "reason": (
                "The official single-cell-type train and val files share wells, so "
                "validation measured on them overstates generalisation."
            ),
        },
        "selection": "deterministic hash-ranked sample within each official split",
        "selection_caps": caps,
        "seed": seed,
        "images": counts,
        "cell_types": dict(sorted(cell_counts.items())),
        "wells_by_split": {
            split: sorted({record.well for record in records})
            for split, records in sorted(records_by_split.items())
        },
        "groups_by_split": {split: len(value) for split, value in sorted(groups.items())},
        "micrometers_per_pixel": micrometers_per_pixel,
        "calibration_note": (
            "micrometers_per_pixel is null unless supplied by the operator. Without it, "
            "morphology is reported in pixels and must not be quoted in micrometres."
        ),
        "note": (
            "This public dataset teaches generic cell boundaries. It contains no Tribonema "
            "treatment labels and cannot validate anti-tumor effects."
        ),
    }
    _atomic_write_text(
        manifests_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return {"counts": counts, "summary": summary}


def prepare_demo(
    root: Path,
    *,
    cell_types: Iterable[str] = ("A172",),
    max_images: int | dict[str, int] = 60,
    seed: int = 42,
    images_zip: Path | None = None,
    group_by: str = "well",
    val_fraction: float = 0.3,
    micrometers_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Prepare deterministic, leakage-free samples from official LIVECell splits.

    ``max_images`` may be a single cap or a per-official-split mapping. The mapping
    exists so that a training set can be grown while the held-out split is held
    byte-identical: a scaling experiment that also changes what it is measured on
    answers nothing.
    """
    caps = (
        {split: int(max_images) for split in ("train", "val", "test")}
        if isinstance(max_images, int)
        else {split: int(max_images.get(split, 60)) for split in ("train", "val", "test")}
    )
    if any(value < 1 for value in caps.values()):
        raise PreparationError("max_images must be at least 1 for every split.")
    if group_by not in ("well", "acquisition"):
        raise PreparationError(f"group_by must be 'well' or 'acquisition', got {group_by!r}.")
    if not 0.0 < val_fraction < 1.0:
        raise PreparationError(f"val_fraction must be between 0 and 1, got {val_fraction}.")
    if micrometers_per_pixel is not None and not micrometers_per_pixel > 0:
        raise PreparationError("micrometers_per_pixel must be positive when provided.")

    root = root.resolve()
    canonical_types: list[str] = []
    by_casefold = {cell_type.casefold(): cell_type for cell_type in CELL_TYPES}
    for requested in cell_types:
        try:
            canonical = by_casefold[requested.casefold()]
        except KeyError as exc:
            raise PreparationError(
                f"Unknown cell type {requested!r}; choose from {', '.join(CELL_TYPES)}."
            ) from exc
        if canonical not in canonical_types:
            canonical_types.append(canonical)

    selections: list[tuple[str, Path, str, dict[str, Any], list[dict[str, Any]]]] = []
    all_selected: list[tuple[str, dict[str, Any]]] = []
    for cell_type in canonical_types:
        for split in ("train", "val", "test"):
            source_url = SINGLE_CELL_ANNOTATION_URLS[cell_type][split]
            annotation_path = root / "annotations" / cell_type.casefold() / f"{split}.json"
            _download(
                source_url,
                annotation_path,
                expected_max_bytes=160 * 1024 * 1024,
                validator=_valid_coco_json,
            )
            coco = _load_coco(annotation_path)
            selected = _select_images(coco, [cell_type], caps[split], seed)
            selections.append((split, annotation_path, source_url, coco, selected))
            all_selected.extend((split, image) for image in selected)

    seen_files: dict[str, str] = {}
    for split, image in all_selected:
        source_name = PurePosixPath(str(image["file_name"])).name.casefold()
        previous_split = seen_files.setdefault(source_name, split)
        if previous_split != split:
            raise PreparationError(
                f"Source data problem: {image['file_name']!r} occurs in both official "
                f"{previous_split} and {split} annotations."
            )

    key = acquisition_group if group_by == "acquisition" else well_group
    triples = []
    group_of: dict[tuple[str, int], str] = {}
    for split, image in all_selected:
        file_name = str(image["file_name"])
        payload = {
            "source_file_name": file_name,
            "cell_type": _cell_type(file_name),
            "well": _well(file_name),
        }
        item_key = (split, int(image["id"]))
        group_key = key(payload)
        group_of[item_key] = group_key
        triples.append((split, group_key, item_key))
    try:
        assignment = assign_group_splits(triples, seed=seed, val_fraction=val_fraction)
    except ManifestError as exc:  # pragma: no cover - defensive
        raise PreparationError(str(exc)) from exc

    image_root = root / "images"
    selected_flat = [image for _, image in all_selected]
    # Images already present and dimension-verified are reused, so a re-preparation
    # (for example after changing the split policy) needs no network access at all.
    extracted, missing = _existing_images(selected_flat, image_root)
    if missing:
        if images_zip:
            extracted.update(_extract_local_members(images_zip.resolve(), missing, image_root))
        else:
            # Selective extraction over HTTP ranges is long-running and a single
            # read timeout used to lose the whole batch. Each image is written
            # atomically, so a retry resumes: recompute what is still missing and
            # only give up when an attempt makes no progress at all.
            for attempt in range(1, REMOTE_ATTEMPTS + 1):
                try:
                    extracted.update(_extract_remote_members(IMAGES_URL, missing, image_root))
                    break
                except PreparationError:
                    extracted, still_missing = _existing_images(selected_flat, image_root)
                    if attempt == REMOTE_ATTEMPTS or len(still_missing) >= len(missing):
                        raise PreparationError(
                            f"Could not fetch {len(still_missing)} of "
                            f"{len(selected_flat)} images after {attempt} attempt(s). "
                            f"{len(extracted)} are already on disk and will be reused, "
                            "so simply re-running resumes where this stopped. For a "
                            "large subset, download the official images.zip once and "
                            "pass --images-zip instead."
                        ) from None
                    missing = still_missing
    image_paths = {
        (split, PurePosixPath(str(image["file_name"])).name.casefold()): extracted[
            PurePosixPath(str(image["file_name"])).name.casefold()
        ]
        for split, image in all_selected
    }
    result = _write_manifests(
        root, selections, image_paths, assignment, seed, group_by, micrometers_per_pixel, caps
    )
    result["split_report"] = summarise(assignment, group_of)
    return result


def download_full(root: Path, *, confirmed: bool) -> dict[str, Path]:
    """Download the complete image archive and official train/val/test annotations."""
    if not confirmed:
        raise PreparationError(
            "Full LIVECell is multiple gigabytes. Re-run with --confirm-full-download."
        )
    root = root.resolve()
    downloads = root / "downloads"
    paths = {"images": _download(IMAGES_URL, downloads / "images.zip", validator=_valid_zip)}
    for split, url in FULL_ANNOTATION_URLS.items():
        paths[split] = _download(
            url,
            root / "annotations" / f"livecell_coco_{split}.json",
            validator=_valid_coco_json,
        )
    return paths
