"""Group-aware train/validation/test assignment.

LIVECell's official single-cell-type ``train`` and ``val`` files share wells: for
the A172 subset used here, wells A7, B7 and D7 all appear in both, and 22
acquisition groups (same well, same field of view, same timestamp) are split
across the two. Validation scores measured that way describe memorisation of a
well, not generalisation to a new one.

This module keeps the official *test* split untouched — that is the number a
reviewer should trust — and re-partitions the official train+val pool so that no
group crosses the train/validation boundary.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

GROUP_LEVELS = ("well", "acquisition")


class SplitError(RuntimeError):
    """Raised when a leakage-free split cannot be produced."""


def _stable_rank(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def assign_group_splits(
    items: Sequence[tuple[str, str, Any]],
    *,
    seed: int,
    val_fraction: float = 0.3,
    holdout_test: bool = True,
) -> dict[Any, str]:
    """Assign each item to ``train``/``val``/``test`` without splitting a group.

    ``items`` are ``(official_split, group_key, item_key)`` triples. Items whose
    official split is ``test`` are pinned to test when *holdout_test* is set; all
    other items are re-partitioned by whole group.
    """
    if not 0.0 < val_fraction < 1.0:
        raise SplitError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}.")
    if not items:
        raise SplitError("No images available to split.")

    groups_by_key: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for official_split, group_key, item_key in items:
        groups_by_key[group_key].append((official_split, item_key))

    test_groups: set[str] = set()
    if holdout_test:
        for group_key, members in groups_by_key.items():
            official = {split for split, _ in members}
            if "test" in official and official != {"test"}:
                raise SplitError(
                    f"Group {group_key!r} appears in the official test split and in "
                    f"{sorted(official - {'test'})}. The source data cannot be separated "
                    "at this group level."
                )
            if official == {"test"}:
                test_groups.add(group_key)

    pool = sorted(set(groups_by_key) - test_groups, key=lambda key: _stable_rank(key, seed))
    if not pool:
        raise SplitError("Every group was pinned to the test split; nothing left to train on.")
    if len(pool) < 2:
        raise SplitError(
            f"Only {len(pool)} development group(s) available. A leakage-free "
            "train/validation split needs at least two distinct groups — prepare more "
            "wells or fields of view."
        )

    pool_size = sum(len(groups_by_key[key]) for key in pool)
    target_val = max(1, round(pool_size * val_fraction))
    validation_groups: set[str] = set()
    validation_count = 0
    for key in pool:
        if validation_count >= target_val:
            break
        # Never take the last group: training must keep at least one group.
        if len(validation_groups) == len(pool) - 1:
            break
        validation_groups.add(key)
        validation_count += len(groups_by_key[key])

    assignment: dict[Any, str] = {}
    for group_key, members in groups_by_key.items():
        if group_key in test_groups:
            split = "test"
        elif group_key in validation_groups:
            split = "val"
        else:
            split = "train"
        for _, item_key in members:
            assignment[item_key] = split

    counts: dict[str, int] = defaultdict(int)
    for split in assignment.values():
        counts[split] += 1
    for required in ("train", "val"):
        if counts[required] == 0:
            raise SplitError(f"The {required} split came out empty; adjust val_fraction or seed.")
    if holdout_test and counts["test"] == 0:
        raise SplitError("The test split came out empty; the official test annotations are needed.")
    return assignment


def summarise(assignment: dict[Any, str], group_of: dict[Any, str]) -> dict[str, Any]:
    """Describe the resulting split and prove that no group is shared."""
    groups: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for item_key, split in assignment.items():
        groups[split].add(group_of[item_key])
        counts[split] += 1
    overlaps: dict[str, list[str]] = {}
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(groups[left] & groups[right])
            if shared:
                overlaps[f"{left}|{right}"] = shared
    return {
        "images": dict(sorted(counts.items())),
        "groups": {name: sorted(value) for name, value in sorted(groups.items())},
        "group_counts": {name: len(value) for name, value in sorted(groups.items())},
        "overlaps": overlaps,
        "clean": not overlaps,
    }
