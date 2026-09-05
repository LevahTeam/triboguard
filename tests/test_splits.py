"""Group-aware split assignment."""

from __future__ import annotations

import pytest

from tribovision.splits import SplitError, assign_group_splits, summarise


def _items() -> list[tuple[str, str, tuple[str, int]]]:
    items = []
    identifier = 0
    for official, group in (
        ("train", "A7"),
        ("train", "B7"),
        ("val", "A7"),
        ("val", "D7"),
        ("test", "C7"),
    ):
        for _ in range(6):
            identifier += 1
            items.append((official, group, (official, identifier)))
    return items


def test_no_group_is_split_across_train_and_validation() -> None:
    items = _items()
    assignment = assign_group_splits(items, seed=42, val_fraction=0.3)
    group_of = {key: group for _, group, key in items}
    report = summarise(assignment, group_of)
    assert report["clean"]
    assert set(report["groups"]["test"]) == {"C7"}
    assert not set(report["groups"]["train"]) & set(report["groups"]["val"])


def test_official_test_images_stay_in_test() -> None:
    items = _items()
    assignment = assign_group_splits(items, seed=7, val_fraction=0.4)
    for official, _, key in items:
        if official == "test":
            assert assignment[key] == "test"


def test_assignment_is_deterministic_for_a_seed() -> None:
    items = _items()
    assert assign_group_splits(items, seed=1) == assign_group_splits(items, seed=1)
    assert assign_group_splits(items, seed=1) != assign_group_splits(items, seed=99)


def test_a_group_spanning_test_and_train_is_refused() -> None:
    items = [("test", "A7", ("test", 1)), ("train", "A7", ("train", 2))]
    with pytest.raises(SplitError, match="official test split"):
        assign_group_splits(items, seed=0)


def test_a_single_development_group_cannot_be_split() -> None:
    items = [("train", "A7", ("train", index)) for index in range(4)]
    items.append(("test", "C7", ("test", 99)))
    with pytest.raises(SplitError, match="at least two distinct groups"):
        assign_group_splits(items, seed=0)


def test_training_always_keeps_at_least_one_group() -> None:
    items = [("train", "A7", ("a", 1)), ("train", "B7", ("b", 2)), ("test", "C7", ("c", 3))]
    assignment = assign_group_splits(items, seed=0, val_fraction=0.99)
    assert "train" in assignment.values() and "val" in assignment.values()


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_invalid_val_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(SplitError, match="val_fraction"):
        assign_group_splits(_items(), seed=0, val_fraction=fraction)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(SplitError, match="No images"):
        assign_group_splits([], seed=0)
