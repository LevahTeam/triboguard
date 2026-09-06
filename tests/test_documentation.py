"""The documentation must agree with the artifacts it cites.

Numbers in prose drift. In this repository they drifted twice: a p-value that one
document disavowed as pseudoreplication was still quoted as evidence in another,
and a test count went stale in three places at three different values. Patching
each instance does not stop the next one, so the figures a reader would act on
are checked against the JSON that produced them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contexts(text: str, pattern: re.Pattern[str], radius: int = 2) -> list[str]:
    """Every match of *pattern*, with the lines around it.

    Prose wraps, so a qualifier routinely lands on the line after the number it
    qualifies. Windows are taken symmetrically around each hit rather than at
    every offset, which would otherwise clip the qualifier off the end.
    """
    lines = text.splitlines()
    found: list[str] = []
    for index, line in enumerate(lines):
        if pattern.search(line):
            found.append("\n".join(lines[max(0, index - radius) : index + radius + 1]))
    return found


NEGATION = re.compile(r"\bnot\b|\bno\b|\bnever\b|\bcannot\b|\bwithout\b|\bnothing\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def comparison() -> dict:
    path = ROOT / "results" / "comparison__comparison.json"
    if not path.is_file():
        pytest.skip("no published comparison artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_document_exists_that_the_others_link_to() -> None:
    """Citing a file that is not in the repository is a dangling claim."""
    missing: list[str] = []
    for document in DOCS:
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", _text(document)):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.name} -> {target}")
    assert not missing, f"dangling links: {missing}"


def test_no_document_cites_the_git_ignored_run_directory_as_evidence() -> None:
    """`runs/` is not in a fresh clone; `results/` is."""
    offenders = [
        document.name
        for document in DOCS
        if re.search(r"`runs/[\w/]+\.(json|csv)`", _text(document))
    ]
    assert not offenders, f"{offenders} cite runs/ instead of the tracked results/"


def test_the_pseudoreplicated_p_value_is_never_quoted_as_evidence() -> None:
    """It may appear only where it is named as an inflated figure."""
    inflated = re.compile(r"1\.7 ?(e-18|× 10⁻¹⁸)")
    disclaimer = re.compile(r"pseudoreplicat|inflat|not (be )?quoted", re.IGNORECASE)
    for document in DOCS:
        for context in _contexts(_text(document), inflated):
            assert disclaimer.search(context), (
                f"{document.name} quotes the pseudoreplicated p-value without naming "
                f"it as one:\n{context}"
            )


def test_the_documented_p_value_matches_the_published_artifact(comparison: dict) -> None:
    expected = comparison["sign_test_p_value"]
    quoted = {
        match for document in DOCS for match in re.findall(r"p = (\d\.\d+)e-(\d+)", _text(document))
    }
    assert quoted, "no p-value quoted anywhere"
    for mantissa, exponent in quoted:
        value = float(f"{mantissa}e-{exponent}")
        assert value == pytest.approx(expected, rel=0.05), (
            f"documented p = {value:.3g} but the artifact says {expected:.3g}"
        )


def test_the_documented_reference_floor_matches_the_published_artifact(
    comparison: dict,
) -> None:
    floor = comparison["reference_floor_macro_dice"]
    readme = _text(ROOT / "README.md")
    assert f"{floor:.3f}" in readme, f"README does not state the {floor:.3f} floor"
    # And the trivial predictor must be named, not just numbered.
    assert "no learning at all" in readme or "no learning" in readme


def test_the_documented_instance_ceiling_matches_the_published_artifact(
    comparison: dict,
) -> None:
    ceiling = comparison.get("instance_ceiling")
    if not ceiling:
        pytest.skip("artifact predates the instance ceiling")
    value = ceiling["matching_score_50_95"]
    readme = _text(ROOT / "README.md")
    assert f"{value:.3f}" in readme, f"README does not state the {value:.3f} instance ceiling"


def test_the_documented_test_count_is_not_stale(request: pytest.FixtureRequest) -> None:
    """A count that drifts by a fifth is worse than no count."""
    collected = request.session.testscollected
    if collected < 100:
        pytest.skip(f"only {collected} tests collected; this check is meaningful on a full run")
    # Only current-state claims. A historical "27 passed" in a before/after column
    # is correct and must not be flagged.
    claims = [
        int(match)
        for document in DOCS
        for match in re.findall(r"\*\*(\d{2,4}) tests[,.]", _text(document))
        + re.findall(r"(\d{2,4}) passed, \d+ skipped", _text(document))
    ]
    stale = [claim for claim in claims if not 0.9 * collected <= claim <= 1.1 * collected]
    assert not stale, (
        f"documentation claims {stale} tests but {collected} were collected; "
        "update the docs or the claim"
    )


def test_the_project_never_claims_a_tribonema_result() -> None:
    """The single claim this project must never make, checked mechanically."""
    forbidden = re.compile(
        r"(tribonema[^.]{0,80}(kills|cures|destroys|eliminates)"
        r"|(we|this) (show|prove|demonstrate)[^.]{0,60}anti-?tumou?r)",
        re.IGNORECASE,
    )
    for document in DOCS:
        for context in _contexts(_text(document), forbidden):
            # The phrase is allowed only inside an explicit denial.
            assert NEGATION.search(context), (
                f"{document.name} claims a Tribonema result:\n{context}"
            )


def test_no_document_claims_a_nanotechnology_component() -> None:
    for document in DOCS:
        for line in _text(document).splitlines():
            if re.search(r"nano(robot|particle|technolog)", line, re.IGNORECASE):
                assert re.search(r"\bno\b|not\b|never|without", line, re.IGNORECASE), (
                    f"{document.name} mentions nanotechnology affirmatively: {line.strip()}"
                )
