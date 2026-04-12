"""Tests for 1-based vs 0-based slide target normalization."""

import pytest

from app.agent.slide_target import normalize_slide_target


@pytest.mark.parametrize(
    ("transcript", "raw_target", "expected"),
    [
        ("go to the fifth slide", 5, 4),
        ("Please show me the 3rd slide", 3, 2),
        ("jump to slide 4", 4, 3),
        ("open the first slide", 1, 0),
        ("go to the sixth slide", 6, 5),
    ],
)
def test_converts_when_model_echoes_human_slide_number(
    transcript: str, raw_target: int, expected: int
) -> None:
    assert normalize_slide_target(transcript, raw_target, num_slides=6) == expected


def test_no_change_when_model_already_0_based() -> None:
    # User asked for fifth slide; model correctly returned 4
    assert normalize_slide_target("go to the fifth slide", 4, num_slides=6) == 4


def test_no_change_when_numbers_differ() -> None:
    # Sixth slide: index 5 — user said sixth, model returned 5
    assert normalize_slide_target("go to the sixth slide", 5, num_slides=6) == 5


def test_skips_when_user_says_zero_based() -> None:
    assert normalize_slide_target("use 0-based index 5", 5, num_slides=6) == 5


def test_word_ordinals() -> None:
    assert normalize_slide_target("show the third slide please", 3, num_slides=6) == 2
    assert normalize_slide_target("the fifth slide", 5, num_slides=6) == 4
