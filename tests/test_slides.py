"""Tests for slide content — no mocks needed, pure logic."""
import pytest
from app.slides.content import SLIDES, get_slide, slides_summary


def test_slide_count():
    assert len(SLIDES) == 6


def test_slide_indices_are_sequential():
    for i, slide in enumerate(SLIDES):
        assert slide.index == i, f"Slide at position {i} has index {slide.index}"


def test_all_slides_have_required_fields():
    for slide in SLIDES:
        assert slide.title, f"Slide {slide.index} missing title"
        assert len(slide.bullets) >= 3, f"Slide {slide.index} has fewer than 3 bullets"
        assert slide.speaker_notes, f"Slide {slide.index} missing speaker_notes"
        assert len(slide.keywords) >= 3, f"Slide {slide.index} has fewer than 3 keywords"


def test_get_slide_returns_correct_slide():
    for i in range(6):
        slide = get_slide(i)
        assert slide.index == i


def test_get_slide_raises_on_out_of_range():
    with pytest.raises(IndexError):
        get_slide(6)
    with pytest.raises(IndexError):
        get_slide(-1)


def test_slides_summary_contains_all_indices():
    summary = slides_summary()
    for i in range(6):
        assert f"[{i}]" in summary


def test_slides_summary_contains_keywords():
    summary = slides_summary()
    # Spot-check a few keywords that must appear
    assert "recruitment" in summary
    assert "FDA" in summary
    assert "protocol" in summary
