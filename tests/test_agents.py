"""Tests for the per-level prompt addendum composer."""
from muninn.agents import compose_muninn_prompt, _LEVEL_ADDENDA


def test_compose_appends_low_addendum() -> None:
    out = compose_muninn_prompt("BASE_PROMPT", "low")
    assert out.startswith("BASE_PROMPT")
    assert out.endswith(_LEVEL_ADDENDA["low"])
    assert "Freedom level: LOW" in out


def test_compose_appends_medium_addendum() -> None:
    out = compose_muninn_prompt("BASE_PROMPT", "medium")
    assert "Freedom level: MEDIUM" in out
    assert "decide" in out.lower()


def test_compose_appends_high_addendum() -> None:
    out = compose_muninn_prompt("BASE_PROMPT", "high")
    assert "Freedom level: HIGH" in out
    assert "autonomously" in out.lower()


def test_compose_unknown_level_falls_back_to_low() -> None:
    """Defensive fallback: a typo or stale value never produces an empty
    addendum (which would silently drop the bias the system expects)."""
    out = compose_muninn_prompt("BASE", "wat")
    assert out == compose_muninn_prompt("BASE", "low")
    assert "Freedom level: LOW" in out
