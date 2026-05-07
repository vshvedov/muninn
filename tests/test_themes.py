"""Theme registration and persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from muninn import bootstrap
from muninn import themes
from tests.test_app_pilot import patched_app  # reuse the headless TUI fixture


def test_default_theme_in_default_config() -> None:
    assert bootstrap.DEFAULT_CONFIG["theme"] == "muninn-dark"
    assert themes.DEFAULT_THEME_NAME == "muninn-dark"


def test_themes_module_exposes_both_themes() -> None:
    names = {t.name for t in themes.ALL_THEMES}
    assert names == {"muninn-dark", "muninn-light"}
    dark = next(t for t in themes.ALL_THEMES if t.name == "muninn-dark")
    light = next(t for t in themes.ALL_THEMES if t.name == "muninn-light")
    assert dark.dark is True
    assert light.dark is False
    # Knotpm rule: pure black is reserved for ink, never used as a background.
    assert dark.background.lower() != "#000000"
    # Light page canvas must be pure white per the design system.
    assert light.background.lower() == "#ffffff"


def test_env_override_for_theme(tmp_path: Path, monkeypatch) -> None:
    md = bootstrap.ensure_muninn_dir(tmp_path)
    monkeypatch.setenv("MUNINN_THEME", "muninn-light")
    cfg = bootstrap.load_config(md)
    assert cfg["theme"] == "muninn-light"


async def test_app_registers_and_activates_default_theme(patched_app, tmp_path) -> None:
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        assert "muninn-dark" in patched_app.available_themes
        assert "muninn-light" in patched_app.available_themes
        assert patched_app.theme == "muninn-dark"


async def test_theme_change_persists_to_config(patched_app, tmp_path) -> None:
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        patched_app.theme = "muninn-light"
        await pilot.pause()
    cfg = bootstrap.load_config(patched_app.muninn_dir)
    assert cfg["theme"] == "muninn-light"
