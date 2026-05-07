import os
from pathlib import Path

from muninn import bootstrap


def test_project_bootstrap_creates_expected_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    # Project dir holds config + state, NOT prompts (those live user-level / bundled).
    assert (md / "config.toml").exists()
    assert (md / "SETUP.md").exists()
    # Modelfile bundle removed - qwen3-coder works out of the box, no patch needed.
    assert not (md / "Modelfile").exists()
    assert (md / "prompts").is_dir()
    assert list((md / "prompts").iterdir()) == []  # empty by default
    # Bug + review prompts are registered alongside feature ones.
    assert "bug_ground" in bootstrap.PROMPT_NAMES
    assert "bug_problem" in bootstrap.PROMPT_NAMES
    assert "bug_critique" in bootstrap.PROMPT_NAMES
    assert "precommit_review" in bootstrap.PROMPT_NAMES
    assert (md / "logs").is_dir()
    # history/ was removed: Phase 2 resume will reconstruct from the JSONL log.
    assert not (md / "history").exists()


def test_bootstrap_does_not_create_user_level_prompts_dir(tmp_path: Path, monkeypatch) -> None:
    """Regression: bootstrap no longer seeds prompts at ~/.muninn/prompts/.
    Bundled prompts are the canonical source; only project-level overrides
    exist on disk. ensure_muninn_dir must not write any files under the
    fake home dir (other than what the test explicitly creates)."""
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    project = tmp_path / "project"
    project.mkdir()
    bootstrap.ensure_muninn_dir(project)
    # The user-level prompts dir must not have been seeded by muninn.
    assert not (fake_home / ".muninn" / "prompts").exists()


def test_load_prompt_resolution_project_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    # Project override wins over the bundled default.
    (md / "prompts" / "muninn.md").write_text("PROJECT OVERRIDE")
    assert bootstrap.load_prompt(md, "muninn") == "PROJECT OVERRIDE"


def test_load_prompt_resolution_bundled_when_no_project_override(
    tmp_path: Path, monkeypatch,
) -> None:
    """No project override -> bundled default is returned. There is no
    user-level layer to fall back through."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    # The project prompts/ dir is empty by default.
    assert list((md / "prompts").iterdir()) == []
    text = bootstrap.load_prompt(md, "muninn")
    assert text.startswith("You are Muninn")  # bundled default


def test_load_prompt_user_level_dir_is_ignored(tmp_path: Path, monkeypatch) -> None:
    """Even if a user has lingering files at ~/.muninn/prompts/ from a prior
    install, load_prompt must not read them. They are orphaned, not loaded."""
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    # Simulate a user with leftover files from the old layout.
    legacy = fake_home / ".muninn" / "prompts"
    legacy.mkdir(parents=True)
    (legacy / "muninn.md").write_text("LEGACY USER LEVEL - SHOULD NOT BE READ")
    project = tmp_path / "project"
    project.mkdir()
    md = bootstrap.ensure_muninn_dir(project)
    text = bootstrap.load_prompt(md, "muninn")
    assert "LEGACY" not in text
    assert text.startswith("You are Muninn")  # bundled default


def test_load_config_defaults_and_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    monkeypatch.setenv("MUNINN_NUM_CTX", "8192")
    monkeypatch.setenv("MUNINN_MODEL", "qwen3-coder:8b")
    monkeypatch.setenv("MUNINN_MAX_REVISION_ROUNDS", "5")
    cfg = bootstrap.load_config(md)
    assert cfg["num_ctx"] == 8192
    assert cfg["model"] == "qwen3-coder:8b"
    assert cfg["max_revision_rounds"] == 5
    assert cfg["base_url"].startswith("http://localhost")
    assert cfg["freedom_level"] in {"low", "medium", "high"}


def test_default_config_has_max_revision_rounds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    assert cfg["max_revision_rounds"] == 3  # default cap mirrors eg-new-feature.md


def test_presets_exposed(tmp_path: Path) -> None:
    """Palette presets exist and are in expected ranges."""
    assert len(bootstrap.NUM_CTX_PRESETS) >= 3
    values = [v for v, _ in bootstrap.NUM_CTX_PRESETS]
    assert values == sorted(values)
    assert all(v >= 1024 for v in values)
    assert bootstrap.MAX_REVISION_ROUNDS_PRESETS == tuple(sorted(bootstrap.MAX_REVISION_ROUNDS_PRESETS))
    assert all(1 <= v <= 10 for v in bootstrap.MAX_REVISION_ROUNDS_PRESETS)


def test_save_config_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    cfg["freedom_level"] = "high"
    bootstrap.save_config(md, cfg)
    cfg2 = bootstrap.load_config(md)
    assert cfg2["freedom_level"] == "high"


def test_schema_version_mismatch_keeps_existing(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 99\nmodel = "weird:1b"\nbase_url = "http://localhost:11434/v1"\n'
        'num_ctx = 1024\nfreedom_level = "low"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["model"] == "weird:1b"
    captured = capsys.readouterr()
    assert "schema_version" in captured.err


# ---------------------------------------------------------------------------
# freedom_level migration + resolution
# ---------------------------------------------------------------------------


def test_freedom_level_default_is_low(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "low"
    assert "auto_mode" not in cfg


def test_freedom_level_migration_from_confirm(tmp_path: Path, monkeypatch) -> None:
    """Legacy disk auto_mode='confirm' must surface as freedom_level='low'."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 1\nmodel = "qwen3-coder:30b"\n'
        'base_url = "http://localhost:11434/v1"\n'
        'num_ctx = 65536\nauto_mode = "confirm"\n'
        'max_revision_rounds = 3\ntheme = "muninn-dark"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "low"
    assert "auto_mode" not in cfg


def test_freedom_level_migration_from_yolo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 1\nmodel = "qwen3-coder:30b"\n'
        'base_url = "http://localhost:11434/v1"\n'
        'num_ctx = 65536\nauto_mode = "yolo"\n'
        'max_revision_rounds = 3\ntheme = "muninn-dark"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "high"


def test_freedom_level_both_keys_freedom_wins(tmp_path: Path, monkeypatch) -> None:
    """When both freedom_level and legacy auto_mode are present on disk,
    freedom_level wins; auto_mode is dropped from the in-memory cfg."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 1\nfreedom_level = "medium"\nauto_mode = "yolo"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "medium"
    assert "auto_mode" not in cfg


def test_freedom_level_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MUNINN_FREEDOM_LEVEL", "medium")
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "medium"


def test_freedom_level_env_auto_mode_back_compat(tmp_path: Path, monkeypatch) -> None:
    """MUNINN_AUTO_MODE=yolo with no MUNINN_FREEDOM_LEVEL migrates to high."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.setenv("MUNINN_AUTO_MODE", "yolo")
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "high"


def test_freedom_level_env_conflict_freedom_wins(tmp_path: Path, monkeypatch) -> None:
    """Both env vars set: MUNINN_FREEDOM_LEVEL wins."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MUNINN_FREEDOM_LEVEL", "low")
    monkeypatch.setenv("MUNINN_AUTO_MODE", "yolo")
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "low"


def test_freedom_level_invalid_falls_back(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 1\nfreedom_level = "wat"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "low"
    err = capsys.readouterr().err
    assert "invalid freedom_level" in err
    assert "wat" in err


def test_freedom_level_invalid_falls_through_to_auto_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """Invalid freedom_level acts as 'not set'; legacy auto_mode then wins."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MUNINN_FREEDOM_LEVEL", raising=False)
    monkeypatch.delenv("MUNINN_AUTO_MODE", raising=False)
    md = bootstrap.ensure_muninn_dir(tmp_path)
    (md / "config.toml").write_text(
        'schema_version = 1\nfreedom_level = "wat"\nauto_mode = "yolo"\n'
    )
    cfg = bootstrap.load_config(md)
    assert cfg["freedom_level"] == "high"


def test_save_config_drops_legacy_auto_mode(tmp_path: Path, monkeypatch) -> None:
    """save_config strips auto_mode (and any private _* keys) on every write."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    md = bootstrap.ensure_muninn_dir(tmp_path)
    cfg = bootstrap.load_config(md)
    cfg["auto_mode"] = "yolo"
    cfg["_internal_marker"] = True
    bootstrap.save_config(md, cfg)
    on_disk = (md / "config.toml").read_text()
    assert "auto_mode" not in on_disk
    assert "_internal_marker" not in on_disk
    assert "freedom_level" in on_disk


def test_freedom_level_presets_shape() -> None:
    """PresetPickerScreen requires (value, label, description) triples."""
    presets = bootstrap.FREEDOM_LEVEL_PRESETS
    values = [p[0] for p in presets]
    assert values == ["low", "medium", "high"]
    for value, label, desc in presets:
        assert isinstance(value, str)
        assert isinstance(label, str) and label
        assert isinstance(desc, str) and len(desc) > 20
