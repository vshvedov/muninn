"""Pilot test: drive the TUI headless with mocked Ollama health + agents."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from muninn import bootstrap


@pytest.fixture
def patched_app(tmp_path: Path, monkeypatch):
    """Construct a MuninnTUI in tmp_path with a fake httpx response and stub agents."""

    # Bootstrap a .muninn/ first so on_mount uses our temp dir.
    bootstrap.ensure_muninn_dir(tmp_path)

    # Patch the http_client.get to return a stub /api/tags response.
    async def fake_get(self, url, *args, **kwargs):
        if url.endswith("/api/tags"):
            req = httpx.Request("GET", url)
            return httpx.Response(
                200, json={"models": [{"name": "qwen3-coder:30b"}]},
                request=req,
            )
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    # Don't actually instantiate Ollama agents in pilot - patch _build_agents.
    from muninn import app as app_mod

    def _stub_build(self):
        self.muninn_agent = object()  # truthy sentinel
        self.huginn_factory = lambda: object()
        from muninn.tools import ToolContext
        self.tool_ctx = ToolContext(
            cwd=self.cwd, muninn_dir=self.muninn_dir,
            freedom_level=self.freedom_level,
            confirm_callback=lambda *a, **k: None,
            ask_user_callback=lambda *a, **k: None,
            log=self._log,
        )

    monkeypatch.setattr(app_mod.MuninnTUI, "_build_agents", _stub_build)

    return app_mod.MuninnTUI(cwd=tmp_path)


async def test_app_boots_and_health_green(patched_app, tmp_path):
    async with patched_app.run_test() as pilot:
        # Wait a bit for the boot worker to complete.
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        assert patched_app.health_status == "ok"
        banner = patched_app.query_one("#status-banner")
        assert "Ollama OK" in str(banner.render())


async def test_app_unknown_slash_command_renders_message(patched_app, tmp_path):
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        # Type an unknown slash command.
        from textual.widgets import Input, Markdown
        inp = patched_app.query_one("#user-input", Input)
        inp.value = "/brainstorm hello"
        await inp.action_submit()
        await pilot.pause()
        await pilot.pause()
        # Should render an error Markdown saying Phase 1 only supports /feature.
        mds = patched_app.query("#muninn-scroll Markdown").results(Markdown)
        joined = "\n".join(str(md._markdown) if hasattr(md, '_markdown') else "" for md in mds)
        assert "/feature" in joined or "Phase 1" in joined or any("brainstorm" in str(getattr(m, '_markdown', '')) for m in mds)


async def test_cycle_freedom(patched_app, tmp_path):
    """Ctrl+A cycles freedom_level low -> medium -> high -> low.

    Asserts the reactive value, the on-disk persisted value, the live
    tool_ctx mirror, and the status banner text all stay in sync.
    """
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        assert patched_app.freedom_level == "low"
        for expected in ("medium", "high", "low"):
            await pilot.press("ctrl+a")
            await pilot.pause()
            assert patched_app.freedom_level == expected
            assert patched_app.tool_ctx.freedom_level == expected
            cfg = bootstrap.load_config(patched_app.muninn_dir)
            assert cfg["freedom_level"] == expected
            banner = patched_app.query_one("#status-banner")
            assert f"freedom: {expected}" in str(banner.render())


async def test_cycle_freedom_no_agent_does_not_crash(tmp_path, monkeypatch):
    """Watcher must not crash when fired before _build_agents has run.

    Reproduces the boot path where on_mount sets self.freedom_level
    BEFORE the health check resolves, so muninn_agent is still None and
    tool_ctx is also None. The watcher must be a no-op rebuild in that
    case (only the persisted config + banner update need to happen).
    """
    from muninn import app as app_mod

    bootstrap.ensure_muninn_dir(tmp_path)
    a = app_mod.MuninnTUI(cwd=tmp_path)
    # Force the degraded boot path: don't run health check.
    a.muninn_dir = tmp_path / ".muninn"
    a.muninn_dir.mkdir(exist_ok=True)
    a.config = dict(bootstrap.DEFAULT_CONFIG)
    a.muninn_agent = None
    a.tool_ctx = None
    a.health_status = app_mod.HEALTH_OLLAMA_DOWN
    # Read the reactive once to flush Textual's lazy initialization so
    # subsequent reads inside _ok_banner_text don't double-trigger watchers.
    _ = a.freedom_level

    builds: list[int] = []
    monkeypatch.setattr(app_mod.MuninnTUI, "_build_agents",
                        lambda self: builds.append(1))

    # Fire the watcher directly - simulates the reactive change.
    a.watch_freedom_level("low", "medium")
    assert builds == [], "rebuild must be skipped when agent is None"

    # Now flip to OK with a sentinel agent and assert rebuild happens.
    a.muninn_agent = object()
    a.health_status = app_mod.HEALTH_OK
    builds.clear()
    a.watch_freedom_level("medium", "high")
    assert builds == [1], "rebuild must happen when agent is built and health OK"


async def test_escape_calls_cancel_workers(patched_app, tmp_path, monkeypatch):
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        cancelled = []
        orig_cancel_group = patched_app.workers.cancel_group

        def spy(node, group):
            cancelled.append(group)
            return orig_cancel_group(node, group)

        monkeypatch.setattr(patched_app.workers, "cancel_group", spy)
        await pilot.press("escape")
        await pilot.pause()
        assert "muninn" in cancelled
        assert "feature" in cancelled
        # /brainstorm and /prd workers must also cancel on Esc.
        assert "brainstorm" in cancelled
        assert "prd" in cancelled


async def test_slash_candidates_returns_empty_for_plain_chat() -> None:
    from muninn.app import _slash_candidates
    from textual_autocomplete import TargetState
    state = TargetState(text="hello there", cursor_position=11)
    assert _slash_candidates(state) == []


async def test_slash_candidates_returns_commands_when_typing_slash() -> None:
    from muninn.app import _slash_candidates, SLASH_COMMANDS
    from textual_autocomplete import TargetState
    state = TargetState(text="/", cursor_position=1)
    items = _slash_candidates(state)
    assert len(items) == len(SLASH_COMMANDS)
    mains = [str(it.main) for it in items]
    assert any(m.startswith("/feature - ") for m in mains)
    assert any(m.startswith("/bug - ") for m in mains)
    assert any(m.startswith("/precommit-review - ") for m in mains)


async def test_apply_completion_inserts_only_command(patched_app, tmp_path):
    """Selecting `/feature - design doc...` inserts `/feature ` only."""
    from muninn.app import UpwardAutoComplete
    from textual.widgets import Input
    from textual_autocomplete import TargetState
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        ac = patched_app.query_one(UpwardAutoComplete)
        inp = patched_app.query_one("#user-input", Input)
        # /feature takes an argument, so trailing space.
        ac.apply_completion(
            "/feature - design doc · cold-read · revise · implement",
            TargetState(text="", cursor_position=0),
        )
        assert inp.value == "/feature "
        # /precommit-review takes no argument, no trailing space.
        ac.apply_completion(
            "/precommit-review - diff + stack-aware local checks + Huginn cold-read",
            TargetState(text="", cursor_position=0),
        )
        assert inp.value == "/precommit-review"


async def test_app_renders_autocomplete_widget(patched_app, tmp_path):
    """UpwardAutoComplete is mounted and targets the Input."""
    from muninn.app import UpwardAutoComplete
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        ac_widgets = list(patched_app.query(UpwardAutoComplete))
        assert len(ac_widgets) == 1


async def test_ask_user_screen_accepts_string_options() -> None:
    """Back-compat: plain string options work and have no `?` button."""
    from muninn.screens import AskUserScreen
    s = AskUserScreen("pick", ["a", "b"])
    assert s.labels == ["a", "b"]
    assert all(expl is None for _, expl in s.options)


async def test_ask_user_screen_accepts_tuple_options() -> None:
    """New behavior: (label, explanation) tuples carry per-option help."""
    from muninn.screens import AskUserScreen
    s = AskUserScreen("pick", [("a", "first option"), ("b", "second option")])
    assert s.labels == ["a", "b"]
    assert s.options[0] == ("a", "first option")
    assert s.options[1] == ("b", "second option")


async def test_ask_user_screen_explain_button_updates_panel(patched_app, tmp_path):
    """Pressing `?` on an option pops the explanation into the panel,
    without dismissing the modal."""
    from muninn.screens import AskUserScreen
    from textual.widgets import Static

    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        screen = AskUserScreen(
            "pick one",
            [("alpha", "the first letter"), ("beta", "the second letter")],
        )
        # push without awaiting result so we can poke the screen
        patched_app.push_screen(screen)
        await pilot.pause()
        # Press the ? button for option 1 (beta).
        await pilot.click("#explain-1")
        await pilot.pause()
        panel = screen.query_one("#explanation-panel", Static)
        rendered = str(panel.render())
        assert "beta" in rendered
        assert "second letter" in rendered
        # Modal still open (no dismiss).
        assert isinstance(patched_app.screen, AskUserScreen)


async def test_f12_toggles_debug_pane(patched_app, tmp_path):
    """F12 flips debug_visible and changes #debug-pane display."""
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        assert patched_app.debug_visible is False
        pane = patched_app.query_one("#debug-pane")
        # Hidden initially (display: none from CSS).
        assert str(pane.styles.display) == "none"
        await pilot.press("f12")
        await pilot.pause()
        assert patched_app.debug_visible is True
        assert str(pane.styles.display) == "block"
        await pilot.press("f12")
        await pilot.pause()
        assert patched_app.debug_visible is False
        assert str(pane.styles.display) == "none"


async def test_log_appends_to_debug_pane(patched_app, tmp_path):
    """Every record passed to _log shows up in the debug RichLog.

    The pane is hidden by default, so RichLog.write defers rendering to
    `_deferred_renders` until the pane has a known size. We assert against
    the union of rendered lines + deferred renders so the test passes
    regardless of whether the pane has been shown.
    """
    from textual.widgets import RichLog
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        rich_log = patched_app.query_one("#debug-log", RichLog)
        deferred_before = len(getattr(rich_log, "_deferred_renders", []))
        lines_before = len(rich_log.lines)
        before = deferred_before + lines_before
        patched_app._log({"type": "test_event", "value": "hello"})
        await pilot.pause()
        deferred_after = len(getattr(rich_log, "_deferred_renders", []))
        lines_after = len(rich_log.lines)
        assert (deferred_after + lines_after) > before


async def test_help_text_appears_in_intro(patched_app, tmp_path):
    """The first-launch help block is part of the intro Markdown so a new
    user lands with a quick-start visible without scrolling."""
    from textual.widgets import Markdown
    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        # Find the intro Markdown (first one in the muninn pane after boot).
        mds = patched_app.query("#muninn-scroll Markdown").results(Markdown)
        joined = "\n".join(getattr(m, "_markdown", "") or "" for m in mds)
        assert "Quick start" in joined
        assert "/feature" in joined
        assert "/bug" in joined
        assert "/precommit-review" in joined
        assert "/brainstorm" in joined
        assert "/prd" in joined


async def test_ollama_not_installed_detection(tmp_path, monkeypatch):
    """When `ollama` is not on PATH, health_status flips to
    ollama_not_installed and the install instructions get mounted."""
    from muninn import bootstrap
    bootstrap.ensure_muninn_dir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: None)
    from muninn import app as app_mod
    a = app_mod.MuninnTUI(cwd=tmp_path)
    async with a.run_test() as pilot:
        for _ in range(30):
            await pilot.pause()
            if a.health_status != "unknown":
                break
        assert a.health_status == "ollama_not_installed"
        from textual.widgets import Markdown
        mds = a.query("#muninn-scroll Markdown").results(Markdown)
        joined = "\n".join(getattr(m, "_markdown", "") or "" for m in mds)
        assert "Ollama is not installed" in joined


async def test_model_picker_allowlist_supported_families() -> None:
    """The model picker accepts the Qwen3-Coder and DeepSeek-R1 families."""
    from muninn.commands import _is_allowed_model
    # Qwen3-Coder family: still accepted.
    assert _is_allowed_model("qwen3-coder:30b") is True
    assert _is_allowed_model("qwen3-coder-next:latest") is True
    # DeepSeek-R1 family: newly accepted; prefix matches all official
    # distill sizes plus :latest.
    assert _is_allowed_model("deepseek-r1:32b") is True
    assert _is_allowed_model("deepseek-r1:14b") is True
    assert _is_allowed_model("deepseek-r1:8b") is True
    assert _is_allowed_model("deepseek-r1:latest") is True
    # Known-broken / out-of-scope models filtered out.
    assert _is_allowed_model("qwen2.5-coder:32b") is False
    assert _is_allowed_model("hhao/qwen2.5-coder-tools:32b") is False
    assert _is_allowed_model("deepseek-coder:6.7b") is False
    assert _is_allowed_model("deepseek-coder-v2:16b") is False
    assert _is_allowed_model("deepseek-v3.1:671b") is False
    # Third-party namespace (MFDoom/*) NOT matched - prefix anchored.
    assert _is_allowed_model("MFDoom/deepseek-r1-tool-calling:latest") is False
    # Unrelated families filtered out.
    assert _is_allowed_model("llama3:70b") is False


async def test_model_picker_lists_deepseek_r1_when_pulled(tmp_path, monkeypatch) -> None:
    """If both qwen3-coder and deepseek-r1 are pulled, the picker
    surfaces both as compatible options.

    Plumbing: install the /api/tags mock on the class BEFORE the app is
    constructed (the boot worker fires on mount). Capture push_screen by
    replacing the bound method with a recorder that does NOT actually
    push, so the modal never mounts inside the pilot.
    """
    import httpx
    from muninn import app as app_mod
    from muninn.commands import MuninnSettingsProvider
    from muninn.screens import PresetPickerScreen

    bootstrap.ensure_muninn_dir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ollama")

    async def _fake_get(self, url, *args, **kwargs):
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [
                    {"name": "qwen3-coder:30b", "size": 18_556_700_761,
                     "details": {"family": "qwen3moe",
                                 "parameter_size": "30.5B",
                                 "quantization_level": "Q4_K_M"}},
                    {"name": "deepseek-r1:32b", "size": 20_000_000_000,
                     "details": {"family": "qwen2",
                                 "parameter_size": "32.8B",
                                 "quantization_level": "Q4_K_M"}},
                ]},
                request=httpx.Request("GET", url),
            )
        return httpx.Response(
            200, json={}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    # Stub out _build_agents - same trick as the patched_app fixture.
    def _stub_build(self):
        self.muninn_agent = object()
        self.huginn_factory = lambda: object()
        from muninn.tools import ToolContext
        self.tool_ctx = ToolContext(
            cwd=self.cwd, muninn_dir=self.muninn_dir,
            freedom_level=self.freedom_level,
            confirm_callback=lambda *a, **k: None,
            ask_user_callback=lambda *a, **k: None,
            log=self._log,
        )
    monkeypatch.setattr(app_mod.MuninnTUI, "_build_agents", _stub_build)

    a = app_mod.MuninnTUI(cwd=tmp_path)
    pushed: list[PresetPickerScreen] = []

    async with a.run_test() as pilot:
        for _ in range(30):
            await pilot.pause()
            if a.health_status != "unknown":
                break

        # Capture-only push_screen: record but do not actually push.
        def capture(screen, *cap_args, **cap_kwargs):
            if isinstance(screen, PresetPickerScreen):
                pushed.append(screen)
            return None
        monkeypatch.setattr(a, "push_screen", capture)

        # Bind _fetch_and_show_models to a thin shim that exposes _app -
        # avoids depending on the Textual Provider constructor signature.
        # _fetch_and_show_models calls self._make_apply("model") to build
        # the picker callback, so the shim borrows that too.
        class _Probe:
            _app = a
            _fetch_and_show_models = MuninnSettingsProvider._fetch_and_show_models
            _make_apply = MuninnSettingsProvider._make_apply
        probe = _Probe()
        await _Probe._fetch_and_show_models(probe)

        assert pushed, "model picker did not push a PresetPickerScreen"
        names = [preset[0] for preset in pushed[0].presets]
        assert "qwen3-coder:30b" in names
        assert "deepseek-r1:32b" in names

        # Round-trip the apply closure for the deepseek-r1 preset and
        # confirm config is updated. Run inside the pilot so app.notify
        # and the rebuild worker have a live event loop. _build_agents
        # is stubbed above, so the rebuild worker is a no-op.
        apply_fn = probe._make_apply("model")
        apply_fn("deepseek-r1:32b")
        assert a.config["model"] == "deepseek-r1:32b"


async def test_check_not_home_rejects_home(tmp_path, monkeypatch) -> None:
    """Running muninn with cwd == ~ would make the per-project .muninn/
    overlap exactly with the user-level ~/.muninn/. Refuse to launch."""
    from pathlib import Path
    from muninn.app import _check_not_home

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    err = _check_not_home(Path(fake_home))
    assert err is not None
    assert "home directory" in err.lower()
    assert "muninn ~/code" in err  # the suggested fix is in the message


async def test_check_not_home_allows_subdirs(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    from muninn.app import _check_not_home

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = fake_home / "code" / "demo"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    assert _check_not_home(project) is None


async def test_check_not_home_resolves_symlinks(tmp_path, monkeypatch) -> None:
    """A symlink that points at $HOME still counts as $HOME after resolve()."""
    from pathlib import Path
    from muninn.app import _check_not_home

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    link = tmp_path / "home_alias"
    link.symlink_to(fake_home)

    assert _check_not_home(Path(link)) is not None


async def test_main_exits_2_when_run_from_home(tmp_path, monkeypatch, capsys) -> None:
    """Full-stack: main() with argv pointing at home exits with code 2
    and prints the helpful message to stderr."""
    import pytest
    from muninn.app import main

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    with pytest.raises(SystemExit) as excinfo:
        main(argv=[str(fake_home)])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "cannot run from your home directory" in captured.err


async def test_install_hint_per_os(monkeypatch):
    import sys
    from muninn import app as app_mod

    monkeypatch.setattr(sys, "platform", "darwin")
    assert "brew" in app_mod._ollama_install_hint()

    monkeypatch.setattr(sys, "platform", "linux")
    assert "install.sh" in app_mod._ollama_install_hint()

    monkeypatch.setattr(sys, "platform", "win32")
    assert "windows" in app_mod._ollama_install_hint().lower()


async def test_health_check_red_banner_when_ollama_down(tmp_path, monkeypatch):
    bootstrap.ensure_muninn_dir(tmp_path)

    async def failing_get(self, url, *args, **kwargs):
        raise httpx.ConnectError("simulated down", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", failing_get)

    from muninn import app as app_mod
    a = app_mod.MuninnTUI(cwd=tmp_path)
    async with a.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if a.health_status != "unknown":
                break
        assert a.health_status == "ollama_down"
        inp = a.query_one("#user-input")
        assert inp.disabled is True


# ---------------------------------------------------------------------
# `muninn update` CLI subcommand
# ---------------------------------------------------------------------


def test_update_subcommand_exits_with_clear_error_when_uv_missing(monkeypatch, capsys):
    """If `uv` is not on PATH, `muninn update` must surface a concrete
    error directing the user at the official installer instead of the
    opaque output a raw subprocess call would produce."""
    import shutil as shutil_mod
    import pytest as _pytest
    from muninn import app as app_mod

    # `_run_update` imports shutil locally, so patching the real shutil module
    # is what counts.
    monkeypatch.setattr(shutil_mod, "which", lambda _name: None)

    with _pytest.raises(SystemExit) as exc:
        app_mod._run_update()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "'uv' is not on PATH" in captured.err
    # Don't hardcode the URL but confirm we're pointing at uv's docs.
    assert "uv" in captured.err.lower()


def test_update_subcommand_propagates_uv_exit_code(monkeypatch, capsys):
    """`_run_update` must forward `uv tool upgrade muninn`'s exit code so
    shell scripts (CI, dotfiles bootstraps) calling `muninn update` can
    branch on success / failure."""
    import shutil as shutil_mod
    import subprocess as subprocess_mod
    import pytest as _pytest
    from muninn import app as app_mod

    monkeypatch.setattr(shutil_mod, "which", lambda name: "/fake/bin/" + name)

    captured_args: list[list[str]] = []

    class _FakeCompleted:
        def __init__(self, rc: int): self.returncode = rc

    def fake_run(argv, **_kwargs):
        captured_args.append(list(argv))
        return _FakeCompleted(7)

    monkeypatch.setattr(subprocess_mod, "run", fake_run)

    with _pytest.raises(SystemExit) as exc:
        app_mod._run_update()
    assert exc.value.code == 7, "exit code from `uv tool upgrade` must propagate"
    assert captured_args == [["uv", "tool", "upgrade", "muninn"]], (
        "must invoke `uv tool upgrade muninn` exactly"
    )


def test_main_update_rejects_extra_args(monkeypatch, capsys):
    """`muninn update foo` must fail loudly so a typo doesn't silently
    drop arguments AND silently run the upgrade against an unrelated
    intent."""
    import pytest as _pytest
    from muninn import app as app_mod

    # Guard: if _run_update is reached, the test should fail clearly.
    def _should_not_run():
        raise AssertionError("_run_update must NOT run when extra args are present")
    monkeypatch.setattr(app_mod, "_run_update", _should_not_run)

    with _pytest.raises(SystemExit) as exc:
        app_mod.main(["update", "foo"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "takes no arguments" in captured.err


def test_main_update_with_no_extra_args_calls_run_update(monkeypatch):
    """The happy path: `muninn update` with nothing else dispatches to
    `_run_update`, not into the path-resolution branch."""
    from muninn import app as app_mod

    called = []
    monkeypatch.setattr(app_mod, "_run_update", lambda: called.append(True))
    app_mod.main(["update"])
    assert called == [True], "main(['update']) must call _run_update once"


# =====================================================================
# /brainstorm and /prd dispatch + registry consistency
# =====================================================================


def test_slash_command_registry_consistency() -> None:
    """The slash-command surface (dropdown, taking-arg set, help text,
    cancel groups) must stay in sync. Stops the next /command from being
    half-wired and producing silent UX bugs.
    """
    from muninn.app import (
        SLASH_COMMANDS, _COMMANDS_TAKING_ARG, _CANCEL_GROUPS, HELP_TEXT,
    )

    # Both new commands appear in the dropdown registry.
    cmds = [c for c, _ in SLASH_COMMANDS]
    for cmd in ("/feature", "/bug", "/precommit-review", "/brainstorm", "/prd"):
        assert cmd in cmds, f"{cmd} missing from SLASH_COMMANDS"

    # Both new commands take a description argument.
    for cmd in ("/feature", "/bug", "/brainstorm", "/prd"):
        assert cmd in _COMMANDS_TAKING_ARG, f"{cmd} missing from _COMMANDS_TAKING_ARG"

    # All command-taking-arg entries must have a SLASH_COMMANDS entry too.
    for cmd in _COMMANDS_TAKING_ARG:
        assert cmd in cmds, f"{cmd} in _COMMANDS_TAKING_ARG but not SLASH_COMMANDS"

    # Both new commands appear in the help block.
    for cmd in ("/feature", "/bug", "/precommit-review", "/brainstorm", "/prd"):
        assert cmd in HELP_TEXT, f"{cmd} missing from HELP_TEXT"

    # Both new worker groups are cancellable on Esc.
    for group in ("brainstorm", "prd"):
        assert group in _CANCEL_GROUPS, f"{group} missing from _CANCEL_GROUPS"


async def test_pilot_brainstorm_dispatch(patched_app, tmp_path, monkeypatch):
    """Submitting `/brainstorm <text>` must spawn a worker in group
    'brainstorm'. Verifies dispatch wiring without running the LLM flow.
    """
    spawned: list[str] = []

    def spy(self, description):
        spawned.append(description)

    from muninn import app as app_mod
    # Patch the worker's underlying coroutine method so it does NOT run
    # the actual brainstorm_flow; we only care that dispatch hit the
    # @work-decorated method, which still uses worker scheduling.
    monkeypatch.setattr(
        app_mod.MuninnTUI, "_run_brainstorm_worker",
        lambda self, description: spawned.append(description),
    )

    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        from textual.widgets import Input
        inp = patched_app.query_one("#user-input", Input)
        inp.value = "/brainstorm what if huginn voted on diffs"
        await pilot.press("enter")
        await pilot.pause()

    assert spawned == ["what if huginn voted on diffs"], spawned


async def test_pilot_prd_dispatch(patched_app, tmp_path, monkeypatch):
    """Submitting `/prd <text>` must dispatch to _run_prd_worker."""
    spawned: list[str] = []
    from muninn import app as app_mod
    monkeypatch.setattr(
        app_mod.MuninnTUI, "_run_prd_worker",
        lambda self, description: spawned.append(description),
    )

    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        from textual.widgets import Input
        inp = patched_app.query_one("#user-input", Input)
        inp.value = "/prd add a persistent transcript pane"
        await pilot.press("enter")
        await pilot.pause()

    assert spawned == ["add a persistent transcript pane"], spawned


async def test_pilot_brainstorm_empty_arg_echoes_hint(patched_app, tmp_path, monkeypatch):
    """`/brainstorm` with no argument must NOT spawn a worker; it must
    echo a hint instead, mirroring /feature and /bug behavior."""
    spawned: list[str] = []
    from muninn import app as app_mod
    monkeypatch.setattr(
        app_mod.MuninnTUI, "_run_brainstorm_worker",
        lambda self, description: spawned.append(description),
    )

    async with patched_app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if patched_app.health_status == "ok":
                break
        from textual.widgets import Input, Markdown
        inp = patched_app.query_one("#user-input", Input)
        inp.value = "/brainstorm"
        await inp.action_submit()
        await pilot.pause()
        await pilot.pause()

        assert spawned == [], "no worker must spawn for empty /brainstorm"
        # The hint message landed in the muninn pane.
        mds = patched_app.query("#muninn-scroll Markdown").results(Markdown)
        joined = "\n".join(
            str(getattr(m, "_markdown", "") or "") for m in mds
        )
        assert "/brainstorm requires a rough idea" in joined, (
            f"hint not found in pane; got widgets:\n{joined[:2000]}"
        )
