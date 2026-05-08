"""Command-palette providers for runtime config tweaks.

Surfaced via Ctrl+P in the TUI. Two top-level entries: one for `num_ctx`,
one for `max_revision_rounds`. Selecting either pushes a `PresetPickerScreen`
modal so the user can drill into preset values without cluttering the palette.
"""
from __future__ import annotations

from textual.command import DiscoveryHit, Hit, Hits, Provider

from . import bootstrap
from .screens import PresetPickerScreen


_MODEL_TITLE = "model · Ollama tag"
_MODEL_HELP = (
    "Switch which Ollama tag Muninn / Huginn talk to. Lists pulled tags that "
    "muninn currently considers compatible (Qwen3-Coder and DeepSeek-R1 "
    "families). To use a different tag, pull it then edit "
    "`.muninn/config.toml` directly."
)


# Compatibility allowlist for the model picker. Only tags whose name starts
# with one of these prefixes are surfaced in the dropdown. The promotion
# gate is dual: a family is allowlisted if it is EITHER curl-verified
# locally to emit OpenAI-compat tool_calls via /v1/chat/completions OR
# Ollama's library page advertises both `tools` and `thinking` capability
# tags for it. Currently allowlisted:
#   - qwen3-coder: curl-verified locally.
#   - deepseek-r1: capability-tag advertised (Ollama library lists
#     `tools thinking` for every distill 1.5b through 70b plus 671b cloud).
#     Streaming is the OpenAI-compat default. Users can curl-verify a
#     specific tag via the snippet in SETUP.md before relying on it.
# Extend this tuple as more families pass either gate.
_ALLOWED_MODEL_PREFIXES: tuple[str, ...] = (
    "qwen3-coder",
    "deepseek-r1",
)


def _is_allowed_model(name: str) -> bool:
    return any(name.startswith(p) for p in _ALLOWED_MODEL_PREFIXES)

_NUM_CTX_TITLE = "num_ctx · Ollama context window"
_NUM_CTX_HELP = (
    "How many tokens of conversation Muninn / Huginn can fit in a single "
    "model call. Higher = more headroom for long /feature flows; trades "
    "RAM/VRAM for KV cache."
)

_REVISIONS_TITLE = "max revisions · /feature backstop cap"
_REVISIONS_HELP = (
    "How many design -> Huginn revision rounds /feature runs before stopping "
    "and asking how to proceed. Higher = more chances for the design to "
    "converge; lower = faster but more likely to escalate to the user."
)

_FREEDOM_TITLE = "freedom · agent autonomy level"
_FREEDOM_HELP = (
    "How much Muninn decides on its own. low = confirm every shell/write "
    "and bias toward ask_user. medium = auto-allow read-only shell, gate "
    "writes, decide routine ambiguity. high = autonomous: skip the "
    "/feature and /bug non-convergence backstop and run through to "
    "implementation. Mirrors Ctrl+A which cycles low/medium/high."
)


class MuninnSettingsProvider(Provider):
    """Single unified palette provider for muninn.

    Replaces Textual's default SystemCommandsProvider so we control the
    listing order across muninn settings AND system commands. Order:

      1. muninn settings (model / num_ctx / max revisions) - top, since
         they're what the user reaches for most often during a session
      2. other system commands (Theme, Keys, Maximize, Screenshot)
      3. Quit - pinned to the bottom so it's not accidentally hit while
         scrolling the palette

    `App.COMMANDS = {MuninnSettingsProvider}` (drops the default
    SystemCommandsProvider from the App-level set).
    """

    @property
    def _app(self):
        return self.screen.app

    def _muninn_entries(self) -> list[tuple[str, str, callable]]:
        return [
            (_MODEL_TITLE, _MODEL_HELP, self._open_model_picker),
            (_FREEDOM_TITLE, _FREEDOM_HELP, self._open_freedom_picker),
            (_NUM_CTX_TITLE, _NUM_CTX_HELP, self._open_num_ctx_picker),
            (_REVISIONS_TITLE, _REVISIONS_HELP, self._open_revisions_picker),
        ]

    def _system_entries(self) -> list[tuple[str, str, callable]]:
        """Pull Textual's system commands via App.get_system_commands and
        re-emit them in the order we want, with Quit pinned last."""
        try:
            cmds = list(self._app.get_system_commands(self._app.screen))
        except Exception:
            return []
        non_quit = [
            (c.title, c.help, c.callback)
            for c in cmds if c.title != "Quit"
        ]
        quit_cmd = next((c for c in cmds if c.title == "Quit"), None)
        result = list(non_quit)
        if quit_cmd is not None:
            result.append((quit_cmd.title, quit_cmd.help, quit_cmd.callback))
        return result

    def _all_entries(self) -> list[tuple[str, str, callable]]:
        return self._muninn_entries() + self._system_entries()

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for display, help_text, callback in self._all_entries():
            score = matcher.match(display)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(display),
                    callback,
                    text=display,
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for display, help_text, callback in self._all_entries():
            yield DiscoveryHit(
                display,
                callback,
                text=display,
                help=help_text,
            )

    # ---- callbacks ------------------------------------------------------

    def _open_model_picker(self) -> None:
        # Fetching pulled models requires an HTTP call to /api/tags; do it
        # in a worker so the palette doesn't block while we wait.
        app = self._app
        app.run_worker(self._fetch_and_show_models(), exclusive=True, group="model-picker")

    async def _fetch_and_show_models(self) -> None:
        app = self._app
        base_url = (app.config.get("base_url") or "").rstrip("/v1").rstrip("/")
        url = base_url + "/api/tags"
        try:
            resp = await app.http_client.get(url, timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
        except Exception as e:
            app.notify(
                f"Failed to list Ollama models: {e}",
                severity="error", timeout=5,
            )
            return
        if not models:
            app.notify(
                "No Ollama models found. Run `ollama pull <tag>` and try again.",
                severity="warning", timeout=5,
            )
            return

        # Filter to compatibility allowlist. Hides tags we know don't work
        # (qwen2.5-coder family) and tags we haven't verified, so testers
        # don't pick a model whose tool calls land as plain text.
        compatible = [m for m in models if _is_allowed_model(m.get("name", ""))]
        if not compatible:
            app.notify(
                "No compatible Ollama models found. Pull `qwen3-coder:30b` "
                "(or `deepseek-r1:32b`) and retry.",
                severity="warning", timeout=6,
            )
            return

        current = app.config.get("model")
        presets: list[tuple[object, str, str]] = []
        for m in sorted(compatible, key=lambda m: m.get("name", "")):
            name = m.get("name", "")
            size_bytes = int(m.get("size", 0) or 0)
            size_gb = size_bytes / (1024 ** 3) if size_bytes else 0.0
            details = m.get("details") or {}
            family = details.get("family", "?")
            param = details.get("parameter_size", "?")
            quant = details.get("quantization_level", "?")
            desc = (
                f"{size_gb:.1f} GB · {family} · {param} parameters · "
                f"{quant} quantization. Switching applies the new tag to "
                f"both Muninn and Huginn agents on the next turn."
            )
            presets.append((name, name, desc))
        screen = PresetPickerScreen(
            title="Set model",
            subtitle="Compatible Ollama tags pulled locally (Qwen3-Coder and DeepSeek-R1)",
            presets=presets,
            current=current,
        )
        app.push_screen(screen, callback=self._make_apply("model"))

    def _open_num_ctx_picker(self) -> None:
        app = self._app
        current = int(app.config.get("num_ctx", 0))
        presets = [
            (value, f"{value:>6}     {blurb.split(' - ', 1)[0]}", blurb)
            for value, blurb in bootstrap.NUM_CTX_PRESETS
        ]
        screen = PresetPickerScreen(
            title="Set num_ctx",
            subtitle="Ollama context window size in tokens",
            presets=presets,
            current=current,
        )
        app.push_screen(screen, callback=self._make_apply("num_ctx"))

    def _open_freedom_picker(self) -> None:
        app = self._app
        current = str(app.config.get("freedom_level", "low"))
        screen = PresetPickerScreen(
            title="Set freedom level",
            subtitle="Agent autonomy + tool-confirm policy",
            presets=list(bootstrap.FREEDOM_LEVEL_PRESETS),
            current=current,
        )
        app.push_screen(screen, callback=self._apply_freedom)

    def _apply_freedom(self, value: object) -> None:
        """Apply a freedom_level pick by writing the reactive on the app.

        Writing the reactive triggers watch_freedom_level, which is the
        single place that mutates tool_ctx, persists config, refreshes
        the banner, and rebuilds the muninn agent. We deliberately do
        NOT use the generic _make_apply path here: that path writes
        cfg dict only and would bypass the watcher.
        """
        if value is None:
            return
        level = str(value)
        if level not in {"low", "medium", "high"}:
            return
        app = self._app
        if app.freedom_level != level:
            app.freedom_level = level  # fires watch_freedom_level
        app.notify(f"set freedom_level = {level}",
                   severity="information", timeout=3)
        try:
            app._log({"type": "config_changed",
                      "key": "freedom_level", "value": level})
        except Exception:
            pass

    def _open_revisions_picker(self) -> None:
        app = self._app
        current = int(app.config.get("max_revision_rounds", 0))
        presets = [
            (
                value,
                f"{value} round{'s' if value != 1 else ''}",
                f"/feature stops after {value} revision round"
                f"{'s' if value != 1 else ''} and asks the user how to proceed",
            )
            for value in bootstrap.MAX_REVISION_ROUNDS_PRESETS
        ]
        screen = PresetPickerScreen(
            title="Set max revisions",
            subtitle="/feature backstop cap (design -> huginn rounds)",
            presets=presets,
            current=current,
        )
        app.push_screen(screen, callback=self._make_apply("max_revision_rounds"))

    def _make_apply(self, key: str):
        app = self._app

        def _apply(value: object) -> None:
            if value is None:
                return  # cancelled, no change
            cfg = dict(app.config)
            cfg[key] = value
            app.config = cfg
            try:
                bootstrap.save_config(app.muninn_dir, cfg)
            except Exception:
                pass
            app.notify(f"set {key} = {value}", severity="information", timeout=3)
            app._log({"type": "config_changed", "key": key, "value": value})
            # Refresh the top status banner so the user sees the new value
            # without waiting for the next health check.
            try:
                app.refresh_status_banner()
            except Exception:
                pass
            # Model change: rebuild the agents so the next turn uses the new
            # tag. Other settings (num_ctx, max_revision_rounds) are read at
            # call-site each turn, no rebuild needed.
            if key == "model" and hasattr(app, "_rebuild_agents_for_new_model"):
                try:
                    app.run_worker(
                        app._rebuild_agents_for_new_model(),
                        exclusive=True,
                        group="agent-rebuild",
                    )
                except Exception:
                    pass

        return _apply
