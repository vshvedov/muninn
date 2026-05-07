"""Muninn CLI - TUI shell.

Phase 1 surface:
- Two-pane layout (Muninn left, Huginn right) with dynamically mounted Markdown widgets.
- Free-chat with Muninn (tools available, freedom_level governs tool gates).
- /feature <description> - design doc -> Huginn cold-read -> revise (up to N rounds) -> implement.
- /precommit-review - gather diff + run local checks + one Huginn cold-read of pending changes.
- Esc cancels in-flight workers; Ctrl+A cycles freedom level low -> medium -> high; Ctrl+C quits.
- Ollama health check on startup; banner explains the failure mode.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Markdown, RichLog, Static
from textual.geometry import Offset, Region, Spacing
from textual_autocomplete import AutoComplete, DropdownItem, TargetState


class UpwardAutoComplete(AutoComplete):
    """AutoComplete variant that opens the dropdown ABOVE the target Input.

    The default `_align_to_target` places the dropdown one row below the
    cursor (`y + 1`); since muninn's Input is docked at the bottom of the
    pane, that direction collides with the Footer and gets clipped. We
    align the dropdown's BOTTOM edge with the cursor row instead by
    computing `y - height`, then let Textual's `constrain` keep it on
    screen when it would otherwise extend above the top.
    """

    def _align_to_target(self) -> None:
        x, y = self.target.cursor_screen_offset
        width, height = self.option_list.outer_size
        x, y, _w, _h = Region(x - 1, y - height, width, height).constrain(
            "inside",
            "none",
            Spacing.all(0),
            self.screen.scrollable_content_region,
        )
        self.absolute_offset = Offset(x, y)

    def apply_completion(self, value: str, state: TargetState) -> None:
        """Insert ONLY the command part of "/cmd - description".

        The dropdown line shows the description for visibility, but we never
        want it in the Input. Split on " - " (first occurrence), then append
        a trailing space for commands that expect an argument so the user can
        keep typing the description directly.
        """
        cmd = value.split(" - ", 1)[0]
        if cmd in _COMMANDS_TAKING_ARG:
            cmd = cmd + " "
        target = self.target
        target.value = ""
        target.insert_text_at_cursor(cmd)
        # Same post-update bookkeeping as the base class so the dropdown
        # rebuilds against the new state instead of leaving the old options
        # visible.
        new_target_state = self._get_target_state()
        self._target_state = new_target_state
        search_string = self.get_search_string(new_target_state)
        self._rebuild_options(new_target_state, search_string)

from . import bootstrap
from .agents import (
    compose_muninn_prompt,
    huginn_agent,
    make_local_model,
    make_provider,
    muninn_agent,
    num_ctx_settings,
)
from .commands import MuninnSettingsProvider
from .egress import EgressDenied, make_localhost_client
from .jsonl_log import JsonlLogger
from .screens import AskUserScreen, ConfirmScreen
from .streaming import run_and_stream
from .themes import ALL_THEMES, DEFAULT_THEME_NAME
from .tools import ToolContext, ToolError, make_tools
from .workflows import (
    BRAINSTORM_LENSES,
    PRD_LENSES,
    BrainstormRunCtx,
    BugRunCtx,
    FeatureRunCtx,
    PRDRunCtx,
    ReviewRunCtx,
    brainstorm_flow,
    bug_flow,
    feature_flow,
    prd_flow,
    precommit_review_flow,
)


# Worker groups cancelled on Esc / action_cancel_workers. Module-level
# constant so tests can import + assert membership without source AST
# inspection. Order is informational; cancellation is set-semantic.
_CANCEL_GROUPS: tuple[str, ...] = (
    "muninn", "feature", "bug", "review",
    "brainstorm", "prd",
    "huginn", "pull",
)


HEALTH_OK = "ok"
HEALTH_OLLAMA_NOT_INSTALLED = "ollama_not_installed"
HEALTH_OLLAMA_DOWN = "ollama_down"
HEALTH_MODEL_MISSING = "model_missing"
HEALTH_UNKNOWN = "unknown"


# Per-OS install instructions for Ollama. Surfaced when `which ollama`
# returns nothing on first launch.
OLLAMA_INSTALL_INSTRUCTIONS: dict[str, str] = {
    "darwin": (
        "Install with Homebrew:\n"
        "    brew install ollama\n"
        "Or download the macOS app from https://ollama.com/download/mac"
    ),
    "linux": (
        "Install with the upstream script:\n"
        "    curl -fsSL https://ollama.com/install.sh | sh\n"
        "Or use your distro's package manager (apt / dnf / pacman) if it ships ollama."
    ),
    "win32": (
        "Download the Windows installer from https://ollama.com/download/windows"
    ),
}


def _ollama_install_hint() -> str:
    import sys
    return OLLAMA_INSTALL_INSTRUCTIONS.get(
        sys.platform,
        "Visit https://ollama.com/download for installation instructions.",
    )


# First-launch help text: surfaced in the Muninn pane on every launch.
# Short by design - users can scroll up to re-read; the goal is "zero
# context cliff" so they know what typing /command vs plain text does.
HELP_TEXT = """\
## Quick start

**Just type** to chat with Muninn. Muninn has tools (read / write files,
run shell, ask you a question) and uses them when helpful. This is the
normal LLM-as-coding-assistant mode - no Huginn review.

**Slash commands** activate the elephant/goldfish pattern. Muninn drafts
the artifact; one or more fresh stateless **Huginns** cold-read it for
blind spots; Muninn revises (or synthesizes, for ideation flows); cycle
repeats until Huginn signs off (or you choose to proceed / cancel from
the modal). Five commands ship:

- `/feature <description>` - design a new feature, get cold-read, implement
- `/bug <symptom>` - diagnose a bug, write a failing test, fix it
- `/precommit-review` - cold-read pending diff with stack-aware local checks
- `/brainstorm <rough idea>` - 3 lens cold-reads (tech/contrarian/UX),
  synthesize, save to `docs/brainstorms/`
- `/prd <idea>` - structured Q&A, 3 research lenses
  (prior-art/edge-cases/integration), synthesize, save to `docs/prds/`

**Keys:**
- `Ctrl+P` palette: model / freedom / num_ctx / max revisions / theme
- `Ctrl+A` cycle freedom level: low (confirm everything) -> medium
  (auto-allow read-only shell) -> high (autonomous, skip backstop)
- `F12` toggle the live event log (debug pane)
- `Esc` cancel an in-flight workflow
- `Ctrl+R` retry the Ollama health check
"""


# Slash-command registry. Display in the dropdown is `<cmd> - <description>`.
# The command portion (left of " - ") is what gets inserted into the Input
# on completion; UpwardAutoComplete.apply_completion below splits on " - "
# and adds a trailing space for commands listed in _COMMANDS_TAKING_ARG so
# the user can immediately type the argument.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/feature", "design doc · cold-read · revise · implement"),
    ("/bug", "ground · problem doc · cold-read · failing test · fix"),
    ("/precommit-review", "diff + stack-aware local checks + Huginn cold-read"),
    ("/brainstorm", "ground · 3 lens cold-reads · synthesis · save to docs/brainstorms/"),
    ("/prd", "ground · structured Q&A · 3 research lenses · synthesis · save to docs/prds/"),
]

_COMMANDS_TAKING_ARG: frozenset[str] = frozenset({
    "/feature", "/bug", "/brainstorm", "/prd",
})


def _slash_candidates(state: TargetState) -> list[DropdownItem]:
    """Surface slash-command suggestions ONLY when the user starts a /command.

    Returning an empty list while the input is plain free-chat text keeps the
    dropdown hidden during normal Muninn conversation. textual-autocomplete's
    built-in fuzzy filter narrows the list as the user keeps typing.
    """
    text = state.text or ""
    if not text.startswith("/"):
        return []
    return [DropdownItem(main=f"{cmd} - {desc}")
            for cmd, desc in SLASH_COMMANDS]


BANNER_ASCII = r"""```
⠀⠀⣀⠠⣀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀
⣈⠒⠤⣍⡒⢍⢦⢑⢤⡀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡈⡆
⠢⣭⣛⣚⣪⠍⢶⡝⡹⣯⣦⣱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⣇⡿
⢲⣾⣵⡖⢡⣭⣬⣶⣬⡞⢹⣿⣾⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⢸⣷⠃
⠀⢤⣷⠞⠓⣊⣕⠭⣮⣧⣵⣿⣿⣿⣷⡀⠀⠀⠀⠀⢀⣠⢤⣺⣮⣿⠏⡠
⠀⠀⠈⠛⣿⣗⣯⣶⣿⣿⣿⣿⣿⣿⣿⣿⣶⣠⣤⡶⢾⡼⡾⠽⠟⠃⠁⠀
⠀⠀⠀⠀⠀⢺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣷⣿⣦⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⣿⡿⣾⣻⣿⣿⣿⡿⠛⠛⠉⠓⠂⠀⠀          Muninn version 0.1.1
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣻⣿⣿⢚⣿⣿⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣿⣿⣿⣷⡟⢻⣿⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠤⢶⣻⣿⣿⣿⣿⣿⣿⡇⢱⢦⠀⠈⠯⠊⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣼⣿⡿⢿⣿⣿⣿⣿⣿⡇⠘⠸⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠐⠚⠉⣴⣿⣻⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠸⠛⣾⣿⡟⢹⡟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠁⠟
```
"""


class MuninnTUI(App):
    """Muninn TUI - two-pane Textual app for the elephant/goldfish workflow."""

    TITLE = "Muninn"
    SUB_TITLE = "Memory and Sight for Software Design"

    CSS = """
    Screen {
        layout: horizontal;
        layers: base overlay;
    }
    /* F12 debug pane: overlay layer so toggling display does not push the
       main panes around; sits above the right pane when visible. */
    #debug-pane {
        layer: overlay;
        dock: right;
        width: 50%;
        height: 1fr;
        background: $panel;
        border: tall $accent;
        padding: 1;
        display: none;
    }
    #debug-pane-header {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    #debug-log {
        height: 1fr;
        background: $surface;
        scrollbar-gutter: stable;
    }
    #panes {
        height: 1fr;
    }
    #muninn-pane {
        width: 65%;
        border-right: solid $accent;
    }
    #huginn-pane {
        width: 35%;
        background: $panel;
    }
    #muninn-scroll, #huginn-scroll {
        height: 1fr;
        scrollbar-gutter: stable;
        padding: 0 1;
    }
    #huginn-header, #muninn-header {
        content-align: center middle;
        padding: 0 1;
        height: 1;
        border-bottom: solid $primary;
    }
    #status-banner {
        dock: top;
        height: 1;
        content-align: left middle;
        padding: 0 1;
    }
    /* Status colors apply to both the top status banner AND the bottom
       Footer, so the chrome bars at top and bottom always match the
       current Ollama state. */
    .status-ok { background: $success; color: $text; }
    .status-warn { background: $warning; color: $text; }
    .status-err { background: $error; color: $text; }
    Footer.status-ok, Footer.status-ok FooterKey,
    Footer.status-ok FooterKey .footer-key--key,
    Footer.status-ok FooterKey .footer-key--description {
        background: $success;
        color: $text;
    }
    Footer.status-warn, Footer.status-warn FooterKey,
    Footer.status-warn FooterKey .footer-key--key,
    Footer.status-warn FooterKey .footer-key--description {
        background: $warning;
        color: $text;
    }
    Footer.status-err, Footer.status-err FooterKey,
    Footer.status-err FooterKey .footer-key--key,
    Footer.status-err FooterKey .footer-key--description {
        background: $error;
        color: $text;
    }
    #user-input {
        dock: bottom;
        margin-top: 1;
        margin-bottom: 1;
    }
    Markdown {
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+a", "cycle_freedom", "Freedom: low/med/high", priority=True),
        Binding("escape", "cancel_workers", "Cancel"),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+r", "retry_health", "Retry Ollama", priority=True),
        Binding("f12", "toggle_debug", "Debug log", priority=True),
    ]

    # Single unified palette provider. MuninnSettingsProvider re-emits
    # Textual's system commands itself (Theme, Keys, Maximize, Screenshot,
    # Quit) so we can control the cross-provider ordering: muninn
    # settings first, system commands middle, Quit last.
    COMMANDS: ClassVar[frozenset] = frozenset({MuninnSettingsProvider})

    freedom_level: reactive[str] = reactive("low")
    health_status: reactive[str] = reactive(HEALTH_UNKNOWN)
    health_detail: reactive[str] = reactive("")
    debug_visible: reactive[bool] = reactive(False)

    def __init__(self, *, cwd: Path | None = None) -> None:
        super().__init__()
        self.cwd = cwd or Path.cwd()
        self.muninn_dir: Path = self.cwd / ".muninn"
        self.config: dict = dict(bootstrap.DEFAULT_CONFIG)
        self.muninn_history: list = []
        self.http_client = make_localhost_client()
        self.logger: JsonlLogger | None = None
        self.tool_ctx: ToolContext | None = None
        self.muninn_agent = None  # set in on_mount
        self.huginn_factory: Callable | None = None

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("⏳ Booting...", id="status-banner")
        with Horizontal(id="panes"):
            with Vertical(id="muninn-pane"):
                yield Static("🐦‍⬛  [b]Muninn (memory)[/b]", id="muninn-header")
                yield VerticalScroll(id="muninn-scroll")
                yield Input(
                    placeholder=(
                        "Chat with Muninn · /feature · /bug · /precommit-review "
                        "· Ctrl+A: freedom · Esc: cancel"
                    ),
                    id="user-input",
                )
                # Slash-command dropdown. Targets #user-input by selector;
                # the candidates callback returns [] for plain free chat
                # so the popup only appears when the user types `/`.
                # Upward variant because the Input is docked at the bottom
                # of the pane.
                yield UpwardAutoComplete(
                    "#user-input",
                    candidates=_slash_candidates,
                )
            with Vertical(id="huginn-pane"):
                yield Static("👁️  [b]Huginn (cold-read)[/b]", id="huginn-header")
                yield VerticalScroll(id="huginn-scroll")
        # F12 debug pane: hidden by default, overlays the right pane on toggle.
        # Receives every record passed to self._log so the user can watch the
        # session JSONL stream in real time.
        with Vertical(id="debug-pane"):
            yield Static("🔍 [b]Debug log[/b] (F12 to hide)", id="debug-pane-header")
            yield RichLog(id="debug-log", markup=True, max_lines=2000)
        yield Footer()

    # ------------------------------------------------------------- lifecycle
    async def on_mount(self) -> None:
        # Bootstrap .muninn/ and load config.
        self.muninn_dir = bootstrap.ensure_muninn_dir(self.cwd)
        self.config = bootstrap.load_config(self.muninn_dir)
        # load_config has already normalized freedom_level (and migrated
        # any legacy auto_mode) to one of low/medium/high.
        self.freedom_level = self.config.get("freedom_level", "low")

        # Register the Muninn-branded themes and activate whichever the
        # config asked for. If the config names something we don't recognize
        # (typo, deleted theme, ...), fall back to muninn-dark so the user
        # is never stuck on a broken theme name.
        for theme in ALL_THEMES:
            self.register_theme(theme)
        wanted = self.config.get("theme", DEFAULT_THEME_NAME)
        self.theme = wanted if wanted in self.available_themes else DEFAULT_THEME_NAME

        self.logger = JsonlLogger(self.muninn_dir / "logs")
        # Log everything needed to fully reproduce a session: cwd, config, and
        # the verbatim text of every prompt template the agents will see.
        self.logger.log({
            "type": "session_start",
            "cwd": str(self.cwd),
            "config": {k: v for k, v in self.config.items() if k != "schema_version"},
            "prompts": {
                name: bootstrap.load_prompt(self.muninn_dir, name)
                for name in bootstrap.PROMPT_NAMES
            },
        })

        # Mount a system-status line + the first-launch help in the muninn pane.
        scroll = self.query_one("#muninn-scroll", VerticalScroll)
        intro = Markdown(
            BANNER_ASCII
            + f"\nBootstrapped `.muninn/`  ·  cwd: `{self.cwd}`  ·  "
            f"model: **{self.config['model']}**  ·  num_ctx: {self.config['num_ctx']}\n\n"
            f"session log: `{self.logger.path}`\n\n"
            + HELP_TEXT
        )
        await scroll.mount(intro)

        # Tail-follow autoscroll: ticks every 100ms and only scrolls panes that
        # are already near the bottom. If the user scrolls up, we leave them be;
        # as soon as they scroll back near the end, autoscroll resumes.
        self.set_interval(0.1, self._autoscroll_follow)

        # Run the health check in the background, then build agents.
        self.run_worker(self._boot(), group="boot", exclusive=True)

    # How many rows from the bottom still counts as "following".
    _AUTOSCROLL_THRESHOLD = 4

    def _autoscroll_follow(self) -> None:
        """Tail-follow autoscroll that respects user-initiated scroll-up
        even during fast streaming.

        Two key behaviours:
        1. We only do work when scrollable content actually grew since the
           last tick. Ticks that find no new content are zero-cost. This
           stops the autoscroll from competing with streaming for the event
           loop and keeps mouse wheel / arrow keys responsive.
        2. The "is the user following?" check compares scroll_y to the
           PREVIOUS max_scroll_y, not the current one. If the user manually
           scrolled up while content was streaming in, prev_max - scroll_y
           will be much greater than the threshold and we leave them be.
           If they scroll back to the bottom, we naturally re-engage on the
           next content growth.
        """
        # Lazy-init the per-pane history so this method works even if
        # __init__ does not pre-populate the dict.
        history = self.__dict__.setdefault(
            "_scroll_max_history",
            {"#muninn-scroll": 0, "#huginn-scroll": 0},
        )
        for scroll_id in ("#muninn-scroll", "#huginn-scroll"):
            try:
                sc = self.query_one(scroll_id, VerticalScroll)
            except Exception:
                continue
            prev_max = history.get(scroll_id, 0)
            cur_max = sc.max_scroll_y
            if cur_max == prev_max:
                continue  # no new content this tick
            # Was the user near the previous end? If yes, follow.
            if prev_max - sc.scroll_y <= self._AUTOSCROLL_THRESHOLD:
                sc.scroll_end(animate=False)
            history[scroll_id] = cur_max

    async def _boot(self) -> None:
        await self._refresh_health()
        if self.health_status == HEALTH_OK:
            self._build_agents()

    def get_system_commands(self, screen):
        """Re-emit Textual's system commands with Quit pinned to the end.

        Default Textual order is: Theme, Quit, Keys, Maximize/Minimize,
        Screenshot. Quit-as-last is more conventional for menus and avoids
        accidental quits when scrolling through the palette.
        """
        from textual.app import SystemCommand
        ours: list[SystemCommand] = []
        quit_cmd: SystemCommand | None = None
        for cmd in super().get_system_commands(screen):
            if cmd.title == "Quit":
                quit_cmd = cmd
            else:
                ours.append(cmd)
        yield from ours
        if quit_cmd is not None:
            yield quit_cmd

    async def on_unmount(self) -> None:
        try:
            await self.http_client.aclose()
        except Exception:
            pass
        if self.logger:
            self.logger.log({"type": "session_end"})

    # ----------------------------------------------------------- health check
    async def _refresh_health(self) -> None:
        import shutil
        self._set_banner("⏳ Checking Ollama...", "status-warn")

        # Step 1: is the `ollama` CLI even on PATH? If not, we can short-circuit
        # with OS-specific install instructions instead of trying to talk to a
        # service that doesn't exist.
        if shutil.which("ollama") is None:
            self.health_status = HEALTH_OLLAMA_NOT_INSTALLED
            self.health_detail = "binary not on PATH"
            self._set_banner(
                "❌ Ollama is not installed. Press Ctrl+R after installing.",
                "status-err",
            )
            await self._show_install_instructions_once()
            self._sync_input_state()
            return

        # Step 2: is the service responding? (ollama serve)
        url = self.config["base_url"].rstrip("/v1").rstrip("/") + "/api/tags"
        try:
            resp = await self.http_client.get(url, timeout=5.0)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {t.get("name", "") for t in tags}
            wanted = self.config["model"]
            normalized = {n.split(":")[0] for n in names}
            if wanted in names or wanted.split(":")[0] in normalized:
                self.health_status = HEALTH_OK
                self.health_detail = wanted
                self._set_banner(self._ok_banner_text(), "status-ok")
            else:
                self.health_status = HEALTH_MODEL_MISSING
                self.health_detail = wanted
                self._set_banner(
                    f"⚠️ Ollama up but model {wanted!r} not pulled.  ·  Ctrl+R to retry",
                    "status-warn",
                )
                # Offer to pull it once per session.
                if not getattr(self, "_offered_pull_for", None) == wanted:
                    self._offered_pull_for = wanted
                    self.run_worker(
                        self._offer_model_pull(wanted),
                        exclusive=True, group="pull-prompt",
                    )
        except (EgressDenied, Exception) as e:
            self.health_status = HEALTH_OLLAMA_DOWN
            self.health_detail = str(e)
            self._set_banner(
                f"❌ Ollama is installed but not running. Run `ollama serve` "
                f"in another shell  ·  Ctrl+R to retry",
                "status-err",
            )

        self._sync_input_state()

    async def _show_install_instructions_once(self) -> None:
        """Mount a one-time block in the Muninn pane explaining how to install
        Ollama for the user's OS. Idempotent: re-pressing Ctrl+R after install
        will pick up the new state without duplicating the message."""
        if getattr(self, "_install_hint_shown", False):
            return
        self._install_hint_shown = True
        try:
            await self._echo_md(
                "❌ **Ollama is not installed**\n\n"
                f"{_ollama_install_hint()}\n\n"
                "After installing, start the service with `ollama serve` "
                "(macOS app does this for you), then press **Ctrl+R** here "
                "to retry the health check.",
                pane="muninn",
            )
        except Exception:
            pass

    async def _offer_model_pull(self, model: str) -> None:
        """Ask the user (via ConfirmScreen) whether to pull the missing
        model now. On yes, runs `ollama pull <model>` and streams progress."""
        try:
            choice = await self.push_screen_wait(
                AskUserScreen(
                    f"Model `{model}` is not pulled. Pull it now?",
                    [
                        ("yes, pull now",
                         f"Runs `ollama pull {model}` and streams progress in "
                         f"the Muninn pane. Big models (30B+) can be 18-20 GB "
                         f"and take 5-30 minutes depending on your connection. "
                         f"Press Esc to cancel."),
                        ("no, I'll do it manually",
                         f"Skip. You can run `ollama pull {model}` yourself in "
                         f"another shell, then press Ctrl+R here to re-check."),
                    ],
                )
            )
        except Exception:
            return
        if not choice or not str(choice).startswith("yes"):
            return
        self.run_worker(self._pull_model(model), exclusive=True, group="pull")

    @work(group="pull", exclusive=True)
    async def _pull_model(self, model: str) -> None:
        """Run `ollama pull <model>` and stream progress lines to the muninn
        pane. Updates a single Static widget in place so the pane doesn't
        flood with each progress line. On success, refreshes health + agents."""
        import shlex
        progress = Static(
            f"📥 **Pulling `{model}`...** starting (Esc to cancel)",
            id="pull-progress",
        )
        scroll = self.query_one("#muninn-scroll", VerticalScroll)
        await scroll.mount(progress)
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                f"ollama pull {shlex.quote(model)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            buf = b""
            while True:
                chunk = await proc.stdout.read(2048)
                if not chunk:
                    break
                buf += chunk
                # Ollama uses \r to overwrite the progress line. Take the last
                # non-empty token after splitting on either separator.
                tokens = buf.replace(b"\r", b"\n").split(b"\n")
                last = next(
                    (t.decode("utf-8", "replace").strip() for t in reversed(tokens) if t.strip()),
                    "",
                )
                buf = tokens[-1]
                if last:
                    progress.update(
                        f"📥 **Pulling `{model}`...**\n```\n{last[-180:]}\n```"
                    )
            await proc.wait()
            if proc.returncode == 0:
                progress.update(f"✅ **Pulled `{model}`.** Re-checking health...")
                await self._refresh_health()
                if self.health_status == HEALTH_OK:
                    self._build_agents()
            else:
                progress.update(
                    f"❌ **Pull failed** (exit {proc.returncode}). "
                    f"Run `ollama pull {model}` manually for full output."
                )
        except asyncio.CancelledError:
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            progress.update("[pull cancelled]")
            raise
        except Exception as e:
            progress.update(f"❌ **Pull failed:** {type(e).__name__}: {e}")

    def _ok_banner_text(self) -> str:
        """Build the green-banner text. Includes the runtime-tweakable settings
        so the user can see current values without opening the palette."""
        return (
            f"🐦‍⬛ Ollama OK · {self.health_detail} · "
            f"freedom: {self.freedom_level} · "
            f"num_ctx: {self.config.get('num_ctx', '?')} · "
            f"revisions: {self.config.get('max_revision_rounds', '?')}"
        )

    def refresh_status_banner(self) -> None:
        """Re-render the OK banner if we're currently healthy. Called by the
        command palette after a config change."""
        if self.health_status == HEALTH_OK:
            self._set_banner(self._ok_banner_text(), "status-ok")

    def _set_banner(self, text: str, css_class: str) -> None:
        banner = self.query_one("#status-banner", Static)
        banner.update(text)
        # Mirror the status class onto the Footer so the bottom chrome bar
        # tracks the same color as the top status banner (green / yellow / red).
        targets = [banner]
        try:
            targets.append(self.query_one(Footer))
        except Exception:
            pass
        for w in targets:
            for c in ("status-ok", "status-warn", "status-err"):
                w.remove_class(c)
            w.add_class(css_class)

    def _sync_input_state(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.disabled = self.health_status != HEALTH_OK
        if not inp.disabled:
            inp.focus()

    # ---------------------------------------------------------------- agents
    def _build_agents(self) -> None:
        provider = make_provider(self.config["base_url"], self.http_client)
        model = make_local_model(self.config["model"], provider)

        async def confirm(label: str, preview: str) -> bool:
            return bool(await self.push_screen_wait(ConfirmScreen(label, preview)))

        async def ask_user(question: str, options: list[str]) -> str:
            return str(await self.push_screen_wait(AskUserScreen(question, options)))

        self.tool_ctx = ToolContext(
            cwd=self.cwd,
            muninn_dir=self.muninn_dir,
            freedom_level=self.freedom_level,
            confirm_callback=confirm,
            ask_user_callback=ask_user,
            log=self._log,
        )
        muninn_prompt = compose_muninn_prompt(
            bootstrap.load_prompt(self.muninn_dir, "muninn"),
            self.freedom_level,
        )
        huginn_prompt = bootstrap.load_prompt(self.muninn_dir, "huginn")

        self.muninn_agent = muninn_agent(model, make_tools(self.tool_ctx), muninn_prompt)
        self._huginn_model = model
        self._huginn_prompt = huginn_prompt
        self.huginn_factory = lambda: huginn_agent(self._huginn_model, self._huginn_prompt)

    def _log(self, record: dict) -> None:
        if self.logger:
            self.logger.log(record)
        self._append_debug_log(record)

    def _append_debug_log(self, record: dict) -> None:
        """Mirror the JSONL record into the F12 debug pane as a one-line
        summary. Truncates long values so the pane stays scannable; full
        detail remains in the on-disk JSONL."""
        try:
            rich_log = self.query_one("#debug-log", RichLog)
        except Exception:
            return
        import time as _time
        ts = record.get("ts") or _time.time()
        ts_str = _time.strftime("%H:%M:%S", _time.localtime(ts))
        type_ = record.get("type", "?")
        parts: list[str] = []
        for k, v in record.items():
            if k in ("ts", "session_id", "type"):
                continue
            s = repr(v)
            if len(s) > 80:
                s = s[:80] + "…"
            parts.append(f"{k}={s}")
        rich_log.write(f"[dim]{ts_str}[/] [b cyan]{type_}[/] {' '.join(parts)}")

    # -------------------------------------------------------------- watchers
    def watch_theme(self, _old: str | None, new: str | None) -> None:
        """Persist theme picks made through the Ctrl+P palette.

        Textual sets `theme` to None briefly during construction; ignore
        that, and also ignore writes that happen before .muninn/ exists
        (the very first time the reactive fires from App.__init__, before
        on_mount has bootstrapped the dir).
        """
        if not new or not self.muninn_dir.exists():
            return
        if self.config.get("theme") == new:
            return
        cfg = dict(self.config)
        cfg["theme"] = new
        self.config = cfg
        try:
            bootstrap.save_config(self.muninn_dir, cfg)
        except Exception:
            pass

    def watch_freedom_level(self, _old: str, new: str) -> None:
        """React to a freedom_level change from Ctrl+A or the palette.

        Mutates tool_ctx in place so the next tool call sees the new
        gate policy immediately. Persists to config.toml. Rebuilds the
        muninn agent so the per-level system_prompt addendum updates;
        the rebuild is skipped when the agent has not been built yet
        (boot before health check) or when Ollama is degraded. In-flight
        feature/bug workers continue against their captured agent
        reference - same policy as model-switch (see
        _rebuild_agents_for_new_model).
        """
        if self.tool_ctx is not None:
            self.tool_ctx.freedom_level = new  # type: ignore[assignment]
        if self.muninn_dir.exists():
            cfg = dict(self.config)
            cfg["freedom_level"] = new
            self.config = cfg
            try:
                bootstrap.save_config(self.muninn_dir, cfg)
            except Exception as e:
                try:
                    self._log({
                        "type": "config_save_failed",
                        "key": "freedom_level",
                        "error": str(e),
                        "exc": type(e).__name__,
                    })
                except Exception:
                    pass
        try:
            self.refresh_status_banner()
        except Exception:
            pass
        if self.muninn_agent is not None and self.health_status == HEALTH_OK:
            self._build_agents()

    # --------------------------------------------------------------- actions
    def action_cycle_freedom(self) -> None:
        """Cycle freedom_level: low -> medium -> high -> low.

        Bound to Ctrl+A. Unknown current values fall back to "low" so a
        hand-edited config or a stale reactive state cannot wedge the
        cycle. The watcher persists, refreshes the banner, and rebuilds
        the muninn agent if applicable.
        """
        order = ("low", "medium", "high")
        cur = self.freedom_level if self.freedom_level in order else "low"
        self.freedom_level = order[(order.index(cur) + 1) % len(order)]

    def action_cancel_workers(self) -> None:
        for group in _CANCEL_GROUPS:
            self.workers.cancel_group(self, group)
        self._log({"type": "user_cancel"})
        try:
            self.query_one("#user-input", Input).focus()
        except Exception:
            pass

    async def action_retry_health(self) -> None:
        await self._refresh_health()
        if self.health_status == HEALTH_OK and self.muninn_agent is None:
            self._build_agents()

    def watch_debug_visible(self, _old: bool, new: bool) -> None:
        try:
            pane = self.query_one("#debug-pane")
        except Exception:
            return
        pane.styles.display = "block" if new else "none"

    def action_toggle_debug(self) -> None:
        self.debug_visible = not self.debug_visible

    async def _rebuild_agents_for_new_model(self) -> None:
        """Re-run the health check and rebuild the Muninn / Huginn agents
        against the newly-configured `model`. Called by the palette's
        /model picker after applying a switch. The next turn will use the
        new tag automatically; in-flight workers continue with the old
        agent (their FeatureRunCtx already captured the old reference).
        """
        self._log({"type": "model_changed", "model": self.config.get("model")})
        await self._refresh_health()
        if self.health_status == HEALTH_OK:
            self._build_agents()

    # ---------------------------------------------------------- input handler
    @on(Input.Submitted, "#user-input")
    async def _on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.health_status != HEALTH_OK or self.muninn_agent is None:
            return
        event.input.value = ""
        await self._echo_user(text)
        self._log({"type": "user_input", "text": text})

        if text.startswith("/feature"):
            description = text[len("/feature"):].strip()
            if not description:
                await self._echo_md("_/feature requires a description._")
                return
            self._run_feature_worker(description)
        elif text.startswith("/bug"):
            description = text[len("/bug"):].strip()
            if not description:
                await self._echo_md("_/bug requires a description of the bug._")
                return
            self._run_bug_worker(description)
        elif text.startswith("/brainstorm"):
            description = text[len("/brainstorm"):].strip()
            if not description:
                await self._echo_md("_/brainstorm requires a rough idea._")
                return
            self._run_brainstorm_worker(description)
        elif text.startswith("/prd"):
            description = text[len("/prd"):].strip()
            if not description:
                await self._echo_md("_/prd requires an idea description._")
                return
            self._run_prd_worker(description)
        elif text.startswith("/precommit-review"):
            self._run_review_worker()
        elif text.startswith("/"):
            available = ", ".join(f"`{c}`" for c, _ in SLASH_COMMANDS)
            await self._echo_md(
                f"Unknown slash command `{text.split()[0]}`. "
                f"Available: {available}."
            )
        else:
            self._run_chat_worker(text)

    # --------------------------------------------------------- mount helpers
    # Note: none of these helpers calls scroll_end() directly. The 100ms
    # _autoscroll_follow interval handles autoscroll, but only when the user
    # is already near the bottom - so scrolling up to read older output is
    # respected during streaming.
    async def _echo_user(self, text: str) -> None:
        scroll = self.query_one("#muninn-scroll", VerticalScroll)
        md = Markdown(f"**❯ You:** {text}")
        await scroll.mount(md)

    async def _echo_md(self, content: str, *, pane: str = "muninn") -> Markdown:
        scroll_id = "#muninn-scroll" if pane == "muninn" else "#huginn-scroll"
        scroll = self.query_one(scroll_id, VerticalScroll)
        md = Markdown(content)
        await scroll.mount(md)
        return md

    async def _new_streaming_md(self, header: str, *, pane: str) -> Markdown:
        """Mount a header widget AND a fresh streaming Markdown widget, return
        the streaming one.

        Splitting them prevents Markdown re-tokenization between the static
        header and the streamed body (bold or italic markers spanning the
        boundary can otherwise be misparsed when fragments are appended).
        """
        scroll_id = "#muninn-scroll" if pane == "muninn" else "#huginn-scroll"
        scroll = self.query_one(scroll_id, VerticalScroll)
        await scroll.mount(Markdown(header))
        stream_md = Markdown("")
        await scroll.mount(stream_md)
        return stream_md

    # ------------------------------------------------------------- workers
    @work(group="muninn", exclusive=False)
    async def _run_chat_worker(self, text: str) -> None:
        md = await self._new_streaming_md(
            "🐦‍⬛ **Muninn · working solo…**", pane="muninn"
        )
        completed_cleanly = False
        try:
            _, self.muninn_history = await run_and_stream(
                self.muninn_agent,
                text,
                md,
                message_history=self.muninn_history,
                log=self._log,
                model_settings=num_ctx_settings(int(self.config["num_ctx"])),
                label="muninn-chat",
            )
            completed_cleanly = True
        except asyncio.CancelledError:
            await md.append("\n\n[cancelled]")
            raise
        except ToolError as e:
            await md.append(f"\n\n**Tool error:** {e}")
            self._log({"type": "tool_error", "error": str(e)})
        except Exception as e:
            await md.append(f"\n\n**Muninn error:** {type(e).__name__}: {e}")
            self._log({"type": "muninn_error", "error": str(e), "exc": type(e).__name__})
        finally:
            if completed_cleanly:
                await self._echo_md("✅ **Muninn · done**", pane="muninn")

    @work(group="feature", exclusive=True)
    async def _run_feature_worker(self, description: str) -> None:
        async def mount_muninn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="muninn")

        async def mount_huginn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="huginn")

        ctx = FeatureRunCtx(
            description=description,
            muninn_agent=self.muninn_agent,
            huginn_agent_factory=self.huginn_factory,
            muninn_history=self.muninn_history,
            feature_ground_prompt=bootstrap.load_prompt(self.muninn_dir, "feature_ground"),
            feature_design_prompt=bootstrap.load_prompt(self.muninn_dir, "feature_design"),
            feature_critique_prompt=bootstrap.load_prompt(self.muninn_dir, "feature_critique"),
            feature_comprehension_prompt=bootstrap.load_prompt(self.muninn_dir, "feature_comprehension"),
            feature_readiness_prompt=bootstrap.load_prompt(self.muninn_dir, "feature_readiness"),
            ask_user=self.tool_ctx.ask_user_callback,
            max_revision_rounds=int(self.config.get("max_revision_rounds", 3)),
            model_settings=num_ctx_settings(int(self.config["num_ctx"])),
            log=self._log,
            mount_muninn_md=mount_muninn,
            mount_huginn_md=mount_huginn,
            freedom_level=self.freedom_level,
        )
        try:
            summary = await feature_flow(ctx)
            self.muninn_history = ctx.muninn_history
            await self._echo_md(
                f"**Feature complete.** design: {summary['design_len']}b · "
                f"implement: {summary['implement_len']}b"
            )
        except asyncio.CancelledError:
            await self._echo_md("[/feature cancelled]")
            raise
        except ToolError as e:
            await self._echo_md(f"**/feature aborted at tool:** {e}")
            self._log({"type": "feature_aborted_tool", "error": str(e)})
        except Exception as e:
            await self._echo_md(f"**/feature failed:** {type(e).__name__}: {e}")
            self._log({"type": "feature_failed", "error": str(e), "exc": type(e).__name__})

    @work(group="bug", exclusive=True)
    async def _run_bug_worker(self, description: str) -> None:
        async def mount_muninn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="muninn")

        async def mount_huginn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="huginn")

        ctx = BugRunCtx(
            description=description,
            muninn_agent=self.muninn_agent,
            huginn_agent_factory=self.huginn_factory,
            muninn_history=self.muninn_history,
            bug_ground_prompt=bootstrap.load_prompt(self.muninn_dir, "bug_ground"),
            bug_problem_prompt=bootstrap.load_prompt(self.muninn_dir, "bug_problem"),
            bug_critique_prompt=bootstrap.load_prompt(self.muninn_dir, "bug_critique"),
            ask_user=self.tool_ctx.ask_user_callback,
            max_revision_rounds=int(self.config.get("max_revision_rounds", 3)),
            model_settings=num_ctx_settings(int(self.config["num_ctx"])),
            log=self._log,
            mount_muninn_md=mount_muninn,
            mount_huginn_md=mount_huginn,
            freedom_level=self.freedom_level,
        )
        try:
            summary = await bug_flow(ctx)
            self.muninn_history = ctx.muninn_history
            if summary["type"] == "bug_complete":
                await self._echo_md(
                    f"**Bug fix complete.** _problem: {summary['problem_len']}b · "
                    f"test: {summary['test_len']}b · fix: {summary['fix_len']}b_"
                )
        except asyncio.CancelledError:
            await self._echo_md("[/bug cancelled]")
            raise
        except ToolError as e:
            await self._echo_md(f"**/bug aborted at tool:** {e}")
            self._log({"type": "bug_aborted_tool", "error": str(e)})
        except Exception as e:
            await self._echo_md(f"**/bug failed:** {type(e).__name__}: {e}")
            self._log({"type": "bug_failed", "error": str(e), "exc": type(e).__name__})

    @work(group="brainstorm", exclusive=True)
    async def _run_brainstorm_worker(self, description: str) -> None:
        async def mount_muninn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="muninn")

        async def mount_huginn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="huginn")

        ctx = BrainstormRunCtx(
            description=description,
            cwd=self.cwd,
            muninn_agent=self.muninn_agent,
            huginn_agent_factory=self.huginn_factory,
            muninn_history=self.muninn_history,
            brainstorm_ground_prompt=bootstrap.load_prompt(self.muninn_dir, "brainstorm_ground"),
            brainstorm_lens_prompts={
                lens: bootstrap.load_prompt(self.muninn_dir, f"brainstorm_lens_{lens}")
                for lens in BRAINSTORM_LENSES
            },
            brainstorm_synthesis_prompt=bootstrap.load_prompt(self.muninn_dir, "brainstorm_synthesis"),
            model_settings=num_ctx_settings(int(self.config["num_ctx"])),
            log=self._log,
            mount_muninn_md=mount_muninn,
            mount_huginn_md=mount_huginn,
        )
        try:
            summary = await brainstorm_flow(ctx)
            self.muninn_history = ctx.muninn_history
            if summary["type"] == "brainstorm_complete":
                # Use the workflow's actual successful_lens_count, not the
                # hardcoded BRAINSTORM_LENSES tuple length - on partial-success
                # those numbers diverge.
                ok = summary["successful_lens_count"]
                bad = summary["failed_lens_count"]
                tail = (
                    f" · {bad} lens(es) failed: "
                    + ", ".join(f"{l}={c}" for l, c in summary["lens_failures"].items())
                ) if bad else ""
                await self._echo_md(
                    f"**Brainstorm complete.** _{ok} lens(es) · "
                    f"synthesis {summary['synthesis_len']}b · "
                    f"saved to {summary['artifact_path']}{tail}_"
                )
            elif summary["type"] == "brainstorm_partial":
                await self._echo_md(
                    f"**Brainstorm partial.** _synthesis failed "
                    f"({summary['synthesis_failed']}); lens outputs saved to "
                    f"{summary['artifact_path']}_"
                )
            else:
                await self._echo_md(f"**/brainstorm aborted:** {summary['reason']}")
        except asyncio.CancelledError:
            await self._echo_md("[/brainstorm cancelled]")
            raise
        except ToolError as e:
            await self._echo_md(f"**/brainstorm aborted at tool:** {e}")
            self._log({"type": "brainstorm_aborted_tool", "error": str(e)})
        except Exception as e:
            await self._echo_md(f"**/brainstorm failed:** {type(e).__name__}: {e}")
            self._log({"type": "brainstorm_failed",
                       "error": str(e), "exc": type(e).__name__})

    @work(group="prd", exclusive=True)
    async def _run_prd_worker(self, description: str) -> None:
        async def mount_muninn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="muninn")

        async def mount_huginn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="huginn")

        ctx = PRDRunCtx(
            description=description,
            cwd=self.cwd,
            muninn_agent=self.muninn_agent,
            huginn_agent_factory=self.huginn_factory,
            muninn_history=self.muninn_history,
            prd_ground_prompt=bootstrap.load_prompt(self.muninn_dir, "prd_ground"),
            prd_qa_prompt=bootstrap.load_prompt(self.muninn_dir, "prd_qa"),
            prd_lens_prompts={
                lens: bootstrap.load_prompt(self.muninn_dir, f"prd_lens_{lens}")
                for lens in PRD_LENSES
            },
            prd_synthesis_prompt=bootstrap.load_prompt(self.muninn_dir, "prd_synthesis"),
            model_settings=num_ctx_settings(int(self.config["num_ctx"])),
            log=self._log,
            mount_muninn_md=mount_muninn,
            mount_huginn_md=mount_huginn,
        )
        try:
            summary = await prd_flow(ctx)
            self.muninn_history = ctx.muninn_history
            if summary["type"] == "prd_complete":
                ok = summary["successful_lens_count"]
                bad = summary["failed_lens_count"]
                tail = (
                    f" · {bad} lens(es) failed: "
                    + ", ".join(f"{l}={c}" for l, c in summary["lens_failures"].items())
                ) if bad else ""
                await self._echo_md(
                    f"**PRD complete.** _{ok} research lens(es) · "
                    f"synthesis {summary['synthesis_len']}b · "
                    f"saved to {summary['artifact_path']}{tail}_"
                )
            elif summary["type"] == "prd_partial":
                await self._echo_md(
                    f"**PRD partial.** _synthesis failed "
                    f"({summary['synthesis_failed']}); lens outputs and Q&A "
                    f"saved to {summary['artifact_path']}_"
                )
            else:
                await self._echo_md(f"**/prd aborted:** {summary['reason']}")
        except asyncio.CancelledError:
            await self._echo_md("[/prd cancelled]")
            raise
        except ToolError as e:
            await self._echo_md(f"**/prd aborted at tool:** {e}")
            self._log({"type": "prd_aborted_tool", "error": str(e)})
        except Exception as e:
            await self._echo_md(f"**/prd failed:** {type(e).__name__}: {e}")
            self._log({"type": "prd_failed",
                       "error": str(e), "exc": type(e).__name__})

    @work(group="review", exclusive=True)
    async def _run_review_worker(self) -> None:
        async def mount_muninn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="muninn")

        async def mount_huginn(header: str) -> Markdown:
            return await self._new_streaming_md(header, pane="huginn")

        ctx = ReviewRunCtx(
            cwd=self.cwd,
            huginn_agent_factory=self.huginn_factory,
            review_prompt=bootstrap.load_prompt(self.muninn_dir, "precommit_review"),
            model_settings=num_ctx_settings(int(self.config["num_ctx"])),
            log=self._log,
            mount_muninn_md=mount_muninn,
            mount_huginn_md=mount_huginn,
        )
        try:
            await precommit_review_flow(ctx)
        except asyncio.CancelledError:
            await self._echo_md("[/precommit-review cancelled]")
            raise
        except Exception as e:
            await self._echo_md(
                f"**/precommit-review failed:** {type(e).__name__}: {e}"
            )
            self._log({"type": "review_failed", "error": str(e),
                       "exc": type(e).__name__})


def _check_not_home(cwd: Path) -> str | None:
    """Return an error message if `cwd` is the user's home directory.

    Running muninn from `~` would create `~/.muninn/` as per-project state,
    which is almost certainly an accident: muninn is per-project by design,
    and using your home directory as the project would scatter
    `.muninn/logs/` and `config.toml` somewhere you didn't mean. Refuse to
    launch and point the user at a real project path. Subdirectories of
    home (`~/code/some-project`) are fine.
    """
    if cwd.resolve() == Path.home().resolve():
        return (
            "error: muninn cannot run from your home directory.\n"
            "  reason: per-project `.muninn/` would land at `~/.muninn/`,\n"
            "  which is almost certainly not what you want. muninn is\n"
            "  per-project; run it against a specific project directory:\n"
            "      muninn ~/code/some-project\n"
            "      muninn /tmp/scratch\n"
            "      cd ~/code/some-project && muninn"
        )
    return None


def _run_update() -> None:
    """Shell out to `uv tool upgrade muninn`.

    Requires muninn to have been installed via the official installer
    (which uses `uv tool install muninn`). If `uv` is missing or muninn was
    installed some other way, uv prints its own error and we propagate the
    exit code.
    """
    import shutil
    import subprocess
    import sys

    if shutil.which("uv") is None:
        print(
            "error: 'uv' is not on PATH. The 'muninn update' command\n"
            "requires uv (https://docs.astral.sh/uv/) which is installed\n"
            "automatically by the official muninn installer.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Upgrading muninn via uv...")
    result = subprocess.run(["uv", "tool", "upgrade", "muninn"])
    sys.exit(result.returncode)


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    Usage:
        muninn                  # run TUI in the current directory
        muninn /path/to/repo    # run TUI against that repo
        muninn update           # upgrade muninn in place via uv
        muninn --help
    """
    import sys
    args = sys.argv[1:] if argv is None else argv

    if args and args[0] in ("-h", "--help"):
        print(
            "usage: muninn [PATH | update]\n"
            "\n"
            "Run muninn TUI against PATH (default: current working directory).\n"
            "All per-project state lives in <PATH>/.muninn/.\n"
            "\n"
            "PATH must NOT be your home directory; run muninn against a\n"
            "specific project directory (e.g. `muninn ~/code/some-project`).\n"
            "\n"
            "Subcommands:\n"
            "  update    upgrade muninn to the latest release (via uv).\n"
        )
        return

    if args and args[0] == "update":
        # Reject trailing args so a typo like `muninn update foo` doesn't silently
        # run the upgrade and drop `foo` on the floor. Use `muninn ./update` if
        # the user really has a directory literally called `update`.
        if len(args) > 1:
            print(
                f"error: 'muninn update' takes no arguments (got: {' '.join(args[1:])})",
                file=sys.stderr,
            )
            sys.exit(2)
        _run_update()
        return

    cwd = Path.cwd()
    if args:
        cwd = Path(args[0]).expanduser().resolve()
        if not cwd.exists():
            print(f"error: {cwd} does not exist", file=sys.stderr)
            sys.exit(1)
        if not cwd.is_dir():
            print(f"error: {cwd} is not a directory", file=sys.stderr)
            sys.exit(1)

    home_err = _check_not_home(cwd)
    if home_err is not None:
        print(home_err, file=sys.stderr)
        sys.exit(2)

    MuninnTUI(cwd=cwd).run()


if __name__ == "__main__":
    main()
