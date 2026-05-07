"""Modal screens for Confirm (y/n), AskUser (multiple choice), and preset
pickers used by the command palette.

AskUserScreen and PresetPickerScreen both support a "?" button per option:
clicking it surfaces a longer explanation in a panel at the bottom of the
modal, without dismissing. This is the answer to "the user is out of
context even when the question is clear" - the option button itself stays
short and scannable, and the explanation is one click away."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmScreen(ModalScreen[bool]):
    """y/n confirmation modal. Returns True for yes, False for no/escape."""

    BINDINGS = [
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
        Binding("escape", "no", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    ConfirmScreen > Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: thick $accent;
        padding: 1 2;
    }
    ConfirmScreen Label.title {
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        text-wrap: wrap;
    }
    ConfirmScreen Static.preview {
        height: auto;
        max-height: 20;
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    ConfirmScreen Label.hint {
        color: $text-muted;
    }
    """

    def __init__(self, label: str, preview: str) -> None:
        super().__init__()
        self.label = label
        self.preview = preview

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Approve: {self.label}?", classes="title")
            yield Static(self.preview, classes="preview")
            yield Label("y = yes    n = no    esc = cancel", classes="hint")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class AskUserScreen(ModalScreen[str]):
    """Multiple-choice ask_user modal. Returns the selected option label.

    `options` may be a list of plain strings (back-compat) OR a list of
    (label, explanation) tuples. When an explanation is provided, a small
    `?` button is rendered next to that option; pressing it pops the
    explanation into a panel at the bottom of the modal without dismissing.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    AskUserScreen {
        align: center middle;
    }
    AskUserScreen > Vertical {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }
    AskUserScreen Label.question {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        height: auto;
        text-wrap: wrap;
    }
    AskUserScreen .option-row {
        layout: horizontal;
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
    }
    AskUserScreen Button.option-main {
        width: 4fr;
    }
    AskUserScreen Button.option-explain {
        width: 1fr;
        min-width: 5;
        max-width: 7;
        margin-left: 1;
    }
    AskUserScreen #explanation-panel {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 10;
        background: $surface;
        border-top: solid $primary;
        margin-top: 1;
        padding: 1;
        color: $text;
    }
    """

    def __init__(
        self,
        question: str,
        options: list[str] | list[tuple[str, str]],
    ) -> None:
        super().__init__()
        self.question = question
        # Normalize every option to (label, explanation_or_None).
        normalized: list[tuple[str, str | None]] = []
        for opt in (options or ["ok"]):
            if isinstance(opt, tuple):
                label = opt[0]
                expl = opt[1] if len(opt) > 1 and opt[1] else None
                normalized.append((label, expl))
            else:
                normalized.append((opt, None))
        self.options = normalized

    @property
    def labels(self) -> list[str]:
        """The plain string labels, for callers that don't care about the
        explanations (e.g. existing test assertions)."""
        return [label for label, _ in self.options]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.question, classes="question")
            for i, (label, expl) in enumerate(self.options):
                with Horizontal(classes="option-row"):
                    yield Button(label, id=f"opt-{i}", classes="option-main")
                    if expl:
                        yield Button("?", id=f"explain-{i}",
                                     classes="option-explain")
            yield Static(
                "press `?` next to an option to see what it does",
                id="explanation-panel",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("opt-"):
            idx = int(bid.split("-", 1)[1])
            self.dismiss(self.options[idx][0])
        elif bid.startswith("explain-"):
            idx = int(bid.split("-", 1)[1])
            label, expl = self.options[idx]
            panel = self.query_one("#explanation-panel", Static)
            panel.update(f"💡 [b]{label}[/b]\n\n{expl or ''}")

    def action_cancel(self) -> None:
        self.dismiss(self.options[0][0])


class PresetPickerScreen(ModalScreen[object]):
    """Pick one preset value from a labelled list. Returns the chosen value
    (any type) or None on cancel/escape.

    Each entry in `presets` is (value, label, description). The screen renders
    a vertical stack of buttons; the button label is the formatted label, and
    pressing it dismisses with that value. The currently-active value is
    visually marked.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    PresetPickerScreen {
        align: center middle;
    }
    PresetPickerScreen > Vertical {
        width: 80%;
        max-width: 110;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }
    PresetPickerScreen Label.title {
        color: $primary;
        text-style: bold;
        margin-bottom: 0;
        width: 100%;
        text-wrap: wrap;
    }
    PresetPickerScreen Label.subtitle {
        color: $text-muted;
        margin-bottom: 1;
        width: 100%;
        height: auto;
        text-wrap: wrap;
    }
    PresetPickerScreen .preset-row {
        layout: horizontal;
        height: auto;
        width: 100%;
        margin: 0 0 1 0;
    }
    PresetPickerScreen Button.preset-main {
        width: 4fr;
    }
    PresetPickerScreen Button.preset-explain {
        width: 1fr;
        min-width: 5;
        max-width: 7;
        margin-left: 1;
    }
    PresetPickerScreen Button.-current {
        border: tall $accent;
    }
    PresetPickerScreen #explanation-panel {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 8;
        background: $surface;
        border-top: solid $primary;
        margin-top: 1;
        padding: 1;
        color: $text;
    }
    PresetPickerScreen Label.hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        title: str,
        subtitle: str,
        presets: list[tuple[object, str, str]],
        current: object = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.subtitle_text = subtitle
        # presets: list of (value, label, description). Description shows
        # behind the `?` button on demand instead of being baked into the
        # button label, keeping the row scannable.
        self.presets = presets
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text, classes="title")
            yield Label(self.subtitle_text, classes="subtitle")
            for i, (value, label, desc) in enumerate(self.presets):
                with Horizontal(classes="preset-row"):
                    btn = Button(label, id=f"preset-{i}", classes="preset-main")
                    if value == self.current:
                        btn.add_class("-current")
                    yield btn
                    if desc:
                        yield Button("?", id=f"explain-{i}",
                                     classes="preset-explain")
            yield Static(
                "press `?` next to a preset to see what it means",
                id="explanation-panel",
            )
            yield Label("esc = cancel", classes="hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("preset-"):
            idx = int(bid.split("-", 1)[1])
            value = self.presets[idx][0]
            self.dismiss(value)
        elif bid.startswith("explain-"):
            idx = int(bid.split("-", 1)[1])
            _value, label, desc = self.presets[idx]
            panel = self.query_one("#explanation-panel", Static)
            panel.update(f"💡 [b]{label}[/b]\n\n{desc or ''}")

    def action_cancel(self) -> None:
        self.dismiss(None)
