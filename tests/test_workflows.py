"""Workflow-level tests with stubbed agent + Markdown widget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.settings import ModelSettings

from muninn.workflows import (
    BRAINSTORM_LENSES,
    PRD_LENSES,
    BrainstormRunCtx,
    BugRunCtx,
    FeatureRunCtx,
    PRDRunCtx,
    ReviewRunCtx,
    _artifact_body,
    _format_lens_outputs,
    _has_no_findings,
    _python_syntax_check,
    _slug_for_path,
    _unique_artifact_path,
    _verdict,
    brainstorm_flow,
    bug_flow,
    feature_flow,
    prd_flow,
    precommit_review_flow,
)


def _label(opt) -> str:
    """Options may be plain strings or (label, explanation) tuples; return
    the label either way."""
    return opt[0] if isinstance(opt, tuple) else opt


async def _ask_user_stub(_question: str, options) -> str:
    """Default ask_user used by tests that don't exercise the unconverged path."""
    return _label(options[0]) if options else "ok"


def _pick(options, prefix: str) -> str:
    """Pick the first option whose label starts with prefix; returns the label."""
    return next(_label(o) for o in options if _label(o).startswith(prefix))


def test_verdict_parses_design_ready() -> None:
    assert _verdict("...some critique...\n\ndesign ready") == "ready"
    assert _verdict("DESIGN READY\n") == "ready"
    assert _verdict("design ready.") == "ready"


def test_verdict_parses_needs_revision() -> None:
    assert _verdict("1. gap\n\ndesign needs revision") == "revise"
    assert _verdict("Design Needs Revision") == "revise"


def test_verdict_unknown() -> None:
    assert _verdict("some unrelated trailing line") == "unknown"


# --- feature_flow integration with stub agents ----------------------------


class _FakeMd:
    """Stand-in for Textual Markdown - captures stream writes."""

    def __init__(self) -> None:
        self.buf: list[str] = []

    # Streaming.run_and_stream calls Markdown.get_stream(self) - we patch that.
    pass


class _FakeStream:
    def __init__(self, md: _FakeMd) -> None:
        self.md = md

    async def write(self, fragment: str) -> None:
        self.md.buf.append(fragment)

    async def stop(self) -> None:
        pass


@dataclass
class _FakeAgent:
    name: str
    output_text: str
    # Optional injection knobs added for /brainstorm and /prd tests.
    # error: when set, raise after yielding one TextPartDelta - simulates
    #        mid-stream Ollama disconnect.
    # received_prompts / call_count: per-instance recorders for tests
    #        that need to verify what the agent saw.
    error: BaseException | None = None

    def __post_init__(self) -> None:
        self.received_prompts: list[str] = []
        self.call_count: int = 0

    async def run_stream_events(self, prompt, *, message_history=None, model_settings=None):
        # Yield a few text deltas + a final result event.
        from pydantic_ai.messages import (
            PartDeltaEvent,
            TextPartDelta,
        )
        from pydantic_ai.run import AgentRunResultEvent

        self.received_prompts.append(prompt)
        self.call_count += 1

        chunks = [self.output_text[i:i+20] for i in range(0, len(self.output_text), 20)] or [""]
        for i, chunk in enumerate(chunks):
            yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=chunk))
            # Inject mid-stream error after the first delta so tests see
            # partial output before the failure (matches real Ollama
            # disconnect behavior).
            if self.error is not None and i == 0:
                raise self.error

        # Build a minimal result-like object.
        class _R:
            def __init__(self, text):
                self.output = text
            def new_messages(self):
                return []

        yield AgentRunResultEvent(result=_R(self.output_text))


async def test_feature_flow_happy_path(monkeypatch) -> None:
    from textual.widgets import Markdown
    from muninn.streaming import run_and_stream
    from muninn import streaming as streaming_mod

    # Replace Markdown.get_stream with our fake.
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    pane_log: list[tuple[str, str]] = []

    async def mount_muninn(header: str):
        pane_log.append(("muninn", header))
        return _FakeMd()

    async def mount_huginn(header: str):
        pane_log.append(("huginn", header))
        return _FakeMd()

    log_records: list[dict] = []

    muninn = _FakeAgent("muninn", "DESIGN DOC ... here is the design.")
    huginn_count = [0]

    def huginn_factory():
        huginn_count[0] += 1
        return _FakeAgent("huginn", "no gaps. design ready")

    ctx = FeatureRunCtx(
        description="dummy feature",
        muninn_agent=muninn,
        huginn_agent_factory=huginn_factory,
        muninn_history=[],
        feature_ground_prompt="ground: {description}",
        feature_design_prompt="design: {description}",
        feature_critique_prompt="critique: {design_doc}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_muninn,
        mount_huginn_md=mount_huginn,
        ask_user=_ask_user_stub,
    )
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    # Muninn pane mounts: kickoff banner + ground + design + implement + final = 5
    assert sum(1 for p, _ in pane_log if p == "muninn") == 5
    # Huginn pane mounts: streaming widget + verdict callout = 2
    assert sum(1 for p, _ in pane_log if p == "huginn") == 2
    assert huginn_count[0] == 1
    # Grounding step ran and was logged.
    assert any(r.get("type") == "muninn_grounded" for r in log_records)
    # Verdict was logged.
    assert any(r.get("type") == "huginn_verdict" and r.get("verdict") == "ready"
               for r in log_records)


async def test_feature_flow_one_revision_round(monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    muninn = _FakeAgent("muninn", "design doc v1")
    huginn_outputs = ["gap 1.\ndesign needs revision", "ok now.\ndesign ready"]
    idx = [0]

    def huginn_factory():
        out = huginn_outputs[idx[0]] if idx[0] < len(huginn_outputs) else "design ready"
        idx[0] += 1
        return _FakeAgent("huginn", out)

    ctx = FeatureRunCtx(
        description="dummy",
        muninn_agent=muninn,
        huginn_agent_factory=huginn_factory,
        muninn_history=[],
        feature_ground_prompt="ground: {description}",
        feature_design_prompt="design: {description}",
        feature_critique_prompt="critique: {design_doc}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
        ask_user=_ask_user_stub,
    )
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    # Two huginn rounds because the first verdict was "revise".
    assert idx[0] == 2
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    assert len(verdicts) == 2
    assert verdicts[0]["verdict"] == "revise"
    assert verdicts[1]["verdict"] == "ready"


def _make_unconverged_ctx(ask_user_fn, log_records):
    """Helper: ctx where Huginn always says 'design needs revision' so the
    backstop fires."""
    async def mount_md(_h): return _FakeMd()
    muninn = _FakeAgent("muninn", "design")
    huginn_factory = lambda: _FakeAgent("huginn", "still bad. design needs revision")
    return FeatureRunCtx(
        description="dummy", muninn_agent=muninn, huginn_agent_factory=huginn_factory,
        muninn_history=[], feature_ground_prompt="g: {description}",
        feature_design_prompt="d: {description}", feature_critique_prompt="c: {design_doc}",
        model_settings=ModelSettings(), log=log_records.append,
        mount_muninn_md=mount_md, mount_huginn_md=mount_md, ask_user=ask_user_fn,
    )


async def test_feature_flow_honors_ctx_max_revision_rounds(monkeypatch) -> None:
    """ctx.max_revision_rounds=1 caps the loop at 2 huginn verdicts then prompts."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []

    async def ask_proceed(_q, opts):
        return _pick(opts, "proceed")

    ctx = _make_unconverged_ctx(ask_proceed, log_records)
    ctx.max_revision_rounds = 1
    summary = await feature_flow(ctx)
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    # max_rounds=1 -> initial cold-read + 1 revision round = 2 verdicts.
    assert len(verdicts) == 2
    assert summary["type"] == "feature_complete"


async def test_feature_flow_unconverged_user_proceeds(monkeypatch) -> None:
    """Huginn never converges; backstop asks user; user picks proceed -> implements anyway."""
    from textual.widgets import Markdown
    from muninn.workflows import MAX_REVISION_ROUNDS
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    log_records: list[dict] = []

    async def ask_proceed(_q, opts):
        return _pick(opts, "proceed")

    ctx = _make_unconverged_ctx(ask_proceed, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    assert len(verdicts) == MAX_REVISION_ROUNDS + 1
    assert all(v["verdict"] == "revise" for v in verdicts)
    assert any(r.get("type") == "feature_unconverged" for r in log_records)


async def test_feature_flow_unconverged_user_cancels(monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []

    async def ask_cancel(_q, opts):
        return _pick(opts, "cancel")

    ctx = _make_unconverged_ctx(ask_cancel, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_cancelled"
    assert summary["implement_len"] == 0


async def test_feature_flow_unconverged_one_more_round_then_proceed(monkeypatch) -> None:
    """User asks for one extra round, still not converged, then proceeds."""
    from textual.widgets import Markdown
    from muninn.workflows import MAX_REVISION_ROUNDS
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    call = [0]

    async def ask_choices(_q, opts):
        call[0] += 1
        if call[0] == 1:
            return _pick(opts, "do one more")
        return _pick(opts, "proceed")

    ctx = _make_unconverged_ctx(ask_choices, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    # Initial MAX+1 critiques + 1 extra round = MAX+2.
    assert len(verdicts) == MAX_REVISION_ROUNDS + 2


async def test_feature_flow_unconverged_user_answers_directly(monkeypatch) -> None:
    """User picks 'let me answer' -> Muninn runs the resolve step -> proceeds."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []

    async def ask_resolve(_q, opts):
        return _pick(opts, "let me answer")

    ctx = _make_unconverged_ctx(ask_resolve, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    assert any(r.get("type") == "muninn_user_resolved" for r in log_records)


# --- three-goldfish design check (article-faithful flow) ------------------
#
# The legacy /feature tests above leave feature_comprehension_prompt and
# feature_readiness_prompt at "" so they hit the critic-only fallback.
# These tests pass concrete prompts and exercise the full
# comprehension + critic + readiness loop the article prescribes.


def _make_seq_huginn_factory(outputs: list[str]):
    """Pop one canned output per call so a test can describe a multi-pass
    round as an ordered list. Useful when the test cares about the exact
    ordering of comprehension / critic / readiness across rounds.
    """
    idx = [0]

    def factory():
        out = outputs[idx[0]] if idx[0] < len(outputs) else outputs[-1]
        idx[0] += 1
        return _FakeAgent("huginn", out)

    return factory, idx


def _make_three_pass_ctx(huginn_factory, log_records, *, max_rounds: int | None = None):
    async def mount_md(_h):
        return _FakeMd()

    muninn = _FakeAgent("muninn", "design doc body")
    ctx = FeatureRunCtx(
        description="dummy",
        muninn_agent=muninn,
        huginn_agent_factory=huginn_factory,
        muninn_history=[],
        feature_ground_prompt="g: {description}",
        feature_design_prompt="d: {description}",
        feature_critique_prompt="critique: {design_doc}",
        feature_comprehension_prompt="comprehension: {design_doc}",
        feature_readiness_prompt="readiness: {design_doc}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
        ask_user=_ask_user_stub,
    )
    if max_rounds is not None:
        ctx.max_revision_rounds = max_rounds
    return ctx


async def test_three_pass_happy_path(monkeypatch) -> None:
    """All three passes return ready on round 1; no revisions; 3 huginn calls."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    huginn_factory, idx = _make_seq_huginn_factory([
        "doc paraphrase.\ncomprehension passed",
        "no gaps.\ndesign ready",
        "no questions.\nimplementation ready",
    ])
    ctx = _make_three_pass_ctx(huginn_factory, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    # One round = 3 huginn calls (comprehension + critic + readiness).
    assert idx[0] == 3
    # Per-pass verdicts are logged distinctly.
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    assert len(verdicts) == 3
    kinds = [v.get("kind") for v in verdicts]
    assert kinds == ["comprehension", "critic", "readiness"]
    assert all(v["verdict"] == "ready" for v in verdicts)
    # The combined-round summary records "ready".
    rounds = [r for r in log_records if r.get("type") == "design_check_round"]
    assert len(rounds) == 1
    assert rounds[0]["combined"] == "ready"


async def test_three_pass_readiness_blocks_critic_pass(monkeypatch) -> None:
    """Critic ready but readiness not ready -> revise. Round 2 runs only
    critic + readiness (no comprehension) and converges."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    huginn_factory, idx = _make_seq_huginn_factory([
        # Round 1: comprehension passes, critic ready, readiness blocks.
        "paraphrase.\ncomprehension passed",
        "no gaps.\ndesign ready",
        "1. how do we handle Ollama down?\nimplementation not ready",
        # Round 2: critic ready, readiness ready (no comprehension call).
        "still no gaps.\ndesign ready",
        "no questions.\nimplementation ready",
    ])
    ctx = _make_three_pass_ctx(huginn_factory, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    # Round 1 = 3 calls, round 2 = 2 calls (no comprehension) -> 5 total.
    assert idx[0] == 5
    # Round summaries: round 1 not ready, round 2 ready.
    rounds = [r for r in log_records if r.get("type") == "design_check_round"]
    assert [r["combined"] for r in rounds] == ["revise", "ready"]
    assert rounds[0]["readiness"] == "revise"
    assert rounds[0]["critic"] == "ready"
    # Round 2 logs comprehension as "skipped" because we don't re-run it.
    assert rounds[1]["comprehension"] == "skipped"
    # Muninn was asked to revise once.
    assert sum(1 for r in log_records if r.get("type") == "muninn_design_revised") == 1


async def test_three_pass_comprehension_unclear_does_not_block(monkeypatch) -> None:
    """Comprehension flagged as unclear is informational - if critic and
    readiness both pass, the round is ready and we go straight to implement.
    The unclear paraphrase is still logged."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    huginn_factory, idx = _make_seq_huginn_factory([
        "section X is too vague.\ncomprehension unclear",
        "no gaps.\ndesign ready",
        "no questions.\nimplementation ready",
    ])
    ctx = _make_three_pass_ctx(huginn_factory, log_records)
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    assert idx[0] == 3
    rounds = [r for r in log_records if r.get("type") == "design_check_round"]
    assert rounds[0]["combined"] == "ready"
    assert rounds[0]["comprehension"] == "revise"
    # No revision happened because the combined verdict was ready.
    assert not any(r.get("type") == "muninn_design_revised" for r in log_records)


async def test_three_pass_revise_bundle_has_both_sections() -> None:
    """The bundle fed to the revise prompt labels critic and readiness
    sections separately so the model can address both."""
    from muninn.workflows import _combined_critique
    bundle = _combined_critique(
        comprehension_text="",
        critic_text="1. interface vague.\ndesign needs revision",
        readiness_text="1. file path missing.\nimplementation not ready",
    )
    assert "=== CRITIC GAPS ===" in bundle
    assert "=== READINESS OPEN QUESTIONS ===" in bundle
    assert "=== COMPREHENSION FEEDBACK" not in bundle  # empty -> omitted

    bundle_with_compr = _combined_critique(
        comprehension_text="section X reads ambiguously.\ncomprehension unclear",
        critic_text="1. gap",
        readiness_text="1. q",
    )
    assert "=== COMPREHENSION FEEDBACK" in bundle_with_compr


async def test_verdict_accepts_custom_tokens() -> None:
    """The generalized parser handles the new comprehension and readiness
    closing strings via explicit tokens."""
    assert _verdict(
        "doc paraphrase\ncomprehension passed",
        ready_token="comprehension passed",
        revise_token="comprehension unclear",
    ) == "ready"
    assert _verdict(
        "section X unclear\ncomprehension unclear",
        ready_token="comprehension passed",
        revise_token="comprehension unclear",
    ) == "revise"
    assert _verdict(
        "open question\nimplementation not ready",
        ready_token="implementation ready",
        revise_token="implementation not ready",
    ) == "revise"


# --- /bug tests ----------------------------------------------------------


def _make_bug_ctx(
    *,
    muninn_output: str,
    huginn_outputs: list[str],
    ask_user_fn,
    log_records: list[dict],
):
    """Helper: build a BugRunCtx wired to fake agents that emit canned text."""
    async def mount_md(_h):
        return _FakeMd()

    muninn = _FakeAgent("muninn", muninn_output)
    idx = [0]

    def huginn_factory():
        out = (huginn_outputs[idx[0]]
               if idx[0] < len(huginn_outputs)
               else "design ready")
        idx[0] += 1
        return _FakeAgent("huginn", out)

    return BugRunCtx(
        description="dummy bug",
        muninn_agent=muninn,
        huginn_agent_factory=huginn_factory,
        muninn_history=[],
        bug_ground_prompt="ground: {description}",
        bug_problem_prompt="problem: {description}",
        bug_critique_prompt="critique: {problem_doc}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
        ask_user=ask_user_fn,
    )


async def test_bug_flow_happy_path(monkeypatch) -> None:
    """Huginn signs off on round 1; flow runs problem doc + test + fix."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["clean.\ndesign ready"],
        ask_user_fn=_ask_user_stub,
        log_records=log_records,
    )
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    # Step 1 (ground) + step 2 (problem) + step 4 (test) + step 5 (fix) =
    # 4 muninn agent calls. Plus the ground log.
    muninn_grounded = [r for r in log_records if r.get("type") == "muninn_grounded"]
    assert len(muninn_grounded) == 1
    test_writes = [r for r in log_records if r.get("type") == "muninn_bug_test_written"]
    assert len(test_writes) == 1


async def test_bug_flow_one_revision_round(monkeypatch) -> None:
    """Round 1 says revise, round 2 says ready, then test + fix proceed."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=[
            "1. gap.\ndesign needs revision",
            "ok.\ndesign ready",
        ],
        ask_user_fn=_ask_user_stub,
        log_records=log_records,
    )
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    verdicts = [r for r in log_records if r.get("type") == "huginn_verdict"]
    assert len(verdicts) == 2
    assert verdicts[0]["verdict"] == "revise"
    assert verdicts[1]["verdict"] == "ready"
    # Revision log entry exists.
    assert any(r.get("type") == "muninn_bug_problem_revised" for r in log_records)


async def test_bug_flow_user_cancels_at_backstop(monkeypatch) -> None:
    """Huginn never converges; user picks cancel at the backstop."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []

    async def ask_cancel(_q, opts):
        return _pick(opts, "cancel")

    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["bad.\ndesign needs revision"] * 10,
        ask_user_fn=ask_cancel,
        log_records=log_records,
    )
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_cancelled"
    assert summary["test_len"] == 0
    assert summary["fix_len"] == 0


async def test_bug_flow_user_resolves_directly(monkeypatch) -> None:
    """Huginn keeps flagging; user picks 'let me answer the remaining gaps'."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []

    async def ask_resolve(_q, opts):
        return _pick(opts, "let me answer")

    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["bad.\ndesign needs revision"] * 10,
        ask_user_fn=ask_resolve,
        log_records=log_records,
    )
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    assert any(r.get("type") == "muninn_bug_user_resolved" for r in log_records)


# --- /precommit-review tests ----------------------------------------------


def test_has_no_findings_parser() -> None:
    assert _has_no_findings("1. foo\n\nno findings") is True
    assert _has_no_findings("1. foo\nfindings flagged") is False
    assert _has_no_findings("rambling output without verdict") is False
    assert _has_no_findings("") is False


def test_python_syntax_check_skips_when_no_py_files(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# hi\n")
    res = _python_syntax_check(tmp_path, ["README.md"])
    assert res["status"] == "skip"
    assert "no python files" in res["summary"]


def test_python_syntax_check_passes_clean_file(tmp_path) -> None:
    (tmp_path / "good.py").write_text("x = 1\n")
    res = _python_syntax_check(tmp_path, ["good.py"])
    assert res["status"] == "pass"


def test_python_syntax_check_reports_failures(tmp_path) -> None:
    (tmp_path / "bad.py").write_text("def f(:\n  pass\n")
    (tmp_path / "good.py").write_text("y = 2\n")
    res = _python_syntax_check(tmp_path, ["bad.py", "good.py"])
    assert res["status"] == "fail"
    assert "bad.py" in res["summary"]


def test_python_syntax_check_skips_files_that_disappeared(tmp_path) -> None:
    """git diff may list a renamed/deleted file - we only check what exists."""
    res = _python_syntax_check(tmp_path, ["nonexistent.py"])
    assert res["status"] == "skip"


def _git_init_with_pending_change(tmp_path) -> None:
    """Create a tmp git repo with one committed file and one pending edit."""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    f = tmp_path / "thing.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    f.write_text("x = 1\ny = 2\n")  # pending change


async def test_review_flow_skips_when_no_pending_changes(tmp_path, monkeypatch) -> None:
    import subprocess
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    f = tmp_path / "thing.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    async def mount_md(_h): return _FakeMd()
    log_records: list[dict] = []

    ctx = ReviewRunCtx(
        cwd=tmp_path,
        huginn_agent_factory=lambda: _FakeAgent("h", "no findings"),
        review_prompt="checks={checks}\ndiff={diff}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await precommit_review_flow(ctx)
    assert summary["type"] == "review_skipped"


async def test_review_flow_aborts_when_not_a_git_repo(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h): return _FakeMd()
    log_records: list[dict] = []

    ctx = ReviewRunCtx(
        cwd=tmp_path,
        huginn_agent_factory=lambda: _FakeAgent("h", "no findings"),
        review_prompt="checks={checks}\ndiff={diff}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await precommit_review_flow(ctx)
    assert summary["type"] == "review_aborted"
    assert "git" in summary["reason"]


async def test_review_flow_logs_detected_stack(tmp_path, monkeypatch) -> None:
    """A repo with Cargo.toml is detected as 'rust' even with no .rs in diff."""
    import subprocess
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\nversion='0.1'\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    # Pending edit to a non-Rust file - rust checks should be skipped (no
    # relevant extension), but the stack detection should still log "rust".
    (tmp_path / "README.md").write_text("# hi\n")

    async def mount_md(_h): return _FakeMd()
    log_records: list[dict] = []
    ctx = ReviewRunCtx(
        cwd=tmp_path,
        huginn_agent_factory=lambda: _FakeAgent("h", "no findings"),
        review_prompt="checks={checks}\ndiff={diff}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await precommit_review_flow(ctx)
    assert summary["type"] == "review_complete"
    assert any(r.get("type") == "stack_detected" and r.get("stack") == "rust"
               for r in log_records)
    # No rust check should have run because the diff has no .rs files.
    rust_check_runs = [r for r in log_records
                       if r.get("type") == "review_check"
                       and r.get("name") in {"cargo clippy", "cargo check", "cargo test"}]
    assert rust_check_runs == []


async def test_review_flow_runs_huginn_with_diff(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    _git_init_with_pending_change(tmp_path)

    seen_prompts: list[str] = []

    class _SpyAgent:
        async def run_stream_events(self, prompt, *, message_history=None, model_settings=None):
            from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
            from pydantic_ai.run import AgentRunResultEvent
            seen_prompts.append(prompt)
            yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="1. thing.py:2 missing newline\nWhy: style.\nFix: add newline.\nfindings flagged"))
            class _R:
                def __init__(self, text): self.output = text
                def new_messages(self): return []
            yield AgentRunResultEvent(result=_R("done"))

    async def mount_md(_h): return _FakeMd()
    log_records: list[dict] = []

    ctx = ReviewRunCtx(
        cwd=tmp_path,
        huginn_agent_factory=_SpyAgent,
        review_prompt="checks={checks}\n---\ndiff={diff}",
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await precommit_review_flow(ctx)
    assert summary["type"] == "review_complete"
    assert summary["no_findings"] is False
    # Huginn was given a real diff.
    assert len(seen_prompts) == 1
    assert "+y = 2" in seen_prompts[0]
    # Verdict was logged.
    assert any(r.get("type") == "review_verdict" for r in log_records)


async def test_review_flow_final_line_does_not_use_success_marker_when_findings_flagged(
    tmp_path, monkeypatch,
) -> None:
    """When Huginn flags findings, the Muninn pane's final summary must NOT
    carry the green ✅ "complete" marker - that visually contradicts the
    verdict and makes Muninn look like it ignored the review and stopped.
    """
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    _git_init_with_pending_change(tmp_path)

    muninn_headers: list[str] = []

    async def mount_muninn(header: str):
        muninn_headers.append(header)
        return _FakeMd()

    async def mount_huginn(_h):
        return _FakeMd()

    ctx = ReviewRunCtx(
        cwd=tmp_path,
        huginn_agent_factory=lambda: _FakeAgent(
            "h",
            "1. thing.py:2 missing newline\nWhy: style.\nFix: add newline.\n"
            "findings flagged",
        ),
        review_prompt="checks={checks}\ndiff={diff}",
        model_settings=ModelSettings(),
        log=lambda _r: None,
        mount_muninn_md=mount_muninn,
        mount_huginn_md=mount_huginn,
    )
    summary = await precommit_review_flow(ctx)
    assert summary["type"] == "review_complete"
    assert summary["no_findings"] is False

    # The final summary line is the one that mentions the reviewer verdict.
    finals = [h for h in muninn_headers
              if "/precommit-review" in h and "reviewer:" in h]
    assert finals, f"no final summary line in headers: {muninn_headers}"
    final = finals[-1]
    assert "findings flagged" in final
    # The bug: ✅ + "complete" appear regardless of Huginn's verdict.
    assert "✅" not in final, (
        "green check on a findings-flagged review reads as success "
        f"and contradicts the verdict: {final!r}"
    )


# ---------------------------------------------------------------------------
# freedom_level backstop short-circuit
# ---------------------------------------------------------------------------


async def test_feature_backstop_auto_proceeds_at_high(monkeypatch) -> None:
    """At freedom_level='high', the unconverged backstop never asks - it
    logs feature_unconverged_autoproceed and falls through to implementation."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    asked: list[int] = []

    async def must_not_ask(_q, _opts):
        asked.append(1)
        return "should-not-be-called"

    ctx = _make_unconverged_ctx(must_not_ask, log_records)
    ctx.freedom_level = "high"
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    assert asked == [], "ask_user must not be called at freedom_level=high"
    assert any(r.get("type") == "feature_unconverged_autoproceed"
               for r in log_records)


async def test_feature_backstop_asks_at_medium(monkeypatch) -> None:
    """At freedom_level='medium', the existing 4-option backstop still fires."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    seen_options: list[list] = []

    async def ask_proceed(_q, opts):
        seen_options.append(list(opts))
        return _pick(opts, "proceed")

    ctx = _make_unconverged_ctx(ask_proceed, log_records)
    ctx.freedom_level = "medium"
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    assert len(seen_options) >= 1, "backstop must call ask_user at medium"
    assert not any(r.get("type") == "feature_unconverged_autoproceed"
                   for r in log_records)


async def test_feature_backstop_asks_at_low(monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    asked: list[int] = []

    async def ask_proceed(_q, opts):
        asked.append(1)
        return _pick(opts, "proceed")

    ctx = _make_unconverged_ctx(ask_proceed, log_records)
    ctx.freedom_level = "low"
    summary = await feature_flow(ctx)
    assert summary["type"] == "feature_complete"
    assert asked, "low must ask"
    assert not any(r.get("type") == "feature_unconverged_autoproceed"
                   for r in log_records)


async def test_bug_backstop_auto_proceeds_at_high(monkeypatch) -> None:
    """Bug-flow parity: high skips the 4-option backstop and runs through
    failing-test + fix steps."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    asked: list[int] = []

    async def must_not_ask(_q, _opts):
        asked.append(1)
        return "should-not-be-called"

    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["bad.\ndesign needs revision"] * 10,
        ask_user_fn=must_not_ask,
        log_records=log_records,
    )
    ctx.freedom_level = "high"
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    assert asked == [], "ask_user must not be called at freedom_level=high"
    assert any(r.get("type") == "bug_unconverged_autoproceed"
               for r in log_records)


async def test_bug_backstop_asks_at_medium(monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    asked: list[int] = []

    async def ask_proceed(_q, opts):
        asked.append(1)
        return _pick(opts, "proceed")

    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["bad.\ndesign needs revision"] * 10,
        ask_user_fn=ask_proceed,
        log_records=log_records,
    )
    ctx.freedom_level = "medium"
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    assert asked, "medium must call ask_user"
    assert not any(r.get("type") == "bug_unconverged_autoproceed"
                   for r in log_records)


async def test_bug_backstop_asks_at_low(monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    log_records: list[dict] = []
    asked: list[int] = []

    async def ask_proceed(_q, opts):
        asked.append(1)
        return _pick(opts, "proceed")

    ctx = _make_bug_ctx(
        muninn_output="problem doc",
        huginn_outputs=["bad.\ndesign needs revision"] * 10,
        ask_user_fn=ask_proceed,
        log_records=log_records,
    )
    ctx.freedom_level = "low"
    summary = await bug_flow(ctx)
    assert summary["type"] == "bug_complete"
    assert asked, "low must call ask_user"


# =====================================================================
# /brainstorm and /prd tests
# =====================================================================

import pytest


@pytest.mark.parametrize("desc, expected", [
    ("hello world", "hello-world"),
    ("Multiple   Spaces!!!", "multiple-spaces"),
    ("", "idea"),
    ("你好", "idea"),                       # non-ASCII stripped -> empty -> 'idea'
    ("***", "idea"),
    ("foo--bar--", "foo-bar"),
    ("a" * 100, "a" * 60),                  # truncated to max_len
    ("Hello, World - 2026!", "hello-world-2026"),
])
def test_slug_for_path_edge_cases(desc, expected) -> None:
    assert _slug_for_path(desc) == expected


def test_slug_for_path_truncate_strips_trailing_dash(tmp_path) -> None:
    # If truncation lands the cut on a non-alnum boundary, no trailing dash.
    desc = "a" * 58 + "!!" + "b" * 10
    out = _slug_for_path(desc, max_len=60)
    assert not out.endswith("-")
    assert len(out) <= 60


def test_unique_artifact_path_no_collision(tmp_path) -> None:
    from datetime import datetime
    fake_now = datetime(2026, 5, 7, 14, 30, 0)
    p = _unique_artifact_path(tmp_path, "brainstorms", "test-idea", now=fake_now)
    assert p == tmp_path / "docs" / "brainstorms" / "test-idea-2026-05-07.md"
    # Helper does NOT create the file or its parents.
    assert not p.exists()
    assert not p.parent.exists()


def test_unique_artifact_path_collision_falls_back_to_hhmmss(tmp_path) -> None:
    from datetime import datetime
    fake_now = datetime(2026, 5, 7, 14, 30, 5)
    primary_dir = tmp_path / "docs" / "brainstorms"
    primary_dir.mkdir(parents=True)
    primary = primary_dir / "test-2026-05-07.md"
    primary.write_text("existing", encoding="utf-8")

    p = _unique_artifact_path(tmp_path, "brainstorms", "test", now=fake_now)
    assert p.name == "test-2026-05-07-143005.md"


def test_unique_artifact_path_double_collision_raises(tmp_path) -> None:
    from datetime import datetime
    import pytest as _pt
    fake_now = datetime(2026, 5, 7, 14, 30, 5)
    d = tmp_path / "docs" / "brainstorms"
    d.mkdir(parents=True)
    (d / "test-2026-05-07.md").write_text("a", encoding="utf-8")
    (d / "test-2026-05-07-143005.md").write_text("b", encoding="utf-8")
    with _pt.raises(FileExistsError):
        _unique_artifact_path(tmp_path, "brainstorms", "test", now=fake_now)


def test_format_lens_outputs_disjoint_invariant() -> None:
    # results and failures must be disjoint and together cover order.
    text = _format_lens_outputs(
        ("a", "b", "c"),
        results={"a": "alpha", "b": "beta"},
        failures={"c": "ConnectionError"},
    )
    assert "--- LENS: a ---\nalpha\n--- END LENS: a ---" in text
    assert "--- LENS: b ---\nbeta\n--- END LENS: b ---" in text
    assert "(lens unavailable: ConnectionError)" in text


def test_format_lens_outputs_overlap_assertion() -> None:
    import pytest as _pt
    with _pt.raises(AssertionError):
        _format_lens_outputs(
            ("a",),
            results={"a": "x"},
            failures={"a": "X"},   # overlap is forbidden
        )


def test_format_lens_outputs_missing_lens_assertion() -> None:
    import pytest as _pt
    with _pt.raises(AssertionError):
        _format_lens_outputs(
            ("a", "b"),
            results={"a": "x"},
            failures={},   # 'b' missing from both
        )


def _make_keyed_huginn_factory(
    lens_order: tuple[str, ...],
    outputs: dict[str, str] | None = None,
    errors: dict[str, BaseException] | None = None,
):
    """Test helper: factory returns _FakeAgent instances keyed by call index
    into lens_order. Outputs and errors are looked up by lens name; missing
    keys default to a generic output / no error.

    Pattern: workflows._fan_out_lenses calls factory() in lens_order tuple
    sequence (Step 1 sequential mint), so call N corresponds to lens_order[N].

    Returns (factory, agents) where agents is a list mutated as instances
    are minted - tests can inspect them after the flow.
    """
    outputs = outputs or {}
    errors = errors or {}
    counter = [0]
    agents: list[_FakeAgent] = []

    def factory():
        idx = counter[0]
        counter[0] += 1
        if idx >= len(lens_order):
            # Defensive: extra calls return a benign agent.
            agent = _FakeAgent(f"huginn-extra-{idx}", "extra")
        else:
            lens = lens_order[idx]
            agent = _FakeAgent(
                f"huginn-{lens}",
                outputs.get(lens, f"{lens} lens output"),
                error=errors.get(lens),
            )
        agents.append(agent)
        return agent

    return factory, agents


# ---- /brainstorm ----------------------------------------------------

def _brainstorm_test_prompts() -> dict:
    """Minimal valid prompt strings for tests; placeholders are mandatory."""
    return {
        "ground": "ground: {description}",
        "lens": {
            lens: f"lens-{lens}: {{description}} :: {{ground_brief}}"
            for lens in BRAINSTORM_LENSES
        },
        "synth": (
            "synth: {description} :: {ground_brief} :: {lens_outputs}"
        ),
    }


async def test_brainstorm_flow_writes_artifact(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    log_records: list[dict] = []

    async def mount_md(_h):
        return _FakeMd()

    muninn = _FakeAgent("muninn", "GROUND BRIEF... ## Brief done.")
    # Distinct outputs per turn: ground, then synthesis. After ground, we
    # need to swap muninn.output_text for synthesis. Simpler: chain two
    # FakeAgents via a mutable holder, but the existing tests show that a
    # single _FakeAgent reused for sequential calls is fine if the output
    # is shared. The synthesis can reuse the same string; len asserts are
    # what matter.
    muninn.output_text = "BRIEF\n## Brief done.\n--SEP--\nSYNTHESIS\nrecommended next: /prd"

    prompts = _brainstorm_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        BRAINSTORM_LENSES,
        outputs={
            "technical": "TECH OUTPUT",
            "contrarian": "CONTRA OUTPUT",
            "ux": "UX OUTPUT",
        },
    )

    ctx = BrainstormRunCtx(
        description="test idea",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        brainstorm_ground_prompt=prompts["ground"],
        brainstorm_lens_prompts=prompts["lens"],
        brainstorm_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await brainstorm_flow(ctx)

    assert summary["type"] == "brainstorm_complete"
    assert summary["description"] == "test idea"
    assert summary["artifact_path"].startswith("docs/brainstorms/test-idea-")
    assert summary["artifact_path"].endswith(".md")
    assert summary["lens_failures"] == {}
    assert set(summary["lens_outputs"].keys()) == set(BRAINSTORM_LENSES)
    # Artifact actually written and contains all lens outputs.
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "# Brainstorm: test idea" in art
    assert "TECH OUTPUT" in art
    assert "CONTRA OUTPUT" in art
    assert "UX OUTPUT" in art
    # JSONL events for each lens.
    types = [r.get("type") for r in log_records]
    assert "brainstorm_started" in types
    assert types.count("lens_started") == 3
    assert types.count("lens_completed") == 3
    assert "brainstorm_synthesizing" in types
    assert "brainstorm_complete" in types


async def test_brainstorm_flow_one_lens_fails(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    muninn = _FakeAgent("muninn", "ground brief\n## Brief done.")

    prompts = _brainstorm_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        BRAINSTORM_LENSES,
        outputs={"technical": "TECH", "ux": "UX"},
        errors={"contrarian": ConnectionError("ollama dead mid-stream")},
    )

    ctx = BrainstormRunCtx(
        description="failing-lens-test",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        brainstorm_ground_prompt=prompts["ground"],
        brainstorm_lens_prompts=prompts["lens"],
        brainstorm_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await brainstorm_flow(ctx)

    # Synthesis still runs (>=1 lens succeeded); artifact written.
    assert summary["type"] == "brainstorm_complete"
    assert summary["lens_failures"] == {"contrarian": "ConnectionError"}
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "(lens unavailable: ConnectionError)" in art
    assert "TECH" in art and "UX" in art
    # Synthesis prompt saw the unavailable placeholder.
    synth_prompts = [p for p in muninn.received_prompts
                     if "synth:" in p and "lens_outputs" not in p]
    assert any("(lens unavailable: ConnectionError)" in p
               for p in muninn.received_prompts)
    assert any("--- LENS: contrarian ---" in p for p in muninn.received_prompts)


async def test_brainstorm_flow_all_lenses_fail(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    muninn = _FakeAgent("muninn", "ground brief\n## Brief done.")

    prompts = _brainstorm_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        BRAINSTORM_LENSES,
        errors={
            lens: ConnectionError(f"{lens} dead") for lens in BRAINSTORM_LENSES
        },
    )

    ctx = BrainstormRunCtx(
        description="all-fail",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        brainstorm_ground_prompt=prompts["ground"],
        brainstorm_lens_prompts=prompts["lens"],
        brainstorm_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await brainstorm_flow(ctx)

    assert summary["type"] == "brainstorm_failed"
    assert "all 3 lenses failed" in summary["reason"]
    assert summary["lens_failures"] == {
        lens: "ConnectionError" for lens in BRAINSTORM_LENSES
    }
    # NO artifact written.
    brainstorms_dir = tmp_path / "docs" / "brainstorms"
    assert not brainstorms_dir.exists() or not list(brainstorms_dir.glob("*.md"))
    # Synthesis NOT called (muninn.call_count == 1: only ground).
    assert muninn.call_count == 1, (
        f"synthesis must not run when all lenses fail; "
        f"muninn.call_count={muninn.call_count}"
    )


# ---- /prd ----------------------------------------------------------

def _prd_test_prompts() -> dict:
    return {
        "ground": "prd-ground: {description}",
        "qa": "prd-qa: {description} :: {ground_brief}",
        "lens": {
            lens: f"prd-lens-{lens}: {{description}} :: {{ground_brief}} :: {{qa_summary}}"
            for lens in PRD_LENSES
        },
        "synth": "prd-synth: {description} :: {ground_brief} :: {qa_summary} :: {lens_outputs}",
    }


async def test_prd_flow_writes_artifact_with_qa(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    muninn = _FakeAgent(
        "muninn",
        # Ground turn output (only the first call uses this; we don't
        # rotate output_text between turns - synthesis call sees the same
        # text, which is fine for happy-path verification).
        "ground brief\n## Brief done.\nQ&A summary\n- Q: x\n  A: y\nqa complete",
    )

    prompts = _prd_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        PRD_LENSES,
        outputs={
            "prior_art": "PRIOR ART",
            "edge_cases": "EDGE CASES",
            "integration": "INTEGRATION",
        },
    )

    ctx = PRDRunCtx(
        description="persistent transcript pane",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        prd_ground_prompt=prompts["ground"],
        prd_qa_prompt=prompts["qa"],
        prd_lens_prompts=prompts["lens"],
        prd_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await prd_flow(ctx)

    assert summary["type"] == "prd_complete"
    assert summary["artifact_path"].startswith("docs/prds/persistent-transcript-pane-")
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "# PRD: persistent transcript pane" in art
    assert "## Q&A summary" in art
    assert "PRIOR ART" in art and "EDGE CASES" in art and "INTEGRATION" in art
    types = [r.get("type") for r in log_records]
    assert "prd_started" in types
    assert "prd_grounded" in types
    assert "prd_qa_started" in types
    # qa_complete recognized via "qa complete" tail token.
    assert any(t in types for t in ("prd_qa_complete", "prd_qa_no_token")), types
    assert "prd_synthesizing" in types
    assert "prd_complete" in types


async def test_prd_flow_zero_questions(tmp_path, monkeypatch) -> None:
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    muninn = _FakeAgent(
        "muninn",
        "ground brief\n## Brief done.\nno clarifications gathered",
    )

    prompts = _prd_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(PRD_LENSES)

    ctx = PRDRunCtx(
        description="trivial-prd",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        prd_ground_prompt=prompts["ground"],
        prd_qa_prompt=prompts["qa"],
        prd_lens_prompts=prompts["lens"],
        prd_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await prd_flow(ctx)

    assert summary["type"] == "prd_complete"
    types = [r.get("type") for r in log_records]
    assert "prd_qa_skipped" in types, (
        f"expected prd_qa_skipped in events; got {types}"
    )
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "no clarifications gathered" in art


# ---- artifact body ------------------------------------------------

def test_artifact_body_brainstorm_full() -> None:
    from datetime import date
    body = _artifact_body(
        flow_kind="brainstorm",
        description="test idea",
        today=date(2026, 5, 7),
        synthesis_text="The synthesis.",
        lens_order=("technical", "contrarian", "ux"),
        lens_results={"technical": "tech-out", "ux": "ux-out"},
        lens_failures={"contrarian": "ConnectionError"},
    )
    assert body.startswith("# Brainstorm: test idea\n")
    assert "Generated 2026-05-07 by `/brainstorm`." in body
    assert "## Synthesis" in body
    assert "tech-out" in body
    assert "(lens unavailable: ConnectionError)" in body


def test_artifact_body_prd_includes_qa() -> None:
    from datetime import date
    body = _artifact_body(
        flow_kind="prd",
        description="some prd",
        today=date(2026, 5, 7),
        synthesis_text="THE PRD",
        lens_order=PRD_LENSES,
        lens_results={lens: f"out-{lens}" for lens in PRD_LENSES},
        lens_failures={},
        qa_summary="- Q: a\n  A: b\nqa complete",
    )
    assert "# PRD: some prd" in body
    assert "## Q&A summary" in body
    assert "- Q: a" in body


# ---- partial-synthesis recovery (synthesis raises but lenses succeeded) ----

class _SequencedFakeAgent(_FakeAgent):
    """Variant of _FakeAgent that switches output (and optionally raises)
    based on call_count. Used for tests that need the same agent reference
    to behave differently across the ground/synthesis turns of a flow.

    `outputs` is a list indexed by call_count (1-based after the first call
    increment). `errors_by_call` maps call_count -> exception to raise
    after the first delta of that call.
    """

    def __init__(
        self,
        name: str,
        outputs: list[str],
        errors_by_call: dict[int, BaseException] | None = None,
    ) -> None:
        super().__init__(name=name, output_text=outputs[0] if outputs else "")
        self._outputs = outputs
        self._errors_by_call = errors_by_call or {}

    async def run_stream_events(self, prompt, *, message_history=None, model_settings=None):
        # Pick the output for this call BEFORE incrementing call_count
        # (parent's __post_init__ initializes call_count to 0).
        idx = self.call_count
        if idx < len(self._outputs):
            self.output_text = self._outputs[idx]
        # Switch error injection per-call.
        # call_count is incremented INSIDE parent's run_stream_events;
        # _errors_by_call uses 0-based indexing matching the call we're
        # about to make.
        self.error = self._errors_by_call.get(idx)
        async for ev in super().run_stream_events(
            prompt, message_history=message_history, model_settings=model_settings,
        ):
            yield ev


async def test_brainstorm_flow_synthesis_fails_lens_outputs_preserved(
    tmp_path, monkeypatch,
) -> None:
    """When synthesis raises mid-stream but lenses succeeded, the flow
    returns brainstorm_partial AND the artifact still gets written with
    lens outputs preserved. Covers the synthesis-recovery fallback path
    that was specified in design-doc gap K but had no test coverage."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    # Muninn: call 0 = ground (success), call 1 = synthesis (raises).
    muninn = _SequencedFakeAgent(
        "muninn",
        outputs=["ground brief\n## Brief done.", "partial synthesis text"],
        errors_by_call={1: ConnectionError("ollama died during synthesis")},
    )

    prompts = _brainstorm_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        BRAINSTORM_LENSES,
        outputs={"technical": "TECH", "contrarian": "CONTRA", "ux": "UX"},
    )

    ctx = BrainstormRunCtx(
        description="synthesis-fail-test",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        brainstorm_ground_prompt=prompts["ground"],
        brainstorm_lens_prompts=prompts["lens"],
        brainstorm_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await brainstorm_flow(ctx)

    assert summary["type"] == "brainstorm_partial"
    assert summary["synthesis_failed"] == "ConnectionError"
    # Artifact still written, with lens outputs and the failure marker.
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "_Synthesis failed: ConnectionError" in art
    assert "TECH" in art and "CONTRA" in art and "UX" in art
    types = [r.get("type") for r in log_records]
    assert "brainstorm_synthesis_failed" in types
    assert "brainstorm_complete" not in types


# ---- cancellation propagation through _fan_out_lenses --------------

async def test_prd_flow_all_lenses_fail(tmp_path, monkeypatch) -> None:
    """PRD twin of test_brainstorm_flow_all_lenses_fail. When all 3 PRD
    research lenses fail, synthesis must NOT run and no artifact must be
    written. Catches a regression that breaks only the PRD copy of the
    all-fail short-circuit."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    # Muninn output covers ground (call 0) and would-be QA (call 1).
    # Synthesis (call 2) must NOT happen.
    muninn = _FakeAgent(
        "muninn", "ground brief\n## Brief done.\n\n## Q&A summary\nqa complete",
    )

    prompts = _prd_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        PRD_LENSES,
        errors={lens: ConnectionError(f"{lens} dead") for lens in PRD_LENSES},
    )

    ctx = PRDRunCtx(
        description="prd-all-fail",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        prd_ground_prompt=prompts["ground"],
        prd_qa_prompt=prompts["qa"],
        prd_lens_prompts=prompts["lens"],
        prd_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await prd_flow(ctx)

    assert summary["type"] == "prd_failed"
    assert "all 3 lenses failed" in summary["reason"]
    # NO artifact written.
    prds_dir = tmp_path / "docs" / "prds"
    assert not prds_dir.exists() or not list(prds_dir.glob("*.md"))
    # Muninn called exactly twice: ground + QA. Synthesis must NOT run.
    assert muninn.call_count == 2, (
        f"synthesis must not run when all PRD lenses fail; "
        f"muninn.call_count={muninn.call_count}"
    )


async def test_prd_flow_synthesis_fails_lens_outputs_preserved(
    tmp_path, monkeypatch,
) -> None:
    """PRD twin of test_brainstorm_flow_synthesis_fails_lens_outputs_preserved.
    Synthesis raises mid-stream; flow returns prd_partial; artifact still
    written with Q&A + lens outputs + the failure marker."""
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []
    # Muninn calls in order: ground (0, ok), qa (1, ok), synthesis (2, raises).
    muninn = _SequencedFakeAgent(
        "muninn",
        outputs=[
            "ground brief\n## Brief done.",
            "## Q&A summary\n- Q: x\n  A: y\nqa complete",
            "partial PRD synthesis text",
        ],
        errors_by_call={2: ConnectionError("ollama died during prd synthesis")},
    )

    prompts = _prd_test_prompts()
    factory, _agents = _make_keyed_huginn_factory(
        PRD_LENSES,
        outputs={
            "prior_art": "PRIOR_ART_BODY",
            "edge_cases": "EDGE_CASES_BODY",
            "integration": "INTEGRATION_BODY",
        },
    )

    ctx = PRDRunCtx(
        description="prd-synthesis-fail",
        cwd=tmp_path,
        muninn_agent=muninn,
        huginn_agent_factory=factory,
        muninn_history=[],
        prd_ground_prompt=prompts["ground"],
        prd_qa_prompt=prompts["qa"],
        prd_lens_prompts=prompts["lens"],
        prd_synthesis_prompt=prompts["synth"],
        model_settings=ModelSettings(),
        log=log_records.append,
        mount_muninn_md=mount_md,
        mount_huginn_md=mount_md,
    )
    summary = await prd_flow(ctx)

    assert summary["type"] == "prd_partial"
    assert summary["synthesis_failed"] == "ConnectionError"
    art = (tmp_path / summary["artifact_path"]).read_text(encoding="utf-8")
    assert "_Synthesis failed: ConnectionError" in art
    assert "## Q&A summary" in art
    assert "PRIOR_ART_BODY" in art
    assert "EDGE_CASES_BODY" in art
    assert "INTEGRATION_BODY" in art
    types = [r.get("type") for r in log_records]
    assert "prd_synthesis_failed" in types
    assert "prd_complete" not in types


async def test_fan_out_lenses_propagates_cancelled_error(tmp_path, monkeypatch) -> None:
    """Esc cancellation must propagate through asyncio.gather. The lens
    task's `except Exception` (NOT BaseException) lets CancelledError
    fall through. A future refactor that widens the catch to bare except
    or BaseException would silently swallow Esc and leave workers
    half-running - this test is the canary against that.
    """
    import asyncio as _asyncio
    from textual.widgets import Markdown
    monkeypatch.setattr(Markdown, "get_stream",
                        classmethod(lambda cls, md: _FakeStream(md)))
    from muninn.workflows import _fan_out_lenses

    async def mount_md(_h):
        return _FakeMd()

    log_records: list[dict] = []

    class _CancelOnFirstChunkAgent:
        async def run_stream_events(self, prompt, *, message_history=None, model_settings=None):
            from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
            yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="x"))
            # Simulate the Textual cancel path: Esc triggers
            # workers.cancel_group, which raises CancelledError into the
            # awaited stream. The `except Exception` in _fan_out_lenses
            # MUST let this propagate (CancelledError inherits from
            # BaseException, not Exception, since Python 3.8).
            raise _asyncio.CancelledError()

    factory = lambda: _CancelOnFirstChunkAgent()

    with __import__('pytest').raises(_asyncio.CancelledError):
        await _fan_out_lenses(
            huginn_agent_factory=factory,
            lens_prompts={lens: f"prompt-{lens}" for lens in BRAINSTORM_LENSES},
            lens_order=BRAINSTORM_LENSES,
            mount_huginn_md=mount_md,
            log=log_records.append,
            model_settings=ModelSettings(),
            flow_label="brainstorm",
        )
