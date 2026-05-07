"""Workflows: /feature (design loop) and /precommit-review (diff cold-read)."""
from __future__ import annotations

import ast
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from textual.widgets import Markdown

from .stacks import (
    CheckSpec,
    StackProfile,
    build_command,
    detect_stack,
    relevant_changes,
)
from .streaming import run_and_stream

LogFn = Callable[[dict[str, Any]], None]
MdFactory = Callable[[str], Awaitable[Markdown]]  # async, mounts a fresh Markdown into a pane and returns it


# Default cap; the App passes the configured value via FeatureRunCtx.
# Mirrors .claude/commands/eg-new-feature.md.
MAX_REVISION_ROUNDS = 3


@dataclass
class FeatureRunCtx:
    description: str
    muninn_agent: Agent
    huginn_agent_factory: Callable[[], Agent]  # returns a fresh stateless Huginn each call
    muninn_history: list[ModelMessage]
    feature_ground_prompt: str   # contains "{description}"
    feature_design_prompt: str   # contains "{description}"
    feature_critique_prompt: str  # contains "{design_doc}"
    model_settings: ModelSettings
    log: LogFn
    mount_muninn_md: MdFactory   # mount in left pane
    mount_huginn_md: MdFactory   # mount in right pane
    ask_user: Callable[[str, list[str]], Awaitable[str]]  # blocks awaiting user choice
    # The article's three-goldfish design check: comprehension (round 1
    # only, sanity-checks the doc reads cleanly), critic (every round,
    # finds gaps), readiness (every round, gates first-pass
    # implementability). A round passes only if BOTH critic and readiness
    # return ready; comprehension is informational. Defaults are empty
    # strings so call sites that don't yet pass them fall back to the
    # bundled prompt via load_prompt at app wiring time. The fields are
    # required at runtime when feature_flow runs.
    feature_comprehension_prompt: str = ""  # contains "{design_doc}"
    feature_readiness_prompt: str = ""      # contains "{design_doc}"
    max_revision_rounds: int = MAX_REVISION_ROUNDS
    # Snapshot of MuninnTUI.freedom_level at workflow start. The non-
    # convergence backstop short-circuits at "high" (auto-proceed to
    # implementation); low and medium keep the 4-option ask_user.
    freedom_level: str = "low"


@dataclass(frozen=True)
class _ColdReadKind:
    """Descriptor for one Huginn cold-read pass.

    Each design-stage pass (comprehension / critic / readiness) prints a
    different closing token so the same _verdict parser can serve all of
    them by being told which tokens to look for. The label drives the
    in-flight header and the verdict callout in the Huginn pane.
    """
    label: str          # human label, e.g. "Comprehension", "Critic", "Readiness"
    ready_token: str    # closing string the goldfish prints when the pass is clean
    revise_token: str   # closing string when the pass flags issues
    ready_callout: str  # markdown shown after the stream when verdict == "ready"
    revise_callout: str
    log_kind: str       # value of the "kind" key in the huginn_verdict log record


_KIND_CRITIC = _ColdReadKind(
    label="Critic",
    ready_token="design ready",
    revise_token="design needs revision",
    ready_callout="✅ design ready",
    revise_callout="❌ design needs revision",
    log_kind="critic",
)

# /bug's single Huginn pass uses the same closing tokens as the critic
# (the bug-critique prompt deliberately reuses "design ready" / "design
# needs revision" so the parser is one place) but it is semantically a
# diagnosis check, not a design-stage critic. Distinct log_kind keeps the
# two readable apart in the JSONL log.
_KIND_DIAGNOSIS = _ColdReadKind(
    label="Diagnosis",
    ready_token="design ready",
    revise_token="design needs revision",
    ready_callout="✅ design ready",
    revise_callout="❌ design needs revision",
    log_kind="diagnosis",
)

_KIND_COMPREHENSION = _ColdReadKind(
    label="Comprehension",
    ready_token="comprehension passed",
    revise_token="comprehension unclear",
    ready_callout="✅ doc reads cleanly",
    revise_callout="⚠️ doc unclear in places",
    log_kind="comprehension",
)

_KIND_READINESS = _ColdReadKind(
    label="Readiness",
    ready_token="implementation ready",
    revise_token="implementation not ready",
    ready_callout="✅ first-pass implementable",
    revise_callout="❌ open questions remain",
    log_kind="readiness",
)


def _verdict_callout(huginn_n: int, verdict: str, kind: _ColdReadKind = _KIND_DIAGNOSIS) -> str:
    """Markdown for a prominent post-stream verdict line in the Huginn pane.

    Two spaces after the eye glyph so the emoji's variation-selector width
    doesn't eat the gap before the bold text (matches the in-flight label).
    """
    body = (
        kind.ready_callout if verdict == "ready"
        else kind.revise_callout if verdict == "revise"
        else "⁉️ verdict unparseable"
    )
    return f"👁️  **Huginn #{huginn_n} ({kind.label}) says:** {body}"


def _build_revise_prompt(critique_text: str, *, multi_section: bool = False) -> str:
    """Force point-by-point engagement with the critique. Without this, local
    models tend to rephrase the original doc with different vague language,
    which produces a near-identical critique on the next round.

    multi_section=True signals the critique bundles output from multiple
    Huginn passes (critic + readiness, sometimes comprehension). The intro
    line then reminds Muninn to address every gap across ALL labeled
    sections, not just one. /bug still calls with the default (single
    Huginn diagnosis pass)."""
    intro = (
        "Huginn ran multiple cold-read passes against your design doc; the "
        "combined NUMBERED gaps from each section are below. Address EVERY "
        "numbered gap across ALL sections (CRITIC GAPS, READINESS OPEN "
        "QUESTIONS, and COMPREHENSION FEEDBACK if listed) - do not collapse "
        "or skip a section because the numbering restarts:"
        if multi_section
        else "Huginn returned this NUMBERED critique of your design doc:"
    )
    return (
        f"{intro}\n\n"
        f"{critique_text}\n\n"
        "**No-code gate still applies.** This turn revises the DESIGN DOC; "
        "do NOT call write_file or modify any source file. read_file and "
        "read-only run_shell are encouraged for sharpening citations. Code "
        "gets written only after both critic and readiness gates close.\n\n"
        "You MUST address every numbered gap. For your output, do TWO things "
        "in this exact order:\n"
        "\n"
        "## Part 1 - Point-by-point response\n"
        "For EACH numbered gap above, write ONE of:\n"
        "\n"
        "  N. <one-line restatement of the gap>\n"
        "     Resolution: <how the rewritten doc addresses it - name the\n"
        "     specific section, the concrete criterion you added, or the\n"
        "     specific file:line you'll cite>.\n"
        "\n"
        "  N. <gap restatement>\n"
        "     Rebuttal: <verbatim reason this gap is invalid, citing what\n"
        "     you saw in the project earlier (CONTEXT BRIEF)>.\n"
        "\n"
        "  N. <gap restatement>\n"
        "     Clarify: <call ask_user(question, options) RIGHT NOW with\n"
        "     concrete labelled options - do not guess. Use this when the gap\n"
        "     genuinely requires user input that you cannot derive from the\n"
        "     project files. After the user answers, write the Resolution\n"
        "     using their answer.>\n"
        "\n"
        "Do NOT skip any number. Do NOT collapse multiple gaps into one bullet.\n"
        "Prefer Clarify over inventing an answer when a gap is about user intent\n"
        "(e.g. naming, behavior, scope, look-and-feel). Prefer Resolution when the\n"
        "gap is about something you can verify by reading more files.\n"
        "\n"
        "## Part 2 - Revised design doc\n"
        "Below the point-by-point response, output the FULL revised design "
        "doc. Every section must be MORE specific than the previous draft:\n"
        "  - replace any vague phrase ('clear', 'meaningful', 'appropriate', "
        "'as needed') with a concrete criterion or quoted snippet;\n"
        "  - add `path/to/file.ext:LINE` citations wherever the doc claims a "
        "fact about the codebase;\n"
        "  - turn every failure mode into a testable observation;\n"
        "  - turn every verification criterion into a runnable command or "
        "an exact pytest test name.\n"
        "\n"
        "If you cannot make a section more specific without reading more "
        "files, call read_file or run_shell BEFORE writing the revised doc."
    )


async def _cold_read_once(
    *,
    huginn_agent_factory: Callable[[], Agent],
    formatted_critique_prompt: str,
    huginn_n: int,
    mount_huginn_md: MdFactory,
    log: LogFn,
    model_settings: ModelSettings,
    kind: _ColdReadKind,
) -> tuple[str, str]:
    """One Huginn cold-read of an artifact.

    Used for the design-stage three-pass loop (comprehension/critic/
    readiness) and for /bug's diagnosis-stage pass. The kind controls the
    closing-token tokens, the in-flight header, the verdict callout, and
    the log_kind tag emitted in the JSONL log so the four call types
    (comprehension/critic/readiness/diagnosis) are readable apart.

    Returns (critique_text, verdict_string).
    """
    in_flight_header = f"👁️  **Huginn #{huginn_n} ({kind.label}) · thinking…**"
    huginn_md = await mount_huginn_md(in_flight_header)
    huginn = huginn_agent_factory()
    critique_text, _ = await run_and_stream(
        huginn,
        formatted_critique_prompt,
        huginn_md,
        message_history=None,
        log=log,
        model_settings=model_settings,
        label=f"huginn-{kind.log_kind}-{huginn_n}",
    )
    verdict = _verdict(
        critique_text,
        ready_token=kind.ready_token,
        revise_token=kind.revise_token,
    )
    log({
        "type": "huginn_verdict",
        "round": huginn_n,
        "verdict": verdict,
        "kind": kind.log_kind,
        "convergence": (
            "agree" if verdict == "ready"
            else "disagree" if verdict == "revise"
            else "partial"
        ),
    })
    await mount_huginn_md(_verdict_callout(huginn_n, verdict, kind))
    return critique_text, verdict


def _verdict(
    text: str,
    *,
    ready_token: str = "design ready",
    revise_token: str = "design needs revision",
) -> str:
    """Parse Huginn's closing verdict.

    Returns "ready" if the text ends with (or contains a trailing) ready_token,
    "revise" if it ends with revise_token, "unknown" otherwise.
    Case-insensitive; tolerates trailing whitespace and final period.

    Defaults match the design-stage tokens so existing callers (/bug
    diagnosis, the legacy single-pass helper) work without changes.
    """
    tail = text.strip().lower().rstrip(".").rstrip()
    ready_lc = ready_token.lower()
    revise_lc = revise_token.lower()
    if tail.endswith(ready_lc):
        return "ready"
    if tail.endswith(revise_lc):
        return "revise"
    if re.search(rf"{re.escape(ready_lc)}\b", tail):
        return "ready"
    if re.search(rf"{re.escape(revise_lc)}\b", tail):
        return "revise"
    return "unknown"


def _combined_critique(
    *,
    comprehension_text: str,
    critic_text: str,
    readiness_text: str,
) -> str:
    """Bundle the three design-stage Huginn outputs into a single critique
    payload for the revise prompt.

    Sections are explicitly labeled so the local Qwen model can tell which
    gaps came from which pass. comprehension_text is included only when
    non-empty (i.e. the round-1 comprehension pass returned "unclear");
    when comprehension passed there's no useful signal to feed back.
    Critic and readiness sections are always present because both are
    gated on every revision round.
    """
    parts: list[str] = []
    if comprehension_text.strip():
        parts.append(
            "=== COMPREHENSION FEEDBACK (informational - the cold reader "
            "could not paraphrase parts of the doc) ===\n"
            + comprehension_text.strip()
        )
    parts.append("=== CRITIC GAPS ===\n" + critic_text.strip())
    parts.append("=== READINESS OPEN QUESTIONS ===\n" + readiness_text.strip())
    return "\n\n".join(parts)


@dataclass
class _DesignCheckResult:
    """One round's worth of design-stage cold reads.

    combined_verdict is "ready" iff BOTH critic and readiness returned
    "ready" - comprehension is informational and never gates progress.
    bundle is the multi-section critique text fed to the revise prompt
    when the round did not converge.
    """
    critic_text: str
    critic_verdict: str
    readiness_text: str
    readiness_verdict: str
    comprehension_text: str   # "" when comprehension was skipped or passed
    comprehension_verdict: str  # "ready" / "revise" / "unknown" / "skipped"
    combined_verdict: str
    bundle: str


async def _run_design_check_round(
    *,
    ctx: FeatureRunCtx,
    design_text: str,
    huginn_n: int,
    include_comprehension: bool,
) -> _DesignCheckResult:
    """Run the article's three Huginn passes against one design-doc revision.

    Comprehension only runs on the first round because revisions are
    gap-driven (not structural rewrites) - if comprehension passed once
    on the original draft, it almost always passes on the revision, and
    re-running it triples the round cost for no information. Skip it
    after round 1.

    Critic and readiness gate progress: combined verdict is "ready" iff
    both pass. Comprehension feedback is folded into the revise bundle
    when it returned "unclear", so the model addresses ambiguous prose
    on the same revision pass that fixes critic/readiness gaps.
    """
    comprehension_text = ""
    comprehension_verdict = "skipped"
    if include_comprehension and ctx.feature_comprehension_prompt:
        c_text, c_verdict = await _cold_read_once(
            huginn_agent_factory=ctx.huginn_agent_factory,
            formatted_critique_prompt=ctx.feature_comprehension_prompt.format(
                design_doc=design_text
            ),
            huginn_n=huginn_n,
            mount_huginn_md=ctx.mount_huginn_md,
            log=ctx.log,
            model_settings=ctx.model_settings,
            kind=_KIND_COMPREHENSION,
        )
        comprehension_verdict = c_verdict
        if c_verdict == "revise":
            comprehension_text = c_text

    critic_text, critic_verdict = await _cold_read_once(
        huginn_agent_factory=ctx.huginn_agent_factory,
        formatted_critique_prompt=ctx.feature_critique_prompt.format(
            design_doc=design_text
        ),
        huginn_n=huginn_n,
        mount_huginn_md=ctx.mount_huginn_md,
        log=ctx.log,
        model_settings=ctx.model_settings,
        kind=_KIND_CRITIC,
    )

    readiness_text = ""
    readiness_verdict = "skipped"
    if ctx.feature_readiness_prompt:
        readiness_text, readiness_verdict = await _cold_read_once(
            huginn_agent_factory=ctx.huginn_agent_factory,
            formatted_critique_prompt=ctx.feature_readiness_prompt.format(
                design_doc=design_text
            ),
            huginn_n=huginn_n,
            mount_huginn_md=ctx.mount_huginn_md,
            log=ctx.log,
            model_settings=ctx.model_settings,
            kind=_KIND_READINESS,
        )

    # Combined gate: both critic and readiness must say ready. If readiness
    # was skipped (prompt not configured by an old caller), fall back to
    # critic-only gating so older test contexts that pre-date this change
    # still pass.
    if readiness_verdict == "skipped":
        combined = critic_verdict
    else:
        combined = (
            "ready"
            if critic_verdict == "ready" and readiness_verdict == "ready"
            else "revise"
        )

    bundle = _combined_critique(
        comprehension_text=comprehension_text,
        critic_text=critic_text,
        readiness_text=readiness_text,
    )

    ctx.log({
        "type": "design_check_round",
        "round": huginn_n,
        "comprehension": comprehension_verdict,
        "critic": critic_verdict,
        "readiness": readiness_verdict,
        "combined": combined,
    })

    return _DesignCheckResult(
        critic_text=critic_text,
        critic_verdict=critic_verdict,
        readiness_text=readiness_text,
        readiness_verdict=readiness_verdict,
        comprehension_text=comprehension_text,
        comprehension_verdict=comprehension_verdict,
        combined_verdict=combined,
        bundle=bundle,
    )


async def feature_flow(ctx: FeatureRunCtx) -> dict[str, Any]:
    """Run the /feature pipeline. Returns a summary dict (logged + can be displayed).

    Phase 1: at most one revision round. After one revision the implementation
    proceeds regardless of the second verdict (logged for research).
    """
    ctx.log({"type": "feature_started", "description": ctx.description})
    await ctx.mount_muninn_md(f"### 🚀 `/feature` - {ctx.description}")

    # 0. GROUND - Muninn explores the project before drafting anything.
    #    This step is the single biggest lever for output quality on local
    #    models: without it, the model falls back on its training prior and
    #    produces boilerplate. The grounding output is left in muninn_history
    #    so the design and implementation steps can reference it implicitly.
    ground_md = await ctx.mount_muninn_md(
        "🐦‍⬛ **Muninn · step 1/4 · grounding**"
    )
    ground_prompt = ctx.feature_ground_prompt.format(description=ctx.description)
    ground_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        ground_prompt,
        ground_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-ground",
    )
    ctx.log({"type": "muninn_grounded", "len": len(ground_text)})

    # 1. DRAFT - Muninn drafts the design doc, informed by the brief above.
    muninn_md = await ctx.mount_muninn_md(
        "🐦‍⬛  **Muninn · step 2/4 · drafting design doc**"
    )
    design_prompt = ctx.feature_design_prompt.format(description=ctx.description)
    design_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        design_prompt,
        muninn_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-design",
    )
    ctx.log({"type": "muninn_design_drafted", "len": len(design_text)})

    # 2. Three-goldfish design-check loop, up to ctx.max_revision_rounds
    #    revisions. Each round runs a critic pass + a readiness pass, plus
    #    a one-shot comprehension pass on round 1 only (see
    #    _run_design_check_round for why comprehension doesn't repeat).
    #    A round passes only when both critic AND readiness return ready.
    #    Each subsequent revision pairs with fresh Huginn instances that
    #    re-read the revised doc. Muninn sees every prior critique via
    #    message_history (carried through run_and_stream).
    max_rounds = max(0, int(ctx.max_revision_rounds))
    last_check: _DesignCheckResult | None = None
    verdict = "unknown"
    huginn_n = 0
    for round_n in range(1, max_rounds + 2):
        huginn_n = round_n
        last_check = await _run_design_check_round(
            ctx=ctx,
            design_text=design_text,
            huginn_n=round_n,
            include_comprehension=(round_n == 1),
        )
        verdict = last_check.combined_verdict

        if verdict == "ready" or round_n > max_rounds:
            break

        # Revise. The revised draft becomes the next round's design_text.
        revise_md = await ctx.mount_muninn_md(
            f"🐦‍⬛  **Muninn · step 3/4 · revising design "
            f"(round {round_n}/{max_rounds})**"
        )
        design_text, ctx.muninn_history = await run_and_stream(
            ctx.muninn_agent,
            _build_revise_prompt(last_check.bundle, multi_section=True),
            revise_md,
            message_history=ctx.muninn_history,
            log=ctx.log,
            model_settings=ctx.model_settings,
            label=f"muninn-revise-{round_n}",
        )
        ctx.log({"type": "muninn_design_revised",
                 "round": round_n, "len": len(design_text)})

    # last_check is non-None here because the loop runs at least once
    # (max_rounds + 2 >= 2). last_critique mirrors the previous code path's
    # variable so the implement-step prompt and the unconverged backstop
    # can keep their structure.
    last_critique = last_check.bundle if last_check else ""

    # Backstop: if not converged after MAX rounds, give the user agile options
    # rather than a binary proceed/cancel. Loops until they pick a terminal
    # action (proceed, cancel) or until verdict flips to "ready".
    #
    # At freedom_level == "high", skip the ask entirely: auto-proceed to
    # implementation with the latest critique inline so Muninn addresses
    # gaps while coding. Low and medium keep the interactive backstop.
    if verdict != "ready" and ctx.freedom_level == "high":
        ctx.log({
            "type": "feature_unconverged_autoproceed",
            "level": "high",
            "rounds": huginn_n,
        })
        await ctx.mount_muninn_md(
            f"⚠️ Design did not converge after {huginn_n} Huginn rounds; "
            f"proceeding to implementation (freedom=high)."
        )
        verdict = "ready"  # short-circuits the while-loop below

    while verdict != "ready":
        await ctx.mount_muninn_md(
            f"⚠️ **Design did not converge** after {huginn_n} Huginn rounds. "
            f"Asking how you want to proceed."
        )
        choice = await ctx.ask_user(
            f"Huginn still flags blocking gaps after {huginn_n} cold-read "
            f"rounds. What now?",
            [
                ("proceed to implementation with current design",
                 "Move to the implementation step with the current design even "
                 "though Huginn flagged gaps. Muninn will see the latest critique "
                 "inline and try to address what it can while coding. Use this "
                 "when you've read the gaps and they're acceptable risks."),
                ("do one more revision round",
                 "Run one more Muninn revise + Huginn cold-read pass. Use this "
                 "when you think the model just needs another go - not a "
                 "structural problem with the description itself."),
                ("let me answer the remaining gaps myself (Muninn will ask)",
                 "Muninn enumerates the still-open numbered gaps and calls "
                 "ask_user once per gap with concrete options derived from each. "
                 "Use this when the gaps are about your intent (naming, behavior, "
                 "scope, look-and-feel) - things only you can answer."),
                ("cancel /feature (no code written)",
                 "Stop /feature. No files modified, no code written, design "
                 "discarded. Use this when the description is too vague to "
                 "converge - re-run /feature with a sharper request."),
            ],
        )
        ctx.log({"type": "feature_unconverged", "choice": choice,
                 "rounds_so_far": huginn_n})

        if choice.startswith("proceed"):
            break

        if choice.startswith("cancel"):
            await ctx.mount_muninn_md(
                "❌ **/feature cancelled** - design did not converge. "
                "Re-run with a sharper description."
            )
            return {
                "type": "feature_cancelled",
                "description": ctx.description,
                "design_len": len(design_text),
                "implement_len": 0,
                "rounds": huginn_n,
            }

        if choice.startswith("do one more"):
            huginn_n += 1
            # Revise once more. Use multi_section=True because last_critique
            # is now the bundled critic + readiness output from the prior
            # round.
            revise_md = await ctx.mount_muninn_md(
                f"🐦‍⬛  **Muninn · revising design (extra round, total {huginn_n})**"
            )
            design_text, ctx.muninn_history = await run_and_stream(
                ctx.muninn_agent,
                _build_revise_prompt(last_critique, multi_section=True),
                revise_md,
                message_history=ctx.muninn_history,
                log=ctx.log,
                model_settings=ctx.model_settings,
                label=f"muninn-revise-{huginn_n}",
            )
            ctx.log({"type": "muninn_design_revised",
                     "round": huginn_n, "len": len(design_text)})

            # Run another full design-check round on the revised doc.
            # Comprehension is skipped (only round 1 runs it). Critic and
            # readiness both fire and gate the combined verdict.
            extra_check = await _run_design_check_round(
                ctx=ctx,
                design_text=design_text,
                huginn_n=huginn_n,
                include_comprehension=False,
            )
            verdict = extra_check.combined_verdict
            last_critique = extra_check.bundle
            # Loop will re-ask the user if still not ready.
            continue

        if choice.startswith("let me answer"):
            # Muninn enumerates the still-open gaps and asks the user per gap
            # via ask_user, then folds the answers into the design and accepts
            # it as ready (verdict flipped to "ready" by the act of resolving
            # all gaps interactively).
            answer_md = await ctx.mount_muninn_md(
                "🐦‍⬛  **Muninn · resolving remaining gaps with user input**"
            )
            answer_prompt = (
                "Huginn's most recent critique listed numbered gaps. The user "
                "has chosen to answer the remaining gaps directly. For EACH "
                "still-unresolved numbered gap below, call `ask_user(question, "
                "options)` with concrete labelled options derived from the gap "
                "(do NOT invent answers, do NOT collapse multiple gaps into "
                "one question). After the user answers each one, output the "
                "FINAL revised design doc folding their answers in.\n\n"
                "Latest critique:\n\n"
                f"{last_critique}\n\n"
                "Skip gaps you have already resolved confidently in the current "
                "design. Ask only about the genuinely-open ones. Be concise: "
                "one ask_user call per gap, options 2-5 entries each."
            )
            design_text, ctx.muninn_history = await run_and_stream(
                ctx.muninn_agent,
                answer_prompt,
                answer_md,
                message_history=ctx.muninn_history,
                log=ctx.log,
                model_settings=ctx.model_settings,
                label="muninn-user-resolve",
            )
            ctx.log({"type": "muninn_user_resolved", "len": len(design_text)})
            # Treat the user-resolved design as accepted and proceed to impl.
            verdict = "ready"
            last_critique = (
                "(User resolved remaining gaps directly via ask_user; design "
                "accepted by user.)"
            )
            break

    # 4. Muninn implements via tool calls.
    impl_md = await ctx.mount_muninn_md(
        "🐦‍⬛ **Muninn · step 4/4 · implementing**"
    )
    impl_prompt = (
        "Now implement the design above, grounded in the CONTEXT BRIEF you produced earlier.\n"
        "\n"
        "The most recent Huginn cold-read of the (final) design said:\n"
        "\n"
        f"{last_critique}\n"
        "\n"
        "Address any still-open gaps inline as you implement: if you can resolve\n"
        "by reading more code, do so; if a gap genuinely needs user input, call\n"
        "ask_user with concrete options. Do not silently paper over Huginn's points.\n"
        "\n"
        "Rules:\n"
        "1. For every file you intend to modify, call read_file FIRST so you write the\n"
        "   FULL new content, not a partial diff. write_file replaces the whole file.\n"
        "2. After each significant change, validate with run_shell - pytest if there are\n"
        "   tests, otherwise `python -c 'import ast; ast.parse(open(\"FILE\").read())'`\n"
        "   for syntax sanity.\n"
        "3. If the design left a real fork open, call ask_user with concrete options.\n"
        "   Do NOT guess silently.\n"
        "4. NO boilerplate UNLESS the original feature request explicitly asked for a\n"
        "   scaffold/template/placeholder. Otherwise every line must be specific to\n"
        "   this project's conventions, naming, and existing patterns from the brief.\n"
        "5. When done, output a one-paragraph summary listing each file you changed and\n"
        "   what the change does. Do not include code in the summary.\n"
    )
    impl_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        impl_prompt,
        impl_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-implement",
    )

    summary = {
        "type": "feature_complete",
        "description": ctx.description,
        "design_len": len(design_text),
        "implement_len": len(impl_text),
    }
    ctx.log(summary)

    await ctx.mount_muninn_md(
        f"✅ **Muninn - feature complete** · design {len(design_text)}b · "
        f"implement {len(impl_text)}b"
    )
    return summary


# =====================================================================
# /bug
# =====================================================================


@dataclass
class BugRunCtx:
    """Context for the /bug flow.

    Mirrors FeatureRunCtx but with bug-flavored prompts (ground -> problem
    doc -> Huginn cold-read -> revise -> failing test -> fix). Reuses the
    same multi-round revision + ask_user backstop as /feature, the same
    _verdict() parser, and the same ToolContext-driven implementation.
    """
    description: str
    muninn_agent: Agent
    huginn_agent_factory: Callable[[], Agent]
    muninn_history: list[ModelMessage]
    bug_ground_prompt: str   # contains "{description}"
    bug_problem_prompt: str  # contains "{description}"
    bug_critique_prompt: str  # contains "{problem_doc}"
    model_settings: ModelSettings
    log: LogFn
    mount_muninn_md: MdFactory
    mount_huginn_md: MdFactory
    ask_user: Callable[[str, list[str]], Awaitable[str]]
    max_revision_rounds: int = MAX_REVISION_ROUNDS
    # Snapshot of MuninnTUI.freedom_level at workflow start. Same
    # short-circuit policy as FeatureRunCtx: "high" auto-proceeds.
    freedom_level: str = "low"


async def bug_flow(ctx: BugRunCtx) -> dict[str, Any]:
    """Run the /bug pipeline: ground -> problem doc -> Huginn loop ->
    failing test -> fix. Returns a summary dict.

    Phase 1.5: same revision-loop shape as /feature with the same
    4-option ask_user backstop on non-convergence. The bug-specific
    additions are the failing-test step (Muninn writes a regression
    test that DEMONSTRATES the bug, runs it to confirm it fails) and
    the fix step (Muninn applies the minimum change to make the failing
    test pass without regressing the rest of the suite).
    """
    ctx.log({"type": "bug_started", "description": ctx.description})
    await ctx.mount_muninn_md(f"### 🐛 `/bug` - {ctx.description}")

    # Step 1/5: Ground the bug area.
    ground_md = await ctx.mount_muninn_md(
        "🐦‍⬛ **Muninn · step 1/5 · grounding the bug area**"
    )
    ground_prompt = ctx.bug_ground_prompt.format(description=ctx.description)
    ground_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        ground_prompt,
        ground_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-bug-ground",
    )
    ctx.log({"type": "muninn_grounded", "len": len(ground_text)})

    # Step 2/5: Problem doc.
    problem_md = await ctx.mount_muninn_md(
        "🐦‍⬛  **Muninn · step 2/5 · writing problem doc**"
    )
    problem_prompt = ctx.bug_problem_prompt.format(description=ctx.description)
    problem_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        problem_prompt,
        problem_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-bug-problem",
    )
    ctx.log({"type": "muninn_bug_problem_drafted", "len": len(problem_text)})

    # Step 3/5: Huginn cold-read loop with revisions, identical shape to
    # feature_flow's loop. Sharing the helper would require passing a lot
    # of parameters; the duplication is small enough to leave for now.
    max_rounds = max(0, int(ctx.max_revision_rounds))
    last_critique = ""
    verdict = "unknown"
    huginn_n = 0
    for round_n in range(1, max_rounds + 2):
        huginn_n = round_n
        critique_text, verdict = await _cold_read_once(
            huginn_agent_factory=ctx.huginn_agent_factory,
            formatted_critique_prompt=ctx.bug_critique_prompt.format(
                problem_doc=problem_text
            ),
            huginn_n=round_n,
            mount_huginn_md=ctx.mount_huginn_md,
            log=ctx.log,
            model_settings=ctx.model_settings,
            kind=_KIND_DIAGNOSIS,
        )
        last_critique = critique_text

        if verdict == "ready" or round_n > max_rounds:
            break

        revise_md = await ctx.mount_muninn_md(
            f"🐦‍⬛  **Muninn · step 3/5 · revising problem doc "
            f"(round {round_n}/{max_rounds})**"
        )
        problem_text, ctx.muninn_history = await run_and_stream(
            ctx.muninn_agent,
            _build_revise_prompt(critique_text),
            revise_md,
            message_history=ctx.muninn_history,
            log=ctx.log,
            model_settings=ctx.model_settings,
            label=f"muninn-bug-revise-{round_n}",
        )
        ctx.log({"type": "muninn_bug_problem_revised",
                 "round": round_n, "len": len(problem_text)})

    # Backstop: 4-option ask_user when the diagnosis didn't converge.
    #
    # At freedom_level == "high", short-circuit: auto-proceed to the
    # failing-test step. Low and medium keep the interactive backstop.
    if verdict != "ready" and ctx.freedom_level == "high":
        ctx.log({
            "type": "bug_unconverged_autoproceed",
            "level": "high",
            "rounds": huginn_n,
        })
        await ctx.mount_muninn_md(
            f"⚠️ Diagnosis did not converge after {huginn_n} Huginn rounds; "
            f"proceeding to failing test + fix (freedom=high)."
        )
        verdict = "ready"

    while verdict != "ready":
        await ctx.mount_muninn_md(
            f"⚠️ **Diagnosis did not converge** after {huginn_n} Huginn rounds. "
            f"Asking how you want to proceed."
        )
        choice = await ctx.ask_user(
            f"Huginn still flags blocking gaps in the diagnosis after "
            f"{huginn_n} cold-read rounds. What now?",
            [
                ("proceed to write failing test + fix with current diagnosis",
                 "Move to the failing-test step with the current problem doc "
                 "even though Huginn flagged gaps. Muninn will see the latest "
                 "critique inline. Use this when you've read the gaps and "
                 "they're acceptable risks for the fix you want."),
                ("do one more revision round",
                 "Run one more Muninn revise + Huginn cold-read pass on the "
                 "problem doc. Use this when you think the diagnosis just "
                 "needs another go - not a structural problem with the bug "
                 "report."),
                ("let me answer the remaining gaps myself (Muninn will ask)",
                 "Muninn enumerates the still-open numbered gaps and calls "
                 "ask_user once per gap with concrete options. Use this when "
                 "the gaps are about facts only you know (the exact symptom, "
                 "the environment, what you've already tried)."),
                ("cancel /bug (no test or fix written)",
                 "Stop /bug. No test written, no code changed, problem doc "
                 "discarded. Use this when the bug report is too vague to "
                 "converge - re-run /bug with a sharper description."),
            ],
        )
        ctx.log({"type": "bug_unconverged", "choice": choice,
                 "rounds_so_far": huginn_n})

        if choice.startswith("proceed"):
            break
        if choice.startswith("cancel"):
            await ctx.mount_muninn_md(
                "❌ **/bug cancelled** - diagnosis did not converge. "
                "Re-run with a sharper bug report."
            )
            return {
                "type": "bug_cancelled",
                "description": ctx.description,
                "problem_len": len(problem_text),
                "test_len": 0, "fix_len": 0, "rounds": huginn_n,
            }
        if choice.startswith("do one more"):
            huginn_n += 1
            revise_md = await ctx.mount_muninn_md(
                f"🐦‍⬛  **Muninn · revising problem doc "
                f"(extra round, total {huginn_n})**"
            )
            problem_text, ctx.muninn_history = await run_and_stream(
                ctx.muninn_agent,
                _build_revise_prompt(last_critique),
                revise_md,
                message_history=ctx.muninn_history,
                log=ctx.log,
                model_settings=ctx.model_settings,
                label=f"muninn-bug-revise-{huginn_n}",
            )
            critique_text, verdict = await _cold_read_once(
                huginn_agent_factory=ctx.huginn_agent_factory,
                formatted_critique_prompt=ctx.bug_critique_prompt.format(
                    problem_doc=problem_text
                ),
                huginn_n=huginn_n,
                mount_huginn_md=ctx.mount_huginn_md,
                log=ctx.log,
                model_settings=ctx.model_settings,
                kind=_KIND_DIAGNOSIS,
            )
            last_critique = critique_text
            continue
        if choice.startswith("let me answer"):
            answer_md = await ctx.mount_muninn_md(
                "🐦‍⬛  **Muninn · resolving remaining gaps with user input**"
            )
            answer_prompt = (
                "Huginn's most recent critique listed numbered gaps in the "
                "PROBLEM DOC. The user has chosen to answer the remaining "
                "gaps directly. For EACH still-unresolved numbered gap below, "
                "call `ask_user(question, options)` with concrete labelled "
                "options derived from the gap (do NOT invent answers). After "
                "the user answers each one, output the FINAL revised problem "
                "doc folding their answers in.\n\n"
                "Latest critique:\n\n"
                f"{last_critique}\n\n"
                "Skip gaps you have already resolved confidently. Ask only "
                "about the genuinely-open ones. Be concise: one ask_user "
                "call per gap, options 2-5 entries each."
            )
            problem_text, ctx.muninn_history = await run_and_stream(
                ctx.muninn_agent,
                answer_prompt,
                answer_md,
                message_history=ctx.muninn_history,
                log=ctx.log,
                model_settings=ctx.model_settings,
                label="muninn-bug-user-resolve",
            )
            ctx.log({"type": "muninn_bug_user_resolved",
                     "len": len(problem_text)})
            verdict = "ready"
            last_critique = (
                "(User resolved remaining gaps directly via ask_user; "
                "diagnosis accepted by user.)"
            )
            break

    # Step 4/5: Failing test. Muninn writes a regression test that fails
    #           on the current code. The test exists BEFORE the fix so the
    #           reproduction is captured as code, not just prose.
    test_md = await ctx.mount_muninn_md(
        "🐦‍⬛  **Muninn · step 4/5 · writing failing test**"
    )
    test_prompt = (
        "Now write a REGRESSION TEST for this bug, using the PROBLEM DOC "
        "above as the spec.\n"
        "\n"
        "The most recent Huginn cold-read of the (final) diagnosis said:\n"
        "\n"
        f"{last_critique}\n"
        "\n"
        "Rules:\n"
        "1. The test MUST FAIL on the current (broken) code. Run it with\n"
        "   run_shell to confirm the failure mode matches the symptom in\n"
        "   the PROBLEM DOC. If the test passes, your test is wrong; rewrite\n"
        "   it to actually exercise the bug.\n"
        "2. The test will pass once the fix is applied. (You'll write the\n"
        "   fix in the next step; for now, lock in the failure.)\n"
        "3. Place the test in the appropriate test file for the project's\n"
        "   conventions (look at existing tests with read_file - do NOT\n"
        "   guess the file location or test framework).\n"
        "4. Name the test descriptively: capture the bug, not just\n"
        "   `test_fix`. Example: `test_huginn_history_carry_isolation`.\n"
        "5. After the test fails as expected, output a one-line summary:\n"
        "   `<file>::<test_name> - <one-line failure mode you observed>`.\n"
        "\n"
        "Do NOT proceed to the fix yet. Stop after the failing test is in\n"
        "place and confirmed to fail.\n"
    )
    test_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        test_prompt,
        test_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-bug-test",
    )
    ctx.log({"type": "muninn_bug_test_written", "len": len(test_text)})

    # Step 5/5: Fix.
    fix_md = await ctx.mount_muninn_md(
        "🐦‍⬛  **Muninn · step 5/5 · fixing**"
    )
    fix_prompt = (
        "Now FIX the bug.\n"
        "\n"
        "Rules:\n"
        "1. Read the suspect file(s) again with read_file in case they\n"
        "   changed since you wrote the problem doc.\n"
        "2. Apply the MINIMUM change that makes the failing test from the\n"
        "   previous step pass. Do not refactor unrelated code, do not\n"
        "   rename, do not 'while we're here'. The diff should be focused\n"
        "   on the diagnosed cause.\n"
        "3. After write_file, run the regression test to confirm it now\n"
        "   passes (run_shell with the exact `<file>::<test_name>` from\n"
        "   step 4).\n"
        "4. Then run the FULL test suite (pytest, npm test, cargo test -\n"
        "   whichever matches the project) to confirm no other test\n"
        "   regressed.\n"
        "5. If the fix breaks another test, do NOT silently rewrite that\n"
        "   test. Call ask_user with concrete options like:\n"
        "     - 'keep this fix and update the broken test (its prior\n"
        "        expectation was wrong)'\n"
        "     - 'pick a different approach to the fix'\n"
        "     - 'leave both broken and abandon /bug'\n"
        "6. Output a one-paragraph summary listing files changed and what\n"
        "   each change does. Do not include code in the summary.\n"
    )
    fix_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        fix_prompt,
        fix_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-bug-fix",
    )

    summary = {
        "type": "bug_complete",
        "description": ctx.description,
        "problem_len": len(problem_text),
        "test_len": len(test_text),
        "fix_len": len(fix_text),
        "rounds": huginn_n,
    }
    ctx.log(summary)

    await ctx.mount_muninn_md(
        f"✅ **Muninn · bug fix complete** · "
        f"problem doc {len(problem_text)}b · "
        f"test {len(test_text)}b · "
        f"fix {len(fix_text)}b"
    )
    return summary


# =====================================================================
# /precommit-review
# =====================================================================


@dataclass
class ReviewRunCtx:
    """Context for the /precommit-review flow.

    Single-shot cold-read of the pending diff by one Huginn. No revision loop,
    no Muninn agent involvement (Muninn only orchestrates the orchestration).
    The multi-round triage from .claude/commands/eg-precommit-review.md is a
    Phase 2 follow-up - it requires interactive fix/rebut UI per finding.
    """
    cwd: Path
    huginn_agent_factory: Callable[[], Agent]
    review_prompt: str   # contains "{checks}" and "{diff}"
    model_settings: ModelSettings
    log: LogFn
    mount_muninn_md: MdFactory
    mount_huginn_md: MdFactory


async def _capture(cmd: str, cwd: Path, *, timeout: float = 60.0) -> dict[str, Any]:
    """Run a shell command and capture exit/stdout/stderr without going
    through the Confirm-mode tool wrapper (no user prompt for review checks)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return {"exit": -1, "stdout": "", "stderr": f"spawn failed: {e}"}
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"exit": -1, "stdout": "", "stderr": f"timed out after {timeout}s"}
    return {
        "exit": proc.returncode if proc.returncode is not None else -1,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


async def _gather_diff(cwd: Path) -> tuple[str, str | None]:
    """Collect git status + diff for the reviewer.

    Returns (combined_diff_text, error_or_none). The error is set when the
    cwd is not a git repo or git is missing.
    """
    sections: list[str] = []
    for label, cmd in (
        ("git status", "git status --porcelain=v1"),
        ("git diff (staged + unstaged vs HEAD)", "git diff HEAD"),
        ("git diff main...HEAD (committed since main)", "git diff main...HEAD"),
        ("untracked files", "git ls-files --others --exclude-standard"),
    ):
        result = await _capture(cmd, cwd, timeout=30)
        # Detect non-git cwd from the very first call only, and bail.
        if result["exit"] != 0 and "not a git repository" in (result["stderr"] or ""):
            return ("", "not a git repository")
        # Detect git-not-installed.
        if "command not found" in (result["stderr"] or "") or result["exit"] == 127:
            return ("", "git is not installed or not on PATH")
        text = (result["stdout"] or "").strip()
        if text:
            sections.append(f"=== {label} ===\n{text}")
    if not sections:
        return ("", None)  # repo exists but nothing pending
    return ("\n\n".join(sections), None)


async def _changed_files(cwd: Path) -> list[str]:
    """Modified / added / untracked files relative to cwd."""
    files: set[str] = set()
    r = await _capture("git diff HEAD --name-only --diff-filter=AMR", cwd, timeout=30)
    for line in (r["stdout"] or "").splitlines():
        line = line.strip()
        if line:
            files.add(line)
    r = await _capture("git ls-files --others --exclude-standard", cwd, timeout=30)
    for line in (r["stdout"] or "").splitlines():
        line = line.strip()
        if line:
            files.add(line)
    return sorted(files)


def _python_syntax_check(cwd: Path, files: list[str]) -> dict[str, Any]:
    """In-process ast.parse on every changed .py file that exists.

    Skip cleanly when no Python files were touched (e.g. non-Python project,
    docs-only diff). Failures are reported with file:line and the parser
    message - no shell quoting concerns, no extra subprocess overhead.
    """
    py = [f for f in files if f.endswith(".py") and (cwd / f).is_file()]
    if not py:
        return {
            "name": "syntax (python)",
            "status": "skip",
            "summary": "(no python files in diff)",
        }
    failures: list[str] = []
    for f in py:
        try:
            ast.parse((cwd / f).read_text(encoding="utf-8"))
        except SyntaxError as e:
            failures.append(f"{f}:{e.lineno}: {e.msg}")
        except Exception as e:
            failures.append(f"{f}: {type(e).__name__}: {e}")
    if failures:
        return {
            "name": "syntax (python)",
            "status": "fail",
            "summary": "\n".join(failures),
        }
    return {
        "name": "syntax (python)",
        "status": "pass",
        "summary": f"{len(py)} file(s) parsed cleanly",
    }


async def _run_check(name: str, cmd: str, cwd: Path) -> dict[str, Any]:
    """Run a local check; classify as pass / fail / skip."""
    result = await _capture(cmd, cwd, timeout=120)
    out = (result["stdout"] or "") + (result["stderr"] or "")
    out_lc = out.lower()
    if (
        "command not found" in out_lc
        or "no module named" in out_lc
        or "executable not found" in out_lc
    ):
        return {"name": name, "status": "skip", "summary": "(not configured)"}
    if "no tests ran" in out_lc or "no tests collected" in out_lc:
        return {"name": name, "status": "skip", "summary": "(no tests configured)"}
    if result["exit"] == 0:
        return {"name": name, "status": "pass", "summary": "passed"}
    summary = out.strip()
    if len(summary) > 800:
        summary = summary[:800] + "\n…(truncated)"
    return {"name": name, "status": "fail", "summary": summary}


def _format_checks_for_prompt(checks: list[dict[str, Any]]) -> str:
    """Inline check results for the Huginn prompt (verbose: include failures)."""
    lines: list[str] = []
    for c in checks:
        icon = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[c["status"]]
        lines.append(f"[{icon}] {c['name']}")
        if c["status"] == "fail" and c.get("summary"):
            lines.append("  " + c["summary"].replace("\n", "\n  "))
    return "\n".join(lines) if lines else "(no checks configured)"


def _format_checks_for_pane(checks: list[dict[str, Any]]) -> str:
    """Compact check results for the Muninn pane.

    When a check skipped because the tool is not installed and the stack
    shipped an install_hint, surface a copy-pasteable install line.
    """
    icons = {"pass": "✅", "fail": "❌", "skip": "·"}
    lines = []
    for c in checks:
        first = (c.get("summary") or "").splitlines()[0] if c.get("summary") else ""
        first = first[:120]
        lines.append(f"- {icons[c['status']]} **{c['name']}** - {first}")
        hint = c.get("install_hint")
        if c["status"] == "skip" and hint:
            lines.append(f"   📦 install: `{hint}`")
    return "\n".join(lines)


def _has_no_findings(text: str) -> bool:
    """Parse the reviewer's closing line."""
    last = text.strip().splitlines()[-1].strip().lower() if text.strip() else ""
    if last == "no findings":
        return True
    if last == "findings flagged":
        return False
    # Tolerate trailing punctuation and extra wording.
    if last.endswith("no findings"):
        return True
    return False


async def precommit_review_flow(ctx: ReviewRunCtx) -> dict[str, Any]:
    """Single-Huginn cold-read of pending diff + local checks.

    Steps:
      1. Gather diff (git status + diff HEAD + main..HEAD + untracked)
      2. Run ruff / pytest / ast.parse(app.py) - each graceful-skips
      3. Format check results into the Muninn pane
      4. Stream Huginn's review of the diff
      5. Display verdict + summary
    """
    ctx.log({"type": "review_started"})
    await ctx.mount_muninn_md("### 🔍 `/precommit-review`")

    # Step 1: gather diff.
    await ctx.mount_muninn_md("🐦‍⬛  **Muninn · gathering diff**")
    diff_text, diff_err = await _gather_diff(ctx.cwd)
    if diff_err:
        await ctx.mount_muninn_md(
            f"❌ **/precommit-review aborted** · {diff_err}"
        )
        ctx.log({"type": "review_aborted", "reason": diff_err})
        return {"type": "review_aborted", "reason": diff_err}
    if not diff_text:
        await ctx.mount_muninn_md(
            "✅ **Nothing to review** · no pending changes"
        )
        ctx.log({"type": "review_skipped", "reason": "no pending changes"})
        return {"type": "review_skipped", "reason": "no pending changes"}

    # Step 2: stack-aware local checks.
    #         detect_stack picks the right profile (Python / Node / Rust /
    #         Go / generic) from project markers in cwd. We run each check
    #         only if at least one changed file has a relevant extension,
    #         and pass the per-file checks the actual list (so e.g. ruff
    #         scans the touched .py files rather than the whole tree).
    stack = detect_stack(ctx.cwd)
    ctx.log({"type": "stack_detected", "stack": stack.name})
    changed = await _changed_files(ctx.cwd)

    await ctx.mount_muninn_md(
        f"🐦‍⬛  **Muninn · running local checks** (stack: {stack.name})"
    )
    checks: list[dict[str, Any]] = []
    for spec in stack.checks:
        relevant = relevant_changes(spec, changed)
        if spec.extensions and not relevant:
            # No changed files of this type; not a "fail", not a "skip" -
            # just irrelevant. Don't surface to keep the pane uncluttered.
            continue
        if spec.builtin_python_syntax:
            result = _python_syntax_check(ctx.cwd, relevant)
        else:
            cmd = build_command(spec, relevant)
            result = await _run_check(spec.name, cmd, ctx.cwd)
            if result["status"] == "skip" and spec.install_hint:
                # Surface the install hint when the tool isn't on PATH.
                result["install_hint"] = spec.install_hint
        checks.append(result)
        ctx.log({"type": "review_check", **result})

    if not checks:
        # Nothing to display. Either generic stack or diff doesn't touch
        # any code in the detected stack. Still let Huginn cold-read.
        await ctx.mount_muninn_md(
            "(no stack-specific checks ran for this diff)"
        )

    await ctx.mount_muninn_md(
        "**Local checks:**\n" + _format_checks_for_pane(checks)
    )

    # Step 3: Huginn cold-read.
    huginn_md = await ctx.mount_huginn_md(
        "👁️  **Huginn · cold-reading the diff**"
    )
    huginn = ctx.huginn_agent_factory()
    review_prompt = ctx.review_prompt.format(
        checks=_format_checks_for_prompt(checks),
        diff=diff_text,
    )
    review_text, _ = await run_and_stream(
        huginn,
        review_prompt,
        huginn_md,
        message_history=None,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="huginn-review",
    )

    no_findings = _has_no_findings(review_text)
    callout = "✅ no findings" if no_findings else "❌ findings flagged"
    await ctx.mount_huginn_md(f"👁️  **Huginn says:** {callout}")
    ctx.log({
        "type": "review_verdict",
        "no_findings": no_findings,
        "review_len": len(review_text),
    })

    fail_count = sum(1 for c in checks if c["status"] == "fail")
    final = (
        f"✅ **/precommit-review complete** · "
        f"local checks: {fail_count} failed · "
        f"reviewer: {'no findings' if no_findings else 'findings flagged'}"
    )
    await ctx.mount_muninn_md(final)

    summary = {
        "type": "review_complete",
        "checks_failed": fail_count,
        "no_findings": no_findings,
        "review_len": len(review_text),
    }
    ctx.log(summary)
    return summary
