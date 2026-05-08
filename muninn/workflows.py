"""Workflows: /feature, /bug, /brainstorm, /prd, /precommit-review."""
from __future__ import annotations

import ast
import asyncio
import re
from dataclasses import dataclass
from datetime import date, datetime
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
    revise_callout="🚨 doc unclear in places",
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
    """Markdown for a prominent post-stream verdict line in the Huginn pane."""
    body = (
        kind.ready_callout if verdict == "ready"
        else kind.revise_callout if verdict == "revise"
        else "❓ verdict unparseable"
    )
    return f"👀 **Huginn #{huginn_n} ({kind.label}) says:** {body}"


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
    in_flight_header = f"👀  **Huginn #{huginn_n} ({kind.label}) · thinking…**"
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
        "🪶 **Muninn · step 1/4 · grounding**"
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
        "🪶  **Muninn · step 2/4 · drafting design doc**"
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
            f"🪶  **Muninn · step 3/4 · revising design "
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
            f"🚨 Design did not converge after {huginn_n} Huginn rounds; "
            f"proceeding to implementation (freedom=high)."
        )
        verdict = "ready"  # short-circuits the while-loop below

    while verdict != "ready":
        await ctx.mount_muninn_md(
            f"🚨 **Design did not converge** after {huginn_n} Huginn rounds. "
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
                f"🪶  **Muninn · revising design (extra round, total {huginn_n})**"
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
                "🪶  **Muninn · resolving remaining gaps with user input**"
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
        "🪶 **Muninn · step 4/4 · implementing**"
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
        "🪶 **Muninn · step 1/5 · grounding the bug area**"
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
        "🪶  **Muninn · step 2/5 · writing problem doc**"
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
            f"🪶  **Muninn · step 3/5 · revising problem doc "
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
            f"🚨 Diagnosis did not converge after {huginn_n} Huginn rounds; "
            f"proceeding to failing test + fix (freedom=high)."
        )
        verdict = "ready"

    while verdict != "ready":
        await ctx.mount_muninn_md(
            f"🚨 **Diagnosis did not converge** after {huginn_n} Huginn rounds. "
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
                f"🪶  **Muninn · revising problem doc "
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
                "🪶  **Muninn · resolving remaining gaps with user input**"
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
        "🪶  **Muninn · step 4/5 · writing failing test**"
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
        "🪶  **Muninn · step 5/5 · fixing**"
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
# /brainstorm and /prd  - parallel-lens fan-out
# =====================================================================
#
# The article's goldfish work is sequential validation (comprehension /
# critic / readiness gates). /brainstorm and /prd extend the pattern to
# upstream ideation and requirements: multiple Huginns evaluate the same
# input through deliberately asymmetric role injections, in parallel,
# then Muninn synthesizes. PRD line 223 acknowledges parallel is "fake
# speedup on a single GPU" because Ollama serializes generations per
# model; we keep parallel-in-code for clean structure and forward
# compatibility with vLLM, not for wall-clock speedup.

# /brainstorm lenses
LENS_TECHNICAL = "technical"
LENS_CONTRARIAN = "contrarian"
LENS_UX = "ux"
BRAINSTORM_LENSES: tuple[str, ...] = (LENS_TECHNICAL, LENS_CONTRARIAN, LENS_UX)

# /prd research lenses
LENS_PRIOR_ART = "prior_art"
LENS_EDGE_CASES = "edge_cases"
LENS_INTEGRATION = "integration"
PRD_LENSES: tuple[str, ...] = (LENS_PRIOR_ART, LENS_EDGE_CASES, LENS_INTEGRATION)


@dataclass
class BrainstormRunCtx:
    """Context for the /brainstorm flow.

    Ground (Muninn) -> 3 parallel Huginn lenses -> Muninn synthesis ->
    workflow writes docs/brainstorms/<slug>-<date>.md via Path.write_text
    (deterministic, bypasses the write_file tool gate, independent of
    freedom_level). No revision loop. No critic/readiness gate on the
    output - user chains /prd or /feature for that.
    """
    description: str
    cwd: Path  # project root, where docs/brainstorms/ lands
    muninn_agent: Agent
    huginn_agent_factory: Callable[[], Agent]
    muninn_history: list[ModelMessage]
    brainstorm_ground_prompt: str            # contains "{description}"
    brainstorm_lens_prompts: dict[str, str]  # keys = BRAINSTORM_LENSES; values "{description}", "{ground_brief}"
    brainstorm_synthesis_prompt: str         # "{description}", "{ground_brief}", "{lens_outputs}"
    model_settings: ModelSettings
    log: LogFn
    mount_muninn_md: MdFactory
    mount_huginn_md: MdFactory


@dataclass
class PRDRunCtx:
    """Context for the /prd flow.

    Ground -> single Muninn QA turn (3-5 ask_user calls + Q&A summary
    block) -> 3 parallel Huginn research lenses -> Muninn synthesis ->
    workflow writes docs/prds/<slug>-<date>.md.
    """
    description: str
    cwd: Path
    muninn_agent: Agent
    huginn_agent_factory: Callable[[], Agent]
    muninn_history: list[ModelMessage]
    prd_ground_prompt: str               # "{description}"
    prd_qa_prompt: str                   # "{description}", "{ground_brief}"
    prd_lens_prompts: dict[str, str]     # keys = PRD_LENSES; values "{description}", "{ground_brief}", "{qa_summary}"
    prd_synthesis_prompt: str            # "{description}", "{ground_brief}", "{qa_summary}", "{lens_outputs}"
    model_settings: ModelSettings
    log: LogFn
    mount_muninn_md: MdFactory
    mount_huginn_md: MdFactory


# ---- helpers --------------------------------------------------------

# Tokens the QA prompt instructs Muninn to emit at the tail of its
# Q&A summary. Tolerant parse (substring on the lower-cased tail) -
# matches the same pattern as _verdict() above. If neither token is
# present, we proceed with whatever Muninn returned as qa_summary
# (logged as prd_qa_no_token).
_QA_TOKEN_COMPLETE = "qa complete"
_QA_TOKEN_SKIPPED = "no clarifications gathered"


def _slug_for_path(description: str, max_len: int = 60) -> str:
    """Lowercase; non-alnum runs collapse to a single '-'; strip leading
    /trailing '-'; truncate to max_len then re-strip; empty result -> 'idea'.

    Pure, synchronous. Non-ASCII characters are dropped (since
    `[^a-z0-9]+` doesn't match unicode word chars). Tested across
    edge cases (empty, all-symbol, very long, double-dash).
    """
    s = description.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "idea"


def _unique_artifact_path(
    cwd: Path,
    subdir: str,
    slug: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Return absolute Path under cwd/docs/<subdir>/.

    Tries `<slug>-<YYYY-MM-DD>.md` first; on collision, `<slug>-<date>-<HHMMSS>.md`;
    on a second collision (impossible single-process per PRD F1), raises
    FileExistsError.

    Does NOT create directories. Does NOT write the file. Caller writes.
    `now` is injectable for deterministic tests.
    """
    when = now or datetime.now()
    base_dir = cwd / "docs" / subdir
    primary = base_dir / f"{slug}-{when.strftime('%Y-%m-%d')}.md"
    if not primary.exists():
        return primary
    secondary = base_dir / f"{slug}-{when.strftime('%Y-%m-%d')}-{when.strftime('%H%M%S')}.md"
    if not secondary.exists():
        return secondary
    raise FileExistsError(
        f"both {primary.name} and {secondary.name} already exist; "
        f"single-process race not expected (PRD F1)"
    )


def _format_lens_outputs(
    order: tuple[str, ...],
    results: dict[str, str],
    failures: dict[str, str],
) -> str:
    """Build the synthesis-prompt input for the {lens_outputs} placeholder.

    Iterates `order` (deterministic; matches BRAINSTORM_LENSES or PRD_LENSES).
    For each lens, emits a fenced block:

        --- LENS: <name> ---
        <text or placeholder>
        --- END LENS: <name> ---

    Failed lenses get `(lens unavailable: <ExcClassName>)` as the body. Joined
    by blank lines.

    Invariant (asserted): every lens in `order` MUST appear in exactly one
    of `results` or `failures` (the dicts must be disjoint and together
    cover `order`). Protects against future refactors that drift the dict
    contracts apart.
    """
    keys_results = set(results.keys())
    keys_failures = set(failures.keys())
    assert keys_results.isdisjoint(keys_failures), (
        f"results and failures must be disjoint, got overlap "
        f"{keys_results & keys_failures}"
    )
    assert set(order) <= (keys_results | keys_failures), (
        f"every lens in order must appear in results or failures; "
        f"missing {set(order) - (keys_results | keys_failures)}"
    )
    parts: list[str] = []
    for lens in order:
        if lens in results:
            body = results[lens] or "(empty)"
        else:
            body = f"(lens unavailable: {failures[lens]})"
        parts.append(
            f"--- LENS: {lens} ---\n{body}\n--- END LENS: {lens} ---"
        )
    return "\n\n".join(parts)


async def _fan_out_lenses(
    *,
    huginn_agent_factory: Callable[[], Agent],
    lens_prompts: dict[str, str],   # already-formatted prompts, keyed by lens name
    lens_order: tuple[str, ...],    # deterministic iteration; same as BRAINSTORM_LENSES or PRD_LENSES
    mount_huginn_md: MdFactory,
    log: LogFn,
    model_settings: ModelSettings,
    flow_label: str,                # "brainstorm" | "prd-research"; tags JSONL events
) -> tuple[dict[str, str], dict[str, str]]:
    """Run N Huginn cold-reads in parallel; return (results, failures).

    Step 1 (SEQUENTIAL): for each lens in lens_order, mount its streaming
    Markdown widget and mint its Huginn agent. Both happen synchronously,
    in tuple order, BEFORE any task starts. This is load-bearing - it
    eliminates the concurrent-mount interleave that would otherwise produce
    header-A / header-B / body-A / body-B ordering. DO NOT rewrite this
    block to use asyncio.gather over the mount calls.

    Step 2 (PARALLEL): asyncio.gather over per-lens coroutines. NO
    return_exceptions=True - that would swallow CancelledError as a
    successful result (Esc would never propagate). Each lens task self-
    catches `Exception` (NOT BaseException, NOT bare except - CancelledError
    must propagate) and records its own failure.

    Step 3: build {lens: text} and {lens: ExcClassName} dicts (disjoint;
    together cover lens_order); return both.
    """
    # Step 1: sequential mount + factory in tuple order.
    prepared: list[tuple[str, Any, Agent, str]] = []
    for lens in lens_order:
        header = f"👀  **Huginn · {lens} lens · thinking…**"
        md = await mount_huginn_md(header)
        agent = huginn_agent_factory()
        prompt = lens_prompts[lens]
        prepared.append((lens, md, agent, prompt))
        log({"type": "lens_started", "flow": flow_label, "lens": lens})

    async def _run_one(lens: str, md, agent: Agent, prompt: str) -> tuple[str, str, str]:
        # MUST be `except Exception`, not bare `except` or `except BaseException`.
        # CancelledError inherits from BaseException (not Exception) and must
        # propagate so Esc cancellation works.
        try:
            text, _ = await run_and_stream(
                agent,
                prompt,
                md,
                message_history=None,   # Huginn is always stateless cold-read
                log=log,
                model_settings=model_settings,
                label=f"huginn-{flow_label}-{lens}",
            )
            log({"type": "lens_completed", "flow": flow_label,
                 "lens": lens, "len": len(text)})
            return (lens, text, "")
        except Exception as exc:
            exc_class = type(exc).__name__
            log({"type": "lens_failed", "flow": flow_label, "lens": lens,
                 "exc": exc_class, "msg": str(exc)})
            try:
                await md.append(
                    f"\n\n❌ **Huginn · {lens} lens · failed:** "
                    f"{exc_class}: {exc}"
                )
            except Exception:
                pass
            return (lens, "", exc_class)

    # Step 2: parallel streaming. NO return_exceptions=True.
    triples = await asyncio.gather(
        *(_run_one(lens, md, agent, prompt)
          for lens, md, agent, prompt in prepared)
    )

    # Step 3: split into disjoint dicts.
    results: dict[str, str] = {}
    failures: dict[str, str] = {}
    for lens, text, exc_class in triples:
        if exc_class:
            failures[lens] = exc_class
        else:
            results[lens] = text
    return results, failures


def _artifact_body(
    *,
    flow_kind: str,                 # "brainstorm" | "prd"
    description: str,
    today: date,
    synthesis_text: str,
    lens_order: tuple[str, ...],
    lens_results: dict[str, str],
    lens_failures: dict[str, str],
    qa_summary: str | None = None,  # /prd only
) -> str:
    """Build the markdown content of the artifact file.

    Header + synthesis + (optional Q&A summary for /prd) + verbatim lens
    outputs (one ### Lens: <name> section each, including failed lenses
    with the unavailable marker). Pure, synchronous. Muninn does NOT
    format the file; this helper is the single source of truth so the
    artifact is deterministic regardless of model output drift.
    """
    heading = "Brainstorm" if flow_kind == "brainstorm" else "PRD"
    parts: list[str] = [
        f"# {heading}: {description}",
        "",
        f"Generated {today.isoformat()} by `/{flow_kind}`.",
        "",
        "## Synthesis",
        "",
        synthesis_text.strip() or "_(synthesis was empty)_",
    ]
    if qa_summary is not None:
        parts.extend([
            "",
            "## Q&A summary",
            "",
            qa_summary.strip() or "_(no Q&A captured)_",
        ])
    parts.extend(["", "## Lens outputs (verbatim)"])
    for lens in lens_order:
        parts.extend(["", f"### Lens: {lens}", ""])
        if lens in lens_results:
            parts.append(lens_results[lens].strip() or "_(empty)_")
        else:
            parts.append(
                f"_(lens unavailable: {lens_failures.get(lens, 'unknown')})_"
            )
    return "\n".join(parts) + "\n"


# ---- /brainstorm flow ----------------------------------------------

async def brainstorm_flow(ctx: BrainstormRunCtx) -> dict[str, Any]:
    """Run the /brainstorm pipeline. Returns a summary dict.

    Steps:
      1. Ground (Muninn): explore the codebase, output a short brief.
      2. Fan out 3 Huginn lenses in parallel: technical / contrarian / UX.
      3. Synthesize (Muninn): convergent themes / divergent / next step.
      4. Save artifact to docs/brainstorms/<slug>-<date>.md (workflow writes,
         not via tool surface).

    Returns one of:
      - {"type": "brainstorm_complete", description, artifact_path,
         synthesis_len, lens_outputs: {lens: len},
         lens_failures: {lens: ExcClass}}
      - {"type": "brainstorm_partial", ...} when synthesis raised but lens
         outputs are still worth saving.
      - {"type": "brainstorm_failed", description, reason, lens_failures}
         when ALL lenses failed (synthesis is skipped).
    """
    ctx.log({"type": "brainstorm_started", "description": ctx.description})
    await ctx.mount_muninn_md(f"### 💡 `/brainstorm` - {ctx.description}")

    # Step 1: Ground.
    ground_md = await ctx.mount_muninn_md(
        "🪶 **Muninn · step 1/4 · grounding the idea space**"
    )
    ground_prompt = ctx.brainstorm_ground_prompt.format(description=ctx.description)
    ground_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        ground_prompt,
        ground_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-brainstorm-ground",
    )
    ctx.log({"type": "brainstorm_grounded", "len": len(ground_text)})

    # Step 2: Fan out 3 lenses in parallel.
    await ctx.mount_muninn_md(
        "🪶 **Muninn · step 2/4 · fanning out 3 Huginn lenses**"
    )
    lens_prompts = {
        lens: ctx.brainstorm_lens_prompts[lens].format(
            description=ctx.description,
            ground_brief=ground_text,
        )
        for lens in BRAINSTORM_LENSES
    }
    lens_results, lens_failures = await _fan_out_lenses(
        huginn_agent_factory=ctx.huginn_agent_factory,
        lens_prompts=lens_prompts,
        lens_order=BRAINSTORM_LENSES,
        mount_huginn_md=ctx.mount_huginn_md,
        log=ctx.log,
        model_settings=ctx.model_settings,
        flow_label="brainstorm",
    )

    # All-fail short-circuit: skip synthesis and save nothing. The worker
    # echoes the user-facing abort line; the flow only logs and returns
    # the summary dict so user-facing prose lives in one place.
    if not lens_results:
        reason = (
            f"all {len(BRAINSTORM_LENSES)} lenses failed: "
            + "; ".join(f"{lens}={exc}" for lens, exc in lens_failures.items())
        )
        ctx.log({"type": "brainstorm_failed",
                 "description": ctx.description,
                 "reason": reason, "lens_failures": lens_failures})
        return {
            "type": "brainstorm_failed",
            "description": ctx.description,
            "reason": reason,
            "lens_failures": lens_failures,
        }

    # Step 3: Synthesize.
    syn_md = await ctx.mount_muninn_md(
        f"🪶 **Muninn · step 3/4 · synthesizing "
        f"{len(lens_results)} lens(es)**"
    )
    ctx.log({"type": "brainstorm_synthesizing",
             "successful_lens_count": len(lens_results)})
    synthesis_prompt = ctx.brainstorm_synthesis_prompt.format(
        description=ctx.description,
        ground_brief=ground_text,
        lens_outputs=_format_lens_outputs(
            BRAINSTORM_LENSES, lens_results, lens_failures
        ),
    )

    synthesis_failed: str | None = None
    try:
        synthesis_text, ctx.muninn_history = await run_and_stream(
            ctx.muninn_agent,
            synthesis_prompt,
            syn_md,
            message_history=ctx.muninn_history,
            log=ctx.log,
            model_settings=ctx.model_settings,
            label="muninn-brainstorm-synthesis",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        synthesis_failed = type(exc).__name__
        synthesis_text = (
            f"_Synthesis failed: {synthesis_failed}: {exc}. Lens outputs "
            f"preserved verbatim below for manual review._"
        )
        ctx.log({"type": "brainstorm_synthesis_failed",
                 "exc": synthesis_failed, "msg": str(exc)})

    # Step 4: Save artifact.
    await ctx.mount_muninn_md("🪶 **Muninn · step 4/4 · saving artifact**")
    slug = _slug_for_path(ctx.description)
    path = _unique_artifact_path(ctx.cwd, "brainstorms", slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _artifact_body(
        flow_kind="brainstorm",
        description=ctx.description,
        today=date.today(),
        synthesis_text=synthesis_text,
        lens_order=BRAINSTORM_LENSES,
        lens_results=lens_results,
        lens_failures=lens_failures,
    )
    path.write_text(body, encoding="utf-8")
    rel = str(path.relative_to(ctx.cwd))

    summary: dict[str, Any] = {
        "type": "brainstorm_partial" if synthesis_failed else "brainstorm_complete",
        "description": ctx.description,
        "artifact_path": rel,
        "synthesis_len": len(synthesis_text),
        "lens_outputs": {lens: len(lens_results.get(lens, ""))
                         for lens in BRAINSTORM_LENSES},
        "lens_failures": dict(lens_failures),
        "successful_lens_count": len(lens_results),
        "failed_lens_count": len(lens_failures),
    }
    if synthesis_failed:
        summary["synthesis_failed"] = synthesis_failed
        await ctx.mount_muninn_md(
            f"🚨 **/brainstorm partial** · synthesis failed "
            f"({synthesis_failed}); lens outputs saved to {rel}"
        )
    else:
        await ctx.mount_muninn_md(
            f"✅ **/brainstorm complete** · _{len(lens_results)} lens(es) · "
            f"synthesis {len(synthesis_text)}b · saved to {rel}_"
        )
    ctx.log(summary)
    return summary


# ---- /prd flow -----------------------------------------------------

async def prd_flow(ctx: PRDRunCtx) -> dict[str, Any]:
    """Run the /prd pipeline. Returns a summary dict.

    Steps:
      1. Ground (Muninn): explore the codebase + identify user-input gaps.
      2. QA (Muninn single turn): 3-5 ask_user calls + Q&A summary block.
      3. Fan out 3 Huginn research lenses: prior_art / edge_cases / integration.
      4. Synthesize (Muninn): full PRD matching docs/prds/ template.
      5. Save artifact to docs/prds/<slug>-<date>.md.

    Returns: prd_complete / prd_partial / prd_failed (same shape as
    /brainstorm with `prd_` prefix).
    """
    ctx.log({"type": "prd_started", "description": ctx.description})
    await ctx.mount_muninn_md(f"### 📋 `/prd` - {ctx.description}")

    # Step 1: Ground.
    ground_md = await ctx.mount_muninn_md(
        "🪶 **Muninn · step 1/5 · grounding (codebase landmarks)**"
    )
    ground_prompt = ctx.prd_ground_prompt.format(description=ctx.description)
    ground_text, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        ground_prompt,
        ground_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-prd-ground",
    )
    ctx.log({"type": "prd_grounded", "len": len(ground_text)})

    # Step 2: Single Muninn QA turn (3-5 ask_user calls + summary block).
    qa_md = await ctx.mount_muninn_md(
        "🪶 **Muninn · step 2/5 · structured Q&A (3-5 gaps)**"
    )
    ctx.log({"type": "prd_qa_started"})
    qa_prompt = ctx.prd_qa_prompt.format(
        description=ctx.description, ground_brief=ground_text,
    )
    qa_summary, ctx.muninn_history = await run_and_stream(
        ctx.muninn_agent,
        qa_prompt,
        qa_md,
        message_history=ctx.muninn_history,
        log=ctx.log,
        model_settings=ctx.model_settings,
        label="muninn-prd-qa",
    )
    qa_tail = qa_summary.strip().lower()
    qa_window = qa_tail[-200:]
    # Both checks use substring-on-tail for symmetry; SKIPPED checked first
    # because it's the rarer outcome and would otherwise be masked if the
    # model's output happened to also include the substring "qa complete"
    # somewhere in the body.
    if _QA_TOKEN_SKIPPED in qa_window:
        ctx.log({"type": "prd_qa_skipped", "len": len(qa_summary)})
    elif _QA_TOKEN_COMPLETE in qa_window:
        ctx.log({"type": "prd_qa_complete",
                 "len": len(qa_summary), "had_questions": True})
    else:
        # Tolerant: model didn't emit either closing token. Proceed with
        # whatever it returned. Mirrors _verdict()'s tolerance pattern.
        ctx.log({"type": "prd_qa_no_token", "len": len(qa_summary)})

    # Step 3: Fan out 3 research lenses in parallel.
    await ctx.mount_muninn_md(
        "🪶 **Muninn · step 3/5 · fanning out 3 research lenses**"
    )
    lens_prompts = {
        lens: ctx.prd_lens_prompts[lens].format(
            description=ctx.description,
            ground_brief=ground_text,
            qa_summary=qa_summary,
        )
        for lens in PRD_LENSES
    }
    lens_results, lens_failures = await _fan_out_lenses(
        huginn_agent_factory=ctx.huginn_agent_factory,
        lens_prompts=lens_prompts,
        lens_order=PRD_LENSES,
        mount_huginn_md=ctx.mount_huginn_md,
        log=ctx.log,
        model_settings=ctx.model_settings,
        flow_label="prd-research",
    )

    if not lens_results:
        # Worker echoes the user-facing abort line; flow only logs.
        reason = (
            f"all {len(PRD_LENSES)} lenses failed: "
            + "; ".join(f"{lens}={exc}" for lens, exc in lens_failures.items())
        )
        ctx.log({"type": "prd_failed",
                 "description": ctx.description,
                 "reason": reason, "lens_failures": lens_failures})
        return {
            "type": "prd_failed",
            "description": ctx.description,
            "reason": reason,
            "lens_failures": lens_failures,
        }

    # Step 4: Synthesize the PRD.
    syn_md = await ctx.mount_muninn_md(
        f"🪶 **Muninn · step 4/5 · synthesizing PRD "
        f"({len(lens_results)} lens(es))**"
    )
    ctx.log({"type": "prd_synthesizing",
             "successful_lens_count": len(lens_results)})
    synthesis_prompt = ctx.prd_synthesis_prompt.format(
        description=ctx.description,
        ground_brief=ground_text,
        qa_summary=qa_summary,
        lens_outputs=_format_lens_outputs(
            PRD_LENSES, lens_results, lens_failures
        ),
    )

    synthesis_failed: str | None = None
    try:
        synthesis_text, ctx.muninn_history = await run_and_stream(
            ctx.muninn_agent,
            synthesis_prompt,
            syn_md,
            message_history=ctx.muninn_history,
            log=ctx.log,
            model_settings=ctx.model_settings,
            label="muninn-prd-synthesis",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        synthesis_failed = type(exc).__name__
        synthesis_text = (
            f"_Synthesis failed: {synthesis_failed}: {exc}. Lens outputs "
            f"and Q&A preserved verbatim below for manual review._"
        )
        ctx.log({"type": "prd_synthesis_failed",
                 "exc": synthesis_failed, "msg": str(exc)})

    # Step 5: Save artifact.
    await ctx.mount_muninn_md("🪶 **Muninn · step 5/5 · saving artifact**")
    slug = _slug_for_path(ctx.description)
    path = _unique_artifact_path(ctx.cwd, "prds", slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _artifact_body(
        flow_kind="prd",
        description=ctx.description,
        today=date.today(),
        synthesis_text=synthesis_text,
        lens_order=PRD_LENSES,
        lens_results=lens_results,
        lens_failures=lens_failures,
        qa_summary=qa_summary,
    )
    path.write_text(body, encoding="utf-8")
    rel = str(path.relative_to(ctx.cwd))

    summary: dict[str, Any] = {
        "type": "prd_partial" if synthesis_failed else "prd_complete",
        "description": ctx.description,
        "artifact_path": rel,
        "synthesis_len": len(synthesis_text),
        "lens_outputs": {lens: len(lens_results.get(lens, ""))
                         for lens in PRD_LENSES},
        "lens_failures": dict(lens_failures),
        "successful_lens_count": len(lens_results),
        "failed_lens_count": len(lens_failures),
    }
    if synthesis_failed:
        summary["synthesis_failed"] = synthesis_failed
        await ctx.mount_muninn_md(
            f"🚨 **/prd partial** · synthesis failed "
            f"({synthesis_failed}); lens outputs saved to {rel}"
        )
    else:
        await ctx.mount_muninn_md(
            f"✅ **/prd complete** · _{len(lens_results)} lens(es) · "
            f"synthesis {len(synthesis_text)}b · saved to {rel}_"
        )
    ctx.log(summary)
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
    await ctx.mount_muninn_md("🪶  **Muninn · gathering diff**")
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
        f"🪶  **Muninn · running local checks** (stack: {stack.name})"
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
        "👀  **Huginn · cold-reading the diff**"
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
    await ctx.mount_huginn_md(f"👀  **Huginn says:** {callout}")
    ctx.log({
        "type": "review_verdict",
        "no_findings": no_findings,
        "review_len": len(review_text),
    })

    fail_count = sum(1 for c in checks if c["status"] == "fail")
    # When the reviewer flagged findings or any local check failed, the
    # summary must NOT carry the green check / "complete" - that visually
    # contradicts the verdict and reads as "Muninn ignored the review".
    clean = no_findings and fail_count == 0
    icon = "✅" if clean else "⚠️"
    status = "complete" if clean else "needs triage"
    final = (
        f"{icon} **/precommit-review {status}** · "
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
