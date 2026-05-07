"""The four tools Muninn can call.

Schemas are flat and all-required by virtue of plain-typed function args
(no `Optional`, no defaults). Returns are TypedDicts so the model sees a
constrained shape on the next turn (mitigates Qwen "Maybe-pattern" hallucination).
"""
from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict

from pydantic_ai import RunContext, Tool
from pydantic_ai.exceptions import ModelRetry


class WriteFileResult(TypedDict):
    ok: bool
    bytes: int


class RunShellResult(TypedDict):
    exit: int
    stdout: str
    stderr: str


# Read-only command classifier for run_shell at freedom_level=medium.
# At medium we auto-allow only commands whose first token (or `git <sub>`)
# is in these allowlists AND whose raw string contains zero shell
# metacharacters. The classifier is intentionally pessimistic: false
# negatives just trigger a confirm modal, false positives auto-run code
# the user did not see, so we err toward the modal.

# Any of these in the raw command string => not read-only.
_FORBIDDEN_SHELL_METACHARS: tuple[str, ...] = (
    "&&", "||", ";", ">", "<", "|", "`", "$(", "\n", "&",
)

# General-purpose read-only first tokens. Excludes interpreters with
# arbitrary -c / -e / run / install (python, node, cargo, go) since those
# can mutate files via embedded code despite being "read-only" at the
# command name level. pytest can mutate via fixtures; we accept that as
# a convention - users running pytest via Muninn already expect side
# effects in test scratch dirs.
_READ_ONLY_FIRST_TOKENS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "pwd",
    "which", "tree", "file", "stat", "du", "df", "echo", "printf",
    "pytest", "ruff", "mypy", "tsc", "eslint",
})

# Strict read-only git subcommands. Dropped from any earlier draft:
# fetch (writes refs + network), stash (default subcommand mutates),
# config (writes), remote (mutates), tag (mutates refs).
_READ_ONLY_GIT_SUBCMDS: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "blame", "branch",
    "ls-files", "rev-parse", "describe",
})


def _is_read_only(cmd: str) -> bool:
    """Return True if `cmd` is structurally a single read-only command.

    Rejects any shell metacharacter (chains, redirects, command
    substitution, pipes, here-docs). Rejects KEY=VAL prefix (env-prefixed
    invocations slip past the allowlist). Rejects sudo/env wrappers.
    For git, the second token must be in the read-only-subcommand set.
    Otherwise the first token must be in the read-only-first-token set.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    if any(meta in cmd for meta in _FORBIDDEN_SHELL_METACHARS):
        return False
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # unbalanced quotes etc. - cannot reason about it safely.
        return False
    if not tokens:
        return False
    head = tokens[0]
    # Reject KEY=VAL prefix. shlex keeps it as a single token; flags
    # legitimately start with '-' so guard on that.
    if "=" in head and not head.startswith(("-", "--")):
        return False
    if head in {"sudo", "env"}:
        return False
    if head == "git":
        return len(tokens) >= 2 and tokens[1] in _READ_ONLY_GIT_SUBCMDS
    return head in _READ_ONLY_FIRST_TOKENS


def _shell_needs_confirm(level: str, cmd: str) -> bool:
    """Per-level policy for run_shell.

    low  -> confirm everything
    medium -> confirm only commands the read-only classifier rejects
    high -> auto-allow everything
    """
    if level == "high":
        return False
    if level == "medium":
        return not _is_read_only(cmd)
    return True  # low or unknown


def _write_needs_confirm(level: str) -> bool:
    """Per-level policy for write_file: only `high` skips the modal."""
    return level != "high"


@dataclass
class ToolContext:
    cwd: Path
    muninn_dir: Path
    # 3-tier freedom level. The watcher in app.py mutates this in place
    # when the user cycles Ctrl+A or picks a level from the palette, so
    # tools see the live value on each call.
    freedom_level: Literal["low", "medium", "high"]
    confirm_callback: Callable[[str, str], Awaitable[bool]]
    # Options may be plain strings OR (label, explanation) tuples - the modal
    # shows a "?" affordance when an explanation is present.
    ask_user_callback: Callable[
        [str, list[str] | list[tuple[str, str] | str]], Awaitable[str]
    ]
    log: Callable[[dict[str, Any]], None]


class ToolError(RuntimeError):
    """Raised by a tool implementation; rendered as an error in the chat pane."""


def _resolve_under_cwd(ctx_cwd: Path, path: str) -> Path:
    """Resolve `path` relative to ctx_cwd; raise ModelRetry if it escapes.

    Expands `~` and `~user` first so the model can use those shorthand
    forms without producing a literal-tilde-as-filename FileNotFoundError.
    Absolute paths (after expansion) are taken as-is; relative paths
    resolve under ctx_cwd.

    The path-escape check (relative_to) is the security boundary. Surfacing
    it as ModelRetry lets the model adjust its argument rather than
    aborting the whole flow, AND nudges the user to relaunch muninn from
    the right directory when their target lives elsewhere.
    """
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        candidate = expanded.resolve()
    else:
        candidate = (ctx_cwd / expanded).resolve()
    cwd_resolved = ctx_cwd.resolve()
    try:
        candidate.relative_to(cwd_resolved)
    except ValueError as e:
        raise ModelRetry(
            f"path {path!r} (resolved to {str(candidate)!r}) is outside the "
            f"working directory {str(cwd_resolved)!r}. Use a relative path "
            f"inside the project, or ask the user to relaunch muninn against "
            f"the directory that contains your target "
            f"(e.g. `muninn /path/to/that/project`)."
        ) from e
    return candidate


def make_tools(ctx: ToolContext) -> list[Tool]:
    """Return the four pydantic-ai Tool objects bound to this ToolContext."""

    async def read_file(path: str) -> str:
        """Read a UTF-8 text file from the working directory and return its contents."""
        target = _resolve_under_cwd(ctx.cwd, path)
        ctx.log({"type": "tool_call_started", "name": "read_file", "args": {"path": path},
                 "freedom_level": ctx.freedom_level})
        try:
            data = target.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            ctx.log({"type": "tool_executed", "name": "read_file",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": str(e), "freedom_level": ctx.freedom_level})
            # Recoverable: tell the model the file isn't there and let it pick
            # a different path or call run_shell(ls) to see what actually exists.
            raise ModelRetry(
                f"file {path!r} does not exist in the working directory. "
                f"List the directory with run_shell(cmd='ls -la', cwd='.') "
                f"to see what is actually there, then try a different path."
            ) from e
        except (UnicodeDecodeError, PermissionError) as e:
            ctx.log({"type": "tool_executed", "name": "read_file",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": str(e), "freedom_level": ctx.freedom_level})
            raise ModelRetry(
                f"could not read {path!r}: {type(e).__name__}: {e}. "
                f"Try a different file or use run_shell with cat/head if appropriate."
            ) from e
        except Exception as e:
            ctx.log({"type": "tool_executed", "name": "read_file",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": str(e), "freedom_level": ctx.freedom_level})
            raise ToolError(str(e)) from e
        ctx.log({"type": "tool_executed", "name": "read_file",
                 "attempted": True, "parsed": True, "validated": True,
                 "executed": True, "result_len": len(data), "freedom_level": ctx.freedom_level})
        return data

    async def write_file(path: str, content: str) -> WriteFileResult:
        """Write the given UTF-8 content to a file inside the working directory."""
        target = _resolve_under_cwd(ctx.cwd, path)
        preview = content if len(content) <= 4000 else content[:4000] + "\n…(truncated)"
        ctx.log({"type": "tool_call_started", "name": "write_file",
                 "args": {"path": path, "bytes": len(content.encode("utf-8"))},
                 "freedom_level": ctx.freedom_level})
        if _write_needs_confirm(ctx.freedom_level):
            approved = await ctx.confirm_callback(f"write_file({path})", preview)
            if not approved:
                ctx.log({"type": "tool_executed", "name": "write_file",
                         "attempted": True, "parsed": True, "validated": True,
                         "executed": False, "error": "user denied",
                         "freedom_level": ctx.freedom_level})
                raise ToolError("user denied write_file")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            n = len(content.encode("utf-8"))
        except (PermissionError, IsADirectoryError, OSError) as e:
            ctx.log({"type": "tool_executed", "name": "write_file",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": str(e), "freedom_level": ctx.freedom_level})
            raise ModelRetry(
                f"could not write {path!r}: {type(e).__name__}: {e}. "
                f"Pick a different path or fix the cause."
            ) from e
        except Exception as e:
            ctx.log({"type": "tool_executed", "name": "write_file",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": str(e), "freedom_level": ctx.freedom_level})
            raise ToolError(str(e)) from e
        ctx.log({"type": "tool_executed", "name": "write_file",
                 "attempted": True, "parsed": True, "validated": True,
                 "executed": True, "result": {"ok": True, "bytes": n},
                 "freedom_level": ctx.freedom_level})
        return WriteFileResult(ok=True, bytes=n)

    async def run_shell(cmd: str, cwd: str) -> RunShellResult:
        """Run a shell command (60s timeout) in the given cwd; capture exit code, stdout, stderr."""
        cwd_path = _resolve_under_cwd(ctx.cwd, cwd) if cwd else ctx.cwd
        ctx.log({"type": "tool_call_started", "name": "run_shell",
                 "args": {"cmd": cmd, "cwd": str(cwd_path)}, "freedom_level": ctx.freedom_level})
        if _shell_needs_confirm(ctx.freedom_level, cmd):
            approved = await ctx.confirm_callback(
                f"run_shell({cmd})", f"cwd: {cwd_path}\n\n$ {cmd}"
            )
            if not approved:
                ctx.log({"type": "tool_executed", "name": "run_shell",
                         "attempted": True, "parsed": True, "validated": True,
                         "executed": False, "error": "user denied",
                         "freedom_level": ctx.freedom_level})
                raise ToolError("user denied run_shell")

        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except asyncio.CancelledError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            raise
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            ctx.log({"type": "tool_executed", "name": "run_shell",
                     "attempted": True, "parsed": True, "validated": True,
                     "executed": False, "error": "timeout (60s)",
                     "freedom_level": ctx.freedom_level})
            raise ModelRetry(
                f"command {cmd!r} exceeded the 60s timeout. "
                f"Try a faster / more targeted command, or split the work."
            ) from None

        result: RunShellResult = {
            "exit": proc.returncode if proc.returncode is not None else -1,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        ctx.log({"type": "tool_executed", "name": "run_shell",
                 "attempted": True, "parsed": True, "validated": True,
                 "executed": True, "result": {"exit": result["exit"],
                                              "stdout_len": len(result["stdout"]),
                                              "stderr_len": len(result["stderr"])},
                 "freedom_level": ctx.freedom_level})
        return result

    async def ask_user(
        question: str,
        options: list[str],
        option_explanations: list[str],
    ) -> str:
        """Ask the user a multiple-choice question and return their selection.

        `option_explanations` is a parallel array to `options` (same length).
        Each entry is a one or two sentence plain-language explanation of what
        picking that option actually does - it is shown only when the user
        clicks the `?` button next to the option, so the option label itself
        stays short and scannable. Pass an empty string for an option that
        truly needs no explanation; otherwise always provide one, because the
        UI advertises the `?` affordance and the user expects it to work.
        """
        if len(option_explanations) != len(options):
            raise ModelRetry(
                f"option_explanations must be the same length as options "
                f"(got {len(option_explanations)} vs {len(options)}). "
                f"Pass one explanation per option; use an empty string for "
                f"options that genuinely need no extra context."
            )
        # Build the (label, explanation) shape AskUserScreen expects.
        # Empty explanation means "no `?` button on this row".
        merged: list[tuple[str, str] | str] = [
            (opt, expl) if expl else opt
            for opt, expl in zip(options, option_explanations)
        ]
        ctx.log({"type": "tool_call_started", "name": "ask_user",
                 "args": {"question": question, "options": options,
                          "option_explanations": option_explanations},
                 "freedom_level": ctx.freedom_level})
        answer = await ctx.ask_user_callback(question, merged)
        ctx.log({"type": "tool_executed", "name": "ask_user",
                 "attempted": True, "parsed": True, "validated": True,
                 "executed": True, "result": answer, "freedom_level": ctx.freedom_level})
        return answer

    # max_retries=3 gives the model real headroom to recover from a wrong path
    # (e.g. tries `requirements.txt` in a Poetry project, gets ModelRetry, then
    # tries `pyproject.toml`). Beyond 3 it's almost certainly a deeper problem.
    return [
        Tool(read_file, name="read_file", max_retries=3),
        Tool(write_file, name="write_file", max_retries=3),
        Tool(run_shell, name="run_shell", max_retries=3),
        Tool(ask_user, name="ask_user", max_retries=1),
    ]
