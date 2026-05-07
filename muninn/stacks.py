"""Stack detection + per-stack check definitions for /precommit-review.

A stack is a project type (Python, Node, Rust, Go, generic). Each stack
declares an ordered list of checks (lint, typecheck, test, syntax) that
make sense for it. detect_stack(cwd) inspects the project root and picks
the most specific match; precommit_review_flow runs only the checks
relevant to that stack and to the actual changed files.

Design notes:
- File-extension filter on each check means we skip the noise: a Python
  ruff check is not run if the diff only touched README.md, even in a
  Python project.
- Each check ships an optional `install_hint` shown in the pane when the
  tool is missing. The hint is a copy-pasteable one-liner (pip / npm /
  rustup / etc.); user installs it themselves and re-runs. We do NOT
  auto-install during a review - that is an explicit follow-up command.
- The `cmd_template` is a shell command. It receives a single
  pre-quoted `{files}` argument when a check is per-file; checks that
  always run on the whole project (pytest, cargo test) ignore it.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckSpec:
    name: str
    """Display name shown in the pane."""

    cmd_template: str
    """Shell command. May contain `{files}` which is replaced by a
    pre-quoted, space-separated list of changed files matching `extensions`."""

    extensions: tuple[str, ...] = ()
    """Filter `changed_files` by these extensions. Empty tuple = run
    regardless of which files changed (e.g. project-wide test runners)."""

    per_file: bool = False
    """If True, only run when at least one changed file matches an
    extension AND substitute `{files}` into the command. If False, run
    if ANY changed file matches the extension list (or always, when
    extensions is empty), but do not substitute `{files}`."""

    install_hint: str = ""
    """One-line copy-pasteable command that installs the tool. Empty
    string means we have no install advice (e.g. tools that ship with
    the language toolchain)."""

    builtin_python_syntax: bool = False
    """Sentinel for the in-process Python ast.parse check. Wired
    specially in the flow so we don't shell out for something free."""


@dataclass(frozen=True)
class StackProfile:
    name: str
    detect_files: tuple[str, ...]
    """Filenames at cwd root that mark this stack."""

    detect_extensions: tuple[str, ...]
    """Fallback: any *.ext in cwd (or one level deep) marks this stack."""

    checks: tuple[CheckSpec, ...]


# ----- per-stack profiles ----------------------------------------------------

_PYTHON = StackProfile(
    name="python",
    detect_files=("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
                  "Pipfile"),
    detect_extensions=(".py",),
    checks=(
        CheckSpec(
            name="ruff",
            cmd_template="ruff check {files}",
            extensions=(".py",),
            per_file=True,
            install_hint="pip install ruff",
        ),
        CheckSpec(
            name="pytest",
            cmd_template="python -m pytest -q --no-header",
            extensions=(".py",),
            per_file=False,
            install_hint="pip install pytest",
        ),
        CheckSpec(
            name="syntax (python)",
            cmd_template="",
            extensions=(".py",),
            builtin_python_syntax=True,
        ),
    ),
)

_NODE = StackProfile(
    name="node",
    detect_files=("package.json",),
    detect_extensions=(".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
    checks=(
        CheckSpec(
            name="eslint",
            cmd_template="npx --no eslint {files}",
            extensions=(".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
            per_file=True,
            install_hint="npm install --save-dev eslint",
        ),
        CheckSpec(
            name="typecheck (tsc)",
            cmd_template="npx --no tsc --noEmit",
            extensions=(".ts", ".tsx"),
            per_file=False,
            install_hint="npm install --save-dev typescript",
        ),
        CheckSpec(
            name="test (npm)",
            cmd_template="npm test --silent",
            extensions=(".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
            per_file=False,
            install_hint="",  # depends on what test runner the project uses
        ),
    ),
)

_RUST = StackProfile(
    name="rust",
    detect_files=("Cargo.toml",),
    detect_extensions=(".rs",),
    checks=(
        CheckSpec(
            name="cargo clippy",
            cmd_template="cargo clippy --quiet --no-deps -- -D warnings",
            extensions=(".rs",),
            per_file=False,
            install_hint="rustup component add clippy",
        ),
        CheckSpec(
            name="cargo check",
            cmd_template="cargo check --quiet",
            extensions=(".rs",),
            per_file=False,
            install_hint="",  # ships with rustup
        ),
        CheckSpec(
            name="cargo test",
            cmd_template="cargo test --quiet --no-run",
            extensions=(".rs",),
            per_file=False,
            install_hint="",
        ),
    ),
)

_GO = StackProfile(
    name="go",
    detect_files=("go.mod",),
    detect_extensions=(".go",),
    checks=(
        CheckSpec(
            name="go vet",
            cmd_template="go vet ./...",
            extensions=(".go",),
            per_file=False,
            install_hint="",  # ships with go
        ),
        CheckSpec(
            name="go test",
            cmd_template="go test ./...",
            extensions=(".go",),
            per_file=False,
            install_hint="",
        ),
    ),
)

_GENERIC = StackProfile(
    name="generic",
    detect_files=(),
    detect_extensions=(),
    checks=(),  # no specific checks; reviewer cold-reads the diff anyway
)

# Order matters: most-specific marker first. The first profile whose
# detect_files OR detect_extensions match wins.
_PROFILES: tuple[StackProfile, ...] = (_RUST, _GO, _NODE, _PYTHON, _GENERIC)


def _has_extension_anywhere(cwd: Path, extensions: tuple[str, ...]) -> bool:
    """Cheap recursive-ish check: any file in cwd / one level / two levels
    deep matches one of the extensions. Walking the entire tree on every
    /precommit-review would be wasteful."""
    if not extensions:
        return False
    for ext in extensions:
        if any(cwd.glob(f"*{ext}")):
            return True
        if any(cwd.glob(f"*/*{ext}")):
            return True
        if any(cwd.glob(f"*/*/*{ext}")):
            return True
    return False


def detect_stack(cwd: Path) -> StackProfile:
    """Pick the most specific stack for cwd. Falls back to generic."""
    for profile in _PROFILES:
        for marker in profile.detect_files:
            if (cwd / marker).exists():
                return profile
        if _has_extension_anywhere(cwd, profile.detect_extensions):
            return profile
    return _GENERIC


def relevant_changes(check: CheckSpec, changed_files: list[str]) -> list[str]:
    """Return the subset of `changed_files` whose extensions match the check.

    Empty extensions tuple means the check is project-wide (no filter):
    in that case we return changed_files as-is so callers can decide
    whether to run based on whether anything changed at all.
    """
    if not check.extensions:
        return list(changed_files)
    return [f for f in changed_files if any(f.endswith(e) for e in check.extensions)]


def build_command(check: CheckSpec, files: list[str]) -> str:
    """Materialize the shell command for a check, substituting changed files
    when the check is per-file."""
    if check.per_file:
        quoted = " ".join(shlex.quote(f) for f in files)
        return check.cmd_template.format(files=quoted)
    return check.cmd_template
