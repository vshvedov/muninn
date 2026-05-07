"""Tests for the install.sh / install.ps1 distribution surface.

We don't try to actually run the installers end-to-end here - that would
need network access, a real uv binary, and would mutate the test machine.
Instead we verify the static contracts: the scripts exist, parse cleanly,
bootstrap uv when missing, and run `uv tool install` against the
configured git source.
"""
from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"


# ---------------------------------------------------------------------
# install.sh (macOS / Linux)
# ---------------------------------------------------------------------


def test_install_sh_exists() -> None:
    assert INSTALL_SH.is_file(), f"missing {INSTALL_SH}"


def test_install_sh_has_shebang_and_is_executable() -> None:
    head = INSTALL_SH.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!"), "install.sh must start with a #! shebang line"
    assert "/sh" in head, f"shebang should point at sh, got: {head}"
    mode = INSTALL_SH.stat().st_mode
    assert mode & stat.S_IXUSR, "install.sh must be executable by owner"


def test_install_sh_uses_posix_sh_idioms() -> None:
    """install.sh ships as /bin/sh - reject bash-only constructs that
    would break on dash / ash. Static lint, not a full parser; we catch
    the easy ones."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "[[" not in text, "use [ ... ] not [[ ... ]] for POSIX sh portability"
    assert "function " not in text, "use `name() { ... }` not `function name { ... }`"


def test_install_sh_parses_under_sh() -> None:
    """`sh -n` does syntax-only parsing; cheap fast feedback that the
    script doesn't have a stray quote or unbalanced brace."""
    result = subprocess.run(
        ["sh", "-n", str(INSTALL_SH)], capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"install.sh failed sh -n parse:\n{result.stdout}{result.stderr}"
    )


def test_install_sh_passes_shellcheck_if_available() -> None:
    """If shellcheck is on PATH, run it. Otherwise skip cleanly."""
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        ["shellcheck", "--shell=sh", "--severity=warning", str(INSTALL_SH)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"shellcheck found issues in install.sh:\n{result.stdout}{result.stderr}"
    )


def test_install_sh_bootstraps_uv_when_missing() -> None:
    """The whole point of the new installer: users do NOT need Python or
    pip pre-installed - we install uv, which then brings its own Python.
    Verify the script gates on `command -v uv` and falls back to Astral's
    official one-liner when uv is absent."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "command -v uv" in text, (
        "install.sh must check for uv before assuming it's available"
    )
    assert "https://astral.sh/uv/install.sh" in text, (
        "install.sh must bootstrap uv via Astral's official installer "
        "(https://astral.sh/uv/install.sh) when uv is not on PATH"
    )


def test_install_sh_runs_uv_tool_install_against_git_source() -> None:
    """The installation step must use `uv tool install` so muninn lands in
    an isolated tool environment with the `muninn` console script on
    PATH. The source is a git URL composed from MUNINN_REPO / MUNINN_BRANCH."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "uv tool install" in text, (
        "install.sh must use `uv tool install` (not `uv pip install` or `pip`)"
    )
    # `--reinstall` is required so re-running the installer pulls the
    # latest commit on $BRANCH instead of uv-cached source.
    assert "--reinstall" in text, (
        "uv tool install must use --reinstall so re-running the installer "
        "pulls the latest commit on the configured branch"
    )
    assert 'git+${REPO}@${BRANCH}' in text or 'git+$REPO@$BRANCH' in text, (
        "install command must build a git+URL from MUNINN_REPO and MUNINN_BRANCH"
    )


def test_install_sh_warns_when_uv_bin_dir_not_on_path() -> None:
    """`uv tool install` puts the binary somewhere like ~/.local/bin -
    if that's not on PATH, the user types `muninn` and gets command-not-
    found. Surface a clear export hint instead of silent failure."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "uv tool dir --bin" in text, (
        "install.sh must ask uv where its bin dir actually is"
    )
    assert 'export PATH=' in text, "must print an `export PATH=...` hint"


def test_install_sh_declares_expected_env_vars() -> None:
    """Documented env vars must all be referenced. Old ones (MUNINN_INSTALL_DIR,
    MUNINN_BIN_DIR) belonged to the venv-based installer and are no longer
    needed - uv owns those locations now."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    for var in ("MUNINN_REPO", "MUNINN_BRANCH"):
        assert var in text, f"install.sh should reference ${var}"
    # Stale vars must NOT linger: they imply behavior the script no longer has.
    for stale in ("MUNINN_INSTALL_DIR", "MUNINN_BIN_DIR"):
        assert stale not in text, (
            f"install.sh must not reference {stale} - that var belonged to "
            "the old venv-based installer"
        )


def test_install_sh_documents_update_and_uninstall_paths() -> None:
    """The bottom-of-script next-steps block must point users at the
    right commands for self-update (`muninn update`) and removal."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "muninn update" in text, "install.sh should document `muninn update`"
    assert "uv tool uninstall muninn" in text, (
        "install.sh should document the uninstall command"
    )


# ---------------------------------------------------------------------
# install.ps1 (Windows)
# ---------------------------------------------------------------------


def test_install_ps1_exists() -> None:
    assert INSTALL_PS1.is_file(), f"missing {INSTALL_PS1}"


def test_install_ps1_bootstraps_uv_when_missing() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    assert "Get-Command uv" in text, "install.ps1 must check for uv"
    assert "https://astral.sh/uv/install.ps1" in text, (
        "install.ps1 must bootstrap uv via Astral's official PowerShell installer"
    )


def test_install_ps1_runs_uv_tool_install_against_git_source() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    assert "uv tool install" in text
    assert "--reinstall" in text
    assert 'git+$Repo@$Branch' in text, (
        "install command must build a git+URL from MUNINN_REPO and MUNINN_BRANCH"
    )


def test_install_ps1_declares_expected_env_vars() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    for var in ("MUNINN_REPO", "MUNINN_BRANCH"):
        assert var in text, f"install.ps1 should reference $env:{var}"


def test_install_ps1_documents_update_and_uninstall_paths() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    assert "muninn update" in text
    assert "uv tool uninstall muninn" in text


def test_install_ps1_parses_under_pwsh_if_available() -> None:
    """Mirror of `test_install_sh_parses_under_sh`: when PowerShell is on
    PATH, parse install.ps1 via PowerShell's own AST parser to catch
    unbalanced braces / quotes the static-string assertions miss. Skip
    cleanly when pwsh is absent (e.g. macOS / Linux dev machines)."""
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("pwsh / powershell not installed")
    # Use the Parser to tokenize without executing the script. Errors are
    # emitted as a [System.Management.Automation.Language.ParseError[]] in
    # the third out-param; we surface them via $errors to the test.
    cmd = (
        "$errors = $null; $tokens = $null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{INSTALL_PS1}', [ref]$tokens, [ref]$errors); "
        "if ($errors) { $errors | ForEach-Object { Write-Output $_ }; exit 1 } else { exit 0 }"
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"install.ps1 failed PowerShell AST parse:\n{result.stdout}{result.stderr}"
    )
