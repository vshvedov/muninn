#!/bin/sh
# muninn installer (macOS / Linux).
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/vshvedov/muninn/main/install.sh | sh
#
# Or, after cloning the repo manually:
#   sh install.sh
#
# Environment variables (all optional):
#   MUNINN_REPO    git URL to install from (default: vshvedov/muninn)
#   MUNINN_BRANCH  branch / tag to install (default: main)
#
# What it does:
#   1. installs uv (https://astral.sh/uv) if missing
#   2. installs muninn into an isolated uv-managed tool environment via
#      `uv tool install`, exposing the `muninn` command on PATH
#   3. warns if uv's bin dir is not on PATH and prints the export line
#
# Update later:
#   muninn update
#
# Uninstall:
#   uv tool uninstall muninn

set -e

REPO="${MUNINN_REPO:-https://github.com/vshvedov/muninn.git}"
BRANCH="${MUNINN_BRANCH:-main}"

# ---------- helpers ---------------------------------------------------

ok()   { printf '[OK]  %s\n' "$1"; }
info() { printf '      %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1" >&2; exit 1; }

detect_os() {
    case "$(uname -s 2>/dev/null || echo unknown)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      echo "unknown" ;;
    esac
}

git_install_hint() {
    case "$1" in
        macos) echo "  brew install git    # or: xcode-select --install" ;;
        linux) echo "  apt install git     # Debian / Ubuntu" ;
                echo "  dnf install git     # Fedora / RHEL" ;
                echo "  pacman -S git       # Arch" ;;
        *)     echo "  Visit https://git-scm.com/downloads" ;;
    esac
}

# ---------- pre-flight ------------------------------------------------

OS=$(detect_os)
echo "Installing muninn..."
echo "  os:       $OS"
echo "  source:   $REPO ($BRANCH)"
echo

if ! command -v git >/dev/null 2>&1; then
    fail "git not found (uv needs it to clone the muninn source). Install git first:
$(git_install_hint "$OS")"
fi
ok "git $(git --version | awk '{print $3}')"

# ---------- uv --------------------------------------------------------
# uv brings its own managed Python interpreter, so the user does NOT need
# Python installed. If uv is missing, install it via Astral's official
# one-line bootstrapper.

if ! command -v uv >/dev/null 2>&1; then
    info "uv not found, installing via https://astral.sh/uv/install.sh ..."
    # uv's installer writes to ~/.local/bin (or $XDG_DATA_HOME) and updates
    # the shell rc files for next-login PATH. Source its env script so the
    # current shell picks it up immediately.
    #
    # Download to a temp file FIRST and check curl's exit status before piping
    # to sh: under POSIX sh `curl ... | sh` doesn't propagate curl's exit
    # status (no pipefail), so a network blip would leave the user with a
    # misleading "uv not on PATH" error several lines later.
    UV_INSTALLER_TMP=$(mktemp 2>/dev/null || mktemp -t uv-install)
    # `set -e` will bail on any failure between here and the explicit cleanup
    # below; an EXIT trap guarantees the temp file is removed regardless of
    # which step (curl, sh, or fail itself) terminated the script.
    trap 'rm -f "$UV_INSTALLER_TMP"' EXIT INT TERM
    if ! curl -LsSf https://astral.sh/uv/install.sh -o "$UV_INSTALLER_TMP"; then
        fail "Could not download the uv installer from https://astral.sh/uv/install.sh.
  Check your network / DNS / proxy and re-run this script."
    fi
    if ! sh "$UV_INSTALLER_TMP"; then
        fail "uv installer reported an error (see output above). Re-run this script after resolving the cause."
    fi
    rm -f "$UV_INSTALLER_TMP"
    trap - EXIT INT TERM
    # The uv installer drops `~/.local/bin/env` (or similar) the first time.
    # Try the conventional locations; fall back to PATH if all are absent.
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    fi
    if ! command -v uv >/dev/null 2>&1; then
        # Last resort: append the canonical install dir to PATH for this run
        # so the install can complete; the warning at the end nudges the
        # user to add it permanently.
        PATH="$HOME/.local/bin:$PATH"
        export PATH
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    fail "uv was installed but is not on PATH. Open a new shell and re-run this script."
fi
ok "uv $(uv --version | awk '{print $2}')"

# ---------- install muninn -------------------------------------------
# `--reinstall` ensures re-running the installer always pulls the latest
# commit on $BRANCH (uv caches git sources by URL+ref otherwise).

info "Installing muninn from $REPO@$BRANCH ..."
uv tool install --reinstall --from "git+${REPO}@${BRANCH}" muninn
ok "muninn installed"

# ---------- PATH check ------------------------------------------------
# `uv tool install` places binaries in `uv tool dir --bin` (typically
# ~/.local/bin). Warn the user if that dir is not on PATH.

UV_BIN_DIR=$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")
# `uv tool dir --bin` can succeed with empty stdout on misconfigured envs;
# `||` only fires on non-zero exit. Guard against the empty-string case so we
# don't print "WARN  is not on your PATH" with a blank path.
[ -z "$UV_BIN_DIR" ] && UV_BIN_DIR="$HOME/.local/bin"
case ":$PATH:" in
    *":$UV_BIN_DIR:"*)
        ;;
    *)
        printf '\n[WARN] %s is not on your PATH.\n' "$UV_BIN_DIR" >&2
        printf '       Add this to your shell rc file (~/.zshrc, ~/.bashrc, etc.):\n' >&2
        printf '           export PATH="%s:$PATH"\n' "$UV_BIN_DIR" >&2
        ;;
esac

# ---------- next steps ------------------------------------------------

cat <<EOF

✅ muninn installed
   command:  $UV_BIN_DIR/muninn

Next steps:
  1. Make sure Ollama is running:   ollama serve
  2. Pull a compatible model:       ollama pull qwen3-coder:30b
  3. Run in any project directory:  muninn ~/code/some-project

To upgrade later:    muninn update
To uninstall:        uv tool uninstall muninn
EOF
