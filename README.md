# 🪶 Muninn CLI

<img width="1004" height="674" alt="mununn_cli" src="https://github.com/user-attachments/assets/7cbacd9f-0b1d-4950-b366-0aef2178209a" />

---

Muninn CLI is an ongoing experiment in building a local AI coding agent with asymmetric context to improve the overall response quality of open-weight models like Qwen, Deepseek and Mistral (`qwen3-coder` and `deepseek-r1` are supported at this time).

## Overview

Design is the new code. Two agents with deliberately asymmetric context (one stateful, one stateless) catch more holes than either alone:

**Muninn** (memory) is a stateful, memory-keeping co-author that owns the conversation history and helps user write definitive design docs and code.

**Huginn** (thinking) is a stateless cold-reader spawned per request to stress-test problem docs, design docs, or diffs with zero prior context.

## Install

> Prerequisite: a local Ollama server with `qwen3-coder:30b` pulled. Install Ollama from https://ollama.com/download, then `ollama pull qwen3-coder:30b`.

**macOS / Linux:**

```sh
curl -LsSf https://raw.githubusercontent.com/vshvedov/muninn/main/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/vshvedov/muninn/main/install.ps1 | iex
```

The installer fetches uv if you don't already have it, then runs `uv tool install` to drop muninn into an isolated environment with its own managed Python. The `muninn` command lands on your PATH.

### Run

```sh
muninn ~/code/some-project    # run the TUI against a project directory
muninn                        # against the current directory
```

### Commands

- **`/feature <what you want to build>`** - kicks off the full feature flow: Muninn drafts a design doc, then Huginn runs three cold-reads against it - comprehension (does it read clearly?), critic (what gaps?), readiness (is it first-pass implementable?) - the doc gets revised until both critic and readiness sign off, then Muninn implements it. Mirrors the article's three-goldfish design check. Best for anything you'd otherwise spend an afternoon scoping.
- **`/bug <what's broken>`** - bug-hunting flow: Muninn grounds the report against the codebase, writes a problem doc, Huginn pokes at the diagnosis, you get a failing test, then a fix. Good for "this thing is wrong and I don't yet know why."
- **`/brainstorm <rough idea>`** - upstream ideation flow: Muninn grounds the idea against the codebase, three Huginns cold-read in parallel through asymmetric lenses (technical / contrarian / UX), Muninn synthesizes convergent themes vs. divergent ideas vs. recommended next step. Saved to `docs/brainstorms/<slug>-<date>.md`. Use it before you know if the idea is even worth a PRD.
- **`/prd <idea>`** - requirements flow: Muninn grounds, runs a structured Q&A round (3-5 `ask_user` calls to fill the gaps the codebase can't), three research Huginns cold-read in parallel (prior-art / edge-cases / integration), Muninn synthesizes a full PRD matching the project's existing PRD style. Saved to `docs/prds/<slug>-<date>.md`. Chain `/feature` after to subject the PRD to the implementation gate.
- **`/precommit-review`** - one-shot review of your current diff: runs stack-aware local checks (lint, tests, syntax) and hands the patch to Huginn for a cold-read. Run it before opening a PR so the substantive review is settled by the time anyone else looks.

### Update

```sh
muninn update                 # upgrades to the latest release
```

### Uninstall

```sh
uv tool uninstall muninn
```

## Development

Local dev environment is also managed by uv.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone and sync:
   ```sh
   git clone https://github.com/vshvedov/muninn.git
   cd muninn
   uv sync
   ```
   uv installs Python 3.14, runtime deps, and dev deps (pytest, pytest-asyncio) into `.venv` automatically.
3. Run the TUI in dev mode:
   ```sh
   uv run muninn ~/code/some-project
   ```
4. Run tests:
   ```sh
   uv run pytest
   ```

Dev/test deps live under `[dependency-groups].dev` in [pyproject.toml](pyproject.toml).

## License

MIT
