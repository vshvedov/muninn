import asyncio
from pathlib import Path

import pytest

from muninn import bootstrap
from muninn.jsonl_log import JsonlLogger
from muninn.tools import ToolContext, ToolError, make_tools


def _make_ctx(tmp_path: Path, *, freedom_level: str = "low",
              confirm_result: bool = True, ask_answer: str = "yes",
              confirm_counter: list[int] | None = None) -> ToolContext:
    """Build a ToolContext for tool tests.

    `confirm_counter` is an optional list passed in by the gate-matrix
    tests; the confirm callback appends to it on each call so a test can
    assert "confirm was called N times" without instrumenting the tool.
    """
    muninn_dir = bootstrap.ensure_muninn_dir(tmp_path)
    logger = JsonlLogger(muninn_dir / "logs")

    async def confirm(_label, _preview):
        if confirm_counter is not None:
            confirm_counter.append(1)
        return confirm_result

    async def ask(_q, options):
        if not options:
            return ask_answer
        first = options[0]
        return first[0] if isinstance(first, tuple) else first

    return ToolContext(
        cwd=tmp_path,
        muninn_dir=muninn_dir,
        freedom_level=freedom_level,
        confirm_callback=confirm,
        ask_user_callback=ask,
        log=logger.log,
    )


def _impl(tools, name):
    """Pull the underlying coroutine function from a Tool object."""
    for t in tools:
        if t.name == name:
            return t.function
    raise KeyError(name)


async def test_read_file_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi from disk", encoding="utf-8")
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    assert await fn("hello.txt") == "hi from disk"


async def test_read_file_escape_rejected(tmp_path: Path) -> None:
    from pydantic_ai.exceptions import ModelRetry
    (tmp_path.parent / "outside.txt").write_text("nope")
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    with pytest.raises(ModelRetry):
        await fn("../outside.txt")


async def test_read_file_expands_tilde_and_rejects_outside_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """`~` shorthand must expand to $HOME, then trigger the clean
    'outside cwd' ModelRetry rather than a literal-tilde FileNotFoundError."""
    from pydantic_ai.exceptions import ModelRetry
    fake_home = tmp_path.parent / "fake-home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    with pytest.raises(ModelRetry, match="outside the working directory"):
        await fn("~/code/something")
    # The retry message must include both the user's original path and the
    # resolved absolute path so the model can correct its next call.
    try:
        await fn("~/foo")
    except ModelRetry as e:
        msg = str(e)
        assert "~/foo" in msg
        assert "fake-home" in msg
    else:
        pytest.fail("expected ModelRetry")


async def test_read_file_expands_tilde_when_inside_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """If $HOME points INTO cwd (unlikely but possible), `~/foo` should
    resolve correctly to a real file and not throw."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "hello.txt").write_text("hi")
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    assert await fn("~/hello.txt") == "hi"


async def test_read_file_absolute_path_inside_cwd(tmp_path: Path) -> None:
    """A literal absolute path that points inside cwd should work."""
    (tmp_path / "abs.txt").write_text("absolute")
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    assert await fn(str(tmp_path / "abs.txt")) == "absolute"


async def test_read_file_missing_raises_model_retry(tmp_path: Path) -> None:
    """File-not-found is recoverable: model gets a hint, doesn't kill the flow."""
    from pydantic_ai.exceptions import ModelRetry
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "read_file")
    with pytest.raises(ModelRetry, match="does not exist"):
        await fn("nonexistent.txt")


async def test_write_file_high_skips_confirm(tmp_path: Path) -> None:
    """At freedom_level=high, write_file does not pop the confirm modal."""
    ctx = _make_ctx(tmp_path, freedom_level="high")
    tools = make_tools(ctx)
    fn = _impl(tools, "write_file")
    res = await fn("subdir/out.txt", "hello world")
    assert res == {"ok": True, "bytes": len(b"hello world")}
    assert (tmp_path / "subdir" / "out.txt").read_text() == "hello world"


async def test_write_file_denied_in_low(tmp_path: Path) -> None:
    """At freedom_level=low, write_file gates and a denied confirm aborts."""
    ctx = _make_ctx(tmp_path, freedom_level="low", confirm_result=False)
    tools = make_tools(ctx)
    fn = _impl(tools, "write_file")
    with pytest.raises(ToolError, match="denied"):
        await fn("nope.txt", "x")
    assert not (tmp_path / "nope.txt").exists()


async def test_run_shell_captures_output(tmp_path: Path) -> None:
    """High auto-allows the shell call so we exercise the capture path."""
    ctx = _make_ctx(tmp_path, freedom_level="high")
    tools = make_tools(ctx)
    fn = _impl(tools, "run_shell")
    res = await fn("echo hi && false", "")
    assert res["exit"] == 1
    assert "hi" in res["stdout"]


async def test_ask_user_returns_first_option(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "ask_user")
    answer = await fn("pick one", ["a", "b", "c"], ["", "", ""])
    assert answer == "a"


async def test_ask_user_forwards_explanations_as_tuples(tmp_path: Path) -> None:
    """When the model passes per-option explanations, the callback should
    receive (label, expl) tuples so the modal can render the `?` affordance.
    Options with empty explanations stay as plain strings (no `?` button).
    """
    seen: list = []

    async def ask(_q, options):
        seen.extend(options)
        first = options[0]
        return first[0] if isinstance(first, tuple) else first

    ctx = _make_ctx(tmp_path)
    ctx.ask_user_callback = ask
    tools = make_tools(ctx)
    fn = _impl(tools, "ask_user")
    answer = await fn(
        "pick one",
        ["a", "b", "c"],
        ["does the A thing", "", "does the C thing"],
    )
    assert answer == "a"
    assert seen == [("a", "does the A thing"), "b", ("c", "does the C thing")]


async def test_ask_user_rejects_mismatched_explanations(tmp_path: Path) -> None:
    """A length mismatch is a model error - surface it as ModelRetry so the
    model can correct rather than silently dropping data."""
    from pydantic_ai.exceptions import ModelRetry

    ctx = _make_ctx(tmp_path)
    tools = make_tools(ctx)
    fn = _impl(tools, "ask_user")
    with pytest.raises(ModelRetry):
        await fn("pick one", ["a", "b"], ["only one"])


# ---------------------------------------------------------------------------
# freedom_level gate matrix + read-only shell classifier
# ---------------------------------------------------------------------------


async def test_low_confirms_writes_and_shell(tmp_path: Path) -> None:
    """At low, both write_file and run_shell call confirm_callback;
    read_file never does (regression guard)."""
    counter: list[int] = []
    ctx = _make_ctx(tmp_path, freedom_level="low", confirm_counter=counter)
    tools = make_tools(ctx)
    (tmp_path / "in.txt").write_text("data")
    await _impl(tools, "read_file")("in.txt")
    assert counter == []
    await _impl(tools, "write_file")("out.txt", "x")
    await _impl(tools, "run_shell")("ls", ".")
    assert len(counter) == 2


async def test_medium_auto_allows_read_only_shell(tmp_path: Path) -> None:
    counter: list[int] = []
    ctx = _make_ctx(tmp_path, freedom_level="medium", confirm_counter=counter)
    tools = make_tools(ctx)
    res = await _impl(tools, "run_shell")("ls -la", ".")
    assert res["exit"] == 0
    assert counter == []


async def test_medium_confirms_mutating_shell(tmp_path: Path) -> None:
    counter: list[int] = []
    ctx = _make_ctx(tmp_path, freedom_level="medium",
                    confirm_counter=counter, confirm_result=False)
    tools = make_tools(ctx)
    from muninn.tools import ToolError
    with pytest.raises(ToolError, match="denied"):
        await _impl(tools, "run_shell")("rm -rf nope", ".")
    assert counter == [1]


async def test_medium_still_confirms_writes(tmp_path: Path) -> None:
    """Writes still gate at medium - only high auto-allows them."""
    counter: list[int] = []
    ctx = _make_ctx(tmp_path, freedom_level="medium", confirm_counter=counter)
    tools = make_tools(ctx)
    await _impl(tools, "write_file")("ok.txt", "data")
    assert counter == [1]


async def test_high_skips_all_confirms(tmp_path: Path) -> None:
    """At high, neither write_file nor run_shell calls confirm."""
    counter: list[int] = []
    ctx = _make_ctx(tmp_path, freedom_level="high", confirm_counter=counter)
    tools = make_tools(ctx)
    await _impl(tools, "write_file")("ok.txt", "data")
    await _impl(tools, "run_shell")("rm -rf nope-not-real", ".")
    assert counter == []


def test_is_read_only_classifier_table() -> None:
    """Table-driven coverage of the medium-level shell allowlist."""
    from muninn.tools import _is_read_only

    cases = [
        # Read-only - first-token allowlist.
        ("ls -la", True),
        ("pytest tests/", True),
        ("ruff check .", True),
        ("cat file.txt", True),
        ("grep -r foo .", True),
        # Read-only git subcommands.
        ("git status", True),
        ("git log --oneline", True),
        ("git diff main...HEAD", True),
        ("git rev-parse HEAD", True),
        # Mutating git subcommands rejected.
        ("git push", False),
        ("git fetch", False),
        ("git stash", False),
        ("git config user.name x", False),
        ("git remote add origin y", False),
        ("git tag v1", False),
        # Outright mutating commands rejected.
        ("rm -rf .", False),
        ("mv a b", False),
        ("touch new.txt", False),
        # Forbidden shell metacharacters.
        ("ls > out.txt", False),
        ("ls >> out.txt", False),
        ("cat a | tee b", False),
        ("ls; rm x", False),
        ("ls && rm x", False),
        ("ls || rm x", False),
        ("$(rm -rf .)", False),
        ("`rm -rf .`", False),
        ("ls < input", False),
        ("ls &", False),
        # Env-prefix and wrapper rejection.
        ("FOO=1 ls", False),
        ("sudo ls", False),
        ("env X=1 ls", False),
        # Interpreter footguns explicitly NOT allowlisted.
        ("python -c 'print(1)'", False),
        ("python3 -m pytest", False),
        ("node -e '1'", False),
        ("cargo run", False),
        ("go run main.go", False),
        # Empty / whitespace / unbalanced quoting.
        ("", False),
        ("   ", False),
        ("'unbalanced", False),
    ]
    for cmd, want in cases:
        got = _is_read_only(cmd)
        assert got is want, f"_is_read_only({cmd!r}) returned {got}, expected {want}"
