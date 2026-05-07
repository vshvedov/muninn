"""Stack detection + check selection tests."""
from __future__ import annotations

from pathlib import Path

from muninn import stacks


def test_detect_python_via_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool]\n")
    assert stacks.detect_stack(tmp_path).name == "python"


def test_detect_python_via_loose_py_files(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n")
    assert stacks.detect_stack(tmp_path).name == "python"


def test_detect_node(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}")
    assert stacks.detect_stack(tmp_path).name == "node"


def test_detect_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    assert stacks.detect_stack(tmp_path).name == "rust"


def test_detect_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module x\n")
    assert stacks.detect_stack(tmp_path).name == "go"


def test_detect_generic_when_empty(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# hi\n")
    assert stacks.detect_stack(tmp_path).name == "generic"


def test_rust_takes_precedence_over_loose_py(tmp_path: Path) -> None:
    """A repo with Cargo.toml + a stray .py is Rust, not Python."""
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "scripts" / "tool.py").parent.mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("x = 1\n")
    assert stacks.detect_stack(tmp_path).name == "rust"


def test_relevant_changes_filters_by_extension() -> None:
    spec = next(c for c in stacks._PYTHON.checks if c.name == "ruff")
    assert spec.extensions == (".py",)
    assert stacks.relevant_changes(spec, ["a.py", "README.md"]) == ["a.py"]
    assert stacks.relevant_changes(spec, ["README.md"]) == []


def test_relevant_changes_no_filter_means_pass_through() -> None:
    spec = stacks.CheckSpec(name="x", cmd_template="echo")
    assert stacks.relevant_changes(spec, ["a.py", "b.md"]) == ["a.py", "b.md"]


def test_build_command_per_file() -> None:
    spec = stacks.CheckSpec(
        name="ruff", cmd_template="ruff check {files}",
        extensions=(".py",), per_file=True,
    )
    cmd = stacks.build_command(spec, ["foo.py", "bar baz.py"])
    assert "foo.py" in cmd
    # spaces in path must be quoted
    assert "'bar baz.py'" in cmd


def test_build_command_project_wide_ignores_files() -> None:
    spec = stacks.CheckSpec(
        name="pytest", cmd_template="python -m pytest -q",
        extensions=(".py",), per_file=False,
    )
    cmd = stacks.build_command(spec, ["foo.py"])
    assert cmd == "python -m pytest -q"
