import pytest


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path_factory):
    """Redirect HOME to a fresh tmp dir for every test so any code path that
    consults Path.home() (e.g. _check_not_home, or legacy ~/.muninn/ probes)
    sees an isolated, empty home directory and never touches the developer's
    real ~/.muninn/."""
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(fake_home))
