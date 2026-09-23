from __future__ import annotations

import hashlib
import socket
import threading
from pathlib import Path

import pytest

from workspace_mcp import cli
from workspace_mcp.cli import build_interactive_setup_guide, find_available_port, is_interactive_stdio, is_loopback_host
from workspace_mcp.core import Workspace, WorkspaceError
from workspace_mcp.server import build_server


def test_rejects_escape_absolute_and_parent_globs(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("print('ok')\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.read_file("../code.py")
    with pytest.raises(WorkspaceError):
        workspace.read_file(str(tmp_path / "code.py"))
    for pattern in ("../*", "**/../*", r"..\\*"):
        with pytest.raises(WorkspaceError):
            workspace.list_files(pattern)


def test_reads_searches_and_replaces_with_hash(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("one\ntwo\nthree\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    assert workspace.read_file("code.py", 2, 2) == "two\n"
    assert workspace.search_text("two")[0]["line"] == 2
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    result = workspace.replace_range("code.py", 2, 2, "changed", digest)
    assert result["old_sha256"] == digest
    assert file.read_text(encoding="utf-8") == "one\nchanged\nthree\n"
    assert all(".workspace-mcp-locks" not in entry.path for entry in workspace.list_files())


def test_two_workspace_instances_enforce_cross_instance_cas(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("one\n", encoding="utf-8")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    workspaces = [Workspace(tmp_path), Workspace(tmp_path)]
    outcomes: list[tuple[str, str]] = []
    barrier = threading.Barrier(2)

    def write(workspace: Workspace, value: str) -> None:
        barrier.wait()
        try:
            workspace.replace_range("code.py", 1, 1, value, digest)
            outcomes.append(("ok", value))
        except WorkspaceError as exc:
            outcomes.append(("stale", str(exc)))

    threads = [
        threading.Thread(target=write, args=(workspaces[0], "two")),
        threading.Thread(target=write, args=(workspaces[1], "three")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(result for result, _ in outcomes) == ["ok", "stale"]
    winning_value = next(value for result, value in outcomes if result == "ok")
    assert file.read_text(encoding="utf-8") == f"{winning_value}\n"
    assert "expected_sha256" in next(value for result, value in outcomes if result == "stale")


def test_stable_lock_handles_nested_path_and_releases_after_error(tmp_path: Path) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    file = nested / "code.py"
    file.write_text("one\n", encoding="utf-8")
    first = Workspace(tmp_path)
    second = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="expected_sha256"):
        first.replace_range("src/code.py", 1, 1, "bad", "stale")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    second.replace_range("src/code.py", 1, 1, "good", digest)
    assert file.read_text(encoding="utf-8") == "good\n"
    assert not list(nested.glob("*.workspace-mcp.*.tmp"))


def test_precreated_lock_symlink_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    outside = tmp_path.parent / "outside-lock.txt"
    file.write_text("one\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    lock_directory = tmp_path / workspace._LOCK_DIRECTORY
    lock_directory.mkdir(mode=0o700)
    (lock_directory / workspace._lock_name("code.py")).symlink_to(outside)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(WorkspaceError, match="stable workspace lock"):
        workspace.replace_range("code.py", 1, 1, "two", digest)
    assert not outside.exists()
    assert file.read_text(encoding="utf-8") == "one\n"


def test_precreated_lock_directory_symlink_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    outside = tmp_path.parent / "outside-lock-directory"
    outside.mkdir()
    file.write_text("one\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    (tmp_path / workspace._LOCK_DIRECTORY).symlink_to(outside, target_is_directory=True)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(WorkspaceError, match="stable workspace lock"):
        workspace.replace_range("code.py", 1, 1, "two", digest)
    assert file.read_text(encoding="utf-8") == "one\n"


def test_rejects_symlink_escape_even_after_path_parsing(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    target = tmp_path / "target"
    target.write_text("safe", encoding="utf-8")

    class SwappingWorkspace(Workspace):
        def _parts(self, relative: str):
            parts = super()._parts(relative)
            if target.exists() and not target.is_symlink():
                target.unlink()
                target.symlink_to(outside)
            return parts

    workspace = SwappingWorkspace(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.read_file("target")


def test_precreated_temp_symlink_cannot_escape(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    outside = tmp_path.parent / "outside-write.txt"
    target.write_text("old\n", encoding="utf-8")
    fixed_temp = tmp_path / ".target.txt.workspace-mcp.tmp"
    fixed_temp.symlink_to(outside)
    workspace = Workspace(tmp_path)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    workspace.replace_range("target.txt", 1, 1, "new", digest)
    assert not outside.exists()
    assert target.read_text(encoding="utf-8") == "new\n"
    assert fixed_temp.is_symlink()


def test_missing_secure_platform_fails_closed(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("one\n", encoding="utf-8")

    class UnsupportedWorkspace(Workspace):
        @property
        def _supports_secure_dir_fd(self) -> bool:
            return False

    workspace = UnsupportedWorkspace(tmp_path)
    with pytest.raises(WorkspaceError, match="fails closed"):
        workspace.read_file("code.py")
    with pytest.raises(WorkspaceError, match="fails closed"):
        workspace.search_text("one")
    with pytest.raises(WorkspaceError, match="fails closed"):
        workspace.replace_range("code.py", 1, 1, "two", hashlib.sha256(file.read_bytes()).hexdigest())


def test_http_defaults_reject_non_loopback(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    with pytest.raises(WorkspaceError, match="non-loopback"):
        build_server(workspace, host="0.0.0.0")


def test_port_probe_skips_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        occupied = busy.getsockname()[1]
        selected = find_available_port("127.0.0.1", occupied, min(occupied + 10, 65535))
        assert selected != occupied


def test_interactive_terminal_does_not_start_stdio_server() -> None:
    assert is_interactive_stdio("stdio", True)
    assert not is_interactive_stdio("stdio", False)
    assert not is_interactive_stdio("streamable-http", True)


class _FakeInput:
    def __init__(self, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_main_prints_setup_guide_without_starting_interactive_stdio(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeInput(True))
    monkeypatch.setattr(cli, "run_server", lambda *args, **kwargs: pytest.fail("server should not start"))
    monkeypatch.setattr(cli.shutil, "which", lambda command: "/Users/test/.local/bin/workspace-mcp")
    assert cli.main(["--workspace", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "codex mcp add workspace-local" in captured.err
    assert "/Users/test/.local/bin/workspace-mcp" in captured.err
    assert str(tmp_path) in captured.err
    assert "--transport streamable-http" in captured.err


def test_main_converts_keyboard_interrupt_to_exit_130(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _FakeInput(False))

    def interrupt(*args, **kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_server", interrupt)
    assert cli.main(["--workspace", str(tmp_path)]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "interrupted" in captured.err
