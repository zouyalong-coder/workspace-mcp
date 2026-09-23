from __future__ import annotations

import hashlib
import os
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(ValueError):
    """Raised when a workspace operation violates its safety contract."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    kind: str
    size: int = 0


class Workspace:
    _LOCK_DIRECTORY = ".workspace-mcp-locks"

    def __init__(self, root: str | os.PathLike[str], max_file_bytes: int = 5 * 1024 * 1024) -> None:
        candidate = Path(root).expanduser().resolve()
        if not candidate.is_dir():
            raise WorkspaceError(f"workspace is not a directory: {candidate}")
        self.root = candidate
        self.max_file_bytes = max_file_bytes

    @staticmethod
    def _parts(relative: str) -> tuple[str, ...]:
        requested = Path(relative)
        if requested.is_absolute():
            raise WorkspaceError("path must be relative to the workspace")
        parts = requested.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise WorkspaceError("path must be a normalized relative file path")
        return parts

    @staticmethod
    def _validate_glob(glob: str) -> None:
        normalized = glob.replace("\\", "/")
        if not normalized or normalized.startswith("/"):
            raise WorkspaceError("glob must be relative to the workspace")
        if len(normalized) >= 2 and normalized[1] == ":":
            raise WorkspaceError("glob must be relative to the workspace")
        if any(part == ".." for part in normalized.split("/")):
            raise WorkspaceError("glob cannot contain parent-directory components")

    @property
    def _supports_secure_dir_fd(self) -> bool:
        return (
            os.name == "posix"
            and hasattr(os, "O_NOFOLLOW")
            and os.open in os.supports_dir_fd
            and os.mkdir in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.rename in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
        )

    def _require_secure_platform(self) -> None:
        if not self._supports_secure_dir_fd:
            raise WorkspaceError(
                "secure workspace file access is unavailable on this platform; "
                "this build fails closed rather than following race-prone paths"
            )

    def _secure_open(self, relative: str, flags: int, mode: int = 0o600) -> int:
        """Open a regular file relative to the workspace without following symlinks."""
        self._require_secure_platform()
        parts = self._parts(relative)
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        current_fd = root_fd
        try:
            for component in parts[:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = next_fd
            descriptor = os.open(parts[-1], flags | os.O_NOFOLLOW, mode, dir_fd=current_fd)
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                os.close(descriptor)
                raise WorkspaceError("path is not a regular file")
            return descriptor
        except OSError as exc:
            raise WorkspaceError(f"cannot safely open workspace file: {relative}") from exc
        finally:
            if current_fd != root_fd:
                os.close(current_fd)
            os.close(root_fd)

    def _secure_parent(self, relative: str) -> tuple[int, str]:
        self._require_secure_platform()
        parts = self._parts(relative)
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        current_fd = root_fd
        try:
            for component in parts[:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = next_fd
            if current_fd != root_fd:
                os.close(root_fd)
            return current_fd, parts[-1]
        except OSError as exc:
            if current_fd != root_fd:
                os.close(current_fd)
            os.close(root_fd)
            raise WorkspaceError(f"cannot safely open parent directory: {relative}") from exc

    def _lock_name(self, relative: str) -> str:
        normalized = "/".join(self._parts(relative))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() + ".lock"

    def _acquire_stable_lock(self, relative: str) -> int:
        """Lock a stable per-path file that is never replaced during target writes."""
        self._require_secure_platform()
        try:
            import fcntl
        except ImportError as exc:
            raise WorkspaceError("cross-process file locking is unavailable") from exc

        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        lock_dir_fd = -1
        try:
            try:
                os.mkdir(self._LOCK_DIRECTORY, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            lock_dir_fd = os.open(
                self._LOCK_DIRECTORY,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
            lock_name = self._lock_name(relative)
            lock_fd = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=lock_dir_fd,
            )
            lock_stat = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_stat.st_mode):
                os.close(lock_fd)
                raise WorkspaceError("workspace lock is not a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            current_lock_stat = os.stat(lock_name, dir_fd=lock_dir_fd, follow_symlinks=False)
            if current_lock_stat.st_dev != lock_stat.st_dev or current_lock_stat.st_ino != lock_stat.st_ino:
                os.close(lock_fd)
                raise WorkspaceError("workspace lock changed while acquiring it")
            return lock_fd
        except OSError as exc:
            raise WorkspaceError(f"cannot acquire stable workspace lock for: {relative}") from exc
        finally:
            if lock_dir_fd >= 0:
                os.close(lock_dir_fd)
            os.close(root_fd)

    def list_files(self, glob: str = "**/*") -> list[FileEntry]:
        self._require_secure_platform()
        self._validate_glob(glob)
        entries: list[FileEntry] = []
        for path in sorted(self.root.glob(glob)):
            try:
                relative = path.relative_to(self.root).as_posix()
            except ValueError:
                continue
            if relative == self._LOCK_DIRECTORY or relative.startswith(f"{self._LOCK_DIRECTORY}/"):
                continue
            if path.is_symlink():
                continue
            if path.is_dir():
                entries.append(FileEntry(relative, "directory"))
            elif path.is_file():
                entries.append(FileEntry(relative, "file", path.stat().st_size))
        return entries

    def read_file(self, relative: str, start_line: int = 1, end_line: int | None = None) -> str:
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise WorkspaceError("invalid line range")
        descriptor = self._secure_open(relative, os.O_RDONLY)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as file_handle:
                raw = file_handle.read(self.max_file_bytes + 1)
        finally:
            os.close(descriptor)
        if len(raw) > self.max_file_bytes:
            raise WorkspaceError(f"file exceeds {self.max_file_bytes} byte limit")
        try:
            lines = raw.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as exc:
            raise WorkspaceError("file is not valid UTF-8 text") from exc
        return "".join(lines[start_line - 1 : end_line])

    def search_text(self, query: str, glob: str = "**/*", max_results: int = 200) -> list[dict[str, object]]:
        if not query:
            raise WorkspaceError("query must not be empty")
        if max_results < 1 or max_results > 10_000:
            raise WorkspaceError("max_results must be between 1 and 10000")
        results: list[dict[str, object]] = []
        for entry in self.list_files(glob):
            if entry.kind != "file" or entry.size > self.max_file_bytes:
                continue
            try:
                lines = self.read_file(entry.path).splitlines()
            except WorkspaceError:
                continue
            for number, line in enumerate(lines, 1):
                if query in line:
                    results.append({"path": entry.path, "line": number, "text": line})
                    if len(results) >= max_results:
                        return results
        return results

    def replace_range(
        self,
        relative: str,
        start_line: int,
        end_line: int,
        replacement: str,
        expected_sha256: str,
    ) -> dict[str, object]:
        if start_line < 1 or end_line < start_line:
            raise WorkspaceError("invalid line range")
        if not expected_sha256:
            raise WorkspaceError("expected_sha256 is required for writes")

        lock_fd = self._acquire_stable_lock(relative)
        descriptor = -1
        parent_fd = -1
        temp_name: str | None = None
        try:
            descriptor = self._secure_open(relative, os.O_RDWR)
            source_stat = os.fstat(descriptor)
            if source_stat.st_nlink != 1:
                raise WorkspaceError("files with multiple hard links cannot be safely edited")
            raw = os.pread(descriptor, self.max_file_bytes + 1, 0)
            if len(raw) > self.max_file_bytes:
                raise WorkspaceError(f"file exceeds {self.max_file_bytes} byte limit")
            digest = hashlib.sha256(raw).hexdigest()
            if expected_sha256 != digest:
                raise WorkspaceError("file changed since it was read; expected_sha256 does not match")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspaceError("file is not valid UTF-8 text") from exc
            lines = text.splitlines(keepends=True)
            if start_line > len(lines) or end_line > len(lines):
                raise WorkspaceError("line range is outside the file")
            normalized = replacement
            if normalized and not normalized.endswith("\n") and (end_line < len(lines) or text.endswith("\n")):
                normalized += "\n"
            updated = "".join(lines[: start_line - 1]) + normalized + "".join(lines[end_line:])
            new_bytes = updated.encode("utf-8")
            if len(new_bytes) > self.max_file_bytes:
                raise WorkspaceError(f"updated file exceeds {self.max_file_bytes} byte limit")

            parent_fd, leaf = self._secure_parent(relative)
            current_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(current_stat.st_mode) or (
                current_stat.st_dev != source_stat.st_dev or current_stat.st_ino != source_stat.st_ino
            ):
                raise WorkspaceError("target file changed during write")

            temp_name = f".{leaf}.workspace-mcp.{secrets.token_hex(12)}.tmp"
            temp_fd = os.open(
                temp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                stat.S_IMODE(source_stat.st_mode),
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(new_bytes)
                while view:
                    view = view[os.write(temp_fd, view) :]
                os.fsync(temp_fd)
            finally:
                os.close(temp_fd)

            current_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if current_stat.st_dev != source_stat.st_dev or current_stat.st_ino != source_stat.st_ino:
                raise WorkspaceError("target file changed before atomic rename")
            os.rename(temp_name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_name = None
            os.fsync(parent_fd)
            return {
                "path": relative,
                "old_sha256": digest,
                "new_sha256": hashlib.sha256(new_bytes).hexdigest(),
                "bytes_written": len(new_bytes),
            }
        finally:
            if temp_name and parent_fd >= 0:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if parent_fd >= 0:
                os.close(parent_fd)
            if descriptor >= 0:
                os.close(descriptor)
            os.close(lock_fd)

    def git_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "diff", "--", "."],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceError(f"git diff failed: {exc}") from exc
        if result.returncode not in (0, 1):
            raise WorkspaceError(result.stderr.strip() or "git diff failed")
        return result.stdout
