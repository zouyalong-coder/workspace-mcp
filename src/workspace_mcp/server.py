from __future__ import annotations

import ipaddress
import warnings

from .core import Workspace, WorkspaceError


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError as exc:
        raise WorkspaceError("host must be localhost or a loopback IP by default") from exc


def build_server(
    workspace: Workspace,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_network: bool = False,
):
    if not allow_network and not _is_loopback(host):
        raise WorkspaceError("refusing non-loopback HTTP host without explicit allow_network")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Field 'lifespan' has an incomplete definition:.*",
            )
            from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("MCP dependency is missing; install with: pip install workspace-mcp") from exc

    server = FastMCP(
        "workspace-mcp",
        host=host,
        port=port,
        instructions=(
            "This server is strictly bounded to one workspace root. "
            "Use read/search/diff before replace_range. Never assume paths outside the workspace are available. "
            "replace_range requires the SHA-256 returned from a prior file read."
        ),
    )

    @server.tool()
    def workspace_root() -> dict[str, str]:
        """Return the single directory this server can access."""
        return {"root": str(workspace.root)}

    @server.tool()
    def list_files(glob: str = "**/*") -> list[dict[str, object]]:
        """List files and directories below the bounded workspace."""
        return [entry.__dict__ for entry in workspace.list_files(glob)]

    @server.tool()
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a UTF-8 text file using a 1-based inclusive line range."""
        return workspace.read_file(path, start_line, end_line)

    @server.tool()
    def search_text(query: str, glob: str = "**/*", max_results: int = 200) -> list[dict[str, object]]:
        """Search UTF-8 text files within the workspace."""
        return workspace.search_text(query, glob, max_results)

    @server.tool()
    def replace_range(
        path: str,
        start_line: int,
        end_line: int,
        replacement: str,
        expected_sha256: str,
    ) -> dict[str, object]:
        """Atomically replace an inclusive line range using a mandatory optimistic-lock hash."""
        return workspace.replace_range(path, start_line, end_line, replacement, expected_sha256)

    @server.tool()
    def git_diff() -> str:
        """Return the current Git diff for the workspace."""
        return workspace.git_diff()

    return server


def run_server(
    workspace: Workspace,
    transport: str,
    host: str,
    port: int,
    allow_network: bool = False,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Field 'lifespan' has an incomplete definition:.*",
        )
        server = build_server(workspace, host, port, allow_network)
        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport="streamable-http")
