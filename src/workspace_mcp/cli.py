from __future__ import annotations

import argparse
import ipaddress
import shlex
import shutil
import socket
import sys
from pathlib import Path

from .core import Workspace, WorkspaceError
from .server import run_server


def find_available_port(host: str, start: int, end: int) -> int:
    if start < 1 or end < start or end > 65535:
        raise WorkspaceError("invalid port range")
    for port in range(start, end + 1):
        try:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise WorkspaceError(f"cannot resolve host: {host}") from exc
        available = False
        for family, socket_type, protocol, _, address in addresses:
            with socket.socket(family, socket_type, protocol) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(address)
                except OSError:
                    continue
                available = True
                break
        if available:
            return port
    raise WorkspaceError(f"no available port in {start}-{end}")


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError as exc:
        raise WorkspaceError("host must be localhost or a loopback IP by default") from exc


def is_interactive_stdio(transport: str, stdin_isatty: bool) -> bool:
    return transport == "stdio" and stdin_isatty


def build_interactive_setup_guide(workspace: Workspace) -> str:
    executable = shutil.which("workspace-mcp") or sys.argv[0]
    command = " ".join(
        shlex.quote(part)
        for part in (
            "codex",
            "mcp",
            "add",
            "workspace-local",
            "--",
            executable,
            "--workspace",
            str(workspace.root),
        )
    )
    http_command = " ".join(
        shlex.quote(part)
        for part in (
            executable,
            "--workspace",
            str(workspace.root),
            "--transport",
            "streamable-http",
        )
    )
    return (
        "workspace-mcp is ready. Its default stdio mode is started by an MCP client, not used as an interactive shell.\n\n"
        "Add this workspace to Codex:\n"
        f"  {command}\n\n"
        "Then run `codex mcp list`, restart Codex, and use `/mcp` to verify the connection.\n\n"
        "For a manual HTTP test instead:\n"
        f"  {http_command}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded local workspace MCP server")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="workspace root; defaults to current directory")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allow-network", action="store_true", help="allow binding the HTTP server beyond loopback")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-port", type=int, default=8865)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = Workspace(args.workspace)
        if is_interactive_stdio(args.transport, sys.stdin.isatty()):
            print(build_interactive_setup_guide(workspace), file=sys.stderr)
            return 0
        if args.transport != "stdio" and not args.allow_network and not is_loopback_host(args.host):
            raise WorkspaceError("refusing non-loopback HTTP host; use --allow-network only with explicit network security")
        port = find_available_port(args.host, args.port, args.max_port) if args.transport != "stdio" else args.port
        if args.transport != "stdio":
            print(f"workspace-mcp serving {workspace.root} at http://{args.host}:{port}/mcp", flush=True)
        run_server(workspace, args.transport, args.host, port, args.allow_network)
    except KeyboardInterrupt:
        print("workspace-mcp: interrupted", file=sys.stderr)
        return 130
    except (WorkspaceError, RuntimeError) as exc:
        print(f"workspace-mcp: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
