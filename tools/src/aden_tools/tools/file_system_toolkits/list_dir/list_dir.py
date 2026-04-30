"""Agent-sandboxed search_files registration.

This toolkit historically registered a separate ``list_dir`` tool that
returned ``{name, type, size_bytes}`` dicts. It has been folded into
``search_files`` — one tool covers grep, find, and ls. We keep this
module as the registration site for the agent-sandboxed variant so
toolkits scoped via ``get_sandboxed_path(path, agent_id)`` continue to
expose file search through the same canonical name.
"""

import os

from mcp.server.fastmcp import FastMCP

from ..security import get_sandboxed_path


def register_tools(mcp: FastMCP) -> None:
    """Register the agent-sandboxed search_files tool with the MCP server."""

    @mcp.tool()
    def search_files(
        pattern: str = "*",
        target: str = "files",
        path: str = ".",
        file_glob: str = "",
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
        agent_id: str = "",
    ) -> str:
        """Search file contents or find files by name within the agent sandbox.

        Use this instead of grep, find, or ls.

          target='files' (default here): list/find files by glob — mtime-sorted.
          target='content': regex search inside files.

        Args:
            pattern: Glob (files mode) or regex (content mode). Defaults to ``*``
                so a bare call lists every file in the sandbox.
            target: 'files' (default) or 'content'. Legacy aliases: 'grep'/'find'/'ls'.
            path: Directory or file relative to the agent sandbox.
            file_glob: Restrict content search to files matching this glob.
            limit: Max results (default 50).
            offset: Pagination offset (default 0).
            output_mode: Content-mode output — 'content' | 'files_only' | 'count'.
            context: Lines of surrounding context for content matches.
            agent_id: Auto-injected — sandbox owner.
        """
        from aden_tools.file_ops import (
            _do_search_content_target,
            _do_search_files_target,
            _SEARCH_TRACKER,
            _SEARCH_TRACKER_LOCK,
        )

        if target == "grep":
            target = "content"
        elif target in ("find", "ls"):
            target = "files"
        if target not in ("content", "files"):
            return f"Error: invalid target '{target}'. Use 'content' or 'files'."
        if output_mode not in ("content", "files_only", "count"):
            return f"Error: invalid output_mode '{output_mode}'."

        try:
            resolved = get_sandboxed_path(path, agent_id)
        except Exception as e:
            return f"Error: {e}"
        if not os.path.exists(resolved):
            return f"Error: Path not found: {path}"

        bucket = agent_id or "_default"
        key = (target, pattern, str(path), file_glob, int(limit), int(offset), output_mode, int(context))
        with _SEARCH_TRACKER_LOCK:
            td = _SEARCH_TRACKER.setdefault(bucket, {"last_key": None, "consecutive": 0})
            if td["last_key"] == key:
                td["consecutive"] += 1
            else:
                td["last_key"] = key
                td["consecutive"] = 1
            consecutive = td["consecutive"]
        if consecutive >= 4:
            return (
                f"BLOCKED: this exact search has run {consecutive} times in a row. "
                "Results have NOT changed. Use the information you already have and proceed."
            )

        # Display paths relative to the sandbox root, not the resolved absolute.
        try:
            sandbox_root = get_sandboxed_path(".", agent_id)
        except Exception:
            sandbox_root = resolved

        if target == "files":
            result = _do_search_files_target(
                pattern=pattern,
                resolved=resolved,
                display_root=sandbox_root,
                limit=limit,
                offset=offset,
            )
        else:
            result = _do_search_content_target(
                pattern=pattern,
                resolved=resolved,
                project_root=sandbox_root,
                file_glob=file_glob,
                limit=limit,
                offset=offset,
                output_mode=output_mode,
                context=context,
                hashline=False,
            )

        if consecutive == 3:
            result += (
                f"\n\n[Warning: this exact search has run {consecutive} times consecutively. "
                "Results have not changed — use what you have instead of re-searching.]"
            )
        return result
