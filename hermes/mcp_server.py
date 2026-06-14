"""A minimal Model Context Protocol (MCP) server exposing hermes memory over stdio.

This implements just enough of the MCP JSON-RPC 2.0 protocol to be useful from an
MCP-aware client (initialize, tools/list, tools/call) and exposes four tools:

    remember(text, tags?, source?, metadata?) -> stored memory
    recall(query, limit?, tag?, min_score?)   -> ranked relevant memories
    forget(id)                                -> bool
    list_memories(limit?, tag?)               -> recent memories

Framing follows the MCP stdio transport: newline-delimited JSON-RPC messages on
stdin/stdout. (We use line framing, which the common stdio clients accept; this keeps
the server pure-stdlib with no transport dependency.) Diagnostics go to stderr so they
never corrupt the protocol stream.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from .memory import MemoryStore

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "cognis-hermes"
SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for inputs)
# ---------------------------------------------------------------------------

TOOLS: list[Dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Store a long-term memory for the agent. Use this to persist facts, "
            "user preferences, decisions, or observations that should survive across "
            "sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory content to store."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels for filtering recall.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional provenance (e.g. 'user', 'tool', 'doc').",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured metadata.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Retrieve the most relevant stored memories for a query, ranked by "
            "TF-IDF cosine similarity with a recency boost. Call this before answering "
            "when prior context might help."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search memory for."},
                "limit": {"type": "integer", "description": "Max results (default 5)."},
                "tag": {"type": "string", "description": "Restrict to this tag."},
                "min_score": {
                    "type": "number",
                    "description": "Drop hits scoring below this value.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a stored memory by its integer id.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "list_memories",
        "description": "List stored memories, most recent first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 20)."},
                "tag": {"type": "string", "description": "Restrict to this tag."},
            },
        },
    },
]


class HermesMCPServer:
    """Dispatches MCP JSON-RPC requests against a MemoryStore."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._initialized = False

    # -- JSON-RPC plumbing -------------------------------------------------

    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one decoded JSON-RPC message; return a response or None (notify)."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications (no id) get no response.
        is_notification = "id" not in message

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "initialized" or method == "notifications/initialized":
                self._initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                if is_notification:
                    return None
                return _error(msg_id, -32601, f"method not found: {method}")
        except _ToolError as exc:
            return _tool_error_result(msg_id, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            if is_notification:
                return None
            return _error(msg_id, -32603, f"internal error: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    # -- tool dispatch -----------------------------------------------------

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise _ToolError(f"unknown tool: {name}")
        payload = handler(args)
        return {
            "content": [
                {"type": "text", "text": json.dumps(payload, default=str, indent=2)}
            ],
            "isError": False,
        }

    def _tool_remember(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _ToolError("'text' is required and must be a non-empty string")
        mem = self.store.remember(
            text,
            tags=args.get("tags") or [],
            source=args.get("source"),
            metadata=args.get("metadata") or {},
        )
        return {"ok": True, "memory": mem.to_dict()}

    def _tool_recall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _ToolError("'query' is required and must be a non-empty string")
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            raise _ToolError("'limit' must be an integer")
        try:
            min_score = float(args.get("min_score", 0.0))
        except (TypeError, ValueError):
            raise _ToolError("'min_score' must be a number")
        hits = self.store.recall(
            query,
            limit=limit,
            tag=args.get("tag"),
            min_score=min_score,
        )
        return {"query": query, "results": [h.to_dict() for h in hits]}

    def _tool_forget(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mem_id = args.get("id")
        if not isinstance(mem_id, int):
            raise _ToolError("'id' is required and must be an integer")
        return {"ok": self.store.forget(mem_id), "id": mem_id}

    def _tool_list_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            raise _ToolError("'limit' must be an integer")
        mems = self.store.list(limit=limit, tag=args.get("tag"))
        return {"memories": [m.to_dict() for m in mems]}


class _ToolError(Exception):
    """Raised when a tool call fails in a way the model should see."""


def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _tool_error_result(msg_id: Any, message: str) -> Dict[str, Any]:
    # MCP convention: tool failures are reported as a successful result with
    # isError=true so the model can read and react to the error text.
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        },
    }


def serve_stdio(store: MemoryStore, stdin=None, stdout=None) -> None:
    """Run the newline-delimited JSON-RPC loop over stdio until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = HermesMCPServer(store)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = _error(None, -32700, f"parse error: {exc}")
            _write(stdout, resp)
            continue
        response = server.handle(message)
        if response is not None:
            _write(stdout, response)


def _write(stdout, obj: Dict[str, Any]) -> None:
    stdout.write(json.dumps(obj) + "\n")
    stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-mcp",
        description="MCP stdio server exposing hermes remember/recall.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("HERMES_DB", "hermes_memory.sqlite"),
        help="Path to the SQLite memory file.",
    )
    args = parser.parse_args(argv)
    try:
        store = MemoryStore(args.db)
    except (OSError, Exception) as exc:
        print(f"[hermes-mcp] error: cannot open database {args.db!r}: {exc}", file=sys.stderr)
        return 1
    print(
        f"[hermes-mcp] serving {SERVER_NAME} v{SERVER_VERSION} over stdio "
        f"(db={args.db})",
        file=sys.stderr,
    )
    try:
        serve_stdio(store)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
