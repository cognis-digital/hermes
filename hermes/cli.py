"""Command-line interface for cognis-hermes.

Examples
--------
    hermes remember "User prefers metric units" --tags preference,units
    hermes recall "what units does the user want" --limit 5
    hermes list --limit 20
    hermes get 3
    hermes forget 3
    hermes stats

The database path resolves to (in order): --db, $HERMES_DB, ./hermes_memory.sqlite.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import List, Optional, Sequence

from .memory import MemoryStore

_DEFAULT_DB = os.environ.get("HERMES_DB", "hermes_memory.sqlite")


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_metadata(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--metadata must be valid JSON: {exc}")
    if not isinstance(obj, dict):
        raise SystemExit("--metadata must be a JSON object")
    return obj


def _fmt_time(ts: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return str(ts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Model-agnostic, portable long-term memory for AI agents.",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"Path to the SQLite memory file (default: {_DEFAULT_DB!r}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rem = sub.add_parser("remember", help="Store a new memory.")
    p_rem.add_argument("text", help="The memory text to store.")
    p_rem.add_argument("--tags", help="Comma-separated tags.")
    p_rem.add_argument("--source", help="Optional provenance label.")
    p_rem.add_argument("--metadata", help="Optional JSON object of extra metadata.")

    p_rec = sub.add_parser("recall", help="Retrieve relevant memories.")
    p_rec.add_argument("query", help="The query to search memory with.")
    p_rec.add_argument("--limit", type=int, default=5, help="Max results (default 5).")
    p_rec.add_argument("--tag", help="Only consider memories with this tag.")
    p_rec.add_argument(
        "--min-score", type=float, default=0.0, help="Drop hits below this score."
    )

    p_list = sub.add_parser("list", help="List stored memories.")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.add_argument("--tag", help="Only list memories with this tag.")
    p_list.add_argument(
        "--oldest-first", action="store_true", help="List oldest first."
    )

    p_get = sub.add_parser("get", help="Fetch a single memory by id.")
    p_get.add_argument("id", type=int)

    p_forget = sub.add_parser("forget", help="Delete a memory by id.")
    p_forget.add_argument("id", type=int)

    sub.add_parser("stats", help="Show store statistics.")

    return parser


def _emit(args: argparse.Namespace, payload) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, default=str))


def _cmd_remember(store: MemoryStore, args: argparse.Namespace) -> int:
    mem = store.remember(
        args.text,
        tags=_parse_tags(args.tags),
        source=args.source,
        metadata=_parse_metadata(args.metadata),
    )
    if args.json:
        _emit(args, mem.to_dict())
    else:
        tagstr = f" [{', '.join(mem.tags)}]" if mem.tags else ""
        print(f"remembered #{mem.id}{tagstr}: {mem.text}")
    return 0


def _cmd_recall(store: MemoryStore, args: argparse.Namespace) -> int:
    hits = store.recall(
        args.query, limit=args.limit, tag=args.tag, min_score=args.min_score
    )
    if args.json:
        _emit(args, [h.to_dict() for h in hits])
        return 0
    if not hits:
        print("(no relevant memories)")
        return 0
    for h in hits:
        m = h.memory
        tagstr = f" [{', '.join(m.tags)}]" if m.tags else ""
        print(f"{h.score:6.3f}  #{m.id}{tagstr}  {m.text}")
    return 0


def _cmd_list(store: MemoryStore, args: argparse.Namespace) -> int:
    mems = store.list(
        limit=args.limit,
        offset=args.offset,
        tag=args.tag,
        newest_first=not args.oldest_first,
    )
    if args.json:
        _emit(args, [m.to_dict() for m in mems])
        return 0
    if not mems:
        print("(no memories stored)")
        return 0
    for m in mems:
        tagstr = f" [{', '.join(m.tags)}]" if m.tags else ""
        print(f"#{m.id}  {_fmt_time(m.created_at)}{tagstr}  {m.text}")
    return 0


def _cmd_get(store: MemoryStore, args: argparse.Namespace) -> int:
    mem = store.get(args.id)
    if mem is None:
        if args.json:
            _emit(args, None)
        else:
            print(f"no memory with id {args.id}", file=sys.stderr)
        return 1
    if args.json:
        _emit(args, mem.to_dict())
    else:
        print(f"#{mem.id}")
        print(f"  text:     {mem.text}")
        print(f"  tags:     {', '.join(mem.tags) or '(none)'}")
        print(f"  source:   {mem.source or '(none)'}")
        print(f"  metadata: {json.dumps(mem.metadata)}")
        print(f"  created:  {_fmt_time(mem.created_at)}")
    return 0


def _cmd_forget(store: MemoryStore, args: argparse.Namespace) -> int:
    ok = store.forget(args.id)
    if args.json:
        _emit(args, {"forgotten": ok, "id": args.id})
    else:
        print(f"forgot #{args.id}" if ok else f"no memory with id {args.id}")
    return 0 if ok else 1


def _cmd_stats(store: MemoryStore, args: argparse.Namespace) -> int:
    s = store.stats()
    if args.json:
        _emit(args, s)
    else:
        for k, v in s.items():
            if k.endswith("created_at") and isinstance(v, (int, float)):
                v = _fmt_time(v)
            print(f"{k:24} {v}")
    return 0


_DISPATCH = {
    "remember": _cmd_remember,
    "recall": _cmd_recall,
    "list": _cmd_list,
    "get": _cmd_get,
    "forget": _cmd_forget,
    "stats": _cmd_stats,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = MemoryStore(args.db)
    try:
        return _DISPATCH[args.command](store, args)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
