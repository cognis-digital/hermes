# cognis-hermes

**A model-agnostic, portable long-term MEMORY framework for AI agents.**

`cognis-hermes` gives any AI agent — regardless of which LLM or runtime drives it — a
durable place to *remember* facts, observations, and decisions, and to *recall* the most
relevant ones later. It is deliberately small, dependency-free (Python standard library
only), and embeddable: a single SQLite file holds everything, and recall uses a
self-contained TF-IDF + cosine-similarity ranker, so there is no embedding API to call,
no vector database to run, and no network access required.

```
remember(text)  ->  persisted memory with metadata
recall(query)   ->  ranked list of the most relevant memories
```

## Why

Most agent "memory" implementations are bolted to a particular model provider, a hosted
embedding endpoint, or a heavyweight vector store. That couples your agent's long-term
memory to infrastructure you may not want and to a vendor you may not keep. `hermes`
takes the opposite stance:

- **Model-agnostic.** No LLM is required to store or retrieve. You can wire it behind
  GPT, Claude, Llama, a local fleet, or a rules engine — hermes does not care.
- **Portable.** The entire memory is one `.sqlite` file you can copy, version, diff, or
  ship between machines. Move it from a laptop to a server and your agent keeps its mind.
- **Stdlib-only.** `sqlite3`, `math`, `re`, `json`, `argparse`. Nothing to `pip install`
  beyond Python itself. Installs and runs anywhere Python runs.
- **Inspectable.** Memories are plain rows. The ranking is plain math. You can read,
  audit, and reason about exactly why a memory was recalled.

## Prior art and credit

This is a clean-room, standard-library reimplementation of the **Hermes memory pattern** —
the broadly-known approach of giving agents a model-agnostic, persistent store with
lexical/TF-IDF recall rather than provider-locked embeddings. We gratefully acknowledge
the wider ecosystem of agent-memory work that inspired this design: persistent
conversation/observation stores, retrieval-augmented memory, and the
`remember`/`recall` tool interface that many agent frameworks have converged on. The
classic TF-IDF and cosine-similarity techniques used here are long-established
information-retrieval methods. This package re-expresses those ideas in a minimal,
portable form for Cognis Digital agents; any resemblance to prior implementations is at
the level of shared, well-known patterns.

## Install

```bash
pip install -e .
# or, since it is stdlib-only, just run it in place:
python -m hermes.cli --help
```

Requires Python 3.9+.

## Quick start (Python)

```python
from hermes import MemoryStore

mem = MemoryStore("agent_memory.sqlite")

mem.remember(
    "The user prefers metric units and concise answers.",
    tags=["preference", "units"],
    source="onboarding",
)
mem.remember("Deployed the trading bot to paper mode on 2026-06-08.", tags=["ops"])

for hit in mem.recall("what units does the user want", limit=3):
    print(f"{hit.score:.3f}  {hit.memory.text}")

mem.close()
```

## Quick start (CLI)

```bash
# store a memory
python -m hermes.cli remember "User's favorite ticker is GEV" --tags watchlist,equities

# retrieve relevant memories
python -m hermes.cli recall "which stock does the user like" --limit 5

# list / inspect / forget
python -m hermes.cli list --limit 20
python -m hermes.cli get 3
python -m hermes.cli forget 3
python -m hermes.cli stats
```

The database path defaults to `$HERMES_DB` or `./hermes_memory.sqlite`. Override with
`--db /path/to/file.sqlite`.

## MCP server

Hermes ships an [MCP](https://modelcontextprotocol.io) server over stdio that exposes
the memory to any MCP-aware client (Claude Code, IDE agents, etc.) using the standard
JSON-RPC framing. It exposes four tools: `remember`, `recall`, `forget`, and
`list_memories`.

```bash
python -m hermes.mcp_server --db agent_memory.sqlite
```

Example client config (e.g. an MCP `mcpServers` block):

```json
{
  "mcpServers": {
    "hermes": {
      "command": "python",
      "args": ["-m", "hermes.mcp_server", "--db", "/data/agent_memory.sqlite"]
    }
  }
}
```

## How recall works

1. Every stored memory is tokenized (lowercased, alphanumeric word splitting, optional
   stopword removal).
2. Document frequencies are maintained so that TF-IDF weights can be computed.
3. At query time the query is tokenized the same way, a TF-IDF vector is built for it,
   and memories are ranked by cosine similarity against that vector.
4. Recency acts as a gentle tie-breaker/booster (configurable), so all else equal a more
   recent memory ranks slightly higher — useful for agents whose world changes over time.

This is intentionally lexical. It is fast, explainable, and needs no model. If you later
want semantic recall, the `MemoryStore` API is the same shape you would wrap around an
embedding backend.

## Layout

```
hermes/
  __init__.py      public API surface (MemoryStore, Memory, RecallHit)
  memory.py        SQLite-backed store + TF-IDF / cosine recall
  cli.py           argparse command-line interface
  mcp_server.py    stdio JSON-RPC MCP server exposing remember/recall
pyproject.toml     packaging (name: cognis-hermes)
tests/             stdlib unittest suite
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

MIT. See `LICENSE`.

---

Built by **Cognis Digital LLC**. Part of the Cognis agent tooling family.
