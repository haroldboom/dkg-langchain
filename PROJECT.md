# dkg-langchain — DKG v10 LangChain Adapter

## What We're Building
A Python package that integrates OriginTrail DKG v10 with LangChain — giving any LangChain agent persistent, verifiable memory backed by the Decentralized Knowledge Graph.

**Three components:**
1. `DKGChatMessageHistory` — `BaseChatMessageHistory` that stores conversation turns in DKG Working Memory via `/api/memory/turn`, retrieves via `/api/memory/search`
2. `DKGMemory` — `BaseMemory` wrapper (drop-in replacement for `ConversationBufferMemory`)
3. `DKGRetriever` — `BaseRetriever` for RAG pipelines using DKG SPARQL queries (`/api/query`)

## Why This Wins the Bounty
- "RAG pipelines that want a verifiable upstream" — explicitly called out as a priority target in Section 5
- "DKG as a memory backend for an existing RAG pipeline" — explicitly in Section 4's build list
- Highest adoption potential of any missing integration (LangChain has millions of users)
- NOT covered by any first-party adapter (OpenClaw, ElizaOS, Hermes, autoresearch, mcp-dkg are all built in)
- Uses ONLY public HTTP API (in-scope per Section 5)
- Maps directly to LLM-Wiki/autoresearch direction: every agent gets durable, queryable, attributable memory
- Flagship tier candidate: 8,000–10,000 TRAC

## Why We Pivoted from MCP Server
The `packages/mcp-dkg` first-party package already ships an MCP server for Claude Code/Cursor.
Community integrations must connect EXTERNAL tools TO the DKG node — not duplicate built-in capabilities.

## Bounty Program Facts
- Program: OriginTrail DKG v10 Integrations Bounty — Round 1
- Round 1 pool: 50,000 TRAC
- Flagship tier: 8,000–10,000 TRAC | High-quality: 3,000–7,000 | Experimental: 1,000–3,000
- Submission: PR to OriginTrail/dkg-integrations + design brief + demo + tests
- Tag: cfi-dkgv10-r1
- Bounty doc: https://docs.origintrail.io/origintrail-v9-v10/origintrail-dkg-v10-bounty-program
- Integrations registry: https://github.com/OriginTrail/dkg-integrations
- DKG v10 node: https://github.com/OriginTrail/dkg-v9 (dkg-v9 repo = v10 codebase)

## DKG v10 HTTP API (the only interface we use)
Base URL: http://localhost:9200 (or configured host)
Auth: `Authorization: Bearer <token>` (token from `dkg auth show`)

### Key Endpoints

**Conversation Memory:**
- `POST /api/memory/turn` — ingest a conversation turn as a tri-modal Knowledge Asset
  - body: `{ contextGraphId, role, content, agentUri?, subGraphName? }`
  - returns: `{ assertionId, ual }`
- `POST /api/memory/search` — tri-modal search (text + graph + vector)
  - body: `{ contextGraphId, query, limit?, layer? }`
  - returns: `{ results: [{ content, ual, score }] }`

**Assertions (Working Memory):**
- `POST /api/assertion/create` — create an assertion
  - body: `{ contextGraphId, name, subGraphName? }`
- `POST /api/assertion/:name/write` — write RDF quads
  - body: `{ contextGraphId, quads, subGraphName? }`
- `POST /api/assertion/:name/promote` — WM -> Shared Memory (SHARE)
  - body: `{ contextGraphId, entities?, subGraphName? }`
- `GET /api/assertion/:name/history?contextGraphId=...` — audit history

**Shared Memory:**
- `POST /api/shared-memory/write` — write triples directly to SWM
- `POST /api/shared-memory/publish` — SWM -> Verified Memory (PUBLISH, costs TRAC)

**Querying:**
- `POST /api/query` — SPARQL query
  - body: `{ sparql, paranetId?, graphSuffix?, includeWorkspace? }`

**Context Graphs:**
- `POST /api/context-graph/create`
- `POST /api/context-graph/register` (on-chain)
- `GET /api/sub-graph/list?contextGraphId=...`

**Status:**
- `GET /api/agents` — list agents (also used for auth check)

## Architecture

```
LangChain Agent
     |
     |-- DKGChatMessageHistory --> POST /api/memory/turn  (store)
     |                        <-- POST /api/memory/search (retrieve)
     |
     |-- DKGMemory -------------- wraps DKGChatMessageHistory
     |                            (drop-in for ConversationBufferMemory)
     |
     +-- DKGRetriever ----------> POST /api/query (SPARQL)
                                  (RAG: retrieves KAs as Documents)

DKG v10 Node (localhost:9200)
     |
     |-- Working Memory (private, free)
     |-- Shared Memory (gossip-replicated, free)
     +-- Verified Memory (on-chain, costs TRAC)
```

## v10 Terminology (must use exactly — Section 7 of bounty)
- **Context Graph** — scoped knowledge domain (NOT "workspace")
- **Knowledge Asset (KA)** — ownable container of knowledge
- **Knowledge Collection** — group of related KAs
- **Working Memory (WM)** — private, free, local
- **Shared Working Memory (SWM)** — team-visible, gossip-replicated
- **Verified Memory (VM)** — permanent, on-chain
- **SHARE** — promote to SWM (Curator-authorized)
- **PUBLISH** — promote to VM (Curator-authorized, on-chain)
- **Curator** — authority for SHARE/PUBLISH operations
- **Projects** — the v10 dashboard view

## Design Principles (non-negotiable per Section 7)
1. Agent-first — DKG is the memory backend, agent decides what to store
2. Trust gradient — every message tagged with layer (WM/SWM/VM)
3. No UI buttons for endorsement — conversational only
4. Project-centric — layers nest inside a Context Graph
5. No merge conflicts — Shared Memory is gossiped
6. Exact v10 vocabulary throughout

## Forward Compatibility (Verified Memory / Round 2)
- Every stored turn includes `promotion_status`: `draft` | `shared` | `verified`
- `DKGChatMessageHistory.promote_to_shared(ual)` — explicit promotion method
- UAL references preserved on all objects for future Verified Memory operations
- Design brief documents promotion path: WM -> SWM -> VM

## Tech Stack
- **Language**: Python 3.10+
- **Dependencies**: `langchain-core`, `httpx` (async HTTP)
- **Test**: `pytest`, `pytest-asyncio`, `respx` (mock httpx)
- **Build**: `pyproject.toml` with hatchling
- **Publish**: PyPI with trusted publishing (GitHub Actions, PEP 740)
- **License**: MIT

## Package Structure
```
dkg-langchain/
|-- src/
|   +-- langchain_dkg/
|       |-- __init__.py         (exports)
|       |-- client.py           (DKG HTTP API client, httpx)
|       |-- chat_history.py     (DKGChatMessageHistory)
|       |-- memory.py           (DKGMemory)
|       +-- retriever.py        (DKGRetriever)
|-- tests/
|   |-- unit/
|   |   |-- test_client.py
|   |   |-- test_chat_history.py
|   |   +-- test_retriever.py
|   +-- integration/
|       +-- test_integration.py (requires local DKG node)
|-- examples/
|   +-- research_agent.py       (demo: autoresearch agent with DKG memory)
|-- pyproject.toml
|-- README.md
|-- LICENSE
+-- DESIGN_BRIEF.md
```

## Integration Registry Entry (for dkg-integrations PR)
```json
{
  "slug": "langchain-dkg",
  "name": "LangChain DKG Memory Adapter",
  "description": "LangChain BaseChatMessageHistory and BaseRetriever backed by DKG v10 Working and Shared Memory. Gives any LangChain agent persistent, verifiable, queryable memory with provenance.",
  "type": "service",
  "tier": "community",
  "license": "MIT",
  "network_egress": ["localhost (DKG node HTTP API, port 9200 by default)"],
  "write_authority": ["Working Memory (assertion create/write)", "Shared Memory (SHARE — Curator-authorized, opt-in only)"],
  "repository": "https://github.com/<user>/dkg-langchain",
  "pypi_package": "langchain-dkg",
  "installation": "manual"
}
```

## Submission Checklist
- [ ] PR against OriginTrail/dkg-integrations with integration JSON entry
- [ ] DESIGN_BRIEF.md (1-3 pages): problem, user, memory layers, primitives, LLM-Wiki mapping, promotion path, oracle-readiness
- [ ] Working demo — recorded walkthrough of LangChain agent using DKG memory
- [ ] Unit tests (respx mocks) + integration tests (local DKG node)
- [ ] Security notes: credentials via env var, no Curator ops without user opt-in
- [ ] Maintenance commitment: 6-month support window
- [ ] PyPI publish with trusted publishing (GitHub Actions)
- [ ] SPDX license in pyproject.toml
- [ ] No postinstall scripts
- [ ] Contributor attestation on PR

## Key URLs
- Bounty doc: https://docs.origintrail.io/origintrail-v9-v10/origintrail-dkg-v10-bounty-program
- DKG v10 node repo: https://github.com/OriginTrail/dkg-v9
- Integrations registry: https://github.com/OriginTrail/dkg-integrations
- Hello world reference: https://github.com/OriginTrail/dkg-hello-world
- LLM-Wiki (Karpathy): https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Autoresearch (Karpathy): https://github.com/karpathy/autoresearch

## Status
- [x] Project folder created (/root/dkg-langchain)
- [x] PROJECT.md written (this file)
- [x] DKG v10 node cloned (/root/dkg-v10-node)
- [ ] DKG CLI installed and node started
- [ ] Python package scaffolded
- [ ] DKG HTTP client implemented
- [ ] LangChain classes implemented
- [ ] Tests written
- [ ] Demo script written
- [ ] Design brief written
- [ ] PyPI published
- [ ] GitHub Actions CI set up
- [ ] PR opened on dkg-integrations
