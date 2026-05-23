# langchain-dkg Design Brief

**Package:** `langchain-dkg`  
**Bounty tag:** `cfi-dkgv10-r1`  
**Tier target:** Flagship (8,000–10,000 TRAC)

---

## 1. Problem

LangChain is the most widely-used AI agent framework (millions of users, thousands of production deployments). Every LangChain agent is stateless by default: memory disappears when the process ends, there is no provenance for what the agent knew, and knowledge cannot be shared between agents without custom plumbing.

Existing memory backends (Redis, SQLite, in-RAM buffers) solve persistence but not verifiability. There is no way to:
- Prove what an agent knew at a specific time
- Share knowledge across agents with attribution
- Promote important insights to a permanent, on-chain record

DKG v10 solves all three. This package connects LangChain to DKG v10.

---

## 2. Target users

- **LangChain application developers** building conversational agents, RAG pipelines, research assistants, or workflow automation — who want durable, queryable, attributable memory without building DKG integration from scratch.
- **Multi-agent system builders** who need agents to share a common knowledge base (Shared Working Memory) with consistent provenance.
- **Teams needing audit trails** for AI-assisted decisions, where Verified Memory (on-chain) provides permanent, tamper-evident records.

---

## 3. Architecture

```
LangChain Application
        │
        ├─ DKGChatMessageHistory ──► POST /api/memory/turn    (store as markdown KA)
        │                       ◄── POST /api/memory/search   (tri-modal retrieval)
        │
        ├─ DKGMemory ──────────────► RunnableWithMessageHistory (modern LCEL wrapper)
        │                            (session scoped via sessionUri)
        │
        └─ DKGRetriever ───────────► POST /api/query (SPARQL SELECT)
                                     returns triples as LangChain Document objects

DKG v10 Node (localhost:9200)
        │
        ├─ Working Memory (WM)         ← private, free, instant
        ├─ Shared Working Memory (SWM) ← gossip-replicated, free
        └─ Verified Memory (VM)        ← on-chain, TRAC, permanent
```

### API surface used

All communication is over the public HTTP API — no internal DKG packages are imported.

| Endpoint | Purpose |
|---|---|
| `POST /api/memory/turn` | Store a conversation turn (markdown → Knowledge Asset) |
| `POST /api/memory/search` | Tri-modal search: vector + SPARQL + text |
| `POST /api/query` | SPARQL SELECT for DKGRetriever |
| `GET /api/agents` | Health check / auth validation |
| `POST /api/assertion/:name/promote` | Explicit SHARE to Shared Working Memory |

### Markdown encoding of turns

Each conversation turn is encoded as markdown with a role prefix:

```
**Human:** What is a Knowledge Asset?
**AI:** An ownable, verifiable container of structured knowledge on the DKG.
```

The DKG node runs structural + optional semantic extraction on this markdown, builds RDF triples, and stores them as a Knowledge Asset in the target graph. The `turnUri` is returned and cached for future promotion.

---

## 4. Memory layer mapping (LLM-Wiki alignment)

Following Karpathy's LLM-Wiki framing:

| LLM-Wiki concept | DKG v10 concept | This package |
|---|---|---|
| In-context window | — | Not stored |
| External memory | Working Memory (WM) | Default `layer="wm"` |
| Team memory | Shared Working Memory (SWM) | Default layer; gossip-replicated |
| Long-term knowledge | Verified Memory (VM) | Via `promote_to_shared()` + PUBLISH |

The trust gradient is explicit: every Knowledge Asset carries a `turnUri` (UAL) that identifies which layer it lives in. Promotion is always agent-initiated, never automatic.

---

## 5. Trust gradient and promotion path

### Working Memory → Shared Working Memory (SHARE)
```python
turn_uri = history.get_turn_uri("**Human:** Key decision made")
await history.promote_to_shared(turn_uri)
```
This calls `POST /api/assertion/:name/promote` — a Curator-authorized operation. The agent must explicitly decide to share; nothing is shared automatically.

### Shared Working Memory → Verified Memory (PUBLISH)
Round 2 / future: call `POST /api/shared-memory/publish` with the context graph ID. This costs TRAC and writes to the paranet. All UALs are preserved through the promotion chain so on-chain provenance traces back to the original turn.

---

## 6. DKGRetriever as a RAG upstream

`DKGRetriever` implements `BaseRetriever`, making it a drop-in for any LangChain RAG pipeline:

```python
retriever = DKGRetriever(limit=20, include_workspace=True)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

Each SPARQL result triple `(subject, predicate, object)` becomes a `Document` with metadata including `subject`, `predicate`, `object`, and `source: "dkg-v10"`. Custom SPARQL templates are supported for domain-specific queries.

This addresses the bounty's explicit target: **"RAG pipelines that want a verifiable upstream."**

---

## 7. Security model

- Credentials are read from `DKG_TOKEN` env var or passed explicitly to `DKGClient` — never hardcoded.
- No Curator operations (SHARE/PUBLISH) are performed automatically. All promotion is explicit and agent-initiated.
- No write access to Verified Memory without a separate, intentional PUBLISH call.
- The package has no postinstall scripts.
- Network egress is limited to the DKG node HTTP API (default: localhost:9200).

---

## 8. v10 vocabulary compliance

All code and documentation uses the exact DKG v10 terminology:

- **Context Graph** (not "workspace" or "namespace")
- **Knowledge Asset** (not "record" or "document")
- **Working Memory** / **Shared Working Memory** / **Verified Memory** (not "private/public/chain")
- **SHARE** / **PUBLISH** (not "sync" or "propagate")
- **Curator** (for SHARE/PUBLISH authorization)

---

## 9. Maintenance commitment

Six-month support window from submission date. Issues and PRs will be reviewed within 5 business days. The package follows semantic versioning; breaking changes will be released as major versions with migration notes.

---

## 10. Positioning vs other Round 1 submissions

The Round 1 queue covers several distinct integration shapes. This submission is intentionally complementary, not competing:

- **Source-side ingestion submissions** (`dkg-arxiv` for papers, `github-dkg` for engineering tacit knowledge, `tracabot` for Telegram moderation, `polymarket-analysis` for markets) populate DKG with upstream material. `langchain-dkg` is the **read-side** counterpart: any LangChain agent built on top can immediately consume that material as memory, retrieval, or RAG context. The two halves close a write/read loop.
- **Agent-plugin submissions** (`openclaw-working-memory`, `dkg-wm-bridge`, `aipharmagent`) bind WM/SWM access to specific agent frameworks (OpenClaw, Hermes, clinical workflows). `langchain-dkg` targets the broader LangChain ecosystem — the dominant production framework for Python LLM apps — which is currently unaddressed in the queue.
- **Platform-level governance submissions** (`agience-flare`, `repnet`) govern which artifacts reach DKG and how trust is assigned to them. `langchain-dkg` operates one layer below: assuming such governance is in place, it provides the standard adapter that lets a LangChain agent author and retrieve through the governed pipeline.
- **First-party `cursor-mcp-dkg`** serves MCP-protocol clients (Cursor, Claude Code, Claude Desktop). `langchain-dkg` serves the orthogonal Python `BaseChatMessageHistory` / `BaseRetriever` interfaces — not MCP-based, no client overlap.

This entry is the only LangChain-framework adapter in the queue. Its value is amplification: every other source-side, agent-plugin, or governance integration in this round becomes consumable by any LangChain agent the moment `langchain-dkg` lands.
