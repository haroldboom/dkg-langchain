# langchain-dkg

[![CI](https://github.com/haroldboom/dkg-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/haroldboom/dkg-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-dkg.svg)](https://pypi.org/project/langchain-dkg/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-dkg.svg)](https://pypi.org/project/langchain-dkg/)
[![License](https://img.shields.io/pypi/l/langchain-dkg.svg)](https://github.com/haroldboom/dkg-langchain/blob/master/LICENSE)

LangChain memory and retriever backed by [OriginTrail DKG v10](https://origintrail.io) Working Memory.

Give any LangChain agent persistent, verifiable, queryable memory — every conversation turn stored as a cryptographically-linked Knowledge Asset on the Decentralized Knowledge Graph.

## Demo

- **Walkthrough video:** [youtu.be/7VEjDqflEpc](https://youtu.be/7VEjDqflEpc) — narrated run of all three components against a live DKG v10 node.
- **Recording script:** [`examples/demo_video.py`](examples/demo_video.py) — the script behind the video.

## Install

```bash
pip install langchain-dkg
```

Requires a running DKG v10 node. Install with:

```bash
npm install -g @origintrail-official/dkg
dkg init && dkg start
export DKG_TOKEN=$(dkg auth show)
```

## Quick start

```python
from langchain_dkg import DKGChatMessageHistory, DKGMemory, DKGRetriever
from langchain_core.messages import HumanMessage, AIMessage

# Store and retrieve conversation turns
history = DKGChatMessageHistory(context_graph_id="my-project")
history.add_message(HumanMessage(content="What is a Knowledge Asset?"))
history.add_message(AIMessage(content="An ownable container of structured knowledge on the DKG."))

messages = history.messages  # tri-modal semantic search
```

### With a LangChain chain (modern LCEL style)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_dkg import DKGMemory

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain_with_memory = DKGMemory.wrap_chain(
    prompt | llm,
    context_graph_id="my-project",
)

response = chain_with_memory.invoke(
    {"input": "What is DKG?"},
    config={"configurable": {"session_id": "user-42"}},
)
```

### RAG retrieval via SPARQL

```python
from langchain_dkg import DKGRetriever
from langchain.chains import RetrievalQA

retriever = DKGRetriever(limit=10)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

## Components

| Class | LangChain base | Purpose |
|---|---|---|
| `DKGChatMessageHistory` | `BaseChatMessageHistory` | Stores turns in DKG WM; retrieves via tri-modal search |
| `DKGMemory` | — | Factory for `RunnableWithMessageHistory` with DKG backend |
| `DKGRetriever` | `BaseRetriever` | SPARQL retriever — returns triples as `Document` objects |
| `DKGVerifiedRetriever` | `BaseRetriever` | SPARQL retriever over Verifiable Memory with a trust floor |
| `DKGClient` | — | Low-level async HTTP client for the DKG v10 API |

## Memory layers

DKG v10 has three memory layers:

| Layer | Scope | Cost | Use |
|---|---|---|---|
| Working Memory (`wm`) | Private to your node | Free | Default for conversation history |
| Shared Working Memory (`swm`) | Gossip-replicated | Free | Team-visible context |
| Verified Memory | On-chain, permanent | TRAC | Auditable, publishable knowledge |

> **Breaking behavior change (0.1.9):** turns are now written to private
> Working Memory (`layer="wm"`) by default. Earlier versions deferred to the
> node's default, which is Shared Working Memory (`swm`, gossiped to peers).
> Pass `layer="swm"` to keep the old gossiped behavior, or `layer=None` to use
> the node's default.

Explicit promotion to Shared Memory:

```python
turn_uri = history.get_turn_uri("**Human:** Summarize this meeting")
await history.promote_to_shared(turn_uri)
```

Promotion is asynchronous on current node builds: `DKGClient.assertion_promote`
submits a job via `POST /api/assertion/{name}/promote-async`, then polls
`GET /api/assertion/promote-async/{jobId}` (about once per second, up to
`poll_timeout=30.0` seconds) and returns the final job view. A failed job
raises `CuratorUnconfirmedError` / `CuratorRejectedError` when the curator did
not confirm or rejected the write, and `DKGError` otherwise (including poll
timeouts). Note: current node builds expose promotion for named Working Memory
assertions; promoting memory *turns* by URI may not be supported.

## Retrieval options

History retrieval is **semantic-relevance based** — `history.messages` runs a
tri-modal search seeded by `search_query`, not a chronological dump.

- `search_query` (`DKGChatMessageHistory`, `DKGMemory`): seed query used to
  retrieve relevant past turns. Defaults to `"conversation history"`; set it
  to the session's topic for sharper retrieval.
- `search_layers` (`DKGChatMessageHistory`): memory layers searched, default
  `["wm", "swm"]`. Current node builds return nothing when `memoryLayers` is
  omitted, so the layers are always sent explicitly.
- `context_graph_id` (`DKGRetriever`): scopes SPARQL queries to a Context
  Graph — required by current node builds to see workspace (Working Memory)
  data.

## Verifiable Memory (preview)

The full promotion chain now runs end to end: draft in Working Memory →
share to Shared Working Memory → publish to Verifiable Memory on-chain →
endorse → M-of-N verify. Published content carries a trust level
(`TrustLevel`: `SELF_ATTESTED` → `ENDORSED` → `PARTIALLY_VERIFIED` →
`CONSENSUS_VERIFIED`) that queries can filter on.

Promote a quad set in one call:

```python
from langchain_dkg import DKGClient, publish_to_verified, turn_to_quads

client = DKGClient()
quads = turn_to_quads("urn:turn:1", "**AI:** Yield was 4.2 t/ha.")  # minimal default shape
receipt = await publish_to_verified(client, "my-project", "harvest-2026", quads)
# receipt: kaId, ual, txHash, ... merged with the EIP-712 seal (eip712Digest, merkleRoot, ...)
```

This orchestrates `assertion_create` → `ka_write` → `ka_finalize` (the
off-chain EIP-712 seal) → the async SWM share → `vm_publish`. Raise the trust
level afterwards with `client.endorse(...)` (stamps *Endorsed*) and
`client.request_verification(...)` (M-of-N verifier quorum → *ConsensusVerified*;
a partial quorum is returned, not raised, so you can poll). Verify content
locally against a published merkle root with `client.verify_batch(...)`, and
fetch chain-side provenance via `client.kc_metadata(...)` / `client.kc_author(...)`.

Retrieve only trusted knowledge with `DKGVerifiedRetriever` — like
`DKGRetriever`, but scoped to the `"verifiable-memory"` query view with a
trust floor:

```python
from langchain_dkg import DKGVerifiedRetriever, TrustLevel

retriever = DKGVerifiedRetriever(context_graph_id="my-project", min_trust=TrustLevel.ENDORSED)
docs = retriever.invoke("wheat prices")  # metadata: subject/predicate/object, source="dkg-v10-vm"
```

Hand the surface to an agent with `make_dkg_tools` (only needs
`langchain_core`):

```python
from langchain_dkg import DKGClient, make_dkg_tools

tools = make_dkg_tools(DKGClient(), "my-project")
# dkg_endorse(ual), dkg_verified_search(query, min_trust="endorsed"),
# dkg_publish_note(title, content)  — docstrings double as tool descriptions
```

> **Preview status:** this surface targets bounty **Round 2** and was built
> against node build `10.0.2`. Publishing and endorsing hit the chain — the
> node needs a **funded, publish-authorized wallet** (an unfunded wallet fails
> with HTTP 400 and per-wallet balances in the body; a publish whose
> preconditions aren't met — e.g. the SWM share hasn't landed yet — raises
> `DKGPublishPreconditionError`).

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DKG_TOKEN` | — | Bearer token from `dkg auth show` |
| `DKG_BASE_URL` | `http://localhost:9200` | DKG node API URL |

Or pass `token=` / `base_url=` directly to `DKGClient`.

## Session isolation

Each `session_id` passed to `chain_with_memory.invoke(config={"configurable": {"session_id": "..."}})` becomes a `sessionUri` in DKG, linking turns together within the shared Context Graph.

When `session_uri` is set on `DKGChatMessageHistory`, retrieval applies a
client-side session filter: the node's search API has no session parameter, so
the history over-fetches, looks up the session's turns via SPARQL
(`<session_uri> <http://schema.org/hasPart> ?turn`), keeps only those turns
and sorts them chronologically. If the lookup fails or the session has no
linked turns, unfiltered search results are returned.

## Node compatibility

DKG v10 nodes auto-update, and pre-release API surfaces drift (e.g. the
synchronous promote route was replaced by `promote-async`, `memory/search`
began requiring `memoryLayers`, and `/api/query` changed its result shape).
This version was verified against node build `10.0.2` (July 2026). If you see
unexpected 404s/400s or empty results after a node update, check for a newer
`langchain-dkg` release.

Known limitations of node `10.0.2`:

- **`/api/query` cannot see Working Memory (`wm`) quads.** The RFC-29
  working-memory isolation gate is fail-closed and the signature plumbing is
  not shipped in this build, so SPARQL queries (including `DKGRetriever`)
  only cover Shared Working Memory and published data —
  `include_workspace=True` maps to the node's `includeSharedMemory`. Use
  `history.messages` / `memory_search` (which does return `wm` results) to
  retrieve private turns, or write turns with `layer="swm"` when they must be
  SPARQL-queryable.
- The legacy `/api/assertion/create` and promote routes moved to the
  `/api/knowledge-assets` surface; this client targets the new routes and
  falls back to the legacy ones on 404. `assertion_write` /
  `assertion_history` / `shared_memory_publish` still target legacy routes
  that 404 on `10.0.2`; use the new equivalents instead — `ka_write`
  (`/api/knowledge-assets/{name}/wm/write`) and the Verifiable Memory publish
  surface (`vm_publish` / `publish_direct`, see above).

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/                                    # unit tests (no node required)
DKG_TOKEN=$(dkg auth show) pytest tests/integration/  # integration tests
python examples/research_agent.py                     # demo script
```

## License

MIT
