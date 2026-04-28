# langchain-dkg

LangChain memory and retriever backed by [OriginTrail DKG v10](https://origintrail.io) Working Memory.

Give any LangChain agent persistent, verifiable, queryable memory — every conversation turn stored as a cryptographically-linked Knowledge Asset on the Decentralized Knowledge Graph.

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
| `DKGClient` | — | Low-level async HTTP client for the DKG v10 API |

## Memory layers

DKG v10 has three memory layers:

| Layer | Scope | Cost | Use |
|---|---|---|---|
| Working Memory (`wm`) | Private to your node | Free | Default for conversation history |
| Shared Working Memory (`swm`) | Gossip-replicated | Free | Team-visible context |
| Verified Memory | On-chain, permanent | TRAC | Auditable, publishable knowledge |

By default, turns are written to Shared Working Memory (`swm`). Use `layer="wm"` for private-only storage.

Explicit promotion to Shared Memory:

```python
turn_uri = history.get_turn_uri("**Human:** Summarize this meeting")
await history.promote_to_shared(turn_uri)
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DKG_TOKEN` | — | Bearer token from `dkg auth show` |
| `DKG_BASE_URL` | `http://localhost:9200` | DKG node API URL |

Or pass `token=` / `base_url=` directly to `DKGClient`.

## Session isolation

Each `session_id` passed to `chain_with_memory.invoke(config={"configurable": {"session_id": "..."}})` becomes a `sessionUri` in DKG, linking turns together within the shared Context Graph.

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/                                    # unit tests (no node required)
DKG_TOKEN=$(dkg auth show) pytest tests/integration/  # integration tests
python examples/research_agent.py                     # demo script
```

## License

MIT
