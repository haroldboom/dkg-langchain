"""
langchain-dkg demo recording script.

Demonstrates all three components against a live DKG v10 node:
  1. DKGChatMessageHistory  — store conversation turns as Knowledge Assets
  2. DKGMemory              — wrap a LangChain chain with DKG-backed memory
  3. DKGRetriever           — SPARQL queries over the knowledge graph

Usage:
    export DKG_TOKEN=$(dkg auth show)
    export OPENAI_API_KEY=sk-...    # optional — falls back to FakeListChatModel
    python examples/demo_video.py
"""

import asyncio
import os
import time

from langchain_core.messages import HumanMessage, AIMessage
from langchain_dkg import DKGChatMessageHistory, DKGMemory, DKGRetriever, DKGClient

TOKEN = os.environ.get("DKG_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
CONTEXT_GRAPH = "langchain-dkg-demo"


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    w = 62
    print("\n" + "═" * w)
    pad = (w - 2 - len(title)) // 2
    print(" " * pad + " " + title)
    print("═" * w + "\n")


def step(msg: str) -> None:   print(f"  ►  {msg}")
def ok(msg: str) -> None:     print(f"  ✓  {msg}")
def info(msg: str) -> None:   print(f"     {msg}")
def pause(s: float = 1.0) -> None: time.sleep(s)


# ─────────────────────────────────────────────────────────────────
#  Demo 1 — DKGChatMessageHistory
# ─────────────────────────────────────────────────────────────────

async def demo_chat_history(client: DKGClient) -> None:
    banner("DEMO 1  —  DKGChatMessageHistory")
    print("  Store conversation turns as tri-modal Knowledge Assets on the DKG.\n")
    pause()

    history = DKGChatMessageHistory(
        context_graph_id=CONTEXT_GRAPH,
        client=client,
        layer="wm",
        session_uri="urn:session:demo-video-001",
    )

    turns = [
        HumanMessage(content="What is OriginTrail?"),
        AIMessage(content="OriginTrail is a decentralized knowledge graph protocol that enables trusted AI with verifiable data provenance."),
        HumanMessage(content="What is a Knowledge Asset?"),
        AIMessage(content="A Knowledge Asset is an ownable, cryptographically-linked container of structured knowledge, identified by a UAL."),
        HumanMessage(content="What memory layers does DKG v10 support?"),
        AIMessage(content="DKG v10 has three layers: Working Memory (private), Shared Working Memory (gossip-replicated), and Verified Memory (on-chain)."),
    ]

    print(f"  Storing {len(turns)} turns in DKG Working Memory...\n")
    pause(0.5)

    for i, msg in enumerate(turns, 1):
        role_label = "Human" if isinstance(msg, HumanMessage) else "AI   "
        markdown = f"**{role_label.strip()}:** {msg.content}"
        step(f"Turn {i}/{len(turns)}  [{role_label}]  {msg.content[:52]}...")
        result = await client.memory_turn(
            context_graph_id=CONTEXT_GRAPH,
            markdown=markdown,
            session_uri="urn:session:demo-video-001",
            layer="wm",
        )
        ok(f"UAL: {result['turnUri']}")
        info(f"Structural triples: {result['structuralTripleCount']}  "
             f"Semantic: {result['semanticTripleCount']}  "
             f"Embeddings: {result['embeddingId'][:20]}...")
        print()
        pause(0.7)

    pause(0.5)
    step("Semantic search: \"Knowledge Asset\"")
    pause(0.5)
    result = await client.memory_search(
        context_graph_id=CONTEXT_GRAPH,
        query="Knowledge Asset",
        limit=4,
    )
    ok(f"Retrieved {result.get('resultCount', 0)} relevant turns:\n")
    for item in result.get("results", []):
        snippet = item.get("snippet") or item.get("label", "")
        sim = item.get("similarity", 0.0)
        print(f"    [{sim:.2f}]  {snippet[:72]}...")
        print(f"           UAL: {item['entityUri']}")
        print()
        pause(0.3)


# ─────────────────────────────────────────────────────────────────
#  Demo 2 — DKGMemory + LangChain LCEL chain
# ─────────────────────────────────────────────────────────────────

async def demo_memory_chain(client: DKGClient) -> None:
    banner("DEMO 2  —  DKGMemory  +  LangChain LCEL Chain")
    print("  Wrap any Runnable with DKG-backed persistent memory.\n")
    pause()

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    if OPENAI_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_label = "gpt-4o-mini"
    else:
        from langchain_core.language_models.fake import FakeListChatModel
        llm = FakeListChatModel(responses=[
            "DKG v10 has three memory layers: Working Memory (local and private), "
            "Shared Working Memory (gossip-replicated across trusted peers), and "
            "Verified Memory (anchored on-chain — permanent and trustless).",
            "For sensitive private data, use Working Memory (layer='wm'). "
            "It never leaves your local node and is never gossiped or published.",
        ])
        llm_label = "FakeListChatModel (set OPENAI_API_KEY for a real LLM)"

    info(f"LLM             : {llm_label}")
    info(f"Memory backend  : DKG v10 Working Memory  ({CONTEXT_GRAPH})")
    info(f"Session ID      : demo-session-01\n")
    pause(0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to a Decentralized Knowledge Graph."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain_with_memory = DKGMemory.wrap_chain(
        prompt | llm,
        context_graph_id=CONTEXT_GRAPH,
        client=client,
        search_limit=4,
        history_messages_key="history",
    )
    ok("Chain assembled: ChatPromptTemplate | LLM | DKGMemory\n")
    pause(0.5)

    questions = [
        "Summarise the DKG v10 memory layer architecture.",
        "Which layer is best for sensitive private data?",
    ]
    for q in questions:
        print(f"  Human: {q}")
        pause(0.4)
        response = chain_with_memory.invoke(
            {"input": q},
            config={"configurable": {"session_id": "demo-session-01"}},
        )
        answer = response.content if hasattr(response, "content") else str(response)
        print(f"  AI:    {answer}")
        print()
        pause(1.5)

    ok("Every turn automatically persisted to DKG Working Memory as a Knowledge Asset")


# ─────────────────────────────────────────────────────────────────
#  Demo 3 — DKGRetriever (SPARQL)
# ─────────────────────────────────────────────────────────────────

async def demo_retriever(client: DKGClient) -> None:
    banner("DEMO 3  —  DKGRetriever  (SPARQL)")
    print("  Query the Knowledge Graph — results returned as LangChain Documents.\n")
    pause()

    retriever = DKGRetriever(client=client, limit=6, include_workspace=True)

    for query in ["OriginTrail", "Knowledge Asset"]:
        step(f"Query: \"{query}\"")
        pause(0.4)
        docs = await retriever._aget_relevant_documents(query)
        ok(f"{len(docs)} triples retrieved as LangChain Documents\n")
        for doc in docs[:3]:
            print(f"    {doc.page_content[:88]}")
            print(f"    metadata → layer={doc.metadata['layer']}  source={doc.metadata['source']}")
            print()
        pause(1.0)


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

async def main() -> None:
    w = 62
    print("\n╔" + "═" * w + "╗")
    print("║  langchain-dkg  —  LangChain × OriginTrail DKG v10        ║")
    print("║  Bounty tag: cfi-dkgv10-r1   |   pip install langchain-dkg║")
    print("╚" + "═" * w + "╝\n")

    if not TOKEN:
        print("ERROR: set DKG_TOKEN first.")
        print("  Run:  export DKG_TOKEN=$(dkg auth show)")
        return

    step("Connecting to DKG v10 node at http://localhost:9200 ...")
    client = DKGClient(token=TOKEN)
    if not await client.ping():
        print("\nERROR: DKG node not reachable.")
        print("  Run:  dkg start && sleep 20")
        return
    ok("Connected to DKG v10 node\n")
    pause(1.0)

    await demo_chat_history(client)
    pause(2.0)
    await demo_memory_chain(client)
    pause(2.0)
    await demo_retriever(client)

    banner("DEMO COMPLETE")
    print("  All conversation turns stored as Knowledge Assets on the DKG.")
    print(f"  Context Graph: {CONTEXT_GRAPH}\n")
    print("  Install :  pip install langchain-dkg")
    print("  GitHub  :  https://github.com/Mungles/dkg-langchain")
    print("  Bounty  :  cfi-dkgv10-r1")
    print()


if __name__ == "__main__":
    asyncio.run(main())
