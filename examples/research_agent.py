"""Demo: LangChain research agent with DKG v10 Working Memory.

This script shows a minimal agent that:
  1. Uses DKGMemory to persist conversation history to DKG Working Memory
  2. Uses DKGRetriever to pull facts from the knowledge graph
  3. Answers questions with full provenance

Run:
    export DKG_TOKEN=$(dkg auth show)
    export OPENAI_API_KEY=<your key>        # or use any LangChain-compatible LLM
    python examples/research_agent.py
"""

import asyncio
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_dkg import DKGChatMessageHistory, DKGMemory, DKGRetriever, DKGClient


CONTEXT_GRAPH_ID = "research-demo"
TOKEN = os.environ.get("DKG_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


async def run_demo():
    if not TOKEN:
        print("Set DKG_TOKEN environment variable first (run: export DKG_TOKEN=$(dkg auth show))")
        return

    client = DKGClient(token=TOKEN)

    print("Checking DKG node connection...")
    if not await client.ping():
        print("ERROR: DKG node not reachable at http://localhost:9200")
        return
    print("Connected.\n")

    # --- Demo 1: Store and retrieve conversation turns ---
    print("=== Demo 1: DKGChatMessageHistory ===")
    history = DKGChatMessageHistory(
        context_graph_id=CONTEXT_GRAPH_ID,
        client=client,
        search_limit=5,
    )

    turns = [
        ("human", "What is OriginTrail?"),
        ("ai",    "OriginTrail is a decentralized knowledge graph protocol that enables trusted data exchange."),
        ("human", "What is a Knowledge Asset?"),
        ("ai",    "A Knowledge Asset is an ownable, verifiable container of structured knowledge stored on the DKG."),
        ("human", "What memory layers does DKG v10 support?"),
        ("ai",    "DKG v10 has three layers: Working Memory (private, free), Shared Working Memory (gossip-replicated), and Verified Memory (on-chain, permanent)."),
    ]

    print("Storing 6 conversation turns in DKG Working Memory...")
    for role, content in turns:
        msg = HumanMessage(content=content) if role == "human" else AIMessage(content=content)
        await history.aadd_message(msg)
        ual = history.get_ual(role, content)
        print(f"  [{role}] stored → UAL: {ual}")

    print("\nRetrieving relevant history for 'Knowledge Asset'...")
    search_history = DKGChatMessageHistory(
        context_graph_id=CONTEXT_GRAPH_ID,
        client=client,
        search_query="Knowledge Asset",
        search_limit=4,
    )
    msgs = await search_history.aget_messages()
    print(f"  Retrieved {len(msgs)} turns:")
    for m in msgs:
        prefix = "Human" if isinstance(m, HumanMessage) else "AI"
        print(f"    {prefix}: {m.content[:80]}...")

    # --- Demo 2: SPARQL retrieval ---
    print("\n=== Demo 2: DKGRetriever (SPARQL) ===")
    retriever = DKGRetriever(client=client, limit=10, include_workspace=True)
    print("Querying DKG for 'OriginTrail'...")
    docs = await retriever._aget_relevant_documents("OriginTrail")
    print(f"  Retrieved {len(docs)} triples as Documents")
    for doc in docs[:3]:
        print(f"    {doc.page_content[:100]}")

    # --- Demo 3: DKGMemory with a simple chat loop ---
    if OPENAI_KEY:
        print("\n=== Demo 3: DKGMemory + LangChain RunnableWithMessageHistory ===")
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to a Decentralized Knowledge Graph."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        chain_with_memory = DKGMemory.wrap_chain(
            chain,
            context_graph_id=CONTEXT_GRAPH_ID,
            client=client,
            search_limit=5,
            history_messages_key="history",
        )

        questions = [
            "Summarize what you know about DKG v10 memory layers.",
            "Which layer is best for sensitive private data?",
        ]
        for q in questions:
            print(f"\nHuman: {q}")
            response = chain_with_memory.invoke(
                {"input": q},
                config={"configurable": {"session_id": "demo-session"}},
            )
            print(f"AI: {response.content}")
    else:
        print("\nSkipping Demo 3 (set OPENAI_API_KEY to run the chain demo)")

    print("\nDemo complete. All conversation turns are stored in DKG Working Memory.")
    print(f"Context Graph ID: {CONTEXT_GRAPH_ID}")


if __name__ == "__main__":
    asyncio.run(run_demo())
