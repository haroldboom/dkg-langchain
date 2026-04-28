"""Integration tests — require a live DKG v10 node at localhost:9200.

Run with:
    DKG_TOKEN=$(dkg auth show) pytest tests/integration/ -v

These tests create real Working Memory Knowledge Assets. They do NOT
call SHARE or PUBLISH (no Shared/Verified Memory writes, no TRAC cost).

NOTE: Python socket connections to localhost require a DKG node running
on the same machine. These tests are skipped unless DKG_TOKEN is set.
"""

import os
import uuid
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DKG_TOKEN"),
    reason="DKG_TOKEN not set — skipping integration tests",
)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_dkg.client import DKGClient
from langchain_dkg.chat_history import DKGChatMessageHistory
from langchain_dkg.retriever import DKGRetriever


@pytest.fixture(scope="module")
def client():
    return DKGClient()


@pytest.fixture
def unique_context(client):
    """Each test gets its own Context Graph to avoid cross-test pollution."""
    return f"integration-test-{uuid.uuid4().hex[:8]}"


async def test_node_reachable(client):
    assert await client.ping(), "DKG node did not respond — is it running?"


async def test_store_and_retrieve_turn(client, unique_context):
    history = DKGChatMessageHistory(
        context_graph_id=unique_context,
        client=client,
        search_limit=5,
    )
    await history.aadd_message(HumanMessage(content="What is a Knowledge Asset?"))
    await history.aadd_message(AIMessage(content="A Knowledge Asset is an ownable container of structured knowledge on the DKG."))

    msgs = await history.aget_messages()
    contents = [m.content for m in msgs]
    assert any("Knowledge Asset" in c for c in contents), f"Expected KA content in: {contents}"


async def test_ual_returned_on_store(client, unique_context):
    history = DKGChatMessageHistory(
        context_graph_id=unique_context,
        client=client,
    )
    await history.aadd_message(HumanMessage(content="UAL test message"))
    ual = history.get_ual("human", "UAL test message")
    assert ual is not None, "Expected a UAL to be returned and stored"
    assert "dkg" in ual.lower() or ":" in ual, f"UAL looks malformed: {ual}"


async def test_sparql_query(client, unique_context):
    retriever = DKGRetriever(client=client, limit=5, include_workspace=True)
    # A broad query that should return at least empty results without error
    docs = await retriever._aget_relevant_documents("knowledge")
    assert isinstance(docs, list)
