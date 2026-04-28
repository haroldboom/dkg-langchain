"""Unit tests for DKGChatMessageHistory."""

import pytest
import respx
import httpx
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_dkg.client import DKGClient
from langchain_dkg.chat_history import DKGChatMessageHistory, _HUMAN_PREFIX, _AI_PREFIX


BASE = "http://localhost:9200"
TOKEN = "test-token"

TURN_RESPONSE = {
    "turnUri": "did:dkg:context-graph:cg-test/turn/peer-ts",
    "fileHash": "abc123",
    "layer": "swm",
    "graph": "did:dkg:context-graph:cg-test/_shared_memory",
    "structuralTripleCount": 2,
    "semanticTripleCount": 0,
    "totalQuads": 4,
    "embeddingId": None,
    "sessionUri": None,
}


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


@pytest.fixture
def history(client):
    return DKGChatMessageHistory(context_graph_id="cg-test", client=client)


@respx.mock
async def test_add_human_message(history):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await history.aadd_message(HumanMessage(content="What is DKG?"))
    body = json.loads(respx.calls.last.request.content)
    assert body["markdown"] == f"{_HUMAN_PREFIX}What is DKG?"
    assert history.get_turn_uri(f"{_HUMAN_PREFIX}What is DKG?") == TURN_RESPONSE["turnUri"]


@respx.mock
async def test_add_ai_message(history):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await history.aadd_message(AIMessage(content="DKG is a Decentralized Knowledge Graph."))
    body = json.loads(respx.calls.last.request.content)
    assert body["markdown"].startswith(_AI_PREFIX)


@respx.mock
async def test_get_messages(history):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "query": "conversation history",
            "contextGraphId": "cg-test",
            "resultCount": 2,
            "results": [
                {
                    "entityUri": "urn:turn:1",
                    "label": None,
                    "sources": ["vector"],
                    "similarity": 1.0,
                    "sourceFile": None,
                    "snippet": f"{_HUMAN_PREFIX}Hello",
                    "memoryLayer": "swm",
                },
                {
                    "entityUri": "urn:turn:2",
                    "label": None,
                    "sources": ["vector"],
                    "similarity": 0.9,
                    "sourceFile": None,
                    "snippet": f"{_AI_PREFIX}Hi there",
                    "memoryLayer": "swm",
                },
            ],
        })
    )
    msgs = await history.aget_messages()
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "Hello"
    assert msgs[1].content == "Hi there"


@respx.mock
async def test_turn_uri_indexed_from_search(history):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "resultCount": 1,
            "results": [{
                "entityUri": "urn:turn:xyz",
                "snippet": f"{_HUMAN_PREFIX}Test",
                "sources": ["vector"],
                "similarity": 1.0,
                "label": None,
                "sourceFile": None,
                "memoryLayer": "swm",
            }],
        })
    )
    await history.aget_messages()
    assert history.get_turn_uri(f"{_HUMAN_PREFIX}Test") == "urn:turn:xyz"


@respx.mock
async def test_promote_to_shared(history):
    # turn URI has no "/" so used verbatim as assertion name
    turn_uri = "did:dkg:cg-test:turn-abc"
    respx.post(f"{BASE}/api/assertion/{turn_uri}/promote").mock(
        return_value=httpx.Response(200, json={"status": "shared"})
    )
    result = await history.promote_to_shared(turn_uri)
    assert result["status"] == "shared"


@respx.mock
async def test_clear_resets_index(history):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await history.aadd_message(HumanMessage(content="test"))
    assert len(history._turn_uri_index) == 1
    history.clear()
    assert len(history._turn_uri_index) == 0


@respx.mock
async def test_session_uri_forwarded(client):
    hist = DKGChatMessageHistory(
        context_graph_id="cg",
        client=client,
        session_uri="urn:session:demo",
    )
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await hist.aadd_message(HumanMessage(content="hi"))
    body = json.loads(respx.calls.last.request.content)
    assert body["sessionUri"] == "urn:session:demo"
