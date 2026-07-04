"""Unit tests for DKGChatMessageHistory."""

import json

import httpx
import pytest
import respx

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_dkg.client import DKGClient
from langchain_dkg.chat_history import (
    _AI_PREFIX,
    _HUMAN_PREFIX,
    _SYSTEM_PREFIX,
    _TOOL_PREFIX,
    _TURN_URI_INDEX_CAP,
    DKGChatMessageHistory,
    _message_to_markdown,
    _snippet_to_message,
)


BASE = "http://localhost:9200"
TOKEN = "test-token"

TURN_RESPONSE = {
    "turnUri": "did:dkg:context-graph:cg-test/turn/peer-ts",
    "fileHash": "abc123",
    "layer": "wm",
    "graph": "did:dkg:context-graph:cg-test/_workspace",
    "structuralTripleCount": 2,
    "semanticTripleCount": 0,
    "totalQuads": 4,
    "embeddingId": None,
    "sessionUri": None,
}


def _search_result(entity_uri, snippet, similarity=1.0):
    return {
        "entityUri": entity_uri,
        "label": None,
        "sources": ["vector"],
        "similarity": similarity,
        "sourceFile": None,
        "snippet": snippet,
        "memoryLayer": "wm",
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
    turn_uri = await history.aadd_message(HumanMessage(content="What is DKG?"))
    body = json.loads(respx.calls.last.request.content)
    assert body["markdown"] == f"{_HUMAN_PREFIX}What is DKG?"
    assert history.get_turn_uri(f"{_HUMAN_PREFIX}What is DKG?") == TURN_RESPONSE["turnUri"]
    # aadd_message also returns the turn URI so callers can capture it directly.
    assert turn_uri == TURN_RESPONSE["turnUri"]


@respx.mock
async def test_add_ai_message(history):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await history.aadd_message(AIMessage(content="DKG is a Decentralized Knowledge Graph."))
    body = json.loads(respx.calls.last.request.content)
    assert body["markdown"].startswith(_AI_PREFIX)


@respx.mock
async def test_layer_defaults_to_wm_on_writes(history):
    # Conversation history is private by default — layer "wm" must be sent
    # explicitly (the node's own default is "swm", which is gossiped).
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json=TURN_RESPONSE)
    )
    await history.aadd_message(HumanMessage(content="private"))
    body = json.loads(respx.calls.last.request.content)
    assert body["layer"] == "wm"


@respx.mock
async def test_get_messages(history):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "query": "conversation history",
            "contextGraphId": "cg-test",
            "resultCount": 2,
            "results": [
                _search_result("urn:turn:1", f"{_HUMAN_PREFIX}Hello"),
                _search_result("urn:turn:2", f"{_AI_PREFIX}Hi there", similarity=0.9),
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
async def test_get_messages_sends_memory_layers(history):
    route = respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={"resultCount": 0, "results": []})
    )
    await history.aget_messages()
    body = json.loads(route.calls.last.request.content)
    assert body["memoryLayers"] == ["wm", "swm"]
    assert body["limit"] == history.search_limit


@respx.mock
async def test_session_filtering_and_sort(client):
    """With session_uri set: over-fetch, filter to the session's turns, sort by URI."""
    hist = DKGChatMessageHistory(
        context_graph_id="cg-test",
        client=client,
        session_uri="urn:session:s1",
        search_limit=2,
    )
    # Turn URIs end with an ISO timestamp — lexical sort is chronological.
    t1 = "did:dkg:cg-test/turn/peer-2026-07-01T10:00:00Z"
    t2 = "did:dkg:cg-test/turn/peer-2026-07-01T11:00:00Z"
    t3 = "did:dkg:cg-test/turn/peer-2026-07-01T12:00:00Z"
    search_route = respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "resultCount": 3,
            "results": [
                # Relevance-ranked: newest first, plus one turn from another session.
                _search_result(t3, f"{_AI_PREFIX}third", 0.99),
                _search_result("urn:turn:other-session", f"{_AI_PREFIX}other", 0.95),
                _search_result(t2, f"{_HUMAN_PREFIX}second", 0.90),
                _search_result(t1, f"{_HUMAN_PREFIX}first", 0.80),
            ],
        })
    )
    query_route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={
            "result": {"bindings": [{"t": t2}, {"t": t3}, {"t": t1}]}
        })
    )
    msgs = await hist.aget_messages()

    # Over-fetch: limit = search_limit * 3
    search_body = json.loads(search_route.calls.last.request.content)
    assert search_body["limit"] == 6

    # The SPARQL lookup targets the session's hasPart triples in-workspace.
    query_body = json.loads(query_route.calls.last.request.content)
    assert "<urn:session:s1>" in query_body["sparql"]
    assert "http://schema.org/hasPart" in query_body["sparql"]
    assert query_body["contextGraphId"] == "cg-test"
    assert query_body["includeWorkspace"] is True

    # Filtered to session turns, sorted chronologically, and truncated to the
    # MOST RECENT search_limit turns.
    assert [m.content for m in msgs] == ["second", "third"]


@respx.mock
async def test_session_filter_falls_back_when_sparql_fails(client):
    hist = DKGChatMessageHistory(
        context_graph_id="cg-test",
        client=client,
        session_uri="urn:session:s1",
        search_limit=2,
    )
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "resultCount": 3,
            "results": [
                _search_result("urn:turn:a", f"{_AI_PREFIX}A"),
                _search_result("urn:turn:b", f"{_HUMAN_PREFIX}B"),
                _search_result("urn:turn:c", f"{_AI_PREFIX}C"),
            ],
        })
    )
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    msgs = await hist.aget_messages()
    # Fallback: unfiltered relevance order, truncated to search_limit.
    assert [m.content for m in msgs] == ["A", "B"]


@respx.mock
async def test_session_filter_falls_back_when_no_haspart(client):
    hist = DKGChatMessageHistory(
        context_graph_id="cg-test",
        client=client,
        session_uri="urn:session:s1",
        search_limit=5,
    )
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "resultCount": 1,
            "results": [_search_result("urn:turn:a", f"{_AI_PREFIX}A")],
        })
    )
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    msgs = await hist.aget_messages()
    assert [m.content for m in msgs] == ["A"]


@respx.mock
async def test_no_session_uri_keeps_current_behavior(history):
    route = respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={"resultCount": 0, "results": []})
    )
    await history.aget_messages()
    # No over-fetch and no SPARQL lookup when session_uri is None.
    body = json.loads(route.calls.last.request.content)
    assert body["limit"] == history.search_limit
    assert len(respx.calls) == 1


@respx.mock
async def test_turn_uri_indexed_from_search(history):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "resultCount": 1,
            "results": [_search_result("urn:turn:xyz", f"{_HUMAN_PREFIX}Test")],
        })
    )
    await history.aget_messages()
    assert history.get_turn_uri(f"{_HUMAN_PREFIX}Test") == "urn:turn:xyz"


@respx.mock
async def test_promote_to_shared(history):
    # turn URI has no "/" so used verbatim as assertion name (URL-quoted by the client)
    turn_uri = "did:dkg:cg-test:turn-abc"
    quoted = "did%3Adkg%3Acg-test%3Aturn-abc"
    respx.post(f"{BASE}/api/knowledge-assets/{quoted}/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "j1", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/j1").mock(
        return_value=httpx.Response(200, json={"jobId": "j1", "state": "succeeded"})
    )
    result = await history.promote_to_shared(turn_uri)
    assert result["state"] == "succeeded"


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


# ---------------------------------------------------------------------------
# Markdown round-tripping
# ---------------------------------------------------------------------------


def test_message_to_markdown_roles():
    assert _message_to_markdown(HumanMessage(content="h")) == ("human", f"{_HUMAN_PREFIX}h")
    assert _message_to_markdown(AIMessage(content="a")) == ("ai", f"{_AI_PREFIX}a")
    assert _message_to_markdown(SystemMessage(content="s")) == ("system", f"{_SYSTEM_PREFIX}s")
    assert _message_to_markdown(ToolMessage(content="t", tool_call_id="c1")) == (
        "tool",
        f"{_TOOL_PREFIX}t",
    )


def test_message_to_markdown_content_blocks():
    msg = HumanMessage(content=[{"type": "text", "text": "hello "}, "world", {"type": "image"}])
    role, markdown = _message_to_markdown(msg)
    assert role == "human"
    assert markdown == f"{_HUMAN_PREFIX}hello world"


def test_snippet_to_message_roles():
    assert isinstance(_snippet_to_message(f"{_HUMAN_PREFIX}x"), HumanMessage)
    assert isinstance(_snippet_to_message(f"{_AI_PREFIX}x"), AIMessage)
    sys_msg = _snippet_to_message(f"{_SYSTEM_PREFIX}rules")
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "rules"
    tool_msg = _snippet_to_message(f"{_TOOL_PREFIX}output")
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "output"
    assert tool_msg.tool_call_id == ""
    # Unlabelled snippets fall back to AI; empty snippets drop.
    assert isinstance(_snippet_to_message("plain"), AIMessage)
    assert _snippet_to_message("") is None


# ---------------------------------------------------------------------------
# LRU cap on the turn-URI index
# ---------------------------------------------------------------------------


def test_turn_uri_index_lru_cap(history):
    for i in range(_TURN_URI_INDEX_CAP + 10):
        history._remember_turn(f"markdown-{i}", f"urn:turn:{i}")
    assert len(history._turn_uri_index) == _TURN_URI_INDEX_CAP
    # Oldest entries were evicted; newest are retained.
    assert history.get_turn_uri("markdown-0") is None
    assert history.get_turn_uri(f"markdown-{_TURN_URI_INDEX_CAP + 9}") == (
        f"urn:turn:{_TURN_URI_INDEX_CAP + 9}"
    )


def test_turn_uri_index_lru_recency(history):
    for i in range(_TURN_URI_INDEX_CAP):
        history._remember_turn(f"markdown-{i}", f"urn:turn:{i}")
    # Touch the oldest entry so it becomes most-recently-used...
    assert history.get_turn_uri("markdown-0") == "urn:turn:0"
    # ...then push one more entry: markdown-1 (now oldest) is evicted instead.
    history._remember_turn("markdown-new", "urn:turn:new")
    assert history.get_turn_uri("markdown-0") == "urn:turn:0"
    assert history.get_turn_uri("markdown-1") is None
