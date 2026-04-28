"""Unit tests for DKGClient using respx to mock httpx."""

import pytest
import respx
import httpx
import json

from langchain_dkg.client import DKGClient


BASE = "http://localhost:9200"
TOKEN = "test-token"


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


@respx.mock
async def test_ping_success(client):
    respx.get(f"{BASE}/api/agents").mock(return_value=httpx.Response(200, json={"agents": []}))
    assert await client.ping() is True


@respx.mock
async def test_ping_failure(client):
    respx.get(f"{BASE}/api/agents").mock(return_value=httpx.Response(401))
    assert await client.ping() is False


@respx.mock
async def test_memory_turn(client):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json={
            "turnUri": "did:dkg:context-graph:cg-1/turn/peer-ts",
            "fileHash": "abc123",
            "layer": "swm",
            "graph": "did:dkg:context-graph:cg-1/_shared_memory",
            "structuralTripleCount": 3,
            "semanticTripleCount": 0,
            "totalQuads": 5,
            "embeddingId": None,
            "sessionUri": None,
        })
    )
    result = await client.memory_turn(
        context_graph_id="cg-1",
        markdown="**Human:** Hello world",
    )
    assert "turnUri" in result
    body = json.loads(respx.calls.last.request.content)
    assert body["contextGraphId"] == "cg-1"
    assert body["markdown"] == "**Human:** Hello world"


@respx.mock
async def test_memory_turn_with_session_uri(client):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json={"turnUri": "t1", "sessionUri": "urn:session:s1"})
    )
    await client.memory_turn(
        context_graph_id="cg-1",
        markdown="**AI:** Hello",
        session_uri="urn:session:s1",
        layer="wm",
    )
    body = json.loads(respx.calls.last.request.content)
    assert body["sessionUri"] == "urn:session:s1"
    assert body["layer"] == "wm"


@respx.mock
async def test_memory_search(client):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={
            "query": "hello",
            "contextGraphId": "cg-1",
            "resultCount": 1,
            "results": [{
                "entityUri": "did:dkg:context-graph:cg-1/turn/t1",
                "label": "Human",
                "sources": ["vector"],
                "similarity": 0.9,
                "sourceFile": None,
                "snippet": "**Human:** Hello world",
                "memoryLayer": "swm",
            }],
        })
    )
    result = await client.memory_search(context_graph_id="cg-1", query="hello", limit=5)
    assert result["resultCount"] == 1
    assert result["results"][0]["snippet"] == "**Human:** Hello world"


@respx.mock
async def test_query(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    result = await client.query(sparql="SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert "results" in result


@respx.mock
async def test_memory_turn_sends_auth_header(client):
    route = respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json={"turnUri": "x"})
    )
    await client.memory_turn(context_graph_id="cg-1", markdown="**AI:** Reply")
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


def test_client_requires_token():
    import os
    os.environ.pop("DKG_TOKEN", None)
    with pytest.raises(ValueError, match="bearer token"):
        DKGClient(base_url=BASE)


@respx.mock
async def test_assertion_promote(client):
    respx.post(f"{BASE}/api/assertion/my-assert/promote").mock(
        return_value=httpx.Response(200, json={"status": "shared"})
    )
    result = await client.assertion_promote(name="my-assert", context_graph_id="cg-1")
    assert result["status"] == "shared"
