"""Unit tests for DKGClient using respx to mock httpx."""

import pytest
import respx
import httpx
import json

from langchain_dkg.client import (
    CuratorAckError,
    CuratorRejectedError,
    CuratorUnconfirmedError,
    DKGClient,
)


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


# ---------------------------------------------------------------------------
# Quad-shaped writes (shared_memory_write / assertion_write)
#
# The node's /api/shared-memory/write and /api/assertion/{name}/write endpoints
# require `quads` as an array of {subject, predicate, object[, graph]} objects;
# since node rc.19 they 400 on string-shaped quads. These guard the wire shape.
# ---------------------------------------------------------------------------


@respx.mock
async def test_shared_memory_write_sends_quads_objects(client):
    respx.post(f"{BASE}/api/shared-memory/write").mock(
        return_value=httpx.Response(200, json={"status": "written"})
    )
    await client.shared_memory_write(
        quads=[{"subject": "ex:s", "predicate": "ex:p", "object": '"hello"'}],
        context_graph_id="cg-1",
    )
    body = json.loads(respx.calls.last.request.content)
    # Correct field name is "quads" (not the old broken "triples")...
    assert "triples" not in body
    assert body["contextGraphId"] == "cg-1"
    # ...and each element is a structured object, not a bare string.
    assert body["quads"] == [{"subject": "ex:s", "predicate": "ex:p", "object": '"hello"'}]


@respx.mock
async def test_shared_memory_write_normalizes_tuples(client):
    respx.post(f"{BASE}/api/shared-memory/write").mock(
        return_value=httpx.Response(200, json={"status": "written"})
    )
    await client.shared_memory_write(
        quads=[("ex:s", "ex:p", "ex:o", "ex:g")],
    )
    body = json.loads(respx.calls.last.request.content)
    assert body["quads"] == [
        {"subject": "ex:s", "predicate": "ex:p", "object": "ex:o", "graph": "ex:g"}
    ]


async def test_shared_memory_write_rejects_raw_strings(client):
    # Regression guard: the old API took list[str] and sent {"triples": [...]},
    # which the node ignored (HTTP 400). Raw strings now fail fast, client-side.
    with pytest.raises(ValueError, match="structured, not raw strings"):
        await client.shared_memory_write(quads=["<ex:s> <ex:p> <ex:o> ."])


async def test_shared_memory_write_rejects_short_tuple(client):
    with pytest.raises(ValueError, match="3 or 4 elements"):
        await client.shared_memory_write(quads=[("ex:s", "ex:p")])


@respx.mock
async def test_assertion_write_sends_quads_objects(client):
    respx.post(f"{BASE}/api/assertion/my-assert/write").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    await client.assertion_write(
        name="my-assert",
        context_graph_id="cg-1",
        quads=[("ex:s", "ex:p", '"v"')],
    )
    body = json.loads(respx.calls.last.request.content)
    assert body["contextGraphId"] == "cg-1"
    assert body["quads"] == [{"subject": "ex:s", "predicate": "ex:p", "object": '"v"'}]


# ---------------------------------------------------------------------------
# Curator-ack mapping (OT-RFC-49 curator-leader)
#
# The SHARE/PUBLISH write paths return 503 CURATOR_UNCONFIRMED / 409
# CURATOR_REJECTED (node rc.19+) instead of persisting. These map to typed,
# catchable exceptions rather than an opaque httpx.HTTPStatusError.
# ---------------------------------------------------------------------------

VALID_QUAD = {"subject": "ex:s", "predicate": "ex:p", "object": '"v"'}


@respx.mock
async def test_shared_memory_write_maps_curator_unconfirmed(client):
    respx.post(f"{BASE}/api/shared-memory/write").mock(
        return_value=httpx.Response(
            503,
            json={
                "error": "curator did not confirm",
                "code": "CURATOR_UNCONFIRMED",
                "curatorDelivery": "unconfirmed",
                "contextGraphId": "cg-1",
            },
        )
    )
    with pytest.raises(CuratorUnconfirmedError) as exc:
        await client.shared_memory_write(quads=[VALID_QUAD], context_graph_id="cg-1")
    err = exc.value
    assert isinstance(err, CuratorAckError)
    assert err.code == "CURATOR_UNCONFIRMED"
    assert err.curator_delivery == "unconfirmed"
    assert err.context_graph_id == "cg-1"
    assert err.response is not None and err.response.status_code == 503


@respx.mock
async def test_shared_memory_publish_maps_curator_rejected(client):
    respx.post(f"{BASE}/api/shared-memory/publish").mock(
        return_value=httpx.Response(
            409,
            json={"error": "curator rejected", "code": "CURATOR_REJECTED"},
        )
    )
    with pytest.raises(CuratorRejectedError) as exc:
        await client.shared_memory_publish(context_graph_id="cg-1")
    assert exc.value.code == "CURATOR_REJECTED"


@respx.mock
async def test_assertion_promote_maps_curator_unconfirmed(client):
    respx.post(f"{BASE}/api/assertion/a1/promote").mock(
        return_value=httpx.Response(
            503, json={"error": "no ack", "code": "CURATOR_UNCONFIRMED"}
        )
    )
    with pytest.raises(CuratorUnconfirmedError):
        await client.assertion_promote(name="a1", context_graph_id="cg-1")


@respx.mock
async def test_generic_503_without_code_stays_http_error(client):
    # A 503 that is NOT a curator-ack failure must remain a plain HTTPStatusError,
    # so we don't mislabel unrelated outages as curator rejections.
    respx.post(f"{BASE}/api/shared-memory/write").mock(
        return_value=httpx.Response(503, json={"error": "node overloaded"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.shared_memory_write(quads=[VALID_QUAD])
