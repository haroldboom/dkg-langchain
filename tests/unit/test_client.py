"""Unit tests for DKGClient using respx to mock httpx."""

import json

import httpx
import pytest
import respx

from langchain_dkg.client import (
    CuratorAckError,
    CuratorRejectedError,
    CuratorUnconfirmedError,
    DKGClient,
    DKGConnectionError,
    DKGError,
    DKGStatusError,
)


BASE = "http://localhost:9200"
TOKEN = "test-token"


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


@respx.mock
async def test_ping_success(client):
    respx.get(f"{BASE}/api/context-graph/list").mock(
        return_value=httpx.Response(200, json={"contextGraphs": []})
    )
    assert await client.ping() is True


@respx.mock
async def test_ping_transport_error_returns_false(client):
    respx.get(f"{BASE}/api/context-graph/list").mock(
        side_effect=httpx.ConnectError("refused")
    )
    assert await client.ping() is False


@respx.mock
async def test_ping_auth_error_raises(client):
    # A bad token must be distinguishable from a node that is down.
    respx.get(f"{BASE}/api/context-graph/list").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.ping()
    assert exc.value.status_code == 401


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
async def test_memory_search_sends_default_layers(client):
    # Current node builds return 0 results when memoryLayers is omitted,
    # so the client must always send it.
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={"resultCount": 0, "results": []})
    )
    await client.memory_search(context_graph_id="cg-1", query="q")
    body = json.loads(respx.calls.last.request.content)
    assert body["memoryLayers"] == ["wm", "swm"]


@respx.mock
async def test_memory_search_honors_explicit_layers(client):
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={"resultCount": 0, "results": []})
    )
    await client.memory_search(context_graph_id="cg-1", query="q", memory_layers=["wm"])
    body = json.loads(respx.calls.last.request.content)
    assert body["memoryLayers"] == ["wm"]


@respx.mock
async def test_memory_search_empty_layers_list_is_sent(client):
    # `is not None` semantics: an explicit empty list is forwarded as-is.
    respx.post(f"{BASE}/api/memory/search").mock(
        return_value=httpx.Response(200, json={"resultCount": 0, "results": []})
    )
    await client.memory_search(context_graph_id="cg-1", query="q", memory_layers=[])
    body = json.loads(respx.calls.last.request.content)
    assert body["memoryLayers"] == []


@respx.mock
async def test_create_context_graph_sends_id_and_name(client):
    # The node 400s on name-only bodies; id defaults to name.
    respx.post(f"{BASE}/api/context-graph/create").mock(
        return_value=httpx.Response(200, json={"created": "cg-1", "uri": "did:dkg:context-graph:cg-1"})
    )
    result = await client.create_context_graph("cg-1")
    body = json.loads(respx.calls.last.request.content)
    assert body == {"id": "cg-1", "name": "cg-1"}
    assert result["created"] == "cg-1"


@respx.mock
async def test_create_context_graph_explicit_id(client):
    respx.post(f"{BASE}/api/context-graph/create").mock(
        return_value=httpx.Response(200, json={"created": "my-id", "uri": "u"})
    )
    await client.create_context_graph("My Graph", id="my-id")
    body = json.loads(respx.calls.last.request.content)
    assert body == {"id": "my-id", "name": "My Graph"}


@respx.mock
async def test_query(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    result = await client.query(sparql="SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert "results" in result


@respx.mock
async def test_query_sends_context_graph_id(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await client.query(sparql="SELECT * WHERE { ?s ?p ?o }", context_graph_id="cg-1")
    body = json.loads(respx.calls.last.request.content)
    assert body["contextGraphId"] == "cg-1"
    assert body["includeWorkspace"] is True


@respx.mock
async def test_memory_turn_sends_auth_header(client):
    route = respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(200, json={"turnUri": "x"})
    )
    await client.memory_turn(context_graph_id="cg-1", markdown="**AI:** Reply")
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


def test_client_requires_token():
    import os
    saved = os.environ.pop("DKG_TOKEN", None)
    try:
        with pytest.raises(ValueError, match="bearer token"):
            DKGClient(base_url=BASE)
    finally:
        if saved is not None:
            os.environ["DKG_TOKEN"] = saved


# ---------------------------------------------------------------------------
# Error wrapping — httpx errors become DKGStatusError / DKGConnectionError
# ---------------------------------------------------------------------------


@respx.mock
async def test_http_status_error_wrapped(client):
    respx.post(f"{BASE}/api/memory/turn").mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.memory_turn(context_graph_id="cg-1", markdown="x")
    err = exc.value
    assert isinstance(err, DKGError)
    assert err.status_code == 400
    assert "bad request" in err.body
    assert isinstance(err.__cause__, httpx.HTTPStatusError)


@respx.mock
async def test_transport_error_wrapped(client):
    respx.post(f"{BASE}/api/memory/search").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(DKGConnectionError) as exc:
        await client.memory_search(context_graph_id="cg-1", query="q")
    assert isinstance(exc.value, DKGError)
    assert isinstance(exc.value.__cause__, httpx.TransportError)


@respx.mock
async def test_timeout_wrapped_as_connection_error(client):
    respx.post(f"{BASE}/api/query").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    with pytest.raises(DKGConnectionError):
        await client.query(sparql="SELECT * WHERE { ?s ?p ?o }")


# ---------------------------------------------------------------------------
# Async promote (POST /api/knowledge-assets/{name}/swm/share-async + job polling)
# ---------------------------------------------------------------------------


@respx.mock
async def test_assertion_promote_async_happy_path(client):
    respx.post(f"{BASE}/api/knowledge-assets/my-assert/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-1", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-1").mock(
        return_value=httpx.Response(200, json={
            "jobId": "job-1",
            "state": "succeeded",
            "contextGraphId": "cg-1",
            "assertionName": "my-assert",
            "result": {"promotedCount": 3},
        })
    )
    result = await client.assertion_promote(name="my-assert", context_graph_id="cg-1")
    assert result["state"] == "succeeded"
    assert result["result"]["promotedCount"] == 3
    submit_body = json.loads(respx.calls[0].request.content)
    assert submit_body["contextGraphId"] == "cg-1"


@respx.mock
async def test_assertion_promote_async_polls_until_done(client):
    respx.post(f"{BASE}/api/knowledge-assets/my-assert/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-2", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-2").mock(
        side_effect=[
            httpx.Response(200, json={"jobId": "job-2", "state": "running"}),
            httpx.Response(200, json={"jobId": "job-2", "state": "succeeded"}),
        ]
    )
    result = await client.assertion_promote(
        name="my-assert", context_graph_id="cg-1", poll_interval=0.01
    )
    assert result["state"] == "succeeded"


@respx.mock
async def test_assertion_promote_async_url_quotes_name(client):
    respx.post(f"{BASE}/api/knowledge-assets/turn%2Fpeer-ts/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "j", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/j").mock(
        return_value=httpx.Response(200, json={"jobId": "j", "state": "succeeded"})
    )
    result = await client.assertion_promote(name="turn/peer-ts", context_graph_id="cg-1")
    assert result["state"] == "succeeded"


@respx.mock
async def test_assertion_promote_async_curator_rejected_job(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-3", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-3").mock(
        return_value=httpx.Response(200, json={
            "jobId": "job-3",
            "state": "failed",
            "lastError": {
                "code": "fatal",
                "message": 'SWM write to private context graph "cg-1" was permanently '
                           "rejected by its curator (allowlist / signature / validation failure).",
                "retryable": False,
            },
        })
    )
    with pytest.raises(CuratorRejectedError) as exc:
        await client.assertion_promote(name="a1", context_graph_id="cg-1")
    assert exc.value.code == "CURATOR_REJECTED"
    assert exc.value.context_graph_id == "cg-1"


@respx.mock
async def test_assertion_promote_async_curator_unconfirmed_job(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-4", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-4").mock(
        return_value=httpx.Response(200, json={
            "jobId": "job-4",
            "state": "failed",
            "lastError": {
                "code": "transient",
                "message": 'SWM write to private context graph "cg-1" was not confirmed '
                           "by its curator (the authoritative replica).",
                "retryable": True,
            },
        })
    )
    with pytest.raises(CuratorUnconfirmedError):
        await client.assertion_promote(name="a1", context_graph_id="cg-1")


@respx.mock
async def test_assertion_promote_async_generic_failure(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-5", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-5").mock(
        return_value=httpx.Response(200, json={
            "jobId": "job-5",
            "state": "failed",
            "lastError": {"code": "fatal", "message": "boom", "retryable": False},
        })
    )
    with pytest.raises(DKGError, match="boom"):
        await client.assertion_promote(name="a1", context_graph_id="cg-1")


@respx.mock
async def test_assertion_promote_async_poll_timeout(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-6", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-6").mock(
        return_value=httpx.Response(200, json={"jobId": "job-6", "state": "running"})
    )
    with pytest.raises(DKGError, match="did not finish"):
        await client.assertion_promote(
            name="a1", context_graph_id="cg-1", poll_timeout=0.05, poll_interval=0.01
        )


@respx.mock
async def test_assertion_promote_async_conflict_polls_existing_job(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(409, json={"error": "already active", "existingJobId": "job-7"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-7").mock(
        return_value=httpx.Response(200, json={"jobId": "job-7", "state": "succeeded"})
    )
    result = await client.assertion_promote(name="a1", context_graph_id="cg-1")
    assert result["jobId"] == "job-7"


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
# catchable exceptions rather than an opaque status error.
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
    assert err.status_code == 503
    assert err.body["code"] == "CURATOR_UNCONFIRMED"


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
async def test_assertion_promote_submit_maps_curator_unconfirmed(client):
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(
            503, json={"error": "no ack", "code": "CURATOR_UNCONFIRMED"}
        )
    )
    with pytest.raises(CuratorUnconfirmedError):
        await client.assertion_promote(name="a1", context_graph_id="cg-1")


@respx.mock
async def test_generic_503_without_code_is_status_error(client):
    # A 503 that is NOT a curator-ack failure must remain a generic status
    # error, so we don't mislabel unrelated outages as curator rejections.
    respx.post(f"{BASE}/api/shared-memory/write").mock(
        return_value=httpx.Response(503, json={"error": "node overloaded"})
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.shared_memory_write(quads=[VALID_QUAD])
    assert not isinstance(exc.value, CuratorAckError)
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@respx.mock
async def test_http_client_reused_within_loop(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await client.query(sparql="SELECT * WHERE { ?s ?p ?o }")
    first = client._http
    await client.query(sparql="SELECT * WHERE { ?s ?p ?o }")
    assert client._http is first


@respx.mock
async def test_aclose_and_context_manager(client):
    respx.get(f"{BASE}/api/context-graph/list").mock(
        return_value=httpx.Response(200, json={})
    )
    async with client as c:
        assert await c.ping() is True
    assert client._http is None


@respx.mock
async def test_assertion_promote_falls_back_to_legacy_routes(client):
    """Older node builds 404 the knowledge-assets surface; the client falls
    back to the legacy /api/assertion promote-async routes."""
    respx.post(f"{BASE}/api/knowledge-assets/a1/swm/share-async").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )
    respx.post(f"{BASE}/api/assertion/a1/promote-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-l", "state": "queued"})
    )
    respx.get(f"{BASE}/api/assertion/promote-async/job-l").mock(
        return_value=httpx.Response(200, json={"jobId": "job-l", "state": "succeeded"})
    )
    job = await client.assertion_promote("a1", context_graph_id="cg-1")
    assert job["state"] == "succeeded"


@respx.mock
async def test_assertion_create_falls_back_to_legacy_route(client):
    respx.post(f"{BASE}/api/knowledge-assets").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )
    legacy = respx.post(f"{BASE}/api/assertion/create").mock(
        return_value=httpx.Response(200, json={"name": "a1"})
    )
    result = await client.assertion_create("cg-1", "a1")
    assert result["name"] == "a1"
    assert legacy.called


@respx.mock
async def test_assertion_create_uses_knowledge_assets_route(client):
    route = respx.post(f"{BASE}/api/knowledge-assets").mock(
        return_value=httpx.Response(200, json={
            "name": "a1", "assertionUri": "did:x", "status": "draft-open",
        })
    )
    result = await client.assertion_create("cg-1", "a1")
    assert result["status"] == "draft-open"
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1", "name": "a1"}
