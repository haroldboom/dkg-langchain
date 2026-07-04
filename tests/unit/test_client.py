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
    DKGPublishPreconditionError,
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


# ---------------------------------------------------------------------------
# Verifiable Memory surface (node build 10.0.2)
#
# Draft lifecycle (wm/write → wm/finalize → vm/publish), the one-shot direct
# publish, the trust-gradient calls (endorse / verify / kc provenance) and the
# query view/minTrust extension.
# ---------------------------------------------------------------------------


@respx.mock
async def test_assertion_create_one_shot_passthrough(client):
    route = respx.post(f"{BASE}/api/knowledge-assets").mock(
        return_value=httpx.Response(200, json={"name": "a1", "status": "draft-open"})
    )
    await client.assertion_create("cg-1", "a1", also_share_swm=True, also_publish_vm=True)
    body = json.loads(route.calls.last.request.content)
    assert body["alsoShareSwm"] is True
    assert body["alsoPublishVm"] is True


@respx.mock
async def test_assertion_create_one_shot_partial_207_passthrough(client):
    # A partial one-shot answers 207; the body is returned, not raised.
    respx.post(f"{BASE}/api/knowledge-assets").mock(
        return_value=httpx.Response(207, json={
            "name": "a1", "status": "draft-open", "alreadyExists": False,
        })
    )
    result = await client.assertion_create("cg-1", "a1", also_share_swm=True)
    assert result["name"] == "a1"


@respx.mock
async def test_ka_write_sends_quads_objects(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/ka-1/wm/write").mock(
        return_value=httpx.Response(200, json={"written": 2})
    )
    result = await client.ka_write(
        name="ka-1",
        context_graph_id="cg-1",
        quads=[("ex:s", "ex:p", '"v"'), {"subject": "ex:s", "predicate": "ex:q", "object": "ex:o"}],
    )
    assert result["written"] == 2
    body = json.loads(route.calls.last.request.content)
    assert body["contextGraphId"] == "cg-1"
    assert body["quads"] == [
        {"subject": "ex:s", "predicate": "ex:p", "object": '"v"'},
        {"subject": "ex:s", "predicate": "ex:q", "object": "ex:o"},
    ]


@respx.mock
async def test_ka_write_url_quotes_name(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/turn%2Fpeer-ts/wm/write").mock(
        return_value=httpx.Response(200, json={"written": 1})
    )
    await client.ka_write(name="turn/peer-ts", context_graph_id="cg-1", quads=[VALID_QUAD])
    assert route.called


@respx.mock
async def test_ka_finalize_returns_seal(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/ka-1/wm/finalize").mock(
        return_value=httpx.Response(200, json={
            "assertionUri": "did:dkg:assertion:ka-1",
            "merkleRoot": "0xroot",
            "authorAddress": "0xauthor",
            "schemeVersion": 1,
            "chainId": 84532,
            "kav10Address": "0xkav10",
            "eip712Digest": "0xdigest",
        })
    )
    seal = await client.ka_finalize("ka-1", "cg-1")
    assert seal["eip712Digest"] == "0xdigest"
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1"}


@respx.mock
async def test_ka_discard(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/ka-1/wm/discard").mock(
        return_value=httpx.Response(200, json={"discarded": True})
    )
    await client.ka_discard("ka-1", "cg-1")
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1"}


@respx.mock
async def test_vm_publish_happy_path(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(200, json={
            "kaId": "ka-1",
            "status": "confirmed",
            "ual": "did:dkg:base:84532/0xabc/1",
            "txHash": "0xtx",
            "merkleRoot": "0xroot",
            "authorAddress": "0xauthor",
            "blockNumber": 42,
        })
    )
    result = await client.vm_publish("ka-1", "cg-1", publish_epochs=2)
    assert result["status"] == "confirmed"
    assert result["ual"] == "did:dkg:base:84532/0xabc/1"
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1", "publishEpochs": 2}


@respx.mock
async def test_vm_publish_omits_publish_epochs_by_default(client):
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(200, json={"kaId": "ka-1", "status": "confirmed"})
    )
    await client.vm_publish("ka-1", "cg-1")
    body = json.loads(respx.calls.last.request.content)
    assert "publishEpochs" not in body


@respx.mock
async def test_vm_publish_207_partial_returned_not_raised(client):
    # 207 = KA minted but the context-graph binding failed: pass through.
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(207, json={
            "kaId": "ka-1", "status": "partial", "ual": "did:dkg:base:84532/0xabc/1",
        })
    )
    result = await client.vm_publish("ka-1", "cg-1")
    assert result["status"] == "partial"


@respx.mock
@pytest.mark.parametrize("code", ["VM_PUBLISH_PRECONDITION", "PUBLISH_NOT_FULL_SHARE"])
async def test_vm_publish_409_maps_precondition_error(client, code):
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(409, json={"code": code, "message": "not ready"})
    )
    with pytest.raises(DKGPublishPreconditionError) as exc:
        await client.vm_publish("ka-1", "cg-1")
    err = exc.value
    assert isinstance(err, DKGError)
    assert err.code == code
    assert err.status_code == 409
    assert "not ready" in str(err)


@respx.mock
async def test_vm_publish_generic_409_stays_status_error(client):
    # A 409 without a precondition code must not be mislabelled.
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(409, json={"error": "conflict"})
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.vm_publish("ka-1", "cg-1")
    assert not isinstance(exc.value, DKGPublishPreconditionError)


@respx.mock
async def test_vm_publish_400_unfunded_wallet_is_status_error(client):
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(400, json={
            "error": "insufficient funds",
            "wallets": [{"address": "0xa", "trac": "0", "native": "0"}],
        })
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.vm_publish("ka-1", "cg-1")
    assert exc.value.status_code == 400
    assert "insufficient funds" in exc.value.body


@respx.mock
async def test_vm_publish_502_tentative_failed_is_dkg_error(client):
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=httpx.Response(502, json={"error": "tentative-failed"})
    )
    with pytest.raises(DKGError):
        await client.vm_publish("ka-1", "cg-1")


@respx.mock
async def test_publish_direct_full_body(client):
    route = respx.post(f"{BASE}/api/knowledge-assets/publish").mock(
        return_value=httpx.Response(200, json={
            "mode": "direct",
            "kaId": "ka-9",
            "status": "confirmed",
            "kas": [{"tokenId": 1, "rootEntity": "ex:s"}],
            "txHash": "0xtx",
        })
    )
    result = await client.publish_direct(
        context_graph_id="cg-1",
        quads=[("ex:s", "ex:p", '"v"')],
        private_quads=[("ex:s", "ex:secret", '"w"')],
        access_policy="allowList",
        allowed_peers=["peer-1"],
        sub_graph_name="sg",
        publish_epochs=3,
    )
    assert result["mode"] == "direct"
    body = json.loads(route.calls.last.request.content)
    assert body["quads"] == [{"subject": "ex:s", "predicate": "ex:p", "object": '"v"'}]
    assert body["privateQuads"] == [{"subject": "ex:s", "predicate": "ex:secret", "object": '"w"'}]
    assert body["accessPolicy"] == "allowList"
    assert body["allowedPeers"] == ["peer-1"]
    assert body["subGraphName"] == "sg"
    assert body["publishEpochs"] == 3


@respx.mock
async def test_publish_direct_minimal_body(client):
    respx.post(f"{BASE}/api/knowledge-assets/publish").mock(
        return_value=httpx.Response(200, json={"mode": "direct", "kaId": "ka-9", "status": "confirmed"})
    )
    await client.publish_direct(context_graph_id="cg-1", quads=[VALID_QUAD])
    body = json.loads(respx.calls.last.request.content)
    assert set(body) == {"contextGraphId", "quads"}


@respx.mock
async def test_endorse_sends_ual(client):
    route = respx.post(f"{BASE}/api/endorse").mock(
        return_value=httpx.Response(200, json={"status": "queued", "trustLevel": "Endorsed"})
    )
    result = await client.endorse("cg-1", "did:dkg:base:84532/0xabc/1")
    assert result["trustLevel"] == "Endorsed"
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1", "ual": "did:dkg:base:84532/0xabc/1"}


@respx.mock
async def test_endorse_identity_mismatch_surfaces_status_error(client):
    respx.post(f"{BASE}/api/endorse").mock(
        return_value=httpx.Response(403, json={"error": "identity mismatch"})
    )
    with pytest.raises(DKGStatusError) as exc:
        await client.endorse("cg-1", "did:dkg:base:84532/0xabc/1")
    assert exc.value.status_code == 403


@respx.mock
async def test_request_verification_quorum_met(client):
    route = respx.post(f"{BASE}/api/verify").mock(
        return_value=httpx.Response(200, json={"status": "verified", "signatures": 3})
    )
    result = await client.request_verification(
        "cg-1", verifiable_memory_id="vm-1", batch_id="batch-1", required_signatures=3
    )
    assert result["status"] == "verified"
    body = json.loads(route.calls.last.request.content)
    assert body == {
        "contextGraphId": "cg-1",
        "verifiableMemoryId": "vm-1",
        "batchId": "batch-1",
        "requiredSignatures": 3,
    }


@respx.mock
@pytest.mark.parametrize("status", ["partial", "no_quorum"])
async def test_request_verification_409_returned_not_raised(client, status):
    # Callers poll/retry on partial quorum — the body is returned, not raised.
    respx.post(f"{BASE}/api/verify").mock(
        return_value=httpx.Response(409, json={"status": status, "signatures": 1})
    )
    result = await client.request_verification("cg-1", verifiable_memory_id="vm-1", batch_id="b-1")
    assert result["status"] == status


@respx.mock
async def test_request_verification_generic_409_raises(client):
    respx.post(f"{BASE}/api/verify").mock(
        return_value=httpx.Response(409, json={"error": "conflict"})
    )
    with pytest.raises(DKGStatusError):
        await client.request_verification("cg-1", verifiable_memory_id="vm-1", batch_id="b-1")


@respx.mock
async def test_kc_metadata(client):
    route = respx.get(f"{BASE}/api/kc/ka-1").mock(
        return_value=httpx.Response(200, json={"merkleRoot": "0xroot", "author": "0xauthor"})
    )
    result = await client.kc_metadata("ka-1")
    assert result["merkleRoot"] == "0xroot"
    assert route.called


@respx.mock
async def test_kc_author(client):
    respx.get(f"{BASE}/api/kc/ka-1/author").mock(
        return_value=httpx.Response(200, json={"author": "0xauthor", "attested": True})
    )
    result = await client.kc_author("ka-1")
    assert result["attested"] is True


@respx.mock
async def test_verify_batch_sends_expected_root(client):
    route = respx.post(f"{BASE}/api/shared-memory/verify-batch").mock(
        return_value=httpx.Response(200, json={"match": True, "computedRoot": "0xroot"})
    )
    result = await client.verify_batch(
        quads=[("ex:s", "ex:p", '"v"')],
        expected_merkle_root="0xroot",
        private_roots=["0xpriv"],
    )
    assert result["match"] is True
    body = json.loads(route.calls.last.request.content)
    assert body == {
        "quads": [{"subject": "ex:s", "predicate": "ex:p", "object": '"v"'}],
        "expectedMerkleRoot": "0xroot",
        "privateRoots": ["0xpriv"],
    }


@respx.mock
async def test_query_sends_view_and_min_trust(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await client.query(
        sparql="SELECT * WHERE { ?s ?p ?o }",
        context_graph_id="cg-1",
        view="verifiable-memory",
        min_trust="endorsed",
    )
    body = json.loads(respx.calls.last.request.content)
    assert body["view"] == "verifiable-memory"
    assert body["minTrust"] == "endorsed"


@respx.mock
async def test_query_min_trust_zero_is_sent(client):
    # `is not None` semantics: SelfAttested (0) is falsy but must be sent.
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await client.query(sparql="SELECT * WHERE { ?s ?p ?o }", min_trust=0)
    body = json.loads(respx.calls.last.request.content)
    assert body["minTrust"] == 0


@respx.mock
async def test_query_omits_view_and_min_trust_by_default(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await client.query(sparql="SELECT * WHERE { ?s ?p ?o }")
    body = json.loads(respx.calls.last.request.content)
    assert "view" not in body
    assert "minTrust" not in body
