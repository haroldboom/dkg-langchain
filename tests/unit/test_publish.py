"""Unit tests for the high-level promotion pipeline (publish_to_verified / turn_to_quads)."""

import json

import httpx
import pytest
import respx

from langchain_dkg.client import DKGClient, DKGPublishPreconditionError
from langchain_dkg.publish import publish_to_verified, turn_to_quads


BASE = "http://localhost:9200"
TOKEN = "test-token"

SEAL = {
    "assertionUri": "did:dkg:assertion:ka-1",
    "merkleRoot": "0xseal-root",
    "authorAddress": "0xauthor",
    "schemeVersion": 1,
    "chainId": 84532,
    "kav10Address": "0xkav10",
    "eip712Digest": "0xdigest",
}

QUADS = [{"subject": "ex:s", "predicate": "ex:p", "object": '"v"'}]


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


def _mock_promotion_routes(vm_publish_response: httpx.Response) -> None:
    """Mock the five-route promotion chain for assertion name ``ka-1``."""
    respx.post(f"{BASE}/api/knowledge-assets").mock(
        return_value=httpx.Response(200, json={
            "name": "ka-1", "assertionUri": "did:x", "alreadyExists": False, "status": "draft-open",
        })
    )
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/wm/write").mock(
        return_value=httpx.Response(200, json={"written": 1})
    )
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/wm/finalize").mock(
        return_value=httpx.Response(200, json=SEAL)
    )
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/swm/share-async").mock(
        return_value=httpx.Response(200, json={"jobId": "job-1", "state": "queued"})
    )
    respx.get(f"{BASE}/api/knowledge-assets/swm/share-jobs/job-1").mock(
        return_value=httpx.Response(200, json={"jobId": "job-1", "state": "succeeded"})
    )
    respx.post(f"{BASE}/api/knowledge-assets/ka-1/vm/publish").mock(
        return_value=vm_publish_response
    )


@respx.mock
async def test_publish_to_verified_orchestration(client):
    _mock_promotion_routes(
        httpx.Response(200, json={
            "kaId": "ka-1",
            "status": "confirmed",
            "ual": "did:dkg:base:84532/0xabc/1",
            "txHash": "0xtx",
            "merkleRoot": "0xchain-root",
        })
    )
    result = await publish_to_verified(client, "cg-1", "ka-1", QUADS)

    # The publish response wins on overlap; the seal fields are retained.
    assert result["status"] == "confirmed"
    assert result["txHash"] == "0xtx"
    assert result["merkleRoot"] == "0xchain-root"
    assert result["eip712Digest"] == "0xdigest"
    assert result["kav10Address"] == "0xkav10"

    # Full chain, in order: create → write → finalize → share (+poll) → publish.
    called = [(c.request.method, c.request.url.path) for c in respx.calls]
    assert called == [
        ("POST", "/api/knowledge-assets"),
        ("POST", "/api/knowledge-assets/ka-1/wm/write"),
        ("POST", "/api/knowledge-assets/ka-1/wm/finalize"),
        ("POST", "/api/knowledge-assets/ka-1/swm/share-async"),
        ("GET", "/api/knowledge-assets/swm/share-jobs/job-1"),
        ("POST", "/api/knowledge-assets/ka-1/vm/publish"),
    ]

    write_body = json.loads(respx.calls[1].request.content)
    assert write_body == {"contextGraphId": "cg-1", "quads": QUADS}


@respx.mock
async def test_publish_to_verified_forwards_options(client):
    _mock_promotion_routes(
        httpx.Response(200, json={"kaId": "ka-1", "status": "confirmed"})
    )
    await publish_to_verified(
        client, "cg-1", "ka-1", QUADS, sub_graph_name="sg", publish_epochs=2
    )
    create_body = json.loads(respx.calls[0].request.content)
    assert create_body["subGraphName"] == "sg"
    share_body = json.loads(respx.calls[3].request.content)
    assert share_body["subGraphName"] == "sg"
    publish_body = json.loads(respx.calls[-1].request.content)
    assert publish_body["publishEpochs"] == 2


@respx.mock
async def test_publish_to_verified_207_partial_returned(client):
    # A 207 partial publish (KA minted, binding failed) is merged and
    # returned like a 200, not raised.
    _mock_promotion_routes(
        httpx.Response(207, json={"kaId": "ka-1", "status": "partial"})
    )
    result = await publish_to_verified(client, "cg-1", "ka-1", QUADS)
    assert result["status"] == "partial"
    assert result["eip712Digest"] == "0xdigest"


@respx.mock
async def test_publish_to_verified_precondition_gets_helpful_message(client):
    _mock_promotion_routes(
        httpx.Response(409, json={"code": "PUBLISH_NOT_FULL_SHARE", "message": "share incomplete"})
    )
    with pytest.raises(DKGPublishPreconditionError) as exc:
        await publish_to_verified(client, "cg-1", "ka-1", QUADS)
    err = exc.value
    assert err.code == "PUBLISH_NOT_FULL_SHARE"
    # The re-raised error explains that the share has not landed yet.
    assert "share" in str(err).lower()
    assert "share_poll_timeout" in str(err)
    assert isinstance(err.__cause__, DKGPublishPreconditionError)


# ---------------------------------------------------------------------------
# turn_to_quads — minimal schema.org shape for a conversation turn
# ---------------------------------------------------------------------------

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _by_predicate(quads, predicate):
    return [q for q in quads if q["predicate"] == predicate]


def test_turn_to_quads_shape():
    quads = turn_to_quads("urn:turn:1", "**Human:** Hello")
    assert all(set(q) == {"subject", "predicate", "object"} for q in quads)
    assert {"subject": "urn:turn:1", "predicate": RDF_TYPE,
            "object": "http://schema.org/Message"} in quads
    (text,) = _by_predicate(quads, "http://schema.org/text")
    assert text["subject"] == "urn:turn:1"
    assert text["object"] == '"**Human:** Hello"'
    (created,) = _by_predicate(quads, "http://schema.org/dateCreated")
    assert created["object"].startswith('"20')
    # No session quads without a session_uri.
    assert not _by_predicate(quads, "http://schema.org/isPartOf")
    assert len(quads) == 3


def test_turn_to_quads_with_session():
    quads = turn_to_quads("urn:turn:1", "hi", session_uri="urn:session:s1")
    assert {"subject": "urn:turn:1", "predicate": "http://schema.org/isPartOf",
            "object": "urn:session:s1"} in quads
    assert {"subject": "urn:session:s1", "predicate": RDF_TYPE,
            "object": "http://schema.org/Conversation"} in quads
    assert len(quads) == 5


def test_turn_to_quads_escapes_literals():
    quads = turn_to_quads("urn:turn:1", 'line one\nsays "hi"')
    (text,) = _by_predicate(quads, "http://schema.org/text")
    assert text["object"] == '"line one\\nsays \\"hi\\""'
