"""Unit tests for DKGVerifiedRetriever and the TrustLevel gradient."""

import json

import httpx
import pytest
import respx

from langchain_dkg.client import DKGClient
from langchain_dkg.trust import TrustLevel
from langchain_dkg.verified_retriever import DKGVerifiedRetriever


BASE = "http://localhost:9200"
TOKEN = "test-token"

# Old node builds: SPARQL-standard cells ({"type": ..., "value": ...}).
STANDARD_BINDINGS = [
    {
        "subject":   {"type": "uri", "value": "did:dkg:entity:farm"},
        "predicate": {"type": "uri", "value": "http://schema.org/name"},
        "object":    {"type": "literal", "value": "The Farm"},
    },
]

# Current node builds: plain string cells.
PLAIN_BINDINGS = [
    {
        "subject": "did:dkg:entity:farm",
        "predicate": "http://schema.org/name",
        "object": "The Farm",
    },
]


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


@pytest.fixture
def retriever(client):
    return DKGVerifiedRetriever(client=client, context_graph_id="cg-1", limit=5)


def test_trust_level_values():
    assert TrustLevel.SELF_ATTESTED == 0
    assert TrustLevel.ENDORSED == 1
    assert TrustLevel.PARTIALLY_VERIFIED == 2
    assert TrustLevel.CONSENSUS_VERIFIED == 3


def test_trust_level_exported_from_package():
    import langchain_dkg

    assert langchain_dkg.TrustLevel is TrustLevel
    assert langchain_dkg.DKGVerifiedRetriever is DKGVerifiedRetriever


@respx.mock
async def test_query_body_scopes_verifiable_memory_view(retriever):
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await retriever.ainvoke("farm")
    body = json.loads(route.calls.last.request.content)
    assert body["view"] == "verifiable-memory"
    assert body["minTrust"] == TrustLevel.SELF_ATTESTED
    assert body["contextGraphId"] == "cg-1"
    # Verifiable Memory is published content — the workspace is out of scope.
    assert body["includeWorkspace"] is False
    assert "farm" in body["sparql"]


@respx.mock
async def test_min_trust_string_forwarded(client):
    retriever = DKGVerifiedRetriever(client=client, context_graph_id="cg-1", min_trust="endorsed")
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await retriever.ainvoke("farm")
    body = json.loads(respx.calls.last.request.content)
    assert body["minTrust"] == "endorsed"


@respx.mock
async def test_min_trust_enum_forwarded_as_int(client):
    retriever = DKGVerifiedRetriever(
        client=client, context_graph_id="cg-1", min_trust=TrustLevel.CONSENSUS_VERIFIED
    )
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await retriever.ainvoke("farm")
    body = json.loads(respx.calls.last.request.content)
    assert body["minTrust"] == 3


@respx.mock
async def test_documents_old_shape(retriever):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": STANDARD_BINDINGS}})
    )
    docs = await retriever.ainvoke("farm")
    assert len(docs) == 1
    assert docs[0].page_content == "did:dkg:entity:farm http://schema.org/name The Farm"
    assert docs[0].metadata["source"] == "dkg-v10-vm"
    assert docs[0].metadata["object"] == "The Farm"
    assert docs[0].metadata["min_trust"] == TrustLevel.SELF_ATTESTED


@respx.mock
async def test_documents_new_shape_plain_strings(client):
    retriever = DKGVerifiedRetriever(client=client, context_graph_id="cg-1", min_trust="endorsed")
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": PLAIN_BINDINGS}})
    )
    docs = await retriever.ainvoke("farm")
    assert len(docs) == 1
    assert docs[0].metadata["subject"] == "did:dkg:entity:farm"
    assert docs[0].metadata["min_trust"] == "endorsed"


@respx.mock
@pytest.mark.parametrize("payload", [
    {},
    {"result": None},
    {"result": "weird"},
    {"results": {}},
    {"result": {"bindings": None}},
])
async def test_missing_or_odd_shapes_no_crash(retriever, payload):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json=payload)
    )
    docs = await retriever.ainvoke("anything")
    assert docs == []


def test_with_sparql_copies_configuration(retriever):
    custom = "SELECT ?s WHERE {{ ?s ?p ?o }} LIMIT {limit}"
    copy = retriever.with_sparql(custom)
    assert copy.sparql_template == custom
    assert copy is not retriever
    assert copy.client is retriever.client
    assert copy.context_graph_id == retriever.context_graph_id


def test_context_graph_id_is_required(client):
    with pytest.raises(Exception):
        DKGVerifiedRetriever(client=client)
