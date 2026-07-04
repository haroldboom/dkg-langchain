"""Unit tests for DKGRetriever."""

import json

import httpx
import pytest
import respx

from langchain_dkg.client import DKGClient
from langchain_dkg.retriever import DKGRetriever


BASE = "http://localhost:9200"
TOKEN = "test-token"

# Old node builds: SPARQL-standard cells ({"type": ..., "value": ...}).
STANDARD_BINDINGS = [
    {
        "subject":   {"type": "uri", "value": "did:dkg:entity:farm"},
        "predicate": {"type": "uri", "value": "http://schema.org/name"},
        "object":    {"type": "literal", "value": "The Farm"},
    },
    {
        "subject":   {"type": "uri", "value": "did:dkg:entity:farm"},
        "predicate": {"type": "uri", "value": "http://schema.org/description"},
        "object":    {"type": "literal", "value": "A grain farming operation in Australia"},
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
    return DKGRetriever(client=client, limit=5)


@respx.mock
async def test_get_documents_old_shape(retriever):
    # Old builds: {"results": {"bindings": [...]}} with {type, value} cells.
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": STANDARD_BINDINGS}})
    )
    docs = await retriever.ainvoke("farm")
    assert len(docs) == 2
    assert "did:dkg:entity:farm" in docs[0].page_content
    assert docs[0].metadata["source"] == "dkg-v10"
    assert docs[0].metadata["object"] == "The Farm"


@respx.mock
async def test_get_documents_new_shape_plain_strings(retriever):
    # Current builds: {"result": {"bindings": [...]}} with plain string cells.
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": PLAIN_BINDINGS}})
    )
    docs = await retriever.ainvoke("farm")
    assert len(docs) == 1
    assert docs[0].page_content == "did:dkg:entity:farm http://schema.org/name The Farm"


@respx.mock
async def test_get_documents_list_shaped_result(retriever):
    # Defensive: some responses carry the bindings list directly.
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": PLAIN_BINDINGS})
    )
    docs = await retriever.ainvoke("farm")
    assert len(docs) == 1


@respx.mock
@pytest.mark.parametrize("payload", [
    {},
    {"result": None},
    {"result": "weird"},
    {"results": {}},
    {"result": {"bindings": None}},
])
async def test_get_documents_missing_or_odd_shapes_no_crash(retriever, payload):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json=payload)
    )
    docs = await retriever.ainvoke("anything")
    assert docs == []


@respx.mock
async def test_metadata_layer_reflects_query_scope(client):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": PLAIN_BINDINGS}})
    )
    workspace_docs = await DKGRetriever(client=client, include_workspace=True).ainvoke("farm")
    assert workspace_docs[0].metadata["layer"] == "workspace"
    published_docs = await DKGRetriever(client=client, include_workspace=False).ainvoke("farm")
    assert published_docs[0].metadata["layer"] == "published"


@respx.mock
async def test_sparql_contains_query_term(retriever):
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    await retriever.ainvoke("wheat prices")
    body = json.loads(route.calls.last.request.content)
    assert "wheat prices" in body["sparql"]


@respx.mock
async def test_context_graph_id_forwarded(client):
    retriever = DKGRetriever(client=client, context_graph_id="cg-42")
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    await retriever.ainvoke("farm")
    body = json.loads(route.calls.last.request.content)
    assert body["contextGraphId"] == "cg-42"


@respx.mock
async def test_custom_sparql_template(client):
    custom = 'SELECT ?s WHERE {{ ?s a <http://schema.org/Farm> }} LIMIT {limit}'
    retriever = DKGRetriever(client=client, sparql_template=custom, limit=10)
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    await retriever.ainvoke("ignored")
    body = json.loads(route.calls.last.request.content)
    assert "schema.org/Farm" in body["sparql"]
    assert "LIMIT 10" in body["sparql"]


def test_with_sparql_copies_configuration(retriever):
    custom = "SELECT ?s WHERE {{ ?s ?p ?o }} LIMIT {limit}"
    copy = retriever.with_sparql(custom)
    assert copy.sparql_template == custom
    assert copy is not retriever
    assert copy.client is retriever.client
    assert copy.limit == retriever.limit
    # Original untouched.
    assert retriever.sparql_template != custom


@respx.mock
async def test_empty_results(retriever):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    docs = await retriever.ainvoke("nonexistent topic")
    assert docs == []
