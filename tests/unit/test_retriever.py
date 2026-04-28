"""Unit tests for DKGRetriever."""

import pytest
import respx
import httpx

from langchain_dkg.client import DKGClient
from langchain_dkg.retriever import DKGRetriever


BASE = "http://localhost:9200"
TOKEN = "test-token"

BINDINGS = [
    {
        "subject":   {"value": "did:dkg:entity:farm"},
        "predicate": {"value": "http://schema.org/name"},
        "object":    {"value": "The Farm"},
    },
    {
        "subject":   {"value": "did:dkg:entity:farm"},
        "predicate": {"value": "http://schema.org/description"},
        "object":    {"value": "A grain farming operation in Australia"},
    },
]


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


@pytest.fixture
def retriever(client):
    return DKGRetriever(client=client, limit=5)


@respx.mock
async def test_get_documents(retriever):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": BINDINGS}})
    )
    docs = await retriever._aget_relevant_documents("farm")
    assert len(docs) == 2
    assert "did:dkg:entity:farm" in docs[0].page_content
    assert docs[0].metadata["source"] == "dkg-v10"


@respx.mock
async def test_sparql_contains_query_term(retriever):
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    await retriever._aget_relevant_documents("wheat prices")
    import json
    body = json.loads(route.calls.last.request.content)
    assert "wheat prices" in body["sparql"]


@respx.mock
async def test_custom_sparql_template(client):
    custom = 'SELECT ?s WHERE {{ ?s a <http://schema.org/Farm> }} LIMIT {limit}'
    retriever = DKGRetriever(client=client, sparql_template=custom, limit=10)
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    await retriever._aget_relevant_documents("ignored")
    import json
    body = json.loads(route.calls.last.request.content)
    assert "schema.org/Farm" in body["sparql"]
    assert "LIMIT 10" in body["sparql"]


@respx.mock
async def test_empty_results(retriever):
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"results": {"bindings": []}})
    )
    docs = await retriever._aget_relevant_documents("nonexistent topic")
    assert docs == []
