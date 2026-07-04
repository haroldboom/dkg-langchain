"""Smoke tests for the LangChain agent tools (make_dkg_tools).

These tests are synchronous on purpose: the tools are sync callables (they
bridge to the async client via run_sync), which is how a LangChain agent
executor invokes them.
"""

import json

import httpx
import respx

from langchain_dkg.client import DKGClient
from langchain_dkg.tools import make_dkg_tools


BASE = "http://localhost:9200"
TOKEN = "test-token"


def _tools():
    client = DKGClient(base_url=BASE, token=TOKEN)
    tools = make_dkg_tools(client, "cg-1")
    return {t.name: t for t in tools}


def test_make_dkg_tools_returns_three_named_tools():
    tools = _tools()
    assert set(tools) == {"dkg_endorse", "dkg_verified_search", "dkg_publish_note"}
    for t in tools.values():
        # Docstrings become the LLM-facing tool descriptions.
        assert t.description.strip()


@respx.mock
def test_dkg_endorse_invokes_endorse_route():
    route = respx.post(f"{BASE}/api/endorse").mock(
        return_value=httpx.Response(200, json={"status": "queued", "trustLevel": "Endorsed"})
    )
    result = _tools()["dkg_endorse"].invoke({"ual": "did:dkg:base:84532/0xabc/1"})
    assert json.loads(result)["trustLevel"] == "Endorsed"
    body = json.loads(route.calls.last.request.content)
    assert body == {"contextGraphId": "cg-1", "ual": "did:dkg:base:84532/0xabc/1"}


@respx.mock
def test_dkg_verified_search_queries_vm_view():
    route = respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": [
            {"subject": "ex:farm", "predicate": "http://schema.org/name", "object": "The Farm"},
        ]}})
    )
    result = _tools()["dkg_verified_search"].invoke({"query": "farm"})
    assert "ex:farm http://schema.org/name The Farm" in result
    body = json.loads(route.calls.last.request.content)
    assert body["view"] == "verifiable-memory"
    assert body["minTrust"] == "endorsed"  # tool default
    assert body["contextGraphId"] == "cg-1"


@respx.mock
def test_dkg_verified_search_honors_min_trust_and_empty_results():
    respx.post(f"{BASE}/api/query").mock(
        return_value=httpx.Response(200, json={"result": {"bindings": []}})
    )
    result = _tools()["dkg_verified_search"].invoke(
        {"query": "farm", "min_trust": "consensus_verified"}
    )
    assert "No results" in result
    body = json.loads(respx.calls.last.request.content)
    assert body["minTrust"] == "consensus_verified"


@respx.mock
def test_dkg_publish_note_publishes_schema_org_quads():
    route = respx.post(f"{BASE}/api/knowledge-assets/publish").mock(
        return_value=httpx.Response(200, json={
            "mode": "direct", "kaId": "ka-7", "status": "confirmed", "txHash": "0xtx",
        })
    )
    result = _tools()["dkg_publish_note"].invoke(
        {"title": "Harvest result", "content": "Yield was 4.2 t/ha."}
    )
    assert json.loads(result)["kaId"] == "ka-7"
    body = json.loads(route.calls.last.request.content)
    assert body["contextGraphId"] == "cg-1"
    quads = body["quads"]
    by_predicate = {q["predicate"]: q for q in quads}
    assert by_predicate["http://schema.org/name"]["object"] == '"Harvest result"'
    assert by_predicate["http://schema.org/text"]["object"] == '"Yield was 4.2 t/ha."'
    assert "http://schema.org/dateCreated" in by_predicate
    # All quads describe the same generated note subject.
    assert len({q["subject"] for q in quads}) == 1
