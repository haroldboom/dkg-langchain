"""Unit tests for DKGMemory."""

import pytest

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_dkg.client import DKGClient
from langchain_dkg.memory import DKGMemory


BASE = "http://localhost:9200"
TOKEN = "test-token"


@pytest.fixture
def client():
    return DKGClient(base_url=BASE, token=TOKEN)


def test_get_history_returns_scoped_history(client):
    mem = DKGMemory(context_graph_id="test-graph", client=client)
    hist = mem.get_history("user-42")
    # context_graph_id is shared; session scoped via session_uri
    assert hist.context_graph_id == "test-graph"
    assert hist.session_uri == "urn:session:user-42"


def test_get_history_different_sessions_isolated(client):
    mem = DKGMemory(context_graph_id="test-graph", client=client)
    h1 = mem.get_history("alice")
    h2 = mem.get_history("bob")
    assert h1.session_uri != h2.session_uri


def test_wrap_returns_runnable_with_history(client):
    from langchain_core.runnables import RunnableLambda
    identity = RunnableLambda(lambda x: x)
    mem = DKGMemory(context_graph_id="test-graph", client=client)
    wrapped = mem.wrap(identity)
    assert isinstance(wrapped, RunnableWithMessageHistory)


def test_wrap_chain_classmethod(client):
    from langchain_core.runnables import RunnableLambda
    identity = RunnableLambda(lambda x: x)
    wrapped = DKGMemory.wrap_chain(
        identity,
        context_graph_id="test-graph",
        client=client,
    )
    assert isinstance(wrapped, RunnableWithMessageHistory)


def test_search_limit_propagated(client):
    mem = DKGMemory(context_graph_id="cg", client=client, search_limit=3)
    hist = mem.get_history("s1")
    assert hist.search_limit == 3


def test_search_query_propagated(client):
    mem = DKGMemory(context_graph_id="cg", client=client, search_query="wheat harvest plans")
    hist = mem.get_history("s1")
    assert hist.search_query == "wheat harvest plans"


def test_layer_defaults_to_wm(client):
    # Private-by-default: history turns go to Working Memory unless overridden.
    mem = DKGMemory(context_graph_id="cg", client=client)
    assert mem.layer == "wm"
    hist = mem.get_history("s1")
    assert hist.layer == "wm"


def test_layer_override_propagated(client):
    mem = DKGMemory(context_graph_id="cg", client=client, layer="swm")
    hist = mem.get_history("s1")
    assert hist.layer == "swm"
