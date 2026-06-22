"""HTTP client for the DKG v10 node API."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

# A quad accepted by the write methods: either a mapping with
# subject/predicate/object (and optional graph) keys, or a 3-/4-tuple in
# (subject, predicate, object[, graph]) order.
QuadLike = Mapping[str, str] | Sequence[str]


def _normalize_quad(quad: QuadLike) -> dict[str, str]:
    """Normalize a quad into the node's ``{subject, predicate, object[, graph]}`` shape.

    The DKG v10 node's ``/api/shared-memory/write`` and ``/api/assertion/{name}/write``
    endpoints require each quad to be an object with string ``subject``, ``predicate``,
    and ``object`` terms (and an optional string ``graph``); since node rc.19 they
    reject string-shaped quads with HTTP 400. Validate client-side so callers get a
    clear error instead of an opaque 400.
    """
    if isinstance(quad, str):
        raise ValueError(
            "Quads must be structured, not raw strings. Pass a mapping "
            '{"subject": ..., "predicate": ..., "object": ...} (optional "graph") '
            "or a (subject, predicate, object[, graph]) tuple."
        )
    if isinstance(quad, Mapping):
        subject, predicate, obj = quad.get("subject"), quad.get("predicate"), quad.get("object")
        graph = quad.get("graph")
    elif isinstance(quad, Sequence):
        if len(quad) not in (3, 4):
            raise ValueError(
                "Quad tuple must have 3 or 4 elements "
                f"(subject, predicate, object[, graph]); got {len(quad)}"
            )
        subject, predicate, obj = quad[0], quad[1], quad[2]
        graph = quad[3] if len(quad) == 4 else None
    else:
        raise ValueError(
            "Each quad must be a mapping with subject/predicate/object keys "
            f"or a 3-/4-tuple; got {type(quad).__name__}"
        )
    for field, value in (("subject", subject), ("predicate", predicate), ("object", obj)):
        if not isinstance(value, str) or not value:
            raise ValueError(f'Quad "{field}" must be a non-empty string; got {value!r}')
    normalized = {"subject": subject, "predicate": predicate, "object": obj}
    if graph is not None:
        if not isinstance(graph, str):
            raise ValueError(f'Quad "graph" must be a string; got {graph!r}')
        normalized["graph"] = graph
    return normalized


class DKGClient:
    """Thin async wrapper around the DKG v10 HTTP API (port 9200).

    All methods raise httpx.HTTPStatusError on non-2xx responses.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DKG_BASE_URL", "http://localhost:9200")).rstrip("/")
        token = token or os.environ.get("DKG_TOKEN", "")
        if not token:
            raise ValueError(
                "DKG bearer token required. Pass token= or set DKG_TOKEN env var."
            )
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Context Graphs
    # ------------------------------------------------------------------

    async def create_context_graph(self, name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/context-graph/create",
                headers=self._headers,
                json={"name": name},
            )
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Working Memory — conversation turns
    # ------------------------------------------------------------------

    async def memory_turn(
        self,
        context_graph_id: str,
        markdown: str,
        session_uri: str | None = None,
        layer: str | None = None,
        sub_graph_name: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a conversation turn as a tri-modal Knowledge Asset.

        Args:
            context_graph_id: Context Graph that scopes this turn.
            markdown: The turn content as Markdown text.
            session_uri: Optional IRI linking related turns into a session.
            layer: Target layer — "wm" for Working Memory (private), "swm" for
                Shared Working Memory (gossiped). Defaults to "swm".
            sub_graph_name: Optional sub-graph within the Context Graph.

        Returns dict with keys: turnUri, fileHash, layer, graph,
            structuralTripleCount, semanticTripleCount, totalQuads, embeddingId.
        """
        body: dict[str, Any] = {
            "contextGraphId": context_graph_id,
            "markdown": markdown,
        }
        if session_uri:
            body["sessionUri"] = session_uri
        if layer:
            body["layer"] = layer
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/memory/turn",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def memory_search(
        self,
        context_graph_id: str,
        query: str,
        limit: int = 20,
        memory_layers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Tri-modal search across vector, SPARQL, and text stores.

        Returns dict with keys: query, contextGraphId, resultCount, results.
        Each result has: entityUri, label, sources, similarity, sourceFile,
            snippet, memoryLayer.
        """
        body: dict[str, Any] = {
            "contextGraphId": context_graph_id,
            "query": query,
            "limit": limit,
        }
        if memory_layers:
            body["memoryLayers"] = memory_layers
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/memory/search",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Assertions (Working Memory)
    # ------------------------------------------------------------------

    async def assertion_create(
        self,
        context_graph_id: str,
        name: str,
        sub_graph_name: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"contextGraphId": context_graph_id, "name": name}
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/assertion/create",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def assertion_write(
        self,
        name: str,
        context_graph_id: str,
        quads: list[QuadLike],
        sub_graph_name: str | None = None,
    ) -> dict[str, Any]:
        """Write RDF quads into a named Working Memory assertion.

        Quads follow the same shape as :meth:`shared_memory_write` — a mapping with
        ``subject``/``predicate``/``object`` (optional ``graph``) keys, or a 3-/4-tuple.
        """
        body: dict[str, Any] = {
            "contextGraphId": context_graph_id,
            "quads": [_normalize_quad(q) for q in quads],
        }
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/assertion/{name}/write",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def assertion_promote(
        self,
        name: str,
        context_graph_id: str,
        entities: list[str] | None = None,
        sub_graph_name: str | None = None,
    ) -> dict[str, Any]:
        """Promote a Working Memory assertion to Shared Working Memory (SHARE)."""
        body: dict[str, Any] = {"contextGraphId": context_graph_id}
        if entities:
            body["entities"] = entities
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/assertion/{name}/promote",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def assertion_history(
        self, name: str, context_graph_id: str
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.get(
                f"{self.base_url}/api/assertion/{name}/history",
                headers=self._headers,
                params={"contextGraphId": context_graph_id},
            )
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # SPARQL query
    # ------------------------------------------------------------------

    async def query(
        self,
        sparql: str,
        paranet_id: str | None = None,
        graph_suffix: str | None = None,
        include_workspace: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "sparql": sparql,
            "includeWorkspace": include_workspace,
        }
        if paranet_id:
            body["paranetId"] = paranet_id
        if graph_suffix:
            body["graphSuffix"] = graph_suffix
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/query",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Shared Memory
    # ------------------------------------------------------------------

    async def shared_memory_write(
        self, quads: list[QuadLike], context_graph_id: str | None = None
    ) -> dict[str, Any]:
        """Write loose RDF quads directly to Shared Working Memory.

        Each quad may be a mapping with ``subject``/``predicate``/``object`` (and
        optional ``graph``) keys, or a 3-/4-tuple in (subject, predicate,
        object[, graph]) order. ``subject``/``predicate`` must be absolute IRIs and
        ``object`` an absolute IRI or a quoted RDF literal (e.g. ``'"hello"'``).
        """
        body: dict[str, Any] = {"quads": [_normalize_quad(q) for q in quads]}
        if context_graph_id:
            body["contextGraphId"] = context_graph_id
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/shared-memory/write",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    async def shared_memory_publish(
        self, context_graph_id: str | None = None
    ) -> dict[str, Any]:
        """Promote Shared Working Memory to Verified Memory (PUBLISH, costs TRAC)."""
        body: dict[str, Any] = {}
        if context_graph_id:
            body["contextGraphId"] = context_graph_id
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            r = await http.post(
                f"{self.base_url}/api/shared-memory/publish",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return True if the node responds and the token is valid."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                r = await http.get(
                    f"{self.base_url}/api/agents", headers=self._headers
                )
                return r.status_code == 200
        except Exception:
            return False
