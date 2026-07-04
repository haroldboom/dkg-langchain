"""HTTP client for the DKG v10 node API."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

# A quad accepted by the write methods: either a mapping with
# subject/predicate/object (and optional graph) keys, or a 3-/4-tuple in
# (subject, predicate, object[, graph]) order.
QuadLike = Mapping[str, str] | Sequence[str]

# Excerpt length for response bodies stored on exceptions.
_BODY_EXCERPT_LEN = 1000


class DKGError(Exception):
    """Base class for errors raised by :class:`DKGClient`."""


class DKGConnectionError(DKGError):
    """The DKG node could not be reached (transport failure or timeout)."""


class DKGStatusError(DKGError):
    """The DKG node returned a non-2xx HTTP status.

    Attributes:
        status_code: The HTTP status code.
        body: An excerpt of the response body (may be empty).
    """

    def __init__(self, message: str, *, status_code: int, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body[:_BODY_EXCERPT_LEN]


class CuratorAckError(DKGError):
    """A SHARE/PUBLISH write was not accepted by the Context Graph curator.

    The DKG v10 node's OT-RFC-49 curator-leader model gates writes to Shared
    Working Memory on confirmation from the curator (the authoritative replica).
    When that confirmation does not arrive (HTTP 503) or the curator rejects the
    write (HTTP 409), the node returns a structured ``code`` instead of a generic
    error; this exception surfaces it as an actionable, catchable failure.

    Attributes:
        code: The node's machine code (``CURATOR_UNCONFIRMED`` / ``CURATOR_REJECTED``).
        curator_delivery: The node's ``curatorDelivery`` field, if present.
        context_graph_id: The originating Context Graph id, if the node reported it.
        status_code: The HTTP status code, if the failure came from an HTTP response.
        body: The parsed response body (or job view), if available.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        curator_delivery: str | None = None,
        context_graph_id: str | None = None,
        status_code: int | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.curator_delivery = curator_delivery
        self.context_graph_id = context_graph_id
        self.status_code = status_code
        self.body = body


class CuratorUnconfirmedError(CuratorAckError):
    """HTTP 503 ``CURATOR_UNCONFIRMED`` — the curator did not confirm; nothing was persisted."""


class CuratorRejectedError(CuratorAckError):
    """HTTP 409 ``CURATOR_REJECTED`` — the curator actively rejected the write."""


def _raise_for_curator_ack(r: httpx.Response) -> None:
    """Map the node's OT-RFC-49 curator-ack failures to typed exceptions.

    The SHARE/PUBLISH write paths return ``503 CURATOR_UNCONFIRMED`` or
    ``409 CURATOR_REJECTED`` (node rc.19+). Translate those into
    :class:`CuratorUnconfirmedError` / :class:`CuratorRejectedError` so callers can
    catch them precisely; any other status (including a generic 503/409 without the
    curator ``code``) is left for the caller's status handling.
    """
    if r.status_code not in (409, 503):
        return
    try:
        body = r.json()
    except Exception:
        return
    if not isinstance(body, dict):
        return
    code = body.get("code")
    if code not in ("CURATOR_UNCONFIRMED", "CURATOR_REJECTED"):
        return
    exc_cls = CuratorUnconfirmedError if code == "CURATOR_UNCONFIRMED" else CuratorRejectedError
    raise exc_cls(
        body.get("error") or code,
        code=code,
        curator_delivery=body.get("curatorDelivery"),
        context_graph_id=body.get("contextGraphId"),
        status_code=r.status_code,
        body=body,
    )


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
    assert isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str)
    normalized = {"subject": subject, "predicate": predicate, "object": obj}
    if graph is not None:
        if not isinstance(graph, str):
            raise ValueError(f'Quad "graph" must be a string; got {graph!r}')
        normalized["graph"] = graph
    return normalized


def _job_error_text(job: dict[str, Any]) -> str:
    """Collect human/machine error fields from an async promote job view."""
    parts: list[str] = []
    for key in ("code", "error", "reason"):
        value = job.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    for last_error in (job.get("lastError"), (job.get("attempt") or {}).get("lastError")):
        if isinstance(last_error, dict):
            for key in ("code", "message"):
                value = last_error.get(key)
                if isinstance(value, str) and value and value not in parts:
                    parts.append(value)
    return " | ".join(parts)


class DKGClient:
    """Thin async wrapper around the DKG v10 HTTP API (port 9200).

    Methods raise :class:`DKGStatusError` on non-2xx responses and
    :class:`DKGConnectionError` when the node cannot be reached. The
    SHARE/PUBLISH write paths additionally raise CuratorUnconfirmedError /
    CuratorRejectedError (subclasses of CuratorAckError) for the node's
    OT-RFC-49 curator-ack failures.

    The client keeps a pooled ``httpx.AsyncClient`` per event loop. Call
    :meth:`aclose` (or use ``async with``) to release the pool explicitly.
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
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_http(self) -> httpx.AsyncClient:
        """Return a pooled AsyncClient bound to the current event loop.

        If the running loop changed since the client was created (e.g. the
        sync wrappers spawn a fresh loop per call), a new client is created;
        the old one is abandoned rather than closed, because closing it from
        a different loop is unsafe.
        """
        loop = asyncio.get_running_loop()
        if self._http is None or self._http_loop is not loop:
            self._http = httpx.AsyncClient(timeout=self._timeout)
            self._http_loop = loop
        return self._http

    async def aclose(self) -> None:
        """Close the pooled HTTP client, if it belongs to the current loop."""
        http, self._http, loop, self._http_loop = self._http, None, self._http_loop, None
        if http is not None and loop is asyncio.get_running_loop():
            await http.aclose()

    async def __aenter__(self) -> "DKGClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        curator_ack: bool = False,
    ) -> dict[str, Any]:
        """Issue one HTTP request; map transport/status failures to DKG errors."""
        http = self._get_http()
        try:
            r = await http.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                json=json_body,
                params=params,
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise DKGConnectionError(
                f"Could not reach DKG node at {self.base_url}: {e!r}"
            ) from e
        # The curator-ack mapping must run first so 409/503 curator failures
        # surface as CuratorRejectedError/CuratorUnconfirmedError, not a
        # generic status error.
        if curator_ack:
            _raise_for_curator_ack(r)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DKGStatusError(
                f"DKG node returned HTTP {r.status_code} for {method} {path}: "
                f"{r.text[:_BODY_EXCERPT_LEN]}",
                status_code=r.status_code,
                body=r.text,
            ) from e
        return r.json()

    # ------------------------------------------------------------------
    # Context Graphs
    # ------------------------------------------------------------------

    async def create_context_graph(self, name: str, id: str | None = None) -> dict[str, Any]:
        """Create a Context Graph.

        The node requires both ``id`` and ``name``; when ``id`` is omitted the
        ``name`` is reused as the id. Returns the node response as-is
        (``{"created": <id>, "uri": <uri>}`` on current builds).
        """
        return await self._request(
            "POST",
            "/api/context-graph/create",
            json_body={"id": id or name, "name": name},
        )

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
                Shared Working Memory (gossiped). When None, the node default
                applies ("swm" on current builds).
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
        return await self._request("POST", "/api/memory/turn", json_body=body)

    async def memory_search(
        self,
        context_graph_id: str,
        query: str,
        limit: int = 20,
        memory_layers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Tri-modal search across vector, SPARQL, and text stores.

        Args:
            context_graph_id: Context Graph to search.
            query: Free-text search query.
            limit: Maximum number of results.
            memory_layers: Layers to search. Defaults to ``["wm", "swm"]`` when
                None — current node builds return 0 results when the request
                omits ``memoryLayers``, so the layers are always sent.

        Returns dict with keys: query, contextGraphId, resultCount, results.
        Each result has: entityUri, label, sources, similarity, sourceFile,
            snippet, memoryLayer.
        """
        body: dict[str, Any] = {
            "contextGraphId": context_graph_id,
            "query": query,
            "limit": limit,
            "memoryLayers": memory_layers if memory_layers is not None else ["wm", "swm"],
        }
        return await self._request("POST", "/api/memory/search", json_body=body)

    # ------------------------------------------------------------------
    # Assertions (Working Memory)
    # ------------------------------------------------------------------

    async def assertion_create(
        self,
        context_graph_id: str,
        name: str,
        sub_graph_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a named Working Memory assertion (Knowledge Asset).

        Current node builds serve this as ``POST /api/knowledge-assets``; the
        legacy ``POST /api/assertion/create`` route is used as a fallback for
        older builds. The response shape follows whichever route answered.
        """
        body: dict[str, Any] = {"contextGraphId": context_graph_id, "name": name}
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        try:
            return await self._request("POST", "/api/knowledge-assets", json_body=body)
        except DKGStatusError as e:
            if e.status_code != 404:
                raise
            return await self._request("POST", "/api/assertion/create", json_body=body)

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
        return await self._request(
            "POST",
            f"/api/assertion/{urllib.parse.quote(name, safe='')}/write",
            json_body=body,
            curator_ack=True,
        )

    async def assertion_promote(
        self,
        name: str,
        context_graph_id: str,
        entities: list[str] | None = None,
        sub_graph_name: str | None = None,
        poll_timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        """Promote a Working Memory assertion to Shared Working Memory (SHARE).

        Current node builds only expose an asynchronous promote: the job is
        submitted via ``POST /api/knowledge-assets/{name}/swm/share-async``
        and polled via ``GET /api/knowledge-assets/swm/share-jobs/{jobId}``
        until it reaches a terminal state (``succeeded`` / ``failed``). Older
        builds that 404 the knowledge-assets surface fall back to the legacy
        ``/api/assertion/{name}/promote-async`` routes.

        Args:
            name: Assertion name (URL-quoted automatically).
            context_graph_id: Context Graph that owns the assertion.
            entities: Optional subset of entities to promote.
            sub_graph_name: Optional sub-graph within the Context Graph.
            poll_timeout: Max seconds to wait for the job to finish.
            poll_interval: Seconds between job-status polls.

        Returns:
            The final job view (``state == "succeeded"``).

        Raises:
            CuratorUnconfirmedError / CuratorRejectedError: when the job (or
                the submission itself) fails with a curator-ack code.
            DKGError: when the job fails otherwise or polling times out.
        """
        body: dict[str, Any] = {"contextGraphId": context_graph_id}
        if entities:
            body["entities"] = entities
        if sub_graph_name:
            body["subGraphName"] = sub_graph_name
        quoted = urllib.parse.quote(name, safe="")
        legacy = False
        try:
            try:
                submitted = await self._request(
                    "POST",
                    f"/api/knowledge-assets/{quoted}/swm/share-async",
                    json_body=body,
                    curator_ack=True,
                )
            except DKGStatusError as e:
                if e.status_code != 404:
                    raise
                # Older builds don't serve the knowledge-assets surface.
                legacy = True
                submitted = await self._request(
                    "POST",
                    f"/api/assertion/{quoted}/promote-async",
                    json_body=body,
                    curator_ack=True,
                )
        except DKGStatusError as e:
            # 409 conflict: another job is already active for this assertion —
            # poll the existing job instead of failing.
            existing_job_id = None
            if e.status_code == 409:
                try:
                    existing_job_id = json.loads(e.body).get("existingJobId")
                except Exception:
                    existing_job_id = None
            if not existing_job_id:
                raise
            submitted = {"jobId": existing_job_id, "state": "queued"}
        job_id = submitted.get("jobId")
        if not job_id:
            raise DKGError(f"Promote submission returned no jobId: {submitted!r}")

        quoted_job = urllib.parse.quote(str(job_id), safe="")
        poll_path = (
            f"/api/assertion/promote-async/{quoted_job}"
            if legacy
            else f"/api/knowledge-assets/swm/share-jobs/{quoted_job}"
        )
        deadline = asyncio.get_running_loop().time() + poll_timeout
        while True:
            try:
                job = await self._request("GET", poll_path)
            except TimeoutError as e:  # asyncio timeouts bubbling out of the poll
                raise DKGError(f"Timed out polling promote job {job_id}") from e
            state = job.get("state")
            if state == "succeeded":
                return job
            if state == "failed":
                text = _job_error_text(job)
                lowered = text.lower()
                if "curator_unconfirmed" in lowered or "not confirmed by its curator" in lowered:
                    raise CuratorUnconfirmedError(
                        text or "curator did not confirm",
                        code="CURATOR_UNCONFIRMED",
                        context_graph_id=context_graph_id,
                        body=job,
                    )
                if "curator_rejected" in lowered or "rejected by its curator" in lowered:
                    raise CuratorRejectedError(
                        text or "curator rejected the promote",
                        code="CURATOR_REJECTED",
                        context_graph_id=context_graph_id,
                        body=job,
                    )
                raise DKGError(f"Promote job {job_id} failed: {text or job!r}")
            if asyncio.get_running_loop().time() >= deadline:
                raise DKGError(
                    f"Promote job {job_id} did not finish within {poll_timeout}s "
                    f"(last state: {state!r})"
                )
            await asyncio.sleep(poll_interval)

    async def assertion_history(
        self, name: str, context_graph_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/assertion/{urllib.parse.quote(name, safe='')}/history",
            params={"contextGraphId": context_graph_id},
        )

    # ------------------------------------------------------------------
    # SPARQL query
    # ------------------------------------------------------------------

    async def query(
        self,
        sparql: str,
        paranet_id: str | None = None,
        graph_suffix: str | None = None,
        include_workspace: bool = True,
        context_graph_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "sparql": sparql,
            "includeWorkspace": include_workspace,
        }
        if context_graph_id:
            body["contextGraphId"] = context_graph_id
        if paranet_id:
            body["paranetId"] = paranet_id
        if graph_suffix:
            body["graphSuffix"] = graph_suffix
        return await self._request("POST", "/api/query", json_body=body)

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
        return await self._request(
            "POST", "/api/shared-memory/write", json_body=body, curator_ack=True
        )

    async def shared_memory_publish(
        self, context_graph_id: str | None = None
    ) -> dict[str, Any]:
        """Promote Shared Working Memory to Verified Memory (PUBLISH, costs TRAC)."""
        body: dict[str, Any] = {}
        if context_graph_id:
            body["contextGraphId"] = context_graph_id
        return await self._request(
            "POST", "/api/shared-memory/publish", json_body=body, curator_ack=True
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return True if the node responds at ``GET /api/context-graph/list``.

        Returns False only when the node cannot be reached (transport error or
        timeout, using the client's configured timeout). Auth failures (401/403)
        and other HTTP errors raise :class:`DKGStatusError` so a bad token is
        distinguishable from a node that is down.
        """
        try:
            await self._request("GET", "/api/context-graph/list")
            return True
        except DKGConnectionError:
            return False
