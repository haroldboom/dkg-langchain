"""DKGRetriever — BaseRetriever that queries DKG v10 via SPARQL."""

from __future__ import annotations

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, model_validator

from ._sync import run_sync
from .client import DKGClient


_DEFAULT_SPARQL_TEMPLATE = """
SELECT ?subject ?predicate ?object
WHERE {{
  ?subject ?predicate ?object .
  FILTER(CONTAINS(LCASE(STR(?object)), LCASE("{query}")))
}}
LIMIT {limit}
"""


def _escape_sparql_literal(s: str) -> str:
    """Escape a string for safe use inside a SPARQL double-quoted literal."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _parse_triple_bindings(result: dict) -> list[dict[str, str]]:
    """Extract SPARQL bindings from a node query response as plain value dicts.

    Node builds < rc.19 return SPARQL-standard ``{"results": {"bindings"}}``;
    newer builds return ``{"result": {"bindings"}}`` — and some return the
    bindings list directly. Cells are either SPARQL-standard
    ``{"type": ..., "value": ...}`` objects or plain strings (newer builds).
    Extract defensively; malformed rows are skipped.
    """
    container = result.get("result") or result.get("results")
    if isinstance(container, dict):
        bindings = container.get("bindings") or []
    elif isinstance(container, list):
        bindings = container
    else:
        bindings = []
    if not isinstance(bindings, list):
        bindings = []
    rows: list[dict[str, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        rows.append(
            {
                var: cell.get("value", "") if isinstance(cell, dict) else str(cell)
                for var, cell in binding.items()
            }
        )
    return rows


def _triple_page_content(values: dict[str, str]) -> str:
    """Render one bound row as Document page content (subject predicate object)."""
    subject = values.get("subject", "")
    predicate = values.get("predicate", "")
    obj = values.get("object", "")
    if subject or predicate or obj:
        return f"{subject} {predicate} {obj}".strip()
    return " ".join(values.values()).strip()


class DKGRetriever(BaseRetriever):
    """LangChain BaseRetriever that executes SPARQL queries against DKG v10.

    Each result triple (subject / predicate / object) becomes a LangChain
    Document so the agent can cite provenance via the UAL embedded in metadata.

    Usage::

        from langchain_dkg import DKGRetriever

        retriever = DKGRetriever(context_graph_id="my-project")
        docs = retriever.invoke("wheat prices")

    Args:
        client: Pre-configured DKGClient (built from env vars if omitted).
        sparql_template: SPARQL SELECT template with {query} and {limit}
            placeholders. Supply your own for domain-specific queries.
        limit: Max triples to retrieve per query.
        paranet_id: Optional paranet to scope the query.
        graph_suffix: Optional graph suffix.
        include_workspace: Whether to include Working Memory in results.
        context_graph_id: Context Graph to scope the query — required by
            current node builds to see workspace data.

    Note:
        Document metadata carries a coarse ``layer`` value ("workspace" when
        ``include_workspace`` is set, else "published") reflecting the query
        scope; per-triple layer provenance is not available from the node.
    """

    client: DKGClient | None = None
    sparql_template: str = _DEFAULT_SPARQL_TEMPLATE
    limit: int = 20
    paranet_id: str | None = None
    graph_suffix: str | None = None
    include_workspace: bool = True
    context_graph_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _default_client(self) -> "DKGRetriever":
        if self.client is None:
            self.client = DKGClient()
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return run_sync(self._aget_relevant_documents_impl(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await self._aget_relevant_documents_impl(query)

    async def _aget_relevant_documents_impl(self, query: str) -> list[Document]:
        sparql = self.sparql_template.format(
            query=_escape_sparql_literal(query),
            limit=self.limit,
        )
        assert self.client is not None  # set by _default_client
        result = await self.client.query(
            sparql=sparql,
            paranet_id=self.paranet_id,
            graph_suffix=self.graph_suffix,
            include_workspace=self.include_workspace,
            context_graph_id=self.context_graph_id,
        )
        docs: list[Document] = []
        for values in _parse_triple_bindings(result):
            docs.append(
                Document(
                    page_content=_triple_page_content(values),
                    metadata={
                        "subject": values.get("subject", ""),
                        "predicate": values.get("predicate", ""),
                        "object": values.get("object", ""),
                        "source": "dkg-v10",
                        # Coarse query-scope indicator; the node does not
                        # report per-triple layer provenance.
                        "layer": "workspace" if self.include_workspace else "published",
                    },
                )
            )
        return docs

    def with_sparql(self, sparql_template: str) -> "DKGRetriever":
        """Return a copy of this retriever with a custom SPARQL template."""
        return self.model_copy(update={"sparql_template": sparql_template})
