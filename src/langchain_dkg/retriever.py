"""DKGRetriever — BaseRetriever that queries DKG v10 via SPARQL."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

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


class DKGRetriever(BaseRetriever):
    """LangChain BaseRetriever that executes SPARQL queries against DKG v10.

    Each result triple (subject / predicate / object) becomes a LangChain
    Document so the agent can cite provenance via the UAL embedded in metadata.

    Usage::

        from langchain_dkg import DKGRetriever
        from langchain.chains import RetrievalQA

        retriever = DKGRetriever(paranet_id="my-paranet")
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    Args:
        client: Pre-configured DKGClient (built from env vars if omitted).
        sparql_template: SPARQL SELECT template with {query} and {limit}
            placeholders. Supply your own for domain-specific queries.
        limit: Max triples to retrieve per query.
        paranet_id: Optional paranet to scope the query.
        graph_suffix: Optional graph suffix.
        include_workspace: Whether to include Working Memory in results.
    """

    client: Any = None
    sparql_template: str = _DEFAULT_SPARQL_TEMPLATE
    limit: int = 20
    paranet_id: str | None = None
    graph_suffix: str | None = None
    include_workspace: bool = True
    context_graph_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.client is None:
            self.client = DKGClient()

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
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return await self._aget_relevant_documents_impl(query)

    async def _aget_relevant_documents_impl(self, query: str) -> list[Document]:
        sparql = self.sparql_template.format(
            query=_escape_sparql_literal(query),
            limit=self.limit,
        )
        result = await self.client.query(
            sparql=sparql,
            paranet_id=self.paranet_id,
            graph_suffix=self.graph_suffix,
            include_workspace=self.include_workspace,
            context_graph_id=self.context_graph_id,
        )
        docs: list[Document] = []
        # Node builds < rc.19 return SPARQL-standard {"results": {"bindings"}};
        # newer builds return {"result": {"bindings"}}.
        bindings = (
            result.get("results", {}).get("bindings")
            or result.get("result", {}).get("bindings")
            or []
        )
        for binding in bindings:
            # SPARQL-standard cells are {"type": ..., "value": ...}; newer
            # node builds bind plain string values instead.
            values = {
                var: cell.get("value", "") if isinstance(cell, dict) else str(cell)
                for var, cell in binding.items()
            }
            subject = values.get("subject", "")
            predicate = values.get("predicate", "")
            obj = values.get("object", "")
            if subject or predicate or obj:
                page_content = f"{subject} {predicate} {obj}".strip()
            else:
                page_content = " ".join(values.values()).strip()
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "source": "dkg-v10",
                        "layer": "working" if self.include_workspace else "verified",
                    },
                )
            )
        return docs

    def with_sparql(self, sparql_template: str) -> "DKGRetriever":
        """Return a copy of this retriever with a custom SPARQL template."""
        return DKGRetriever(
            client=self.client,
            sparql_template=sparql_template,
            limit=self.limit,
            paranet_id=self.paranet_id,
            graph_suffix=self.graph_suffix,
            include_workspace=self.include_workspace,
            context_graph_id=self.context_graph_id,
        )
