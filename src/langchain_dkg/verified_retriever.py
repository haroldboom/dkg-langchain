"""DKGVerifiedRetriever — BaseRetriever over DKG v10 Verifiable Memory with a trust floor."""

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
from .retriever import (
    _DEFAULT_SPARQL_TEMPLATE,
    _escape_sparql_literal,
    _parse_triple_bindings,
    _triple_page_content,
)
from .trust import TrustLevel


class DKGVerifiedRetriever(BaseRetriever):
    """LangChain BaseRetriever over DKG v10 Verifiable Memory with a trust floor.

    Like :class:`~langchain_dkg.retriever.DKGRetriever`, but queries the
    ``"verifiable-memory"`` view: only content that has been published
    on-chain, filtered to at least ``min_trust`` on the trust gradient
    (SelfAttested → Endorsed → PartiallyVerified → ConsensusVerified).

    Usage::

        from langchain_dkg import DKGVerifiedRetriever, TrustLevel

        retriever = DKGVerifiedRetriever(
            context_graph_id="my-project",
            min_trust=TrustLevel.ENDORSED,
        )
        docs = retriever.invoke("wheat prices")

    Args:
        client: Pre-configured DKGClient (built from env vars if omitted).
        sparql_template: SPARQL SELECT template with {query} and {limit}
            placeholders. Supply your own for domain-specific queries.
        limit: Max triples to retrieve per query.
        context_graph_id: Context Graph to scope the query (required).
        min_trust: Minimum trust level — a :class:`TrustLevel` / int (0-3)
            or a string name (e.g. ``"endorsed"``). The node fails closed
            (HTTP 400) on values it does not recognize.
    """

    client: DKGClient | None = None
    sparql_template: str = _DEFAULT_SPARQL_TEMPLATE
    limit: int = 20
    context_graph_id: str
    min_trust: int | str = TrustLevel.SELF_ATTESTED

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _default_client(self) -> "DKGVerifiedRetriever":
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
            context_graph_id=self.context_graph_id,
            # Verifiable Memory is published content; Working Memory drafts
            # are out of scope for this retriever.
            include_workspace=False,
            view="verifiable-memory",
            min_trust=self.min_trust,
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
                        "source": "dkg-v10-vm",
                        "min_trust": self.min_trust,
                    },
                )
            )
        return docs

    def with_sparql(self, sparql_template: str) -> "DKGVerifiedRetriever":
        """Return a copy of this retriever with a custom SPARQL template."""
        return self.model_copy(update={"sparql_template": sparql_template})
