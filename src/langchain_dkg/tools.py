"""LangChain agent tools for the DKG v10 Verifiable Memory trust gradient."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from langchain_core.tools import BaseTool, tool

from ._sync import run_sync
from .client import DKGClient, QuadLike
from .publish import _RDF_TYPE, _SCHEMA, _rdf_literal
from .verified_retriever import DKGVerifiedRetriever


def make_dkg_tools(client: DKGClient, context_graph_id: str) -> list[BaseTool]:
    """Build the DKG agent tools, bound to a client and a Context Graph.

    Returns three ``langchain_core`` tools ready to hand to an agent:

    - ``dkg_endorse`` — endorse a published Knowledge Asset by UAL,
    - ``dkg_verified_search`` — search Verifiable Memory with a trust floor,
    - ``dkg_publish_note`` — publish a titled note on-chain (costs TRAC).

    The tool docstrings below are written for LLM consumption — they become
    the tool descriptions the agent sees.
    """

    @tool
    def dkg_endorse(ual: str) -> str:
        """Endorse a published Knowledge Asset on the OriginTrail DKG.

        Use this when you have checked a piece of published knowledge and
        want to vouch for it. Endorsing raises the asset's trust level from
        SelfAttested to Endorsed; your endorsement is recorded on the graph
        under your node identity and rides the next publish batch.

        Args:
            ual: The Universal Asset Locator of the Knowledge Asset to
                endorse (e.g. "did:dkg:base:84532/0xabc.../123").

        Returns the node's acknowledgement as JSON.
        """
        result = run_sync(client.endorse(context_graph_id, ual))
        return json.dumps(result)

    @tool
    def dkg_verified_search(query: str, min_trust: str = "endorsed") -> str:
        """Search the DKG's Verifiable Memory for published, trusted knowledge.

        Use this when the answer must be backed by verifiable, on-chain
        knowledge rather than unverified working notes. Only content at or
        above the requested trust level is returned.

        Args:
            query: Free-text search terms (matched against stored values).
            min_trust: Minimum trust level — one of "self_attested",
                "endorsed", "partially_verified", "consensus_verified".
                Defaults to "endorsed" (at least one third party vouched).

        Returns matching facts, one "subject predicate object" triple per
        line, or a message saying nothing was found.
        """
        retriever = DKGVerifiedRetriever(
            client=client,
            context_graph_id=context_graph_id,
            min_trust=min_trust,
        )
        docs = retriever.invoke(query)
        if not docs:
            return f"No results in Verifiable Memory at trust level {min_trust!r} or above."
        return "\n".join(doc.page_content for doc in docs)

    @tool
    def dkg_publish_note(title: str, content: str) -> str:
        """Publish a note as a new Knowledge Asset on the OriginTrail DKG.

        Use this to permanently record a conclusion, finding, or decision so
        other agents can discover and verify it later. Publishing anchors the
        note on-chain (this spends TRAC from the node's wallet) at trust
        level SelfAttested; others can endorse or verify it afterwards.

        Args:
            title: A short, descriptive title for the note.
            content: The note body (plain text or markdown).

        Returns the publish receipt as JSON (kaId, status, transaction hash).
        """
        note_uri = f"urn:dkg:note:{uuid.uuid4()}"
        quads: list[QuadLike] = [
            {"subject": note_uri, "predicate": _RDF_TYPE, "object": f"{_SCHEMA}CreativeWork"},
            {"subject": note_uri, "predicate": f"{_SCHEMA}name", "object": _rdf_literal(title)},
            {"subject": note_uri, "predicate": f"{_SCHEMA}text", "object": _rdf_literal(content)},
            {
                "subject": note_uri,
                "predicate": f"{_SCHEMA}dateCreated",
                "object": _rdf_literal(datetime.now(timezone.utc).isoformat()),
            },
        ]
        result = run_sync(client.publish_direct(context_graph_id, quads))
        return json.dumps(result)

    return [dkg_endorse, dkg_verified_search, dkg_publish_note]
