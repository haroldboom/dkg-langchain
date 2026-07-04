"""High-level promotion to Verifiable Memory (WM → SWM → VM) and quad shaping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .client import DKGClient, DKGPublishPreconditionError, QuadLike

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_SCHEMA = "http://schema.org/"


def _rdf_literal(text: str) -> str:
    """Quote a string as an RDF literal for the node's quad ``object`` term."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def turn_to_quads(
    turn_uri: str,
    markdown: str,
    *,
    session_uri: str | None = None,
) -> list[dict[str, str]]:
    """Build a minimal schema.org quad set for a conversation turn.

    Gives a chat turn a publishable RDF shape: the turn is typed
    ``schema:Message`` carrying ``schema:text`` (the markdown) and
    ``schema:dateCreated``; when ``session_uri`` is given, the turn is linked
    to the session via ``schema:isPartOf`` and the session is typed
    ``schema:Conversation``.

    This is a deliberately minimal default shape — replace it with your own
    quad builder for richer domain modelling (speaker identity, tool calls,
    citations, ...).
    """
    quads = [
        {"subject": turn_uri, "predicate": _RDF_TYPE, "object": f"{_SCHEMA}Message"},
        {"subject": turn_uri, "predicate": f"{_SCHEMA}text", "object": _rdf_literal(markdown)},
        {
            "subject": turn_uri,
            "predicate": f"{_SCHEMA}dateCreated",
            "object": _rdf_literal(datetime.now(timezone.utc).isoformat()),
        },
    ]
    if session_uri:
        quads.append(
            {"subject": session_uri, "predicate": _RDF_TYPE, "object": f"{_SCHEMA}Conversation"}
        )
        quads.append(
            {"subject": turn_uri, "predicate": f"{_SCHEMA}isPartOf", "object": session_uri}
        )
    return quads


async def publish_to_verified(
    client: DKGClient,
    context_graph_id: str,
    name: str,
    quads: list[QuadLike],
    *,
    sub_graph_name: str | None = None,
    publish_epochs: int | None = None,
    share_poll_timeout: float = 30.0,
) -> dict[str, Any]:
    """Promote a quad set end-to-end from draft to Verifiable Memory.

    Orchestrates the full promotion chain on the DKG v10 node:

    1. create the named Knowledge Asset draft (``assertion_create``),
    2. write the quads into its Working Memory (``ka_write``),
    3. finalize the draft — the off-chain EIP-712 seal (``ka_finalize``),
    4. share it to Shared Working Memory, polling the async share job to
       completion (``assertion_promote``),
    5. publish it to Verifiable Memory on-chain (``vm_publish``, costs TRAC).

    Returns:
        The ``vm_publish`` response merged over the finalize seal fields, so
        callers get both the chain result (kaId, ual, txHash, ...) and the
        seal (eip712Digest, schemeVersion, chainId, kav10Address, ...) in one
        dict. A 207 partial publish (KA minted, context-graph binding failed)
        is merged and returned the same way, not raised.

    Raises:
        DKGPublishPreconditionError: the VM publish was refused because its
            preconditions were not met — most commonly the SWM share has not
            fully landed on the network yet.
        CuratorAckError / DKGError: from the underlying share/publish steps.
    """
    await client.assertion_create(context_graph_id, name, sub_graph_name=sub_graph_name)
    await client.ka_write(name, context_graph_id, quads)
    seal = await client.ka_finalize(name, context_graph_id)
    await client.assertion_promote(
        name,
        context_graph_id=context_graph_id,
        sub_graph_name=sub_graph_name,
        poll_timeout=share_poll_timeout,
    )
    try:
        published = await client.vm_publish(name, context_graph_id, publish_epochs=publish_epochs)
    except DKGPublishPreconditionError as e:
        raise DKGPublishPreconditionError(
            f"VM publish of {name!r} was refused ({e.code}): {e}. The SWM share "
            "job succeeded locally but the share may not have fully landed on "
            "the network yet — retry once it settles (or raise share_poll_timeout).",
            code=e.code,
            status_code=e.status_code,
            body=e.body,
        ) from e
    result = dict(seal)
    result.update(published)
    return result
