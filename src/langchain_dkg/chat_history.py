"""DKGChatMessageHistory — BaseChatMessageHistory backed by DKG v10 Working Memory."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ._sync import run_sync
from .client import DKGClient

# Role prefixes embedded in the stored Markdown so role can be recovered on search.
_HUMAN_PREFIX = "**Human:** "
_AI_PREFIX = "**AI:** "
_SYSTEM_PREFIX = "**System:** "
_TOOL_PREFIX = "**Tool:** "

# Max entries kept in the markdown -> turnUri LRU index.
_TURN_URI_INDEX_CAP = 512


def _content_to_text(content: Any) -> str:
    """Extract plain text from a message ``content`` field.

    Handles both plain-string content and content-block lists (where each
    block is either a string or a dict with a ``"text"`` field).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _message_to_markdown(message: BaseMessage) -> tuple[str, str]:
    """Return (role, markdown) for a message, branching on ``message.type``."""
    text = _content_to_text(message.content)
    msg_type = getattr(message, "type", "ai")
    if msg_type == "human":
        return "human", f"{_HUMAN_PREFIX}{text}"
    if msg_type == "system":
        return "system", f"{_SYSTEM_PREFIX}{text}"
    if msg_type == "tool":
        return "tool", f"{_TOOL_PREFIX}{text}"
    if msg_type == "ai":
        return "ai", f"{_AI_PREFIX}{text}"
    return "ai", f"{_AI_PREFIX}{text}"


def _snippet_to_message(snippet: str) -> BaseMessage | None:
    """Reconstruct a BaseMessage from a stored markdown snippet."""
    if snippet.startswith(_HUMAN_PREFIX):
        return HumanMessage(content=snippet[len(_HUMAN_PREFIX):])
    if snippet.startswith(_AI_PREFIX):
        return AIMessage(content=snippet[len(_AI_PREFIX):])
    if snippet.startswith(_SYSTEM_PREFIX):
        return SystemMessage(content=snippet[len(_SYSTEM_PREFIX):])
    if snippet.startswith(_TOOL_PREFIX):
        # The original tool_call_id is not stored in the markdown snippet.
        return ToolMessage(content=snippet[len(_TOOL_PREFIX):], tool_call_id="")
    # Fallback: treat unlabelled snippets as AI messages
    return AIMessage(content=snippet) if snippet else None


class DKGChatMessageHistory(BaseChatMessageHistory):
    """Stores LangChain conversation turns in DKG v10 Working Memory.

    Each add_message call writes a Knowledge Asset via POST /api/memory/turn,
    passing the message as Markdown with a role prefix (**Human:** / **AI:** /
    **System:** / **Tool:**).

    Retrieval is semantic-relevance based: get_messages performs tri-modal
    search via POST /api/memory/search using ``search_query``, NOT a
    chronological dump of the conversation. When ``session_uri`` is set, a
    session filter is applied client-side (the node's search API has no
    session parameter): the session's turn set is looked up via SPARQL
    (``<session_uri> <http://schema.org/hasPart> ?turn``), search results are
    filtered to that set and sorted chronologically. If the SPARQL lookup
    fails or returns nothing, the unfiltered search results are returned.

    Args:
        context_graph_id: The DKG Context Graph that scopes this conversation.
        client: A configured DKGClient (or None to build from env vars).
        search_query: Seed query used for history retrieval.
        search_limit: Maximum number of past turns to retrieve.
        session_uri: Optional IRI linking all turns in this history together.
        sub_graph_name: Optional sub-graph within the Context Graph.
        layer: Memory layer — "wm" (Working Memory, private) or "swm"
            (Shared Working Memory, gossiped). Defaults to "wm" so
            conversation history stays private by default; pass
            ``layer="swm"`` to gossip turns, or ``layer=None`` to use the
            node's default (currently "swm").
        search_layers: Memory layers searched by get_messages. Defaults to
            ["wm", "swm"] — newer node builds return nothing when the request
            omits memoryLayers, so the layers are always sent explicitly.
    """

    def __init__(
        self,
        context_graph_id: str,
        client: DKGClient | None = None,
        search_query: str = "conversation history",
        search_limit: int = 10,
        session_uri: str | None = None,
        sub_graph_name: str | None = None,
        layer: str | None = "wm",
        search_layers: list[str] | None = None,
    ) -> None:
        self.context_graph_id = context_graph_id
        self.client = client or DKGClient()
        self.search_query = search_query
        self.search_limit = search_limit
        self.session_uri = session_uri
        self.sub_graph_name = sub_graph_name
        self.layer = layer
        self.search_layers = search_layers or ["wm", "swm"]
        # Maps markdown content -> turnUri for subsequent promote_to_shared
        # calls. LRU-capped at _TURN_URI_INDEX_CAP entries.
        self._turn_uri_index: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]  # base declares a writeable attribute
        """Retrieve conversation history from DKG (synchronous wrapper)."""
        return run_sync(self.aget_messages())

    async def _session_turn_set(self) -> set[str]:
        """Fetch the session's turn URIs via SPARQL (empty set on failure)."""
        if not self.session_uri:
            return set()
        sparql = (
            f"SELECT ?t WHERE {{ <{self.session_uri}> "
            f"<http://schema.org/hasPart> ?t }}"
        )
        try:
            result = await self.client.query(
                sparql=sparql,
                context_graph_id=self.context_graph_id,
                include_workspace=True,
            )
        except Exception:
            return set()
        container = result.get("result") or result.get("results")
        if isinstance(container, dict):
            bindings = container.get("bindings") or []
        elif isinstance(container, list):
            bindings = container
        else:
            bindings = []
        if not isinstance(bindings, list):
            bindings = []
        turns: set[str] = set()
        for row in bindings:
            if not isinstance(row, dict):
                continue
            cell = row.get("t")
            if isinstance(cell, dict):
                cell = cell.get("value")
            if isinstance(cell, str) and cell:
                turns.add(cell)
        return turns

    async def aget_messages(self) -> list[BaseMessage]:
        """Retrieve relevant turns via tri-modal semantic search.

        Note: results are relevance-ranked by the node, not chronological.
        When ``session_uri`` is set the search over-fetches, filters to the
        session's turns (client-side) and sorts them chronologically by turn
        URI (turn URIs end with an ISO timestamp, so a lexical sort on the
        URI is chronological).
        """
        limit = self.search_limit * 3 if self.session_uri else self.search_limit
        result = await self.client.memory_search(
            context_graph_id=self.context_graph_id,
            query=self.search_query,
            limit=limit,
            memory_layers=self.search_layers,
        )
        results = list(result.get("results", []))
        if self.session_uri:
            session_turns = await self._session_turn_set()
            if session_turns:
                filtered = [r for r in results if r.get("entityUri") in session_turns]
                filtered.sort(key=lambda r: r.get("entityUri") or "")
                # Keep the MOST RECENT search_limit turns, in chronological order.
                results = filtered[-self.search_limit :]
            else:
                # SPARQL lookup failed or session has no linked turns — fall
                # back to relevance-ranked search results (head of the list).
                results = results[: self.search_limit]
        messages: list[BaseMessage] = []
        for item in results:
            snippet = item.get("snippet") or item.get("label") or ""
            turn_uri = item.get("entityUri")
            msg = _snippet_to_message(snippet)
            if msg is not None:
                if turn_uri:
                    self._remember_turn(snippet, turn_uri)
                messages.append(msg)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Store a single message in DKG Working Memory (synchronous wrapper)."""
        run_sync(self.aadd_message(message))

    async def aadd_message(self, message: BaseMessage) -> str | None:
        """Store a message; returns the turnUri assigned by the node (if any)."""
        _, markdown = _message_to_markdown(message)
        result = await self.client.memory_turn(
            context_graph_id=self.context_graph_id,
            markdown=markdown,
            session_uri=self.session_uri,
            layer=self.layer,
            sub_graph_name=self.sub_graph_name,
        )
        turn_uri = result.get("turnUri")
        if turn_uri:
            self._remember_turn(markdown, turn_uri)
        return turn_uri

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            await self.aadd_message(msg)

    def clear(self) -> None:
        # DKG Working Memory is append-only — clear the local index only.
        self._turn_uri_index.clear()

    # ------------------------------------------------------------------
    # DKG-specific: promotion to Shared Working Memory
    # ------------------------------------------------------------------

    async def promote_to_shared(self, turn_uri: str) -> dict[str, Any]:
        """Promote a stored Knowledge Asset to Shared Working Memory (SHARE).

        This is an explicit, Curator-authorized operation — never called
        automatically. The client submits an async promote job and polls it
        to completion.

        Note: current node builds expose promotion for named Working Memory
        assertions; promoting memory *turns* by their URI may not be
        supported and can fail with a node-side error.
        """
        assertion_name = turn_uri.split("/")[-1] if "/" in turn_uri else turn_uri
        return await self.client.assertion_promote(
            name=assertion_name,
            context_graph_id=self.context_graph_id,
            sub_graph_name=self.sub_graph_name,
        )

    def get_turn_uri(self, markdown: str) -> str | None:
        """Return the turnUri for a previously stored markdown string."""
        turn_uri = self._turn_uri_index.get(markdown)
        if turn_uri is not None:
            self._turn_uri_index.move_to_end(markdown)
        return turn_uri

    def _remember_turn(self, markdown: str, turn_uri: str) -> None:
        """Record markdown -> turnUri, evicting least-recently-used entries."""
        self._turn_uri_index[markdown] = turn_uri
        self._turn_uri_index.move_to_end(markdown)
        while len(self._turn_uri_index) > _TURN_URI_INDEX_CAP:
            self._turn_uri_index.popitem(last=False)
