"""DKGChatMessageHistory — BaseChatMessageHistory backed by DKG v10 Working Memory."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ._sync import run_sync
from .client import DKGClient

# Role prefixes embedded in the stored Markdown so role can be recovered on search.
_HUMAN_PREFIX = "**Human:** "
_AI_PREFIX = "**AI:** "


def _message_to_markdown(message: BaseMessage) -> tuple[str, str]:
    """Return (role, markdown) for a message."""
    if isinstance(message, HumanMessage):
        return "human", f"{_HUMAN_PREFIX}{message.content}"
    return "ai", f"{_AI_PREFIX}{message.content}"


def _snippet_to_message(snippet: str) -> BaseMessage | None:
    """Reconstruct a BaseMessage from a stored markdown snippet."""
    if snippet.startswith(_HUMAN_PREFIX):
        return HumanMessage(content=snippet[len(_HUMAN_PREFIX):])
    if snippet.startswith(_AI_PREFIX):
        return AIMessage(content=snippet[len(_AI_PREFIX):])
    # Fallback: treat unlabelled snippets as AI messages
    return AIMessage(content=snippet) if snippet else None


class DKGChatMessageHistory(BaseChatMessageHistory):
    """Stores LangChain conversation turns in DKG v10 Working Memory.

    Each add_message call writes a Knowledge Asset via POST /api/memory/turn,
    passing the message as Markdown with a role prefix (**Human:** / **AI:**).

    get_messages performs tri-modal semantic search via POST /api/memory/search
    to retrieve the most relevant prior turns for the current context.

    Args:
        context_graph_id: The DKG Context Graph that scopes this conversation.
        client: A configured DKGClient (or None to build from env vars).
        search_query: Seed query used for history retrieval.
        search_limit: Maximum number of past turns to retrieve.
        session_uri: Optional IRI linking all turns in this history together.
        sub_graph_name: Optional sub-graph within the Context Graph.
        layer: Memory layer — "wm" (Working Memory, private) or "swm"
            (Shared Working Memory, gossiped). Defaults to "swm".
    """

    def __init__(
        self,
        context_graph_id: str,
        client: DKGClient | None = None,
        search_query: str = "conversation history",
        search_limit: int = 20,
        session_uri: str | None = None,
        sub_graph_name: str | None = None,
        layer: str | None = None,
    ) -> None:
        self.context_graph_id = context_graph_id
        self.client = client or DKGClient()
        self.search_query = search_query
        self.search_limit = search_limit
        self.session_uri = session_uri
        self.sub_graph_name = sub_graph_name
        self.layer = layer
        # Maps markdown content -> turnUri for subsequent promote_to_shared calls
        self._turn_uri_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve conversation history from DKG (synchronous wrapper)."""
        return run_sync(self.aget_messages())

    async def aget_messages(self) -> list[BaseMessage]:
        result = await self.client.memory_search(
            context_graph_id=self.context_graph_id,
            query=self.search_query,
            limit=self.search_limit,
        )
        messages: list[BaseMessage] = []
        for item in result.get("results", []):
            snippet = item.get("snippet") or item.get("label") or ""
            turn_uri = item.get("entityUri")
            msg = _snippet_to_message(snippet)
            if msg is not None:
                if turn_uri:
                    self._turn_uri_index[snippet] = turn_uri
                messages.append(msg)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Store a single message in DKG Working Memory (synchronous wrapper)."""
        run_sync(self.aadd_message(message))

    async def aadd_message(self, message: BaseMessage) -> None:
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
            self._turn_uri_index[markdown] = turn_uri

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
        automatically. Call it when the agent decides a turn should be
        visible beyond the local node.
        """
        assertion_name = turn_uri.split("/")[-1] if "/" in turn_uri else turn_uri
        return await self.client.assertion_promote(
            name=assertion_name,
            context_graph_id=self.context_graph_id,
            sub_graph_name=self.sub_graph_name,
        )

    def get_turn_uri(self, markdown: str) -> str | None:
        """Return the turnUri for a previously stored markdown string."""
        return self._turn_uri_index.get(markdown)
