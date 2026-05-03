"""DKGMemory — helper that wires DKGChatMessageHistory into a LangChain Runnable.

Modern LangChain (v0.3+) replaced the old BaseMemory API with
RunnableWithMessageHistory. DKGMemory is a factory that creates one,
pre-configured to use DKG Working Memory as the history backend.

Usage::

    from langchain_dkg import DKGMemory
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    chain_with_memory = DKGMemory.wrap_chain(
        chain,
        context_graph_id="my-project",
        history_messages_key="history",
    )
    response = chain_with_memory.invoke(
        {"input": "What is DKG?"},
        config={"configurable": {"session_id": "user-42"}},
    )
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from .chat_history import DKGChatMessageHistory
from .client import DKGClient


class DKGMemory:
    """Factory for RunnableWithMessageHistory backed by DKG v10 Working Memory.

    Args:
        context_graph_id: DKG Context Graph scoping this memory.
        client: Pre-configured DKGClient (built from env vars if omitted).
        search_limit: Max turns to retrieve per chain call.
        sub_graph_name: Optional sub-graph name within the Context Graph.
        layer: Memory layer for stored turns — "wm" (Working Memory, private)
            or "swm" (Shared Working Memory, gossiped). Defaults to the node's
            default ("swm") when omitted.
    """

    def __init__(
        self,
        context_graph_id: str,
        client: DKGClient | None = None,
        search_limit: int = 10,
        sub_graph_name: str | None = None,
        layer: str | None = None,
    ) -> None:
        self.context_graph_id = context_graph_id
        self.client = client or DKGClient()
        self.search_limit = search_limit
        self.sub_graph_name = sub_graph_name
        self.layer = layer

    def get_history(self, session_id: str) -> DKGChatMessageHistory:
        """Return a DKGChatMessageHistory scoped to session_id.

        The session_id becomes the sessionUri, grouping turns together.
        The context graph is shared across sessions within this memory instance.
        """
        return DKGChatMessageHistory(
            context_graph_id=self.context_graph_id,
            client=self.client,
            search_query="conversation history",
            search_limit=self.search_limit,
            session_uri=f"urn:session:{session_id}",
            sub_graph_name=self.sub_graph_name,
            layer=self.layer,
        )

    def wrap(
        self,
        runnable: Runnable,
        input_messages_key: str = "input",
        history_messages_key: str = "history",
        output_messages_key: str | None = None,
    ) -> RunnableWithMessageHistory:
        """Wrap a Runnable with DKG-backed message history.

        Returns a RunnableWithMessageHistory that expects a
        ``{"configurable": {"session_id": "<id>"}}`` config dict.
        """
        return RunnableWithMessageHistory(
            runnable,
            self.get_history,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            output_messages_key=output_messages_key,
        )

    @classmethod
    def wrap_chain(
        cls,
        runnable: Runnable,
        context_graph_id: str,
        client: DKGClient | None = None,
        input_messages_key: str = "input",
        history_messages_key: str = "history",
        output_messages_key: str | None = None,
        **kwargs: Any,
    ) -> RunnableWithMessageHistory:
        """Convenience classmethod: create DKGMemory and wrap a runnable in one call."""
        mem = cls(context_graph_id=context_graph_id, client=client, **kwargs)  # type: ignore[arg-type]
        return mem.wrap(
            runnable,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            output_messages_key=output_messages_key,
        )
