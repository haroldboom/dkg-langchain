"""langchain-dkg — LangChain memory and retriever backed by OriginTrail DKG v10."""

from .client import (
    CuratorAckError,
    CuratorRejectedError,
    CuratorUnconfirmedError,
    DKGClient,
    DKGError,
)
from .chat_history import DKGChatMessageHistory
from .memory import DKGMemory
from .retriever import DKGRetriever

__version__ = "0.1.8"
__all__ = [
    "DKGClient",
    "DKGChatMessageHistory",
    "DKGMemory",
    "DKGRetriever",
    "DKGError",
    "CuratorAckError",
    "CuratorUnconfirmedError",
    "CuratorRejectedError",
]
