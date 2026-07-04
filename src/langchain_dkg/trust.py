"""Trust levels for the DKG v10 Verifiable Memory trust gradient."""

from __future__ import annotations

from enum import IntEnum


class TrustLevel(IntEnum):
    """Trust gradient stamped on Verifiable Memory content by the DKG v10 node.

    The node accepts either the integer value or the string name (e.g.
    ``"endorsed"``) wherever a minimum trust level is expected, and fails
    closed (HTTP 400) on anything it does not recognize.

    Levels:
        SELF_ATTESTED: Published by the author; no third-party attestation.
        ENDORSED: At least one other identity endorsed the content
            (``dkg:endorses``).
        PARTIALLY_VERIFIED: Some, but not a quorum, of verifier signatures
            landed for the batch.
        CONSENSUS_VERIFIED: M-of-N verifier quorum met and anchored on-chain.
    """

    SELF_ATTESTED = 0
    ENDORSED = 1
    PARTIALLY_VERIFIED = 2
    CONSENSUS_VERIFIED = 3
