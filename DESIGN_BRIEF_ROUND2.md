# langchain-dkg 0.2 Design Brief — the verified-memory RAG loop

**Package:** `langchain-dkg` (0.2)
**Bounty tag:** `cfi-dkgv10-r2` (provisional — this brief targets the roadmapped Round 2 scope, *Verifiable Memory & context oracles*, and will be tuned when the round officially opens)
**Round 1 foundation:** `langchain-dkg` 0.1 — registry entry [dkg-integrations PR #4](https://github.com/OriginTrail/dkg-integrations/pull/4). Follow-on submission: same package, same maintainer, new version.

---

## 1. Problem

Round 1 gave every LangChain agent durable, attributable memory in Working Memory and Shared Working Memory. But a RAG pipeline built on that memory still cannot answer the question that matters for high-stakes agent decisions: **how much should I trust this retrieved fact?** A passing remark in a chat session and a claim that three independent verifiers co-signed on-chain look identical to `DKGRetriever`.

DKG v10's trust gradient solves this — Knowledge Assets in Verifiable Memory carry a protocol-stamped `trustLevel` that climbs from SelfAttested through Endorsed and PartiallyVerified to ConsensusVerified. What is missing is the framework plumbing: a retriever that filters on that gradient, a one-call promotion path that gets agent knowledge *into* Verifiable Memory, and agent tools that let trust be raised conversationally rather than through dashboards.

`langchain-dkg` 0.2 closes that loop. It is the read-and-write surface for the exact Round 2 pattern: **oracle pipelines consuming matured Shared Memory artifacts**.

---

## 2. Target users

- **RAG pipeline builders** who need a verifiable upstream with a tunable trust floor — "answer only from knowledge at least Endorsed" becomes one constructor argument.
- **Multi-agent system builders** whose agents publish, endorse, and verify each other's claims as part of normal conversation — consensus formation without any human-facing UI.
- **Teams with audit requirements** who need agent conclusions anchored on-chain (ERC-721 Knowledge Asset, UAL, Merkle root, transaction hash) with the full promotion chain traceable back to the originating conversation turn.

---

## 3. What 0.2 adds

```
LangChain Application
        │
        ├─ publish_to_verified() ──► ka create → wm/write → wm/finalize (EIP-712 seal)
        │                            → swm/share (SHARE) → vm/publish (PUBLISH, on-chain mint)
        │
        ├─ DKGVerifiedRetriever ───► POST /api/query  {view: "verifiable-memory", minTrust: 0..3}
        │                            Documents carry UAL, trustLevel, merkleRoot, txHash
        │
        ├─ make_dkg_tools() ───────► dkg_publish_note / dkg_endorse / dkg_verified_search
        │                            (conversational consensus — agent tool calls, no UI)
        │
        └─ DKGClient additions ────► endorse, request_verification (M-of-N),
                                     verify_batch (local root recompute), kc_metadata / kc_author
```

### 3.1 `publish_to_verified()` — completing the Round 1 promotion path

The Round 1 brief documented the promotion path and explicitly deferred its final step to Round 2 ("Shared Working Memory → Verified Memory: Round 2 / future"). 0.2 delivers it. One call orchestrates the full v10 lifecycle: create the named Knowledge Asset, write quads to Working Memory, **finalize** (the off-chain seal — Merkle root plus the author's EIP-712 attestation), **SHARE** to Shared Working Memory (async job, polled to completion), then **PUBLISH** to Verifiable Memory — minting the ERC-721 Knowledge Asset and anchoring the Merkle root on-chain. The receipt merges the publish response (`kaId`, `ual`, `txHash`, `merkleRoot`) with the seal fields, so callers hold the complete provenance bundle from a single return value. UALs are preserved through every promotion, exactly as the Round 1 brief promised.

### 3.2 `DKGVerifiedRetriever` — trust-floor retrieval, an oracle-consuming RAG pipeline

A drop-in `BaseRetriever` like Round 1's `DKGRetriever`, but scoped to the `verifiable-memory` query view with a `min_trust` floor. `DKGVerifiedRetriever(context_graph_id="my-project", min_trust=TrustLevel.ENDORSED)` returns only knowledge that has matured past self-attestation. This is the public oracle-consumer read path in node build 10.0.2, and it makes any downstream LangChain chain an **oracle pipeline consuming matured Shared Memory artifacts** — the Round 2 criterion, verbatim.

### 3.3 Agent tools — conversational consensus

`make_dkg_tools(client, context_graph_id)` returns three LangChain tools: `dkg_publish_note(title, content)`, `dkg_endorse(ual)`, and `dkg_verified_search(query, min_trust)`. Trust is raised the way v10's design principles demand — by agents, in conversation, never by UI buttons. One agent publishes a claim; a second agent, having independent grounds to believe it, calls `dkg_endorse` on its UAL; a third retrieves it at `min_trust="endorsed"`. Endorsement writes `dkg:endorses` triples that ride the next publish batch and stamp the asset *Endorsed*. The tools depend only on `langchain_core`, so they work in any LangChain / LangGraph agent runtime.

### 3.4 TrustLevel gradient fidelity

The package models the gradient exactly as the protocol defines it: `TrustLevel.SELF_ATTESTED (0) → ENDORSED (1) → PARTIALLY_VERIFIED (2) → CONSENSUS_VERIFIED (3)`. `client.request_verification(...)` drives the M-of-N verifier co-signature quorum toward ConsensusVerified; a partial quorum is returned as a status (`partial` / `no_quorum` with signer count), not raised, so agents can poll. Trust levels are protocol-stamped — the package never writes `trustLevel` quads itself (the node rejects user-authored ones), and it never misrepresents a level it did not read from the node.

---

## 4. Memory layers and v10 primitives

| Layer / primitive | Round 1 (0.1) | Round 2 (0.2) |
|---|---|---|
| Working Memory | read + write (turns) | draft stage of `publish_to_verified()` |
| Shared Working Memory | SHARE via promote | SHARE as publish precondition (async, polled) |
| Verifiable Memory | — (documented as future) | PUBLISH, endorse, M-of-N verify, trust-floor query |
| Knowledge Asset / UAL | `turnUri` per turn | ERC-721 mint; UAL in every retriever Document |
| Context Graph | session scoping | publish scope; auto-registered on-chain at first publish |
| Curator | SHARE authorization | SHARE/PUBLISH authorization (funded, publish-authorized wallet) |
| Trust gradient | n/a | first-class: `TrustLevel`, `min_trust`, endorse/verify |

**LLM-Wiki / autoresearch alignment:** Round 1 mapped external memory → WM and team memory → SWM. 0.2 completes the column Karpathy's framing calls *long-term knowledge*: claims that survived team scrutiny become permanent, trust-stamped, on-chain records that any future agent — including autonomous research pipelines — can retrieve with an explicit trust bar. An autoresearch loop can now *cite* its substrate: every retrieved Document names the UAL and trust level it cleared.

---

## 5. Oracle-readiness and forward compatibility

- **Provenance on every Document.** `DKGVerifiedRetriever` results carry `ual`, `trustLevel`, `merkleRoot`, and `txHash` metadata — everything a downstream consumer needs to independently locate and check the on-chain anchor.
- **Local verification today.** `client.verify_batch(...)` recomputes a batch's Merkle root node-side against published content, and `client.kc_metadata(...)` / `client.kc_author(...)` fetch the chain-side root and author for comparison.
- **Client-side Merkle proofs tomorrow.** Full client-side inclusion-proof verification (per-triple proofs against the on-chain root) is roadmapped, composed **exclusively from public interfaces** — `query view=verifiable-memory`, `GET /api/kc/:id`, `verify-batch`, and direct RPC reads of the Knowledge Asset storage contract. The bounty rules prohibit importing internal node packages (`-publisher`, `-core`, …); we comply strictly, reimplementing the check from observable protocol behavior. When the node exposes a public proof endpoint, `DKGVerifiedRetriever(verify=True)` adopts it without breaking changes.

---

## 6. Terminology

0.2 code and docs use exact v10 vocabulary: Context Graph, Knowledge Asset, UAL, Working / Shared Working / **Verifiable** Memory, SHARE, PUBLISH, Curator, trust gradient. One correction from Round 1: the 0.1 brief said "Verified Memory"; current v10 materials use **Verifiable Memory**, and 0.2 standardizes on that. No other deviations.

---

## 7. Status & verification (honest accounting)

- **Implemented and unit-tested:** the full 0.2 surface (`publish_to_verified`, `DKGVerifiedRetriever`, `TrustLevel`, `make_dkg_tools`, endorse / request_verification / verify_batch / kc_metadata) ships today with **134 passing tests**, built and verified against node build `10.0.2`.
- **Verified live:** on our Base Sepolia testnet node — Context Graph creation with on-chain registration, Knowledge Asset create, quad write, **finalize (EIP-712 seal confirmed)**, and SWM share completing with `publishReady: true`.
- **Currently blocked, network-side:** the final `vm/publish` step reaches the network ACK stage and fails with `storage_ack_insufficient (0/3)` — every dialled core peer is either unresponsive or fails on-chain key verification (`ACK_VERIFY: key-not-registered`). The Base Sepolia core-peer set appears unregistered following the June 29 mainnet launch; publish quorum is unreachable on testnet for all publishers, not just us. **Reported to OriginTrail.** Our publish smoke script re-runs the moment peers re-register; no client-side changes are expected.

---

## 8. Demo plan

Re-using the Round 1 recording pipeline (`examples/demo_video.py` → narrated walkthrough video), once testnet publish quorum is restored:

1. Agent A publishes a claim via `dkg_publish_note` — show the receipt: UAL, txHash, merkleRoot, EIP-712 seal.
2. Agent B endorses it conversationally via `dkg_endorse(ual)` — no UI, just a tool call.
3. Agent C runs `dkg_verified_search(query, min_trust="endorsed")` and retrieves it; the same query at the same trust floor before endorsement returns nothing — the trust gradient, visible on screen.
4. Close with `verify_batch` + `kc_metadata` matching the retrieved content to the on-chain root.

---

## 9. Positioning

Round 1's `langchain-dkg` was the read-side amplifier for every source-side integration in the queue. 0.2 extends that role up the trust gradient: any knowledge ingested by any Round 1 integration (including our sister submission `github-dkg`) can now be promoted, endorsed, verified, and consumed by any LangChain agent at a chosen trust floor. Same maintainer, same repository, same six-month maintenance commitment, extended from the 0.2 release date.
