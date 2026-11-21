# ShopSense — Architecture Decision Records (ADRs)

**Course:** CS5903: Distributed AI Training  
**Project:** ShopSense Multi-Agent E-Commerce Customer Care Assistant  
**Repository:** https://github.com/cs23mtech15027/shopsense_ai  
**Documentation Date:** 21 November 2026

---

## ADR-001 — Initial Methodology Selection: Local-Only Multi-Agent Stack (9 October 2026 Baseline)

* **Date, status & scope:** Decision date: 9 October 2026 (documented 21 November 2026) | **Status:** Superseded by ADR-002 | **Scope:** Full runtime stack (LLM inference, Policy RAG, OMS backend, UI, memory)
* **Context & decision drivers:** By the 9 October 2026, 11:59 PM IST methodology checkpoint, the project needed a self-contained, zero-cost, reproducible multi-agent customer support prototype for high-volume e-commerce operations (order tracking, policy Q&A, and deterministic return/refund eligibility). Key constraints included zero external API billing risk, offline reproducibility on a student laptop, and strict deterministic enforcement of refund rules.
* **Options considered:**
  1. **Local-only stack (Ollama Qwen 4B + local `BAAI/bge-small-en-v1.5` + local Qdrant + FastAPI/Chainlit + JSON summary memory):** Zero external API cost and full local control, but constrained by local RAM/disk (~5 GB RAM needed for Qwen 4B plus ~133 MB embedding weights) and slow CPU/single-node inference.
  2. **Fully cloud-hosted stack (Managed LLM + Cloud Vector DB + Hosted backend/MCP):** Lower inference latency and stronger models, but higher configuration complexity, external credential handling, and cloud deployment overhead for early milestone bring-up.
* **Decision & rationale:** Selected the **local-only stack** on 9 October 2026 to validate the core LangGraph multi-agent architecture (input guardrail -> triage -> care/order/refund specialists -> output guardrail) and deterministic SQLite OMS rules without external service dependencies.
* **Consequences & trade-offs:**
  * *Benefits:* Verified end-to-end LangGraph routing, multi-format policy ingestion (`.txt`, `.csv`, `.pdf`), and deterministic refund checks locally.
  * *Costs & risks:* Severe inference latency (~30 seconds per request locally), tight local disk/RAM ceiling that locked the system to a 4B-parameter model (Qwen 4B), and weaker greeting/triage consistency on nuanced queries.
* **Evidence & validation:** Initial prototype architecture (Report Fig. 2 & Table 1), local SQLite OMS schema, and baseline golden-dataset runs. Link: `[INSERT_COMMIT_OR_PR_LINK]`.
* **Revisit trigger:** End-to-end per-turn latency persistently exceeding 20 seconds or local hardware memory/disk constraints blocking upgrade to a stronger tool-capable LLM.

---

## ADR-002 — Post-Checkpoint Methodology Pivot: Hybrid Cloud-Local Architecture (Mandatory Methodology Change Record)

* **Date, status & scope:** Documentation date: 21 November 2026 *(revises the 9 October 2026 local selection; mentor approval record: `[INSERT_MENTOR_APPROVAL_DATE_OR_EMAIL_LINK]`)* | **Status:** Accepted | **Supersedes:** ADR-001 | **Scope:** Inference layer, Policy RAG embeddings/vector DB, UI, MCP tool boundary, memory, and observability.
* **Context & decision drivers:**
  * **Old approach (as of 9 October 2026):** Same LangGraph multi-agent design, but all services ran locally (Ollama Qwen 4B, local `BAAI/bge-small-en-v1.5` ~133 MB weights, local Qdrant, Chainlit UI, local JSON memory consolidation).
  * **New evidence:** Local testing exposed two critical bottlenecks: (1) average response latency reached **~30 seconds per request**, making interactive support demos and 30-query golden-dataset sweeps impractical; (2) limited local RAM and disk space prevented upgrading beyond Qwen 4B, causing weaker greeting classification and policy synthesis.
* **Options considered:**
  1. **Keep local-only stack (ADR-001):** Preserves zero external dependencies, but remains stuck at ~30 s/request latency and 4B-parameter model quality.
  2. **Fully cloud-deployed stack (including remote Hugging Face Space MCP server):** Eliminates all local processes, but introduces private-Space Hugging Face token vs. user JWT `Authorization` header collisions, cold starts, and harder local grading setup.
  3. **Hybrid Cloud-Local stack (Selected):** Keep the exact same multi-agent architecture, run compute-heavy/stateless services in free-tier cloud (**Groq API** with `openai/gpt-oss-120b`, **Qdrant Cloud Inference** with `sentence-transformers/all-minilm-l6-v2`, optional **Supermemory**, optional **Langfuse EU**), and run the operational **Streamlit UI**, **loopback Streamable HTTP MCP server**, **SQLite OMS**, and **SQLite LangGraph checkpointer** locally.
* **Decision & rationale:** Selected **Option 3 (Hybrid Architecture)**. Moving LLM inference to Groq's open-weight `openai/gpt-oss-120b` (Apache 2.0) and moving policy embeddings to Qdrant Cloud Inference (`cloud_inference=True`) eliminated local model disk/RAM bottlenecks while preserving the exact same agent graph and deterministic OMS rules. Keeping the MCP server on `127.0.0.1` avoided cloud token friction while maintaining production-style JWT-authenticated protocol separation.
* **Consequences & trade-offs:**
  * *Benefits:* Latency dropped from **~30 s/request** (local Qwen 4B) to **p90 = 15 s** measured across **108 Langfuse traces**; greeting handling and golden-dataset response quality improved substantially; zero local GPU or embedding model download required.
  * *Trade-offs:* Introduces external API dependencies, Groq free-tier rate limits (requiring retry logic), and external network hops. Moving from `BAAI/bge-small-en-v1.5` (512-token window) to Qdrant Cloud's free server-side `sentence-transformers/all-minilm-l6-v2` (256-token window) trades some standalone dense retrieval ceiling for zero local footprint—mitigated by 180-word chunking.
  * *Impact on milestones & tests:* Frozen acceptance criteria (order ownership isolation, deterministic refund eligibility, confirmation before mutation, prompt-injection blocking, 30-query golden dataset) remained intact; `tests/` expanded to 63 automated tests covering cloud retries, Qdrant Cloud Inference, MCP JWT auth, and redacted observability.
* **Evidence & validation:**
  * 108 Langfuse EU traces confirming **p90 latency = 15 s** (Report Fig. 9).
  * 30-query golden dataset (`data/golden_dataset/golden_test_results.csv`): Policy RAG Mean Recall@3 = `0.7391`, Mean Precision@3 = `0.2899`.
  * Automated test suite (`pytest`): 62/63 passing (1 Langfuse env-toggle test tracked for fix prior to final tag). Link: `[INSERT_COMMIT_OR_PR_LINK]`.
* **Revisit trigger:** Groq free-tier rate limits or latency regularly exceeding SLA (>20 s p90), or policy retrieval Recall@3 remaining below 0.80 on Returns & Refunds queries (triggering hybrid BM25 + dense retrieval in Qdrant).

---

## ADR-003 — Security & Tool Boundary: Loopback Streamable HTTP MCP with Per-Request JWT

* **Date, status & scope:** Documentation date: 21 November 2026 | **Status:** Accepted | **Scope:** `app/mcp_server.py`, `app/auth_context.py`, `app/persistence/orders.py`, `ui.py`
* **Context & decision drivers:** Order lookups, cancellations, and refund initiations mutate customer account state. Allowing an LLM to supply an arbitrary `user_id` creates an Insecure Direct Object Reference (IDOR) vulnerability where one shopper could view or refund another shopper's orders via prompt injection.
* **Options considered:**
  1. **Direct SQLite calls from agent tools using LLM-supplied `user_id`:** Simplest code, but vulnerable to prompt-injected cross-account access.
  2. **Remote cloud-hosted MCP server on Hugging Face Spaces:** Publicly addressable, but requires managing Space auth tokens alongside shopper JWTs and adds cold-start delays.
  3. **Local loopback Streamable HTTP MCP server (`127.0.0.1`) with mandatory Bearer JWT verification (Selected):** Enforces protocol separation and per-request identity verification without remote hosting friction.
* **Decision & rationale:** Selected **Option 3**. Every MCP tool call requires a signed shopper JWT in `Authorization: Bearer <token>`, passed via runtime context (never stored in LangGraph checkpoints or Supermemory). The MCP server verifies signature, issuer, audience, expiry, and token ID, extracts `sub` (`user_id`), and enforces owner-scoped SQLite queries (`WHERE user_id = ?`) plus deterministic refund/Tier-2 rules in code.
* **Consequences & trade-offs:** Prevents LLM-driven IDOR and enforces strict ₹5,000 Tier-2 escalation and non-returnable hygiene rules deterministically. Loopback binding (`127.0.0.1`) limits this prototype to single-host demo execution until deployed behind TLS and an enterprise identity provider.
* **Evidence & validation:** 13 passing integration tests in `tests/` covering loopback/JWT validation, shopper ownership isolation, idempotency, and refund rules. Link: `[INSERT_COMMIT_OR_PR_LINK]`.
* **Revisit trigger:** Deployment beyond local demo to multi-user cloud hosting (requires TLS termination, asymmetric JWKS verification, and external OAuth/OIDC).

---

## ADR-004 — Persistence Split: Local SQLite Checkpoints vs. Optional Supermemory

* **Date, status & scope:** Documentation date: 21 November 2026 | **Status:** Accepted | **Scope:** `app/persistence/checkpoints.py`, `app/persistence/supermemory.py`, `app/persistence/memory.py`
* **Context & decision drivers:** The initial design ran manual LLM summarization to local JSON files (`memories/customer-*/history_summarization.json`), consuming extra LLM tokens/latency and producing lossy static snapshots. Additionally, combining within-thread graph execution state with cross-thread shopper memory risked leaking an old active `order_id` into a new conversation.
* **Options considered:**
  1. **Keep local JSON history consolidation:** No external dependency, but eats Groq free-tier TPM/RPD quota and accumulates stale local files.
  2. **Store LangGraph checkpoints inside Supermemory or MCP:** Misaligned with LangGraph's transactional `BaseCheckpointSaver` interface and adds HTTP overhead to every graph super-step.
  3. **Local SQLite LangGraph checkpoints + optional shopper-scoped Supermemory (Selected):** Cleanly separates active thread execution state from cross-session preferences.
* **Decision & rationale:** Selected **Option 3**. Local SQLite checkpoints preserve exact per-thread message history and active order context, while optional Supermemory stores sanitized cross-thread shopper preferences (`dreaming` reconsolidation) without burning Groq quota. Active order selection stays strictly bound to the current `thread_id`.
* **Consequences & trade-offs:** Clean separation of concerns and graceful fallback when `SUPERMEMORY_API_KEY` is omitted; requires WAL/busy-timeout discipline on local SQLite checkpoints and strict PII/JWT redaction before writing to Supermemory.
* **Evidence & validation:** 4 passing component tests for optional memory retrieval, sanitized writes, and failure fallback; checkpoint integration tests. Link: `[INSERT_COMMIT_OR_PR_LINK]`.
* **Revisit trigger:** Multi-instance horizontal scaling of the backend (migrate local SQLite checkpointer to `AsyncPostgresSaver`).

---

## ADR-005 — UI Layer: Switch from Chainlit to Streamlit

* **Date, status & scope:** Documentation date: 21 November 2026 | **Status:** Accepted | **Supersedes:** Chainlit UI choice in ADR-001 | **Scope:** `ui.py`
* **Context & decision drivers:** Evaluators needed a simple single-command local UI capable of JWT shopper login/switching, thread resets, displaying policy citations, and surfacing per-turn Langfuse trace IDs alongside chat without async WebSocket lifecycle conflicts.
* **Options considered:**
  1. **Keep Chainlit:** Polished chat-first UX, but adds a separate ASGI/WebSocket runtime and requires extra customization for operational controls.
  2. **Streamlit (Selected):** Single-command local startup (`streamlit run ui.py`), straightforward session-state management for JWT + `thread_id`, and easy rendering of citations and trace IDs.
* **Decision & rationale:** Selected **Streamlit** for faster operational testing and cleaner local orchestration alongside the loopback MCP server.
* **Consequences & trade-offs:** Simplifies local setup and grading inspection; responses currently await completion before rendering rather than streaming tokens live.
* **Evidence & validation:** Working UI snapshots in Report Figs. 10–13 showing shopper sign-in, thread isolation, citations, and Langfuse trace IDs.
* **Revisit trigger:** Requirement for live token-by-token streaming UX or customer-facing web production deployment (migrate to Next.js/React frontend over SSE).