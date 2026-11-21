# ShopSense — Architecture Decision Records (ADRs)

**Course:** CS5903: Distributed AI Training  
**Project:** ShopSense - Customer Care & Order Operations Assistant  
**Student:** Yashkumar Joshi (`CS23MTECH15027`)  
**Repository:** https://github.com/cs23mtech15027/shopsense_ai  
**Documentation Date:** 21 November 2026

---

## ADR-001 — Initial Local-Only Multi-Agent Architecture (9 October 2026 Baseline)

### Date, status & scope
* **Date:** 9 October 2026 *(documented 21 November 2026)*
* **Status:** Superseded by `ADR-002`
* **Scope:** Complete initial runtime stack — local LLM inference, local Policy RAG embeddings/vector DB, FastAPI + Chainlit UI, local JSON memory consolidation, and SQLite OMS.

### Context & decision drivers
By the 9 October 2026 methodology checkpoint, ShopSense required a self-contained, zero-cost prototype to validate stateful multi-agent e-commerce customer support (order tracking, multi-format policy Q&A, and deterministic return/refund eligibility). Key drivers were avoiding external API billing during early bring-up, proving deterministic policy/refund enforcement in Python/SQLite (mitigating chatbot policy-hallucination liability), and running offline on a local system.

### Options considered
1. **Local-Only Stack:** Local Ollama (`Qwen 4B`), local `BAAI/bge-small-en-v1.5` embeddings (~133 MB), local Qdrant, FastAPI + Chainlit UI, local JSON history summarization, and direct SQLite OMS queries with static `X-Approval-Token` approval endpoints.
2. **Hybrid Cloud-Local Multi-Agent Architecture:** Hosted LLM inference, cloud vector DB, and remote backend services. Offered lower latency and stronger models, but added early credential management, quota setup, and deployment complexity before the core LangGraph workflow was verified.

### Decision & rationale
Selected **Option 1 (Local-Only Stack)** on 9 October 2026 to establish a working, offline-reproducible baseline for the LangGraph triage-and-specialist architecture (`Care`, `Order`, `Refund`) and verify multi-format `.txt`/`.csv`/`.pdf` ingestion and SQLite eligibility rules with zero external dependencies.

### Consequences & trade-offs
* **Benefits:** Proved end-to-end LangGraph routing, table-aware PDF/CSV policy extraction, and deterministic refund eligibility checks without API costs.
* **Costs & limitations:** High end-to-end latency (**~30 seconds per request** on local hardware); tight local RAM/disk limits (≥5 GB RAM needed for `Qwen 4B` plus ~133 MB for local embeddings) blocked upgrading to a stronger tool-capable LLM; `Qwen 4B` produced weaker greeting and nuanced triage quality; manual LLM summarization into local JSON files added extra latency and created lossy static snapshots.

### Evidence & validation
* Baseline architecture and component breakdown documented in `docs/Project_Report.pdf` (Section 2 & Baseline Table).
* Early prototype code history and baseline SQLite OMS schema (`customers`, `auth_sessions`, `orders`, `approvals`, `refunds`). Commit/PR reference: `06375bdbcf312fbae4ddabd034aba53e52047988`.

### Revisit trigger
Observed per-request latency persistently exceeding 30 seconds, or local RAM/disk constraints preventing upgrade beyond a 4B-parameter model for reliable triage and tool use.

---

## ADR-002 — Transition to Hybrid Cloud-Local Multi-Agent Architecture

### Date, status & scope
* **Decision Date:** Post-9 October 2026 pivot *(documented 21 November 2026; mentor approval record: Animesh Singh)*
* **Status:** Accepted
* **Supersedes:** `ADR-001`
* **Scope:** Full hybrid architecture consolidation — cloud LLM inference (Groq), cloud Policy RAG embeddings (Qdrant Cloud Inference), loopback Streamable HTTP MCP server with per-request JWT auth, persistence split (local SQLite checkpoints + optional Supermemory), Streamlit UI, and optional redacted Langfuse EU observability.

### Context & decision drivers
Live evaluation of the `ADR-001` local baseline triggered both revisit conditions:
1. **Latency bottleneck:** Local inference on a personal laptop (Intel Core i3, 4 GB RAM) averaged **~30 seconds per request**, making interactive demos and 30-query golden-dataset sweeps impractical.
2. **Hardware ceiling:** Limited local RAM and disk space locked inference to `Qwen 4B` and required local embedding model downloads (~133 MB), limiting greeting/triage accuracy and evaluator reproducibility.
3. **Security, state & operational gaps:** Direct SQLite calls and static approval tokens lacked per-user JWT isolation (IDOR protection), local JSON summarization burned extra LLM calls and created stale files, and Chainlit lacked simple side-by-side operational controls (JWT login, thread switching, citations, trace IDs).

### Options considered
1. **Keep Local-Only Baseline (`ADR-001`):** Preserves offline execution, but remains stuck at ~25–30 s/request latency and 4B-parameter model limits.
2. **Hybrid Cloud-Local Architecture (Selected):** Preserves the exact same LangGraph multi-agent architecture while moving compute-heavy stateless inference and vector search to free-tier cloud services and keeping the UI, JWT-protected MCP tool boundary, SQLite OMS, and thread checkpoints local.

### Decision & rationale
Selected **Option 2 (Hybrid Cloud-Local Architecture)**, consolidating six coordinated upgrades under a single strict topology (removing dual-mode local/cloud toggles):

| Subsystem | Previous (`ADR-001` Local) | Selected (`ADR-002` Hybrid) | Rationale |
| :--- | :--- | :--- | :--- |
| **LLM Inference** | Local Ollama (`Qwen 4B`) | **Groq Cloud API** (`openai/gpt-oss-120b`, Apache 2.0) | Eliminates local RAM/GPU bottlenecks, cuts latency sharply, and improves greeting, schema-constrained triage, and policy synthesis. |
| **Policy RAG & Embeddings** | Local `BAAI/bge-small-en-v1.5` (~133 MB, 512-tok) + local Qdrant | **Qdrant Cloud Inference** (`sentence-transformers/all-minilm-l6-v2`, 384-dim, 256-tok) | Eliminates local ML/CPU runtime and disk downloads using free server-side inference in a single network hop; chunks kept at 180 words to fit the 256-token window. |
| **OMS Tool Boundary & Auth** | Direct SQLite queries + `X-Approval-Token` | **Loopback Streamable HTTP MCP** (`127.0.0.1:8001/mcp`) + **per-request Bearer JWT** | Enforces protocol separation and IDOR defense: MCP validates JWT signature/expiry/session, extracts `sub` (`customer_id`), enforces `WHERE customer_id = ?`, and runs deterministic refund & ₹5,000 Tier-2 rules in code. |
| **State & Long-Term Memory** | Local JSON history consolidation (`memories/*.json`) | **Local SQLite Checkpoints** (`data/checkpoints.db`) + **Optional Supermemory** | Keeps exact per-thread execution state and active `order_id` local while offloading sanitized cross-thread shopper context to Supermemory (managed reconsolidation), preserving Groq free-tier token quota. |
| **Frontend UI** | Chainlit + FastAPI | **Local Streamlit UI** (`ui.py`) | Single-command startup (`streamlit run ui.py`) with integrated JWT shopper login, thread switcher, policy source citations, and per-turn trace IDs. |
| **Observability & Guardrails** | Basic local logs & minimal output trim | **Input/Output Guardrails + PII/JWT Redaction + Optional Langfuse EU + Rotating Local Logs** | Defense-in-depth injection/compliance checks; raw JWTs stay in runtime context only and are redacted before cloud tracing or logging. |

### Consequences & trade-offs
* **Measured benefits:** End-to-end latency dropped from **~30 s/request** (local baseline) to **p90 = 15.0 s** across **108 Langfuse EU traces**; greeting handling and golden-dataset response quality improved; evaluators need zero local GPU or embedding downloads.
* **Trade-offs & risks:**
  * Introduces external internet dependencies and free-tier Groq rate limits (30 RPM / 8K TPM), mitigated via retry backoff and graceful no-op fallbacks for optional Supermemory and Langfuse.
  * Moving from `BAAI/bge-small-en-v1.5` (33.4M params, 512-token window) to Qdrant Cloud's free server-side `sentence-transformers/all-minilm-l6-v2` (22.7M params, 256-token window) trades some standalone dense retrieval ceiling for zero local footprint—contributing to Returns & Refunds Recall@3 of `0.5455` (Overall Recall@3 = `0.7391`).
  * Loopback MCP (`127.0.0.1`) and local SQLite are single-host demo boundaries; multi-user cloud production requires TLS, OIDC/JWKS, and `AsyncPostgresSaver`.
  * Streamlit awaits full turn completion rather than streaming tokens live.
* **Impact on milestones & frozen tests:** All frozen acceptance tests (IDOR isolation, deterministic refund eligibility, explicit confirmation before mutation, prompt-injection blocking, and 30-query golden evaluation) remained intact without weakening criteria; automated test coverage expanded to 63 tests.

### Evidence & validation
* **Latency & Observability:** 108 `shopsense-turn` root traces in Langfuse EU confirming **p90 = 15.0 s** (`docs/Project_Report.pdf`, Fig. 4).
* **Golden Dataset (30 queries):** Policy RAG Mean Recall@3 = `0.7391` (Delivery = `1.0000`, Damage/Safety = `0.7500`, Returns/Refunds = `0.5455`), Mean Precision@3 = `0.2899` (`data/golden_dataset/golden_test_results.csv`, `golden_test_execution.log`).
* **Automated Test Suite (`pytest`):** 63/63 tests passing across guardrails (6/6), LLM retries (6/6), intent detection (6/6), confirmation gates (4/4), MCP/JWT & OMS isolation (13/13), Qdrant Cloud Inference ingestion (3/3), RAG metrics (5/5), response composition (11/11), optional Supermemory (4/4), and observability (5/5).
* **Implementation Links:** `app/workflow/router.py`, `app/mcp_server.py`, `app/auth_context.py`, `app/retrieval/policies.py`, `app/persistence/checkpoints.py`, `app/persistence/supermemory.py`, `app/observability.py`, `ui.py`, and commit/PR `60e3c8738f21c3b50382721f93069ad6a826f3b7`.
