# ShopSense

ShopSense is a local-first shopping support assistant prototype. It routes customer
requests to focused care, order, and refund agents; retrieves return policies from
Qdrant; and requires explicit approval before it executes a refund.

## Architecture

```text
FastAPI -> input guardrails -> LangGraph intent/sentiment triage
                                | care: Qdrant policy retrieval -> Ollama answer
                                | order: deterministic order lookup
                                | refund: deterministic eligibility -> approval
                         output guardrails -> customer
```

The model is an explainer, not an authority for order or refund decisions. Orders
and approvals in this prototype are demo data persisted in SQLite. Replace the
repository adapter with an authenticated commerce API before handling real orders.

## Run locally

Requirements: Python 3.11+, Docker Compose, and (for the local model) a machine
with enough memory for Qwen3 8B. The model and embeddings are served locally by
Ollama; the initial Qdrant embedding model is downloaded by FastEmbed.

1. Copy `.env.example` to `.env`, set `SHOPSENSE_APPROVAL_TOKEN` to a unique
   development secret, and adjust settings if needed.
2. Start services: `docker compose up -d`
3. Pull the local chat model: `docker compose exec ollama ollama pull qwen3:8b`
4. Install and run the API:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   uvicorn app.main:app --reload
   ```

5. Open `http://localhost:8000/docs`.

The LangGraph workflow persists the most recent six conversational turns per
conversation in a separate SQLite checkpointer. Reuse the response's
`conversation_id` on follow-up chat requests. The first policy retrieval
initializes the Qdrant collection and downloads the FastEmbed model; subsequent
retrievals reuse the local model and indexed chunks. Policy ingestion reads `.txt`
and `.pdf` files from `data/policies/`, splitting PDFs by page before chunking and
retaining file/page source labels. Changed source content triggers a collection
re-index. Health is available at `GET /health`. Demo orders and policy text are under
`data/`; do not use them as production policy or customer records.

## API examples

Chat:

```json
{
  "customer_id": "customer-1001",
  "conversation_id": "your-conversation-id",
  "message": "Where is order ORD-1001?"
}
```

The first request can omit `conversation_id`; the response provides one to reuse.
The sample order is intentionally tied to `customer-1001`. The API checks order
ownership against the submitted demo customer ID, but does not authenticate it.

Refund flow:

1. Ask to return/refund an order and include its order ID. Demo order `ORD-1001`
   represents an item already returned and inspected; the other sample orders are
   not refund-eligible.
2. The assistant returns a `pending_approval` request if the order is eligible.
3. Call `POST /v1/approvals/{approval_id}` with `{"approved": true}` and the
   configured `X-Approval-Token` header to record the demo refund. Declining
   cancels it. Repeated approvals are idempotent.

## Design and safety notes

- Routing uses inexpensive deterministic rules, avoiding an extra model call for
  intent and basic sentiment classification. Ambiguous requests are sent to care
  for clarification.
- Order lookup and refund eligibility are deterministic. The refund route cannot
  call the refund operation until a persisted approval is approved.
- The refund repository uses a transaction and an idempotency key.
- Refund eligibility checks delivery date, unused condition, original tags,
  final-sale status, return receipt, inspection outcome, and prior refund status.
- Refund approval calls require the configured `X-Approval-Token` header. Approval
  is disabled when no token is configured.
- Policy chunks are embedded locally, indexed in Qdrant, and cited by source.
- The chat model receives the current request, up to two recent conversational
  turns, and a small set of retrieved policy chunks; order/refund agents do not
  send conversation history to a model.
- Tool access is allowlisted by agent. This demo uses SQLite order data and is not
  connected to a real commerce backend, customer authentication provider, or
  payment rail. The `customer_id` in `/v1/chat` is demo input, not verified identity;
  never expose this API to customers or connect real customer records until an
  authentication dependency resolves identity server-side.
- `agent.json` and prompt text are agent-local configuration; never put secrets
  or customer data in prompt files.

## Project layout

```text
app/
  agents/{care,order,refund}_agent/
    prompts/
    tools/
    agent.py
    agent.json
    callback.py
  api/
  guardrails/
  llm/
  persistence/
  retrieval/
  workflow/
data/
tests/
```

Run focused tests with `pytest`.
