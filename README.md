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
with enough memory for Qwen3 8B. The chat model is served locally by Ollama;
FastEmbed downloads and runs the embedding model locally on CPU.

1. Copy `.env.example` to `.env`, set `SHOPSENSE_APPROVAL_TOKEN` to a unique
   development secret, and adjust settings if needed.
2. Start services: `docker compose up -d`
3. Pull the local chat model: `docker compose exec ollama ollama pull qwen3:8b`
4. Install and run the API:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev,ui]"
   uvicorn app.main:app --reload
   ```

5. In a second terminal with the virtual environment activated, run the Chainlit UI:

   ```powershell
   python run_ui.py
   ```

6. Open `http://localhost:8001` for the chat UI, or `http://localhost:8000/docs`
   for the API documentation.

The LangGraph workflow persists the most recent six conversational turns per
conversation in a separate SQLite checkpointer. Reuse the response's
`conversation_id` on follow-up chat requests. The first policy retrieval
initializes the Qdrant collection and downloads the FastEmbed model; subsequent
retrievals reuse the local model and indexed chunks. Policy ingestion reads `.txt`,
`.pdf`, and `.csv` files from `data/policies/`. PDF text is extracted page by page;
PDF and CSV table rows are converted to header/value records so each embedding
retains the column context. Prose uses paragraph-aware chunks with a 30-word
overlap. Changed source content triggers a collection re-index. Health is
available at `GET /health`. Demo orders and policy text are under `data/`; do not
use them as production policy or customer records.

The sample policy handbook PDF contains formatted tables and can be regenerated
after installing the development extras with:

```powershell
python tools/generate_policy_handbook.py
```

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
The Chainlit UI uses `SHOPSENSE_DEMO_CUSTOMER_ID` from `.env` and the `SHOPSENSE_API_URL`
setting to connect to the API. The default API address is `http://localhost:8000`.
To connect the UI to a different API address, set `SHOPSENSE_API_URL` in `.env`
before starting Chainlit.

The Chainlit **Help topic** selector can route automatically or explicitly select
order status/delivery, refund/return, or care/policy. Automatic routing remains
the default.

Refund flow:

1. Ask to return/refund an order and include its order ID. Twelve demo orders
   (`ORD-1001`–`ORD-1012`) have distinct customer IDs and products, and cover
   shipped, in-transit, eligible, outside-window, final-sale, missing-tag,
   inspection-pending, and already-initiated examples.
2. If deterministic eligibility checks pass, the assistant creates or reuses a
   pending approval. A pending request does not set `refund_initiated`.
3. Approve with `POST /v1/approvals/{approval_id}`, `{"approved": true}`, and the
   configured `X-Approval-Token` header. Only then are `refund_initiated=1`,
   `refund_initiated_date`, and the refund ledger record written. Declining leaves
   the order flags unchanged; repeating an approval is idempotent.
4. This records a demo refund initiation only. It does not call a payment provider
   or transfer funds.

## Design and safety notes

- Routing uses inexpensive deterministic rules, avoiding an extra model call for
  intent and basic sentiment classification. Ambiguous requests are sent to care
  for clarification.
- Order lookup and refund eligibility are deterministic. The refund route cannot
  call the refund operation until a persisted approval is approved.
- The refund repository uses a transaction and an idempotency key.
- Refund eligibility checks delivery date, unused condition, original tags,
  final-sale status, return receipt, inspection outcome, prior refund status, and
  the initiated flag.
- Refund approval calls require the configured `X-Approval-Token` header. Approval
  is disabled when no token is configured.
- Text, PDF, and CSV policy chunks/records are embedded locally, indexed in
  Qdrant, and cited by source. CSV/PDF table records retain each column name beside
  its value, avoiding embeddings of detached cell text.
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
- Chainlit approval actions send the approval token from server-side settings to
  the API; the token is not embedded in browser chat content. Customer IDs remain
  demo-only and must be replaced with authenticated identity before real use.

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
tools/
tests/
ui.py
run_ui.py
```

Run focused tests with `pytest`.
