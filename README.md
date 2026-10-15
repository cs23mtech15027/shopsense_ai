# ShopSense

ShopSense is a local-first shopping support assistant prototype. It routes customer
requests to focused care, order, and refund agents; retrieves return policies from
Qdrant; and requires explicit approval before it executes a refund.

## Architecture

```text
FastAPI -> LLM safety gate -> manual input checks -> checkpointed LangGraph state
                                                     | care: policy retrieval
                                                     | order: verified order lookup
                                                     | refund: eligibility -> approval
                                      verified facts -> Ollama response generation
Chainlit <--------------------------- SSE response stream
```

All customer-facing answers are composed by the configured Ollama model from
verified policy/order/refund facts. The model is not an authority for order or
refund decisions. Orders and approvals in this prototype are demo data persisted
in SQLite. Replace the repository adapter with an authenticated commerce API
before handling real orders.

For an order-specific return question, care retrieves the relevant policy using
the order's item name and checks the verified order's delivery date, return
window, condition, tags, and final-sale status. It explains the eligibility
decision and asks whether the customer wants to proceed; this does not submit a
return or approve a refund.

## Run locally

Requirements: Python 3.11+, Docker Compose, and (for the local model) a machine
with enough memory for Qwen3 8B. The chat model is served locally by Ollama;
FastEmbed downloads and runs the embedding model locally on CPU.

1. Copy `.env.example` to `.env`, set `SHOPSENSE_APPROVAL_TOKEN` to a unique,
   non-empty development secret, and adjust settings if needed. Both the API and
   Chainlit UI read this setting; restart both processes after changing it.
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

There is no login screen. Use the **Demo customer** dropdown in Chainlit to choose
Customer 1, Customer 2, or Customer 3. Changing customers starts a fresh
conversation and issues a local demo session for the selected account.
The current read-only **Thread ID** is shown in the Chainlit Chat Settings panel
and updates when you switch customers or start a conversation with `/new`.

Each account has ten separate demo orders. IDs are `ORD-1101`–`ORD-1110`,
`ORD-2101`–`ORD-2110`, and `ORD-3101`–`ORD-3110` respectively. The first
order for each user is eligible for a return after approval; the second is
final-sale and ineligible. The other orders cover varied delivery and return
conditions. If a customer asks about an order without specifying one, ShopSense
asks for the order ID instead of assuming an order from another conversation.
These fixed credentials are only for local demonstration.

To restore seeded demo orders and clear test approvals/refunds before another
test, run from the repository root:

```powershell
python -m app.persistence.reset_demo_data
```

Confirm the prompt to proceed, or append `--yes` for a non-interactive reset.
This resets only seeded demo order records and their approval/refund artifacts;
it leaves auth sessions and conversation checkpoints intact. Unseeded orders
are not changed.

The LangGraph workflow persists its state after each step in a separate SQLite
checkpointer. Reuse the response's `conversation_id`, `thread_id`, and
`session_id` on follow-up chat requests. Thread IDs are random 12-character
keys. Each response also includes `user_id` and `turn_id`.
Raw turns are stored before the response completes; longer-term user memory is
consolidated asynchronously by the LLM every
`SHOPSENSE_MEMORY_DREAMING_INTERVAL` turns and stored in the ignored local path
`memories/{user_id}/history_summarization.json`; raw turn records retain all four
identifiers. The first policy retrieval
initializes the Qdrant collection and downloads the FastEmbed model; subsequent
retrievals reuse the local model and indexed chunks. Policy ingestion reads `.txt`,
`.pdf`, and `.csv` files from `data/policies/`. PDF text is extracted page by page;
PDF and CSV table rows are converted to header/value records so each embedding
retains the column context. Prose uses paragraph-aware chunks with a 30-word
overlap. Changed source content triggers a collection re-index. Health is
available at `GET /health`. Demo orders and policy text are under `data/`; do not
use them as production policy or customer records.

API processing logs are partitioned into rotating `logs/{thread_id}.log` files.
Records without an active conversation thread are written to
`logs/shopsense.log`. Logs include stage timings and completion/failure events.
Set `SHOPSENSE_LOG_DIRECTORY` to change the log folder. Logs omit customer
messages, model prompts, and generated response text.

The sample policy handbook PDF contains formatted tables and can be regenerated
after installing the development extras with:

```powershell
python tools/generate_policy_handbook.py
```

## API examples

Chat:

```json
{
  "conversation_id": "your-conversation-id",
  "session_id": "your-session-id",
  "message": "Where is order ORD-1101?"
}
```

For streamed answers, send the same JSON to `POST /v1/chat/stream` and read
Server-Sent Events. Status events report progress while a request is being
checked or an answer is prepared; validated customer-facing answer text is sent
as `token` events, followed by `complete`. Each stage is bounded by
`SHOPSENSE_CHAT_STAGE_TIMEOUT_SECONDS` (60 seconds by default). The regular
`/v1/chat` endpoint remains available for clients that prefer a single JSON
response.

Care, order, and refund specialists run directly in-process within the
checkpointed workflow. This avoids an unnecessary network/protocol hop while
keeping each specialist's tools and responsibilities separate.

The Chainlit customer dropdown obtains a session from
`POST /v1/auth/demo-session` with one of the three seeded customer IDs; the
returned token is attached as `Authorization: Bearer <access_token>` on chat
requests. The first request can omit `conversation_id`; the response provides one
to reuse. `session_id` can also be omitted; the API derives a stable value from
the authenticated user and conversation. The API resolves the customer from the
session and checks order ownership server-side. Sessions expire after 12 hours.
This no-login selector is for the local demo only and is not real customer
authentication. The UI's API address is controlled by `SHOPSENSE_API_URL` in
`.env`, defaulting to `http://localhost:8000`.

The Chainlit **Help topic** selector can route automatically or explicitly select
order status/delivery, refund/return, or care/policy. Automatic routing remains
the default.

Questions about return eligibility, policy, or steps (for example, "How do I
return this product?") are answered by the policy retrieval agent and do not
create a refund approval. A direct refund request (for example, "Please initiate
a refund for ORD-1101") checks eligibility and asks the customer to confirm the
order and amount with **Yes, proceed** or **No, cancel**. Only the Yes action
records the demo refund initiation; No cancels without initiating it.
General return/refund information uses the canonical `returns.txt` policy passage;
the assistant answers directly without including prior assistant messages as
policy context, and Chainlit appends the policy source after the answer.

Refund flow:

1. Ask to return/refund an order and include its order ID. Each demo account has
   ten customer-owned orders covering shipped, in-transit, eligible,
   outside-window, final-sale, missing-tag, and inspection-pending examples.
2. If deterministic eligibility checks pass, the assistant creates or reuses a
   pending approval. A pending request does not set `refund_initiated`.
3. Approve with `POST /v1/approvals/{approval_id}`, `{"approved": true}`, and the
   configured `X-Approval-Token` header. Only then are `refund_initiated=1`,
   `refund_initiated_date`, and the refund ledger record written. Declining leaves
   the order flags unchanged; repeating an approval is idempotent.
4. This records a demo refund initiation only. It does not call a payment provider
   or transfer funds.

## Design and safety notes

- Manual size and injection-string checks run before the model-based input safety
  gate. The agent workflow retains checkpointed state between every node.
- An LLM intent agent classifies requests from the current message and recent
  conversation; intent routing does not use keyword rules. It extracts and
  validates order IDs against the current message, conversation, and memory.
  Sentiment uses lightweight deterministic rules. All customer
  answers—including greetings, missing-information requests, order/refund
  outcomes, and approval outcomes—are generated by Ollama from verified facts.
- Order lookup and refund eligibility are deterministic. The refund route cannot
  call the refund operation until a persisted approval is approved.
- The refund repository uses a transaction and an idempotency key.
- Refund eligibility checks delivery date, unused condition, original tags,
  final-sale status, return receipt, inspection outcome, prior refund status, and
  the initiated flag.
- Refund approval calls require the configured `X-Approval-Token` header. Approval
  is disabled when no token is configured. The API and Chainlit UI must load the
  same non-empty `SHOPSENSE_APPROVAL_TOKEN` value from `.env`.
- Text, PDF, and CSV policy chunks/records are embedded locally, indexed in
  Qdrant, and cited by source. CSV/PDF table records retain each column name beside
  its value, avoiding embeddings of detached cell text.
- The latest order ID is retained in checkpointed state and user memory, then
  reused for relevant follow-up order/refund questions that omit it. Each
  specialist receives recent customer-visible conversation history and the
  consolidated memory; internal assistant reasoning is excluded from that
  history. Memory consolidation records the customer's request, the agent's
  answer, and the order ID with the turn identifiers. Greetings are generated by
  the care agent without invoking Qdrant. Care answers also receive retrieved
  policy chunks.
- Tool access is allowlisted by agent. This demo uses SQLite order data and local
  demo accounts; it is not connected to a production customer authentication
  provider, commerce backend, or payment rail. Do not use the demo credentials or
  session scheme for real customers.
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
