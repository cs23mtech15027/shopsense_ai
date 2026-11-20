# Hybrid end-to-end evaluation

The golden runner uses the configured Groq model, Qdrant Cloud Inference, the
local Streamable HTTP MCP server, local SQLite checkpoints, and the same JWT
boundary as the UI. It has no local vector/LLM fallback and does not read the
OMS directly.

## Prerequisites

1. Complete the setup in the root README, seed the demo database, and ingest
   policies.
2. Start the local MCP server in one terminal:

   ```powershell
   python -m app.mcp_server
   ```

3. Generate a JWT, then add it to `.env` as `USER_JWT`:

   ```powershell
   python scripts/get_user_jwt.py --user-id customer-1
   ```

4. Ensure the configured Groq/Qdrant credentials are valid. Policy retrieval
   metrics require the cloud collection to be ingested.

## Run

```powershell
python golden_test/run_golden_test.py
```

The runner evaluates routing, Qdrant top-3 source relevance, return/refund
guardrails, and the workflow/MCP integration. Results are written to
`data/golden_dataset/golden_test_results.csv`. Prompt-injection cases must be
blocked before workflow invocation. Refund action cases are intentionally
limited to seeded demo orders; the local database records simulated status
changes and does not move money.

Report the dataset version, model, Qdrant collection, retrieval precision/recall,
intent accuracy, guardrail outcomes, timestamp, and dependency versions with each
run. Do not compare a provider-backed run with old keyword-ranking or synthetic
approval scores; those are not end-to-end results.
