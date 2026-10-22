# Invoice Agent

An A2A JSON-RPC invoice agent intended for deployment on Render. It publishes an agent card at `/.well-known/agent-card.json` and accepts `message/send` (plus the legacy `tasks/send`) and `tasks/get` at `/` and `/a2a`.

## Local run

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export AI_API_KEY=your-provider-key
uvicorn app:app --host 0.0.0.0 --port 8000
```

The AI provider is OpenAI-compatible. Set `AI_BASE_URL` and `AI_MODEL` to use another hosted or local compatible provider. Analysis fails explicitly if no provider is configured.

## A2A requests

Check an invoice with JSON-RPC `message/send`:

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "message-1",
      "role": "user",
      "parts": [{"kind": "text", "text": "Vendor: Acme\\nInvoice number: A-42\\nAmount: USD 120.00\\nApproved for payment"}]
    }
  }
}
```

Attachments may be sent as A2A file parts with inline base64 `data`, `name`, and `mimeType`. Text, PDF, and DOCX are supported. Remote file URLs are intentionally not fetched.

To save the result, send another `message/send` request with `message.metadata.operation` set to `"save"` and `message.metadata.taskId` set to the checked task ID. The latest Save per principal is persisted. Check and Save share a rolling limit of 10 operations per 24 hours. Repeating a completed task ID returns its stored response without repeating analysis or actions.

## Payment safety and receipts

The agent only attempts payment if the analysis recommends `pay`, source text contains explicit payment approval, and vendor plus invoice number are both present. Invoice deduplication uses the pair `(vendor, invoice number)` so a different invoice is not treated as a duplicate.

No payment endpoint is configured by default. To enable actions, set `PAYMENT_ACTION_URL` to an authorized payment adapter that accepts a POST containing the invoice and approval evidence and returns JSON with a non-empty `receipt`. Requests include a stable `Idempotency-Key`; the adapter must honor this key for safe recovery after a timeout. The agent stores the actual returned receipt and never reports completion without one. Set `PAYMENT_ACTION_TOKEN` if the adapter requires bearer authentication.

Set `A2A_API_KEY` to require bearer authentication for agent calls. If unset, the service is publicly callable, with a shared anonymous operation quota.

## Render deployment

Create a new Render Web Service from this directory/repository using `render.yaml`. The persistent disk stores task, operation, payment, and receipt state at `/var/data/agent.db`. Configure `AI_API_KEY` in the Render dashboard; only configure payment environment variables when an authorized payment adapter is ready. Render assigns the public HTTPS service URL after deployment. Submit that URL (or its `/` A2A endpoint, depending on the evaluator instructions) and advertise the agent card URL at `/.well-known/agent-card.json`.
