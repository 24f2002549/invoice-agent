import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app


class InvoiceAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "agent.db"
        self.path_patch = patch.object(app, "DATABASE_PATH", self.db_path)
        self.path_patch.start()
        app.initialize_database()
        self.client = TestClient(app.app)

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def send(self, task_id, text):
        return self.client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": task_id,
                "method": "message/send",
                "params": {
                    "taskId": task_id,
                    "message": {"parts": [{"kind": "text", "text": text}]},
                },
            },
        )

    def test_agent_card_and_health(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        card = self.client.get("/.well-known/agent-card.json").json()
        self.assertEqual(card["protocolVersion"], "0.3.0")
        self.assertEqual(card["preferredTransport"], "JSONRPC")

    def test_task_retry_returns_saved_result_without_second_check(self):
        fake_analysis = {
            "vendor": "Acme",
            "invoice_number": "A-42",
            "amount": 120,
            "currency": "USD",
            "due_date": None,
            "recommended_action": "hold",
            "summary": "Await approval.",
            "evidence": [],
        }
        with patch.object(app, "_call_ai", return_value=fake_analysis) as mocked:
            first = self.send("same-task", "Invoice A-42")
            second = self.send("same-task", "Invoice A-42")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(mocked.call_count, 1)

    def test_message_send_route_accepts_jsonrpc_and_direct_message(self):
        analysis = {
            "vendor": "Acme",
            "invoice_number": "A-42",
            "recommended_action": "hold",
            "summary": "Needs review.",
            "evidence": [],
        }
        with patch.object(app, "_call_ai", return_value=analysis):
            rpc_response = self.client.post(
                "/message:send",
                json={
                    "jsonrpc": "2.0",
                    "id": "rpc-message",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "parts": [{"kind": "text", "text": "Acme invoice A-42"}]
                        }
                    },
                },
            )
            direct_response = self.client.post(
                "/message:send",
                json={
                    "messageId": "direct-message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Acme invoice A-42"}],
                },
            )
        self.assertEqual(rpc_response.status_code, 200)
        self.assertEqual(rpc_response.headers["content-type"], "application/json")
        self.assertEqual(rpc_response.json()["jsonrpc"], "2.0")
        self.assertEqual(direct_response.status_code, 200)
        self.assertEqual(direct_response.headers["content-type"], "application/json")
        self.assertEqual(direct_response.json()["kind"], "task")

    def test_only_same_vendor_and_invoice_is_duplicate(self):
        invoice = {
            "vendor": "Acme",
            "invoice_number": "A-42",
            "amount": 120,
            "currency": "USD",
            "due_date": None,
            "recommended_action": "pay",
            "summary": "Approved invoice.",
            "evidence": [],
        }
        def analyze(text):
            result = dict(invoice)
            if "A-43" in text:
                result["invoice_number"] = "A-43"
            return result

        with patch.object(app, "_call_ai", side_effect=analyze), patch.dict(
            "os.environ", {"PAYMENT_ACTION_URL": "https://payments.invalid"}
        ), patch.object(
            app.httpx, "post", return_value=type(
                "Response",
                (),
                {
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"receipt": {"id": "r-1"}},
                },
            )()
        ) as gateway:
            first = self.send("task-1", "Acme A-42 approved for payment")
            same = self.send("task-2", "Acme A-42 approved for payment")
            other = self.send("task-3", "Acme A-43 approved for payment")
        first_data = first.json()["result"]["artifacts"][0]["parts"][0]["data"]
        same_data = same.json()["result"]["artifacts"][0]["parts"][0]["data"]
        other_data = other.json()["result"]["artifacts"][0]["parts"][0]["data"]
        self.assertEqual(first_data["payment"]["state"], "completed")
        self.assertEqual(same_data["payment"]["state"], "duplicate")
        self.assertEqual(other_data["payment"]["state"], "completed")
        self.assertEqual(gateway.call_count, 2)

    def test_rejects_payment_without_explicit_approval(self):
        analysis = {
            "vendor": "Acme",
            "invoice_number": "A-42",
            "amount": 120,
            "currency": "USD",
            "recommended_action": "pay",
            "evidence": [],
        }
        with patch.object(app, "_call_ai", return_value=analysis), patch.object(app, "_payment_action") as action:
            response = self.send("unapproved", "Acme invoice A-42")
        result = response.json()["result"]["artifacts"][0]["parts"][0]["data"]
        self.assertEqual(result["recommended_action"], "hold")
        action.assert_not_called()

    def test_save_consumes_quota_and_updates_most_recent_save(self):
        result = {"result": "stored"}
        task = app._task_response("checked", "context", result)
        with app._transaction() as connection:
            connection.execute(
                "INSERT INTO tasks(task_id, principal, state, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("checked", "public", "completed", json.dumps(task), 1),
            )
        response = self.client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "save-1",
                "method": "message/send",
                "params": {
                    "taskId": "save-task-1",
                    "message": {
                        "metadata": {"operation": "save", "taskId": "checked"},
                        "parts": [{"kind": "text", "text": "Save"}],
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        saved_data = response.json()["result"]["artifacts"][0]["parts"][0]["data"]
        self.assertTrue(saved_data["saved"])
        fetched = self.client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "get-checked",
                "method": "tasks/get",
                "params": {"id": "checked"},
            },
        )
        self.assertTrue(
            fetched.json()["result"]["artifacts"][0]["parts"][0]["data"]["saved"]
        )
        with app._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
            saved = connection.execute("SELECT task_id FROM saves WHERE principal='public'").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(saved, "checked")

    def test_check_and_save_operations_share_daily_limit(self):
        for _ in range(app.MAX_OPERATIONS_PER_24H):
            app._consume_operation("public", "check")
        with self.assertRaises(HTTPException) as error:
            app._consume_operation("public", "save")
        self.assertEqual(error.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
