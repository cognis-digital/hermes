"""Tests for the MCP stdio server dispatch logic."""

import io
import json
import unittest

from hermes.memory import MemoryStore
from hermes.mcp_server import HermesMCPServer, PROTOCOL_VERSION, TOOLS, serve_stdio


class MCPDispatchTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.server = HermesMCPServer(self.store)

    def tearDown(self):
        self.store.close()

    def _req(self, method, params=None, msg_id=1):
        return self.server.handle(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        )

    def test_initialize(self):
        resp = self._req("initialize", {"protocolVersion": PROTOCOL_VERSION})
        self.assertEqual(resp["result"]["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(resp["result"]["serverInfo"]["name"], "cognis-hermes")

    def test_tools_list_exposes_all_tools(self):
        resp = self._req("tools/list")
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertEqual(
            names, {"remember", "recall", "forget", "list_memories"}
        )
        self.assertEqual(len(resp["result"]["tools"]), len(TOOLS))

    def test_remember_then_recall_via_tools(self):
        rem = self._req(
            "tools/call",
            {"name": "remember", "arguments": {"text": "the user likes uranium plays"}},
        )
        self.assertFalse(rem["result"]["isError"])
        payload = json.loads(rem["result"]["content"][0]["text"])
        self.assertTrue(payload["ok"])

        rec = self._req(
            "tools/call",
            {"name": "recall", "arguments": {"query": "what does the user like"}},
        )
        results = json.loads(rec["result"]["content"][0]["text"])["results"]
        self.assertTrue(results)
        self.assertIn("uranium", results[0]["text"].lower())

    def test_forget_tool(self):
        rem = self._req(
            "tools/call",
            {"name": "remember", "arguments": {"text": "forget this one"}},
        )
        mem_id = json.loads(rem["result"]["content"][0]["text"])["memory"]["id"]
        forget = self._req(
            "tools/call", {"name": "forget", "arguments": {"id": mem_id}}
        )
        self.assertTrue(json.loads(forget["result"]["content"][0]["text"])["ok"])

    def test_remember_missing_text_is_tool_error(self):
        resp = self._req(
            "tools/call", {"name": "remember", "arguments": {}}
        )
        self.assertTrue(resp["result"]["isError"])

    def test_unknown_method(self):
        resp = self._req("does/not/exist")
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    def test_notification_returns_none(self):
        # No 'id' key -> notification -> no response.
        out = self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertIsNone(out)

    def test_serve_stdio_roundtrip(self):
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "remember",
                        "arguments": {"text": "stdio works end to end"},
                    },
                }
            ),
        ]
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        serve_stdio(self.store, stdin=stdin, stdout=stdout)
        out_lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(out_lines), 2)
        init_resp = json.loads(out_lines[0])
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "cognis-hermes")


class MCPHardeningTests(unittest.TestCase):
    """Error-path and edge-case tests added during hardening."""

    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.server = HermesMCPServer(self.store)

    def tearDown(self):
        self.store.close()

    def _req(self, method, params=None, msg_id=1):
        return self.server.handle(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        )

    def test_recall_non_integer_limit_returns_tool_error(self):
        resp = self._req(
            "tools/call",
            {"name": "recall", "arguments": {"query": "hello", "limit": "five"}},
        )
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("limit", resp["result"]["content"][0]["text"])

    def test_recall_non_number_min_score_returns_tool_error(self):
        resp = self._req(
            "tools/call",
            {"name": "recall", "arguments": {"query": "hello", "min_score": "high"}},
        )
        self.assertTrue(resp["result"]["isError"])

    def test_list_memories_non_integer_limit_returns_tool_error(self):
        resp = self._req(
            "tools/call",
            {"name": "list_memories", "arguments": {"limit": "all"}},
        )
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("limit", resp["result"]["content"][0]["text"])

    def test_forget_non_integer_id_returns_tool_error(self):
        resp = self._req(
            "tools/call",
            {"name": "forget", "arguments": {"id": "abc"}},
        )
        self.assertTrue(resp["result"]["isError"])

    def test_unknown_tool_name_returns_tool_error(self):
        resp = self._req(
            "tools/call",
            {"name": "does_not_exist", "arguments": {}},
        )
        self.assertTrue(resp["result"]["isError"])

    def test_serve_stdio_malformed_json_returns_parse_error(self):
        stdin = io.StringIO("this is not json\n")
        stdout = io.StringIO()
        serve_stdio(self.store, stdin=stdin, stdout=stdout)
        line = stdout.getvalue().strip()
        self.assertTrue(line)
        resp = json.loads(line)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32700)


if __name__ == "__main__":
    unittest.main()
