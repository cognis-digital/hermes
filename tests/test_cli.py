"""Tests for the hermes command-line interface."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from hermes import cli


class CLITests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "cli.sqlite")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["--db", self.db, *args])
        return code, buf.getvalue()

    def test_remember_and_recall(self):
        code, out = self._run(
            "remember", "The user's favorite ticker is GEV", "--tags", "watchlist,equities"
        )
        self.assertEqual(code, 0)
        self.assertIn("remembered #1", out)

        code, out = self._run("recall", "which ticker does the user favor")
        self.assertEqual(code, 0)
        self.assertIn("GEV", out)

    def test_json_output(self):
        self._run("remember", "json test memory")
        code, out = self._run("--json", "recall", "json test")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data)
        self.assertIn("score", data[0])

    def test_list_and_get_and_forget(self):
        self._run("remember", "alpha")
        self._run("remember", "beta")
        code, out = self._run("list")
        self.assertEqual(code, 0)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)

        code, out = self._run("get", "1")
        self.assertEqual(code, 0)
        self.assertIn("#1", out)

        code, out = self._run("forget", "1")
        self.assertEqual(code, 0)
        self.assertIn("forgot #1", out)

        code, _ = self._run("get", "1")
        self.assertEqual(code, 1)

    def test_stats(self):
        self._run("remember", "stat me")
        code, out = self._run("stats")
        self.assertEqual(code, 0)
        self.assertIn("memories", out)

    def test_metadata_must_be_json_object(self):
        with self.assertRaises(SystemExit):
            self._run("remember", "x", "--metadata", "not json")


if __name__ == "__main__":
    unittest.main()
