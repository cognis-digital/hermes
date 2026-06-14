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


class CLIHardeningTests(unittest.TestCase):
    """Error-path and edge-case tests added during hardening."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "harden.sqlite")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args, capture_stderr=False):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = cli.main(["--db", self.db, *args])
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_bad_db_path_exits_nonzero(self):
        """Opening a DB in a non-existent directory should exit with code 2."""
        import io
        from contextlib import redirect_stderr, redirect_stdout
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        bad_db = os.path.join(self._tmp.name, "no_such_dir", "mem.sqlite")
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = cli.main(["--db", bad_db, "stats"])
        self.assertEqual(code, 2)
        self.assertIn("error", buf_err.getvalue().lower())

    def test_negative_limit_exits_nonzero(self):
        """Passing --limit -1 should print an error to stderr and exit 2."""
        code, _out, err = self._run("list", "--limit", "-1")
        self.assertEqual(code, 2)
        self.assertIn("limit", err.lower())

    def test_negative_offset_exits_nonzero(self):
        code, _out, err = self._run("list", "--offset", "-3")
        self.assertEqual(code, 2)
        self.assertIn("offset", err.lower())

    def test_remember_empty_text_exits_nonzero(self):
        code, _out, err = self._run("remember", "   ")
        self.assertNotEqual(code, 0)

    def test_forget_missing_id_exits_nonzero(self):
        code, _out, _err = self._run("forget", "999")
        self.assertEqual(code, 1)

    def test_get_missing_id_prints_error_to_stderr(self):
        code, _out, err = self._run("get", "999")
        self.assertEqual(code, 1)
        self.assertIn("999", err)


if __name__ == "__main__":
    unittest.main()
