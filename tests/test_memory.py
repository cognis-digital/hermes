"""Tests for the SQLite-backed memory store and TF-IDF recall."""

import os
import tempfile
import time
import unittest

from hermes import MemoryStore, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(
            tokenize("Hello, WORLD! 42", remove_stopwords=False),
            ["hello", "world", "42"],
        )

    def test_removes_stopwords(self):
        toks = tokenize("the user prefers the metric system")
        self.assertNotIn("the", toks)
        self.assertIn("user", toks)
        self.assertIn("metric", toks)


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_remember_returns_memory_with_id(self):
        mem = self.store.remember("the sky is blue", tags=["fact", "color"])
        self.assertGreater(mem.id, 0)
        self.assertEqual(mem.text, "the sky is blue")
        self.assertEqual(mem.tags, ["fact", "color"])
        self.assertEqual(self.store.count(), 1)

    def test_remember_empty_raises(self):
        with self.assertRaises(ValueError):
            self.store.remember("   ")

    def test_get_roundtrip(self):
        mem = self.store.remember("remember me", source="unit-test", metadata={"k": 1})
        fetched = self.store.get(mem.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.text, "remember me")
        self.assertEqual(fetched.source, "unit-test")
        self.assertEqual(fetched.metadata, {"k": 1})

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get(999))

    def test_recall_ranks_relevant_first(self):
        self.store.remember("The user prefers metric units and concise answers.")
        self.store.remember("Deployed the trading bot to paper mode today.")
        self.store.remember("The capital of France is Paris.")
        hits = self.store.recall("what units does the user prefer", limit=3)
        self.assertTrue(hits)
        self.assertIn("metric", hits[0].memory.text.lower())
        self.assertGreater(hits[0].score, 0.0)

    def test_recall_filters_irrelevant(self):
        self.store.remember("apples and oranges")
        self.store.remember("quantum chromodynamics")
        hits = self.store.recall("banana smoothie recipe")
        # No term overlap -> nothing returned.
        self.assertEqual(hits, [])

    def test_recall_empty_query(self):
        self.store.remember("something")
        self.assertEqual(self.store.recall(""), [])
        self.assertEqual(self.store.recall("   "), [])

    def test_recall_on_empty_store(self):
        self.assertEqual(self.store.recall("anything"), [])

    def test_tag_filter_in_recall(self):
        self.store.remember("buy GEV stock soon", tags=["watchlist"])
        self.store.remember("GEV is a great company", tags=["note"])
        hits = self.store.recall("GEV", tag="watchlist")
        self.assertEqual(len(hits), 1)
        self.assertIn("watchlist", hits[0].memory.tags)

    def test_forget_removes_and_updates_df(self):
        m1 = self.store.remember("uranium mining is capital intensive")
        self.store.remember("uranium prices rose this quarter")
        self.assertEqual(self.store.count(), 2)
        self.assertTrue(self.store.forget(m1.id))
        self.assertEqual(self.store.count(), 1)
        self.assertIsNone(self.store.get(m1.id))
        # Recall still works after a deletion.
        hits = self.store.recall("uranium")
        self.assertEqual(len(hits), 1)

    def test_forget_missing_returns_false(self):
        self.assertFalse(self.store.forget(123))

    def test_list_newest_first(self):
        a = self.store.remember("first")
        time.sleep(0.01)
        b = self.store.remember("second")
        listed = self.store.list()
        self.assertEqual(listed[0].id, b.id)
        self.assertEqual(listed[1].id, a.id)
        listed_old = self.store.list(newest_first=False)
        self.assertEqual(listed_old[0].id, a.id)

    def test_recency_boost_orders_ties(self):
        # Two memories with identical text; the newer one should win on recency.
        store = MemoryStore(":memory:", recency_halflife_days=1.0, recency_weight=0.5)
        old = store.remember("alpha beta gamma")
        # Backdate the first memory by 10 days directly in the DB.
        store._conn.execute(
            "UPDATE memories SET created_at = created_at - ? WHERE id = ?",
            (10 * 86400, old.id),
        )
        store._conn.commit()
        new = store.remember("alpha beta gamma")
        hits = store.recall("alpha beta gamma", limit=2)
        self.assertEqual(hits[0].memory.id, new.id)
        store.close()

    def test_clear(self):
        self.store.remember("x")
        self.store.remember("y")
        self.store.clear()
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.recall("x"), [])

    def test_stats(self):
        self.store.remember("hello world")
        s = self.store.stats()
        self.assertEqual(s["memories"], 1)
        self.assertGreaterEqual(s["vocabulary_terms"], 2)

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.sqlite")
            store = MemoryStore(path)
            store.remember("persistent fact about nuclear energy", tags=["energy"])
            store.close()
            # Reopen and confirm the memory and recall survived.
            store2 = MemoryStore(path)
            self.assertEqual(store2.count(), 1)
            hits = store2.recall("nuclear energy")
            self.assertTrue(hits)
            self.assertIn("nuclear", hits[0].memory.text.lower())
            store2.close()


if __name__ == "__main__":
    unittest.main()
