import os, tempfile, unittest
from nitwit.memory import MemoryStore, propose_memory


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "mem.db")

    def test_add_list_facts_delete_persist(self):
        s = MemoryStore(self.db)
        i = s.add("uses pnpm not npm")
        s.add("prefers tabs")
        self.assertEqual(len(s.list()), 2)
        self.assertIn("uses pnpm not npm", s.facts())
        # dedupe: adding same text again doesn't duplicate
        s.add("uses pnpm not npm")
        self.assertEqual(len(s.list()), 2)
        # persists across instances
        self.assertIn("prefers tabs", MemoryStore(self.db).facts())
        self.assertTrue(s.delete(i))
        self.assertNotIn("uses pnpm not npm", MemoryStore(self.db).facts())

    def test_add_ignores_empty(self):
        s = MemoryStore(self.db)
        s.add("   ")
        self.assertEqual(s.list(), [])


class TestProposeMemory(unittest.TestCase):
    def test_durable_facts_proposed(self):
        for t in ["I use pnpm not npm", "my name is Nit", "call me Wit",
                  "we use FastAPI for the backend", "I always use type hints",
                  "remember that the db is postgres", "I prefer tabs over spaces"]:
            self.assertIsNotNone(propose_memory(t), t)

    def test_ordinary_chat_not_proposed(self):
        for t in ["what does parse() do?", "hi", "thanks", "add a health endpoint",
                  "how does the loop work", "is this thread-safe?"]:
            self.assertIsNone(propose_memory(t), t)


if __name__ == "__main__":
    unittest.main()
