import unittest
from nitwit.tools import web_search, needs_web_search


class TestNeedsWebSearch(unittest.TestCase):
    def test_current_info_true(self):
        for t in ["what's the latest Next.js version?", "current price of an rtx 4090",
                  "who is the current ceo of tesla", "when does the next iphone come out",
                  "any news on the EU AI act in 2026"]:
            self.assertTrue(needs_web_search(t), t)

    def test_ordinary_chat_false(self):
        for t in ["what does parse() do?", "explain closures", "hi", "thanks",
                  "refactor this function", "how does a hash map work",
                  "what's the current branch?", "what version of python are we using", "fix this now"]:
            self.assertFalse(needs_web_search(t), t)


class TestWebSearch(unittest.TestCase):
    def test_formats_results(self):
        fake = {"result": {"results": [
            {"title": "Next.js", "snippet": "v15 is latest", "url": "https://nextjs.org"},
        ]}}
        out = web_search("next.js version", _run=lambda cap, arg: fake)
        self.assertIn("WEB RESULTS", out)
        self.assertIn("Next.js", out)
        self.assertIn("https://nextjs.org", out)

    def test_empty_and_failure_never_raise(self):
        self.assertIn("no results", web_search("x", _run=lambda cap, arg: {"result": {"results": []}}))
        def boom(cap, arg): raise RuntimeError("down")
        self.assertIn("no results", web_search("x", _run=boom))  # must not raise


if __name__ == "__main__":
    unittest.main()
