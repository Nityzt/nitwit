import unittest
from nitwit.tools import web_search, needs_web_search, fetch_url, gather_context


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


class TestFetchUrl(unittest.TestCase):
    def test_extracts_text_and_strips_tags_and_scripts(self):
        html = ("<html><head><title>t</title><style>.x{}</style></head>"
                "<body><h1>One Piece</h1><script>var x=1;</script>"
                "<p>Chapter 1140 is the latest.</p></body></html>")
        out = fetch_url("http://x", _get=lambda u: html)
        self.assertIn("One Piece", out)
        self.assertIn("Chapter 1140 is the latest.", out)
        self.assertNotIn("var x", out)     # script dropped
        self.assertNotIn(".x{}", out)      # style dropped

    def test_caps_length(self):
        html = "<p>" + ("ab " * 2000) + "</p>"
        self.assertLessEqual(len(fetch_url("http://x", max_chars=100, _get=lambda u: html)), 100)

    def test_never_raises_on_failure(self):
        def boom(u): raise RuntimeError("net down")
        self.assertEqual(fetch_url("http://x", _get=boom), "")
        self.assertEqual(fetch_url("http://x", _get=lambda u: ""), "")


class TestGatherContext(unittest.TestCase):
    def test_builds_context_with_page_content_and_sources(self):
        def search(q, limit=6):
            return "WEB RESULTS:\n- A: blurb (https://a.example/x)\n- B: blurb (https://b.example/y)"
        def fetch(u):
            return f"real page text for {u}"
        ctx = gather_context("q", _search=search, _fetch=fetch)
        self.assertIn("PAGE CONTENT", ctx["context"])
        self.assertIn("real page text for https://a.example/x", ctx["context"])
        self.assertEqual(ctx["sources"], ["https://a.example/x", "https://b.example/y"])

    def test_never_raises_when_search_and_fetch_fail(self):
        def bad_search(q, limit=6): raise RuntimeError("x")
        def bad_fetch(u): raise RuntimeError("y")
        ctx = gather_context("q", _search=bad_search, _fetch=bad_fetch)
        self.assertIn("context", ctx)
        self.assertIsInstance(ctx["sources"], list)


if __name__ == "__main__":
    unittest.main()
