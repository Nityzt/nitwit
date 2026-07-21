import unittest
from orchestrator import ModelResponse
from nitwit.coder import MissionContext
from nitwit.model_coder import ModelCoder, parse_file_edits, build_coder_messages


def _resp(content):
    return ModelResponse(content=content, elapsed_s=0.1, usage={}, timings={}, raw={})


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.last_messages = messages
        self.last_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        return _resp(self.content)


def _ctx(**kw):
    base = dict(goal="make add() return a+b", constraints=["python only"], notes="",
               last_test_output="", repo_files={"feature.py": "def add(a,b):\n    return 0\n"})
    base.update(kw)
    return MissionContext(**base)


class TestParseFileEdits(unittest.TestCase):
    def test_parses_single_block(self):
        text = "Here:\n```file:feature.py\ndef add(a, b):\n    return a + b\n```\n"
        edits = parse_file_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].path, "feature.py")
        self.assertEqual(edits[0].content, "def add(a, b):\n    return a + b\n")

    def test_parses_multiple_blocks(self):
        text = ("```file:a.py\nx = 1\n```\n"
                "```file:sub/b.py\ny = 2\n```\n")
        edits = parse_file_edits(text)
        self.assertEqual([e.path for e in edits], ["a.py", "sub/b.py"])

    def test_no_blocks_returns_empty(self):
        self.assertEqual(parse_file_edits("no edits here, just prose"), [])

    def test_strips_think_block(self):
        text = "<think>let me plan</think>\n```file:a.py\nz = 3\n```"
        edits = parse_file_edits(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].content, "z = 3\n")


class TestBuildMessages(unittest.TestCase):
    def test_prompt_includes_goal_constraints_tests_and_files(self):
        msgs = build_coder_messages(_ctx(last_test_output="AssertionError: 0 != 5"))
        joined = "\n".join(m["content"] for m in msgs)
        self.assertIn("make add() return a+b", joined)
        self.assertIn("python only", joined)
        self.assertIn("AssertionError", joined)
        self.assertIn("feature.py", joined)
        self.assertEqual(msgs[0]["role"], "system")


class TestModelCoder(unittest.TestCase):
    def test_propose_returns_parsed_edits(self):
        client = FakeClient("```file:feature.py\ndef add(a, b):\n    return a + b\n```")
        coder = ModelCoder(client)
        out = coder.propose(_ctx())
        self.assertEqual(len(out.edits), 1)
        self.assertEqual(out.edits[0].path, "feature.py")
        self.assertEqual(client.last_kwargs["temperature"], 0.0)

    def test_propose_no_edits_when_model_emits_prose(self):
        coder = ModelCoder(FakeClient("I think this looks fine already."))
        self.assertEqual(coder.propose(_ctx()).edits, [])


if __name__ == "__main__":
    unittest.main()
