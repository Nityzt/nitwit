import unittest
from orchestrator import ModelResponse
from nitwit.coder import MissionContext
from nitwit.model_verifier import ModelVerifier, build_verifier_messages


def _resp(content):
    return ModelResponse(content=content, elapsed_s=0.1, usage={}, timings={}, raw={})


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.last_messages = None

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.last_messages = messages
        return _resp(self.content)


def _ctx():
    return MissionContext(goal="g", constraints=[], notes="", last_test_output="all passed",
                          repo_files={"feature.py": "def add(a,b): return a+b"})


class TestModelVerifier(unittest.TestCase):
    def test_pass_true(self):
        v = ModelVerifier(FakeClient('{"pass": true, "reason": "meets the goal"}'))
        self.assertTrue(v.judge("implementation is meaningful", _ctx()))

    def test_pass_false(self):
        v = ModelVerifier(FakeClient('{"pass": false, "reason": "stub only"}'))
        self.assertFalse(v.judge("implementation is meaningful", _ctx()))

    def test_stringy_verdict_normalized(self):
        v = ModelVerifier(FakeClient('{"pass": "yes", "reason": "ok"}'))
        self.assertTrue(v.judge("x", _ctx()))

    def test_unparseable_is_lenient_true(self):
        v = ModelVerifier(FakeClient("I cannot produce JSON here, sorry."))
        self.assertTrue(v.judge("x", _ctx()))

    def test_prompt_includes_description_and_evidence(self):
        client = FakeClient('{"pass": true, "reason": "y"}')
        ModelVerifier(client).judge("the endpoint returns 201", _ctx())
        joined = "\n".join(m["content"] for m in client.last_messages)
        self.assertIn("the endpoint returns 201", joined)
        self.assertIn("all passed", joined)  # last_test_output as evidence


if __name__ == "__main__":
    unittest.main()
