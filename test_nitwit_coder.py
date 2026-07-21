import unittest
from nitwit.workspace import FileEdit
from nitwit.coder import MissionContext, CoderResponse, FakeCoder, FakeVerifier


class TestFakes(unittest.TestCase):
    def ctx(self):
        return MissionContext(goal="g", constraints=[], notes="", last_test_output="", repo_files={})

    def test_fake_coder_returns_scripted_then_empty(self):
        r1 = CoderResponse(edits=[FileEdit("a.py", "x=1\n")])
        coder = FakeCoder([r1])
        out = coder.propose(self.ctx())
        self.assertEqual(out.edits[0].path, "a.py")
        # after the script is exhausted, returns an empty response (no edits)
        self.assertEqual(coder.propose(self.ctx()).edits, [])
        self.assertEqual(coder.calls, 2)

    def test_fake_verifier(self):
        v = FakeVerifier(verdict=False)
        self.assertFalse(v.judge("is it good?", self.ctx()))
        self.assertEqual(v.calls, 1)


if __name__ == "__main__":
    unittest.main()
