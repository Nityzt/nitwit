import unittest
from nitwit.missions import Mission, slugify


class TestMissionObject(unittest.TestCase):
    def test_defaults(self):
        m = Mission(id="m1", goal="add a health endpoint")
        self.assertEqual(m.state, "queued")
        self.assertEqual(m.iteration, 0)
        self.assertEqual(m.constraints, [])
        self.assertEqual(m.success_criteria, [])
        self.assertEqual(m.repos, [])
        self.assertEqual(m.notes, "")

    def test_row_round_trip_preserves_structured_fields(self):
        m = Mission(
            id="m1", goal="g", title="t",
            constraints=["no new deps"],
            success_criteria=[{"type": "tests", "repo": "/r", "cmd": "pytest"}],
            repos=[{"path": "/r", "branch": "agent/t", "test_cmd": "pytest", "checkpoint_commit": ""}],
            notes="started",
        )
        row = m.to_row()
        # list/dict fields must be JSON strings in the row (SQLite-friendly)
        self.assertIsInstance(row["success_criteria"], str)
        back = Mission.from_row(row)
        self.assertEqual(back.success_criteria, m.success_criteria)
        self.assertEqual(back.repos, m.repos)
        self.assertEqual(back.constraints, m.constraints)
        self.assertEqual(back, m)

    def test_slugify(self):
        self.assertEqual(slugify("Add a /health Endpoint!"), "add-a-health-endpoint")
        self.assertEqual(slugify("  Fix   the BUG  "), "fix-the-bug")
        self.assertTrue(slugify("").startswith("mission"))


if __name__ == "__main__":
    unittest.main()
