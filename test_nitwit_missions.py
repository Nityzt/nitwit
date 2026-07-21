import unittest
import tempfile
import os
from nitwit.missions import Mission, slugify, MissionStore, InvalidTransition


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


class TestMissionStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "missions.db"))

    def test_create_assigns_id_and_queued(self):
        m = self.store.create("add health endpoint")
        self.assertTrue(m.id)
        self.assertEqual(m.state, "queued")
        self.assertGreater(m.created, 0)

    def test_get_and_list(self):
        a = self.store.create("task a")
        self.store.create("task b")
        self.assertEqual(self.store.get(a.id).goal, "task a")
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(len(self.store.list(state="queued")), 2)
        self.assertEqual(len(self.store.list(state="done")), 0)

    def test_persistence_across_instances(self):
        m = self.store.create("persist me", success_criteria=[{"type": "verifier", "description": "x"}])
        reopened = MissionStore(os.path.join(self.tmp, "missions.db"))
        got = reopened.get(m.id)
        self.assertEqual(got.success_criteria, [{"type": "verifier", "description": "x"}])

    def test_valid_transition(self):
        m = self.store.create("t")
        self.store.set_state(m.id, "running")
        self.assertEqual(self.store.get(m.id).state, "running")

    def test_invalid_transition_raises(self):
        m = self.store.create("t")  # queued
        with self.assertRaises(InvalidTransition):
            self.store.set_state(m.id, "done")  # queued -> done is illegal

    def test_bump_iteration_and_notes(self):
        m = self.store.create("t")
        self.store.bump_iteration(m.id)
        self.assertEqual(self.store.get(m.id).iteration, 1)
        self.store.append_note(m.id, "first")
        self.store.append_note(m.id, "second")
        self.assertIn("first", self.store.get(m.id).notes)
        self.assertIn("second", self.store.get(m.id).notes)

    def test_timestamps_are_floats_not_strings_after_fetch_and_reopen(self):
        m = self.store.create("t")
        fetched = self.store.get(m.id)
        self.assertIs(type(fetched.created), float)
        self.assertIs(type(fetched.updated), float)

        reopened = MissionStore(os.path.join(self.tmp, "missions.db"))
        refetched = reopened.get(m.id)
        self.assertIs(type(refetched.created), float)
        self.assertIs(type(refetched.updated), float)
        self.assertAlmostEqual(refetched.created, m.created)
        self.assertAlmostEqual(refetched.updated, m.updated)

    def test_bump_iteration_missing_id_raises(self):
        with self.assertRaises(InvalidTransition):
            self.store.bump_iteration("does-not-exist")

    def test_append_note_missing_id_raises(self):
        with self.assertRaises(InvalidTransition):
            self.store.append_note("does-not-exist", "x")


if __name__ == "__main__":
    unittest.main()
