import os
import tempfile
import threading
import unittest

from nitwit.missions import MissionStore


class TestMissionStoreConcurrency(unittest.TestCase):
    """Stress MissionStore with many threads hammering the same connection.

    Guards against the pre-fix hazard: one shared sqlite3 connection with
    check_same_thread=False but NO mutual exclusion. Concurrent threads
    calling create/get/list/set_state on the same connection can raise
    sqlite3.ProgrammingError ("recursive use of cursors") or interleave a
    read mid-write. After the fix (a single threading.RLock guarding every
    method that touches self._conn), this must pass reliably.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MissionStore(os.path.join(self.tmp, "missions.db"))

    def test_concurrent_create_get_list_set_state(self):
        n_threads = 8
        n_iterations = 50
        errors = []
        created_ids = []
        created_lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker(worker_idx):
            try:
                barrier.wait()
                for i in range(n_iterations):
                    m = self.store.create(f"worker-{worker_idx}-mission-{i}")
                    with created_lock:
                        created_ids.append(m.id)

                    # get
                    got = self.store.get(m.id)
                    assert got is not None
                    assert got.id == m.id

                    # list (both filtered and unfiltered)
                    self.store.list()
                    self.store.list(state="queued")

                    # drive this mission's own state machine: queued -> running -> done
                    self.store.set_state(m.id, "running")
                    self.store.set_state(m.id, "done")
            except Exception as exc:  # noqa: BLE001 - we want to catch everything
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"threads raised: {errors!r}")
        self.assertEqual(len(self.store.list()), len(created_ids))
        self.assertEqual(len(created_ids), n_threads * n_iterations)


if __name__ == "__main__":
    unittest.main()
