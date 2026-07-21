"""`python3 -m nitwit` — run the mission daemon + loopback API server."""
from __future__ import annotations

import os

from nitwit.api import make_server
from nitwit.daemon import MissionDaemon
from nitwit.factory import build_model_engine
from nitwit.missions import MissionStore

DB = os.environ.get("NITWIT_DB", os.path.expanduser("~/.local/share/nitwit/missions.db"))
PORT = int(os.environ.get("NITWIT_PORT", "8807"))


def main() -> None:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    store = MissionStore(DB)
    engine = build_model_engine(store)
    daemon = MissionDaemon(store, engine)
    daemon.start()
    server = make_server(daemon, port=PORT)
    print(f"nitwit daemon on http://127.0.0.1:{PORT} (db {DB})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
