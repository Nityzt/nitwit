"""`wit` — the CLI client over the loopback daemon API. REPL + one-shot + subcommands."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_URL = os.environ.get("NITWIT_URL", "http://127.0.0.1:8807")


def api_call(base: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return 0, {"error": f"cannot reach daemon at {base}: {e.reason}. Is it running? (python3 -m nitwit)"}


def humanize_event(event: dict) -> str:
    kind = event.get("event", "?")
    mid = event.get("mission_id", "")
    if kind == "mission_started":   return f"[{mid}] mission started"
    if kind == "iteration_started": return f"[{mid}] iteration {event.get('iteration')}"
    if kind == "edits_applied":     return f"[{mid}] edited: {', '.join(event.get('paths', []))}"
    if kind == "criteria_evaluated":return f"[{mid}] checked: {event.get('summary','')} -> {'PASS' if event.get('passed') else 'not yet'}"
    if kind == "mission_finished":  return f"[{mid}] finished: {event.get('state')}"
    if kind == "mission_error":     return f"[{mid}] ERROR: {event.get('error')}"
    return f"[{mid}] {kind}"


def cmd_status(args, base):
    _, s = api_call(base, "GET", "/status")
    print(json.dumps(s, indent=2) if isinstance(s, dict) else s)


def cmd_ls(args, base):
    _, missions = api_call(base, "GET", "/missions")
    if not isinstance(missions, list):
        print(missions); return
    for m in missions:
        print(f"{m['id']:14} {m['state']:12} iter={m.get('iteration',0):<3} {m.get('goal','')[:60]}")


def cmd_new(args, base):
    repos = []
    if args.repo:
        repos = [{"path": os.path.abspath(args.repo), "branch": f"agent/{args.branch}",
                  "test_cmd": args.test or "", "checkpoint_commit": ""}]
    crit = []
    if args.test:
        crit.append({"type": "tests", "repo": os.path.abspath(args.repo), "cmd": args.test})
    crit.append({"type": "verifier", "description": "the goal is meaningfully complete"})
    _, m = api_call(base, "POST", "/missions",
                    {"goal": args.goal, "repos": repos, "success_criteria": crit})
    print(f"created {m.get('id')} ({m.get('state')})" if isinstance(m, dict) and m.get("id") else m)


def _simple(action):
    def fn(args, base):
        _, m = api_call(base, "POST", f"/missions/{args.id}/{action}")
        print(f"{args.id}: {m.get('state', m)}" if isinstance(m, dict) else m)
    return fn


def cmd_answer(args, base):
    _, m = api_call(base, "POST", f"/missions/{args.id}/answer", {"answer": args.text})
    print(f"{args.id}: {m.get('state', m)}" if isinstance(m, dict) else m)


def cmd_toggle(on):
    def fn(args, base):
        _, s = api_call(base, "POST", "/control/on" if on else "/control/off")
        print(f"daemon {'ON' if on else 'OFF'}: {s}")
    return fn


def stream(base, out=print):
    req = urllib.request.Request(base + "/events")
    try:
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    out(humanize_event(json.loads(line[6:])))
    except KeyboardInterrupt:
        pass
    except urllib.error.URLError as e:
        out(f"stream ended: {e}")


def cmd_tail(args, base):
    stream(base)


def repl(base):
    print("wit — interactive. /help for commands, /quit to exit. Missions keep running in the daemon.")
    while True:
        try:
            line = input("wit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print("/ls /status /tail /new <goal> --repo P --test CMD /pause <id> /resume <id> "
                  "/cancel <id> /answer <id> <text> /on /off /quit")
            continue
        argv = line[1:].split() if line.startswith("/") else ["new"] + [line]
        try:
            main(argv + ["--url", base])
        except SystemExit:
            pass


def build_parser():
    p = argparse.ArgumentParser(prog="wit")
    p.add_argument("-p", "--prompt", help="one-shot: create a mission from this goal and exit")
    p.add_argument("--url", default=DEFAULT_URL)
    sub = p.add_subparsers(dest="cmd")
    for name in ("status", "ls", "tail", "on", "off"):
        s = sub.add_parser(name); s.add_argument("--url", default=DEFAULT_URL)
    n = sub.add_parser("new"); n.add_argument("goal"); n.add_argument("--repo"); n.add_argument("--test"); n.add_argument("--branch", default="mission"); n.add_argument("--url", default=DEFAULT_URL)
    for name in ("pause", "resume", "cancel"):
        s = sub.add_parser(name); s.add_argument("id"); s.add_argument("--url", default=DEFAULT_URL)
    a = sub.add_parser("answer"); a.add_argument("id"); a.add_argument("text"); a.add_argument("--url", default=DEFAULT_URL)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = args.url
    dispatch = {
        "status": cmd_status, "ls": cmd_ls, "new": cmd_new, "tail": cmd_tail,
        "pause": _simple("pause"), "resume": _simple("resume"), "cancel": _simple("cancel"),
        "answer": cmd_answer, "on": cmd_toggle(True), "off": cmd_toggle(False),
    }
    if args.prompt:
        args.goal, args.repo, args.test, args.branch = args.prompt, None, None, "mission"
        return cmd_new(args, base)
    if not args.cmd:
        return repl(base)
    dispatch[args.cmd](args, base)


if __name__ == "__main__":
    main()
