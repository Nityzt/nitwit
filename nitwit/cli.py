"""`wit` — the CLI client over the loopback daemon API. REPL + one-shot + subcommands."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import urllib.request
from nitwit import session

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


def stream_events_for(base, mission_id, out=print):
    try:
        with urllib.request.urlopen(base + "/events") as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("mission_id") != mission_id:
                    continue
                out(humanize_event(ev))
                if ev.get("event") in ("mission_finished", "mission_error"):
                    return
    except KeyboardInterrupt:
        out("(detached — mission keeps running; `/missions` to check, `wit diff <id>` to review)")
    except Exception as exc:
        out(f"(stream ended: {exc})")


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
    goal = " ".join(args.goal) if isinstance(args.goal, list) else args.goal
    if args.test and not args.repo:
        print("--test requires --repo")
        return
    repos = []
    if args.repo:
        repos = [{"path": os.path.abspath(args.repo), "branch": f"agent/{args.branch}",
                  "test_cmd": args.test or "", "checkpoint_commit": ""}]
    crit = []
    if args.repo and args.test:
        crit.append({"type": "tests", "repo": os.path.abspath(args.repo), "cmd": args.test})
    crit.append({"type": "verifier", "description": "the goal is meaningfully complete"})
    _, m = api_call(base, "POST", "/missions",
                    {"goal": goal, "repos": repos, "success_criteria": crit})
    print(f"created {m.get('id')} ({m.get('state')})" if isinstance(m, dict) and m.get("id") else m)


def _simple(action):
    def fn(args, base):
        _, m = api_call(base, "POST", f"/missions/{args.id}/{action}")
        print(f"{args.id}: {m.get('state', m)}" if isinstance(m, dict) else m)
    return fn


def cmd_answer(args, base):
    text = " ".join(args.text) if isinstance(args.text, list) else args.text
    _, m = api_call(base, "POST", f"/missions/{args.id}/answer", {"answer": text})
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
    except (ConnectionError, http.client.IncompleteRead) as e:
        out(f"stream ended: {e}")


def cmd_tail(args, base):
    stream(base)


def _start_mission(base, repo, test_cmd, goal):
    if not repo:
        print("this looks like a task, but you're not in a git repo. cd into one and try again.")
        return
    crit = []
    if test_cmd:
        crit.append({"type": "tests", "repo": repo, "cmd": test_cmd})
    crit.append({"type": "verifier", "description": "the request is meaningfully and correctly done"})
    branch = "agent/" + __import__("nitwit.missions", fromlist=["slugify"]).slugify(goal)[:40]
    _, m = api_call(base, "POST", "/missions",
                    {"goal": goal, "repos": [{"path": repo, "branch": branch, "test_cmd": test_cmd or "",
                                              "checkpoint_commit": ""}], "success_criteria": crit})
    if not (isinstance(m, dict) and m.get("id")):
        print(m); return
    api_call(base, "POST", "/control/on")
    print(f"→ mission {m['id']} on {branch} (Ctrl-C to detach; it keeps running)")
    stream_events_for(base, m["id"])
    _, fin = api_call(base, "GET", f"/missions/{m['id']}")
    if isinstance(fin, dict):
        print(f"mission {m['id']}: {fin.get('state')} · review: wit diff {m['id']} (or git -C {repo} diff main..{branch})")


def interactive(base, cwd, coder_url, coder_model):
    if not session.ensure_daemon(base):
        print("could not start the nitwit daemon; check ~/.local/share/nitwit/daemon.log")
        return
    repo = session.repo_root(cwd)
    test_cmd = session.detect_test_cmd(repo) if repo else None
    where = repo or f"{cwd} (not a git repo — tasks need a repo)"
    print(f"nitwit · {where} · tests: {test_cmd or 'none detected'} · /help, /quit")
    while True:
        try:
            line = input("wit ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/help":
            print("Talk naturally: a question is answered here; a task ('add ...', 'fix ...') "
                  "runs as a mission on a branch (Ctrl-C detaches).\n"
                  "/missions  /diff <id>  /status  /on  /off  /mission <goal>  /quit")
            continue
        if line.startswith("/"):
            parts = line[1:].split()
            cmd = parts[0] if parts else ""
            if cmd == "missions": cmd_ls(argparse.Namespace(), base); continue
            if cmd == "status":   cmd_status(argparse.Namespace(), base); continue
            if cmd in ("on", "off"): cmd_toggle(cmd == "on")(argparse.Namespace(), base); continue
            if cmd == "diff" and len(parts) > 1:
                _, m = api_call(base, "GET", f"/missions/{parts[1]}")
                print(json.dumps(m, indent=2) if isinstance(m, dict) else m); continue
            if cmd == "mission" and len(parts) > 1:
                _start_mission(base, repo, test_cmd, " ".join(parts[1:])); continue
            print("unknown command; /help"); continue
        if session.classify_intent(line) == "task":
            _start_mission(base, repo, test_cmd, line)
        else:
            session.stream_answer(line, repo, coder_url=coder_url, coder_model=coder_model)


def build_parser():
    p = argparse.ArgumentParser(prog="wit")
    p.add_argument("-p", "--prompt", help="one-shot: create a mission from this goal and exit")
    p.add_argument("--url", default=DEFAULT_URL)
    sub = p.add_subparsers(dest="cmd")
    # Each subparser's --url defaults to argparse.SUPPRESS: when the subcommand
    # doesn't repeat --url, argparse's namespace merge leaves the parent's
    # value alone instead of stomping it with a subparser default. When --url
    # IS given after the subcommand, it still overrides normally.
    for name in ("status", "ls", "tail", "on", "off"):
        s = sub.add_parser(name); s.add_argument("--url", default=argparse.SUPPRESS)
    n = sub.add_parser("new")
    n.add_argument("goal", nargs="+")
    n.add_argument("--repo"); n.add_argument("--test"); n.add_argument("--branch", default="mission")
    n.add_argument("--url", default=argparse.SUPPRESS)
    for name in ("pause", "resume", "cancel"):
        s = sub.add_parser(name); s.add_argument("id"); s.add_argument("--url", default=argparse.SUPPRESS)
    a = sub.add_parser("answer")
    a.add_argument("id"); a.add_argument("text", nargs="+")
    a.add_argument("--url", default=argparse.SUPPRESS)
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
        return interactive(base, os.getcwd(),
                           os.environ.get("NITWIT_CODER_URL", "http://127.0.0.1:8080"),
                           os.environ.get("NITWIT_CODER_MODEL", "qwen2.5-coder-7b"))
    dispatch[args.cmd](args, base)


if __name__ == "__main__":
    main()
