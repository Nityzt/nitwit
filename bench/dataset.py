"""Fixed, labeled task set for the orchestrator benchmark harness.

Ground-truth labels are what a human considers the *correct* behaviour, so the
harness measures how well each stage (heuristic today, MiniCPM/Qwen later) matches
human judgement — not just what the current code happens to do.
"""
from __future__ import annotations

# --- routing: prompt -> the mode a human would want it routed to -------------
# Modes come from MODE_CONFIGS / classify_request in webui.py. Project modes are
# excluded here (they need real on-disk paths); this set is the non-project router.
ROUTING_CASES: list[dict] = [
    {"prompt": "hey", "mode": "chat"},
    {"prompt": "hello there", "mode": "chat"},
    {"prompt": "find the release date for the next one piece manga chapter", "mode": "web_research"},
    {"prompt": "when does the next iphone come out", "mode": "web_research"},
    {"prompt": "how much is a tesla model 3 right now", "mode": "web_research"},
    {"prompt": "what is the latest news on eu ai regulation", "mode": "web_research"},
    {"prompt": "who won the last formula 1 race", "mode": "web_research"},
    {"prompt": "search for python asyncio tutorials", "mode": "search_results"},
    {"prompt": "google the best mechanical keyboards", "mode": "search_results"},
    {"prompt": "explain how a hash map handles collisions", "mode": "direct_answer"},
    {"prompt": "what is the difference between a process and a thread", "mode": "direct_answer"},
    {"prompt": "write a python function to reverse a linked list", "mode": "implementation"},
    {"prompt": "implement a rate limiter in go", "mode": "implementation"},
    {"prompt": "refactor this loop to use a comprehension", "mode": "implementation"},
    {"prompt": "debug why my flask server crashes on startup", "mode": "debug"},
    {"prompt": "my sort function returns the wrong order, whats the bug", "mode": "debug"},
    {"prompt": "plan a migration from flask to fastapi", "mode": "plan"},
    {"prompt": "give me a roadmap for building a cli backup tool", "mode": "plan"},
]

# --- verifier: canned worker results -> whether a human would pass them -------
VERIFIER_CASES: list[dict] = [
    {
        "request": "List three tradeoffs of using SQLite vs Postgres for a small web app.",
        "worker_results": [
            {"id": "t1", "answer": "SQLite is serverless and zero-config; Postgres needs a running server. "
                                    "SQLite handles low concurrency; Postgres scales to many writers. "
                                    "Postgres has richer types and extensions; SQLite is simpler to back up."},
        ],
        "pass": True,
    },
    {
        "request": "Give the exact CLI command to create a Postgres database named 'shop'.",
        "worker_results": [
            {"id": "t1", "answer": "You can create a database in Postgres using various tools and interfaces "
                                    "depending on your setup and preferences."},
        ],
        "pass": False,   # dodges the concrete ask
    },
    {
        "request": "Summarize the plot of a book.",
        "worker_results": [
            {"id": "t1", "answer": "The book is about a journey. It has characters. Things happen and it ends."},
        ],
        "pass": False,   # non-answer
    },
    {
        "request": "What are two reasons to use a message queue?",
        "worker_results": [{"id": "t1", "answer": "1) Decoupling: producers and consumers need not run at the same time. "
                                                    "2) Load leveling: the queue absorbs bursts so consumers work at a steady rate."}],
        "pass": True,
    },
    {
        "request": "Convert 100 F to Celsius and show the formula.",
        "worker_results": [{"id": "t1", "answer": "Temperature conversion is common; many online tools can help you with it."}],
        "pass": False,   # ignores the concrete ask (formula + value)
    },
    {
        "request": "Name the HTTP status code for 'Not Found'.",
        "worker_results": [{"id": "t1", "answer": "The HTTP status code for Not Found is 404."}],
        "pass": True,
    },
    {
        "request": "Give a regex to match a US ZIP code (exactly 5 digits).",
        "worker_results": [{"id": "t1", "answer": "Use ^[0-9]{5}$ to match exactly five digits."}],
        "pass": True,
    },
    {
        "request": "Explain why the given code throws a NullPointerException and how to fix it.",
        "worker_results": [{"id": "t1", "answer": "NullPointerExceptions happen in Java. Review your code carefully and add null checks where appropriate."}],
        "pass": False,   # generic; never diagnoses the specific code
    },
]

# --- query rewrite: request -> what a good web-search query should look like --
QUERY_REWRITE_CASES: list[dict] = [
    {"request": "can you search when the next one piece chapter comes out please",
     "must_include": ["one", "piece", "chapter"],
     "must_exclude": ["can", "you", "please", "search"]},
    {"request": "hey could you look up the current price of an rtx 4090",
     "must_include": ["price", "rtx", "4090"],
     "must_exclude": ["hey", "could", "you", "look"]},
]

# --- memory suggestion extraction: answer text -> expected proposed key -------
MEMORY_CASES: list[dict] = [
    {"answer": 'Sure. {"memory_suggestion":{"scope":"user","key":"editor","value":"prefers tabs over spaces",'
               '"tags":["style"],"reason":"stated as a firm preference"}}',
     "expect_key": "editor"},
    {"answer": "Here is a one-off calculation with no durable fact to remember.",
     "expect_key": None},
]

# --- tool request extraction: answer text -> expected requested capability ----
TOOL_CASES: list[dict] = [
    {"answer": 'Need more evidence. {"tool_request":{"capability":"git_status","input":{"path":"/home/nit/x"},'
               '"reason":"check branch"}}',
     "expect_capability": "git_status"},
    {"answer": "A plain answer with no tool request at all.",
     "expect_capability": None},
]
