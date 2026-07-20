#!/usr/bin/env python3
import json
import datetime as dt
import threading
import time
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from orchestrator import ModelResponse, Orchestrator, extract_json, extract_truncated_planner_json
from webui import (
    AUTH_TOKEN,
    BingParser,
    Handler,
    JobStore,
    Persistence,
    capability_searxng_search,
    capability_python_eval,
    capability_list_dir,
    classify_request,
    collect_attached_capabilities,
    compact_web_context_for_prompt,
    extract_memory_suggestions_from_text,
    extract_trace_memory_suggestions,
    docs_result_quality_score,
    docs_search_query,
    DuckDuckGoParser,
    extract_tool_requests_from_text,
    PageTextParser,
    job_metrics,
    ObservableClient,
    parsed_json_mapping,
    project_memory_scope,
    resolve_run_settings,
    run_capability,
    sanitize_conversation_history,
    search_results_are_relevant,
    should_prefetch_docs,
    source_authority_score,
    sports_result_search_queries,
    tool_evidence_prompt_block,
    web_result_quality_score,
    web_search_query_from_request,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []
        self.formats = []

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        self.calls.append(messages)
        self.formats.append(response_format)
        system = messages[0]["content"]
        if "planner" in system:
            content = json.dumps(
                {
                    "tasks": [
                        {"id": "t1", "title": "API shape", "prompt": "Design the API.", "depends_on": []},
                        {"id": "t2", "title": "Tests", "prompt": "Design test cases.", "depends_on": []},
                    ]
                }
            )
            return ModelResponse(content, 0.1, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, {}, {})
        if "focused worker" in system:
            content = "Worker answer for " + messages[-1]["content"].split("Assigned subtask", 1)[1][:30]
            return ModelResponse(content, 0.2, {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}, {}, {})
        if "context compactor" in system:
            content = json.dumps(
                {
                    "summary": "Compact worker result.",
                    "key_points": ["point"],
                    "decisions": [],
                    "risks": [],
                    "open_questions": [],
                    "use_later": ["point"],
                }
            )
            return ModelResponse(content, 0.1, {"prompt_tokens": 12, "completion_tokens": 12, "total_tokens": 24}, {}, {})
        if "synthesize" in system:
            return ModelResponse("Final synthesized answer.", 0.1, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, {}, {})
        if "verifier" in system:
            return ModelResponse('{"pass": true, "issues": [], "missing_tasks": []}', 0.1, {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}, {}, {})
        raise AssertionError(system)


class SlowClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, *, temperature, max_tokens, response_format=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return ModelResponse("ok", 0.05, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, {}, {})


class OrchestratorTests(unittest.TestCase):
    def test_extract_json_from_fenced_text(self):
        self.assertEqual(extract_json('```json\n{"ok": true}\n```'), {"ok": True})

    def test_bearer_token_auth_helper(self):
        webui = __import__("webui")
        original_admin = webui.ADMIN_TOKEN
        original_restricted = set(webui.RESTRICTED_TOKENS)
        webui.ADMIN_TOKEN = "admin-secret"
        webui.RESTRICTED_TOKENS = {"roommate-secret"}
        try:
            handler = object.__new__(Handler)
            handler.headers = {"Authorization": "Bearer admin-secret"}
            self.assertTrue(handler.is_authenticated())
            self.assertTrue(handler.is_admin())
            handler.headers = {"Authorization": "Bearer roommate-secret"}
            self.assertTrue(handler.is_authenticated())
            self.assertFalse(handler.is_admin())
            handler.headers = {"Authorization": "Bearer wrong"}
            self.assertFalse(handler.is_authenticated())
        finally:
            webui.ADMIN_TOKEN = original_admin
            webui.RESTRICTED_TOKENS = original_restricted

    def test_restricted_job_visibility(self):
        webui = __import__("webui")
        original_admin = webui.ADMIN_TOKEN
        original_restricted = set(webui.RESTRICTED_TOKENS)
        webui.ADMIN_TOKEN = "admin-secret"
        webui.RESTRICTED_TOKENS = {"roommate-secret"}
        try:
            handler = object.__new__(Handler)
            handler.headers = {"Authorization": "Bearer roommate-secret"}
            self.assertTrue(handler.can_access_job({"config": {"access_role": "restricted"}}))
            self.assertFalse(handler.can_access_job({"config": {"access_role": "admin"}}))
            handler.headers = {"Authorization": "Bearer admin-secret"}
            self.assertTrue(handler.can_access_job({"config": {"access_role": "admin"}}))
        finally:
            webui.ADMIN_TOKEN = original_admin
            webui.RESTRICTED_TOKENS = original_restricted

    def test_observable_client_serializes_model_calls_per_store(self):
        store = JobStore()
        first = store.create({"request": "a"})
        second = store.create({"request": "b"})
        client = SlowClient()
        wrapped_a = ObservableClient(client, store, first)
        wrapped_b = ObservableClient(client, store, second)

        def call(wrapped):
            wrapped.chat([{"role": "system", "content": "direct"}, {"role": "user", "content": "x"}], temperature=0, max_tokens=10)

        t1 = threading.Thread(target=call, args=(wrapped_a,))
        t2 = threading.Thread(target=call, args=(wrapped_b,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(client.max_active, 1)

    def test_extract_truncated_planner_json_recovers_complete_tasks(self):
        text = '''```json
{
  "strategy": "split it up",
  "tasks": [
    {"id": "t1", "title": "A", "prompt": "Do A", "depends_on": []},
    {"id": "t2", "title": "B", "prompt": "Do B"
```'''
        parsed = extract_truncated_planner_json(text)
        self.assertEqual(parsed["strategy"], "split it up")
        self.assertEqual(len(parsed["tasks"]), 1)
        self.assertEqual(parsed["tasks"][0]["id"], "t1")

    def test_json_stages_request_constrained_output(self):
        client = FakeClient()
        orch = Orchestrator(client, max_workers=2)
        orch.run("Build an orchestrator")
        # planner + verifier (+ compactors) must request a json_schema constraint
        set_formats = [f for f in client.formats if f is not None]
        self.assertGreaterEqual(len(set_formats), 2)
        self.assertTrue(all(f.get("type") == "json_schema" for f in set_formats))
        names = {f["json_schema"]["name"] for f in set_formats}
        self.assertIn("plan", names)
        self.assertIn("verify", names)
        # the verifier schema must force a real boolean pass, killing stringy verdicts
        verify_schema = next(f for f in set_formats if f["json_schema"]["name"] == "verify")
        self.assertEqual(verify_schema["json_schema"]["schema"]["properties"]["pass"]["type"], "boolean")
        # worker and synthesizer stages produce prose, so they must NOT be constrained
        self.assertIn(None, client.formats)

    def test_run_returns_trace(self):
        orch = Orchestrator(FakeClient(), max_workers=2)
        trace = orch.run("Build an orchestrator")
        self.assertEqual(len(trace["tasks"]), 2)
        self.assertTrue(trace["verifier"]["pass"])
        self.assertEqual(trace["final"], "Final synthesized answer.")
        self.assertEqual(len(trace["worker_results"]), 2)
        self.assertEqual(trace["usage_summary"]["calls"], 7)
        self.assertGreater(trace["usage_summary"]["total_tokens"], 0)
        self.assertIn("compact", trace["worker_results"][0])

    def test_python_eval_capability_is_restricted(self):
        self.assertEqual(capability_python_eval({"expression": "sum([1, 2, 3])"})["result"], 6)
        with self.assertRaises(ValueError):
            capability_python_eval({"expression": "__import__('os').system('pwd')"})

    def test_list_dir_capability(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as temp:
            Path(temp, "example.txt").write_text("hello")
            result = capability_list_dir({"path": temp})
            self.assertEqual(result["entries"][0]["name"], "example.txt")

    def test_capability_registry(self):
        result = run_capability("python_eval", {"expression": "max([3, 9, 2])"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["result"], 9)

    def test_duckduckgo_parser_extracts_result_links(self):
        parser = DuckDuckGoParser("https://html.duckduckgo.com/html/?q=test")
        parser.feed(
            '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Example Doc</a>'
            '<a class="result__snippet">Useful snippet text.</a>'
        )
        self.assertEqual(parser.results[0]["title"], "Example Doc")
        self.assertEqual(parser.results[0]["url"], "https://example.com/doc")
        self.assertEqual(parser.results[0]["snippet"], "Useful snippet text.")

    def test_bing_parser_extracts_result_links(self):
        parser = BingParser("https://www.bing.com/search?q=test")
        parser.feed(
            '<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9kb2M">'
            "Example Doc</a></h2><p>Useful snippet text.</p></li>"
        )
        self.assertEqual(parser.results[0]["title"], "Example Doc")
        self.assertEqual(parser.results[0]["url"], "https://example.com/doc")
        self.assertEqual(parser.results[0]["snippet"], "Useful snippet text.")

    def test_web_search_query_removes_conversational_wrapper(self):
        self.assertEqual(
            web_search_query_from_request("can you search the latest fifa game and give me the results"),
            "latest fifa game",
        )
        self.assertEqual(
            web_search_query_from_request("can you search the fifa world cup and give me the results for the last game"),
            f"fifa world cup last game latest completed result score as of {dt.date.today().isoformat()}",
        )
        self.assertEqual(
            web_search_query_from_request("fifa world cup last game latest completed result score as of 2026-07-17"),
            "fifa world cup last game latest completed result score as of 2026-07-17",
        )

    def test_sports_result_search_adds_targeted_queries(self):
        queries = sports_result_search_queries(
            "can you search the fifa world cup and give me the results for the last game",
            "fifa world cup last game latest completed result score as of 2026-07-17",
        )
        self.assertGreater(len(queries), 1)
        self.assertTrue(any("ESPN" in query for query in queries))

    def test_web_result_quality_prefers_final_score_sources(self):
        good = {
            "title": "England 1-2 Argentina Final Score - ESPN",
            "url": "https://www.espn.com/soccer/match/_/gameId/1",
            "snippet": "Game summary, final score 1-2, from July 15, 2026.",
        }
        bad = {
            "title": "FIFA World Cup schedule and tickets",
            "url": "https://www.facebook.com/example",
            "snippet": "Where to watch and buy tickets for an upcoming fixture.",
        }
        self.assertGreater(web_result_quality_score(good), web_result_quality_score(bad))

    def test_compact_web_context_bounds_prompt_evidence(self):
        context = {
            "query": "q",
            "raw_query": "raw",
            "search_results": [{"title": "t" * 500, "url": "https://example.com", "snippet": "s" * 2000} for _ in range(12)],
            "pages": [{"title": "p", "url": "https://example.com", "search_snippet": "x" * 1000, "text": "body" * 1000} for _ in range(8)],
        }
        compact = compact_web_context_for_prompt(context)
        self.assertEqual(len(compact["search_results"]), 8)
        self.assertEqual(len(compact["pages"]), 4)
        self.assertLess(len(json.dumps(compact)), 7000)

    def test_simple_search_routes_to_search_results(self):
        route = classify_request("can you search the latest fifa game and give me the results", "")
        self.assertEqual(route["mode"], "search_results")
        self.assertTrue(route["direct"])

    def test_deeper_current_info_routes_to_web_research(self):
        route = classify_request("Search the web for current llama.cpp Vulkan information relevant to AMD RX 580 local inference", "")
        self.assertEqual(route["mode"], "web_research")

    def test_last_sports_result_routes_to_web_research(self):
        route = classify_request("can you search the fifa world cup and give me the results for the last game", "")
        self.assertEqual(route["mode"], "web_research")

    def test_capability_question_uses_direct_answer_not_project_research(self):
        route = classify_request("can you tell me which files on my system you have access to read?", "")
        self.assertEqual(route["mode"], "direct_answer")

    def test_generic_files_request_without_project_stays_direct(self):
        route = classify_request("explain how file permissions work on linux", "")
        self.assertEqual(route["mode"], "direct_answer")

    def test_code_modes_prefetch_docs_when_relevant(self):
        route = {"mode": "implementation"}
        self.assertTrue(should_prefetch_docs(route, "build FastAPI auth middleware with best practices", {}))
        self.assertFalse(should_prefetch_docs({"mode": "direct_answer"}, "build FastAPI auth middleware with best practices", {}))

    def test_docs_query_uses_request_and_project_hints(self):
        memory = {"retrieval": {"snippets": [{"path": "package.json", "text": "next react typescript"}]}}
        query = docs_search_query("plan an auth feature", memory)
        self.assertIn("official documentation best practices", query)
        self.assertTrue("react" in query or "next" in query or "typescript" in query)

    def test_docs_result_quality_prefers_official_docs(self):
        good = {"title": "FastAPI Security - Official Docs", "url": "https://fastapi.tiangolo.com/tutorial/security/", "snippet": "Security guide"}
        bad = {"title": "My FastAPI Tips", "url": "https://medium.com/example", "snippet": "Some tips"}
        self.assertGreater(docs_result_quality_score(good), docs_result_quality_score(bad))

    def test_search_relevance_rejects_unrelated_results(self):
        self.assertFalse(
            search_results_are_relevant(
                "latest fifa game",
                [
                    {
                        "title": "CTV News - Breaking News",
                        "url": "https://www.ctvnews.ca/",
                        "snippet": "Get live updates on the latest local and international news.",
                    }
                ],
            )
        )
        self.assertTrue(
            search_results_are_relevant(
                "latest fifa game",
                [
                    {
                        "title": "EA Sports FC",
                        "url": "https://www.ea.com/games/ea-sports-fc",
                        "snippet": "The latest football game from EA Sports.",
                    }
                ],
            )
        )

    def test_searxng_search_parses_json_results(self):
        class FakeResponse:
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "results": [
                            {
                                "title": "England 1-2 Argentina Final Score",
                                "url": "https://example.com/match",
                                "content": "Final score from July 15, 2026.",
                                "engines": ["duckduckgo"],
                                "score": 1.0,
                            }
                        ]
                    }
                ).encode()

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = capability_searxng_search("world cup", 5, "http://127.0.0.1:8888")
        self.assertEqual(result["engine"], "searxng")
        self.assertEqual(result["results"][0]["url"], "https://example.com/match")
        self.assertEqual(result["results"][0]["engine"], "duckduckgo")

    def test_page_text_parser_extracts_title_and_blocks(self):
        parser = PageTextParser()
        parser.feed("<html><title>Example</title><script>bad()</script><h1>Heading</h1><p>This is a useful paragraph with enough text.</p></html>")
        self.assertEqual(parser.title, "Example")
        self.assertIn("This is a useful paragraph", "\n".join(parser.blocks))

    def test_tool_request_extraction(self):
        text = 'Need more evidence. {"tool_request":{"capability":"git_status","input":{"path":"/home/nit/x"},"reason":"check branch"}}'
        requests = extract_tool_requests_from_text(text)
        self.assertEqual(requests[0]["capability"], "git_status")
        self.assertEqual(requests[0]["input"]["path"], "/home/nit/x")

    def test_current_info_queries_route_to_web_research(self):
        for q in [
            "find the release date for the next one piece manga chapter",
            "when does the next iphone come out",
            "how much is a tesla model 3",
        ]:
            self.assertEqual(classify_request(q, "")["mode"], "web_research", q)
        # a plain reasoning/coding question must NOT route to web research
        self.assertNotEqual(classify_request("explain how a hash map handles collisions", "")["mode"], "web_research")

    def test_source_authority_ranks_primary_over_spam(self):
        self.assertGreater(source_authority_score("www.reuters.com"), 5)
        self.assertGreater(source_authority_score("nps.gov"), 5)
        self.assertLess(source_authority_score("claystage.com"), 0)
        self.assertLess(source_authority_score("m.youtube.com"), 0)
        # a reputable source must outrank a rumor content-farm on the same query
        good = {"title": "Next chapter release date confirmed", "snippet": "official", "url": "https://www.reuters.com/x"}
        spam = {"title": "Next chapter LEAKS and spoilers", "snippet": "rumor prediction", "url": "https://claystage.com/y"}
        self.assertGreater(web_result_quality_score(good), web_result_quality_score(spam))

    def test_memory_suggestion_extraction_and_dedupe(self):
        text = ('Here is the answer.\n{"memory_suggestion":{"scope":"user","key":"editor",'
                '"value":"prefers tabs","tags":["style"],"reason":"stated twice"}}')
        got = extract_memory_suggestions_from_text(text)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["key"], "editor")
        self.assertEqual(got[0]["scope"], "user")
        # dedupe against an already-saved memory with the same key (case-insensitive)
        deduped = extract_trace_memory_suggestions({"final": text}, existing=[{"key": "Editor"}])
        self.assertEqual(deduped, [])
        # no suggestion -> empty, and bad scope falls back to user
        self.assertEqual(extract_memory_suggestions_from_text("no json here"), [])

    def test_sessions_and_queue_position(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as temp:
            persistence = Persistence(Path(temp) / "s.sqlite3")
            store = JobStore(persistence)
            s = persistence.create_session("GPU debugging")
            self.assertEqual(persistence.list_sessions()[0]["title"], "GPU debugging")
            a = store.create({"request": "first", "session_id": s["id"]})
            b = store.create({"request": "second", "session_id": s["id"]})
            rows = {r["id"]: r for r in store.list(session_id=s["id"])}
            self.assertEqual(rows[a]["queue_position"], 1)
            self.assertEqual(rows[b]["queue_position"], 2)
            self.assertEqual(store.pending_count(s["id"]), 2)
            # a job in another session is filtered out
            store.create({"request": "other", "session_id": "zzz"})
            self.assertEqual(len(store.list(session_id=s["id"])), 2)
            # session_messages reflects answered jobs
            store.update(a, answer="done first")
            msgs = persistence.session_messages(s["id"])
            self.assertIn({"role": "user", "content": "first"}, msgs)
            self.assertIn({"role": "assistant", "content": "done first"}, msgs)

    def test_attached_capabilities_carry_into_evidence_block(self):
        store = JobStore()
        job_id = store.create({"request": "x"})
        store.event(job_id, {"event": "capability_run_attached", "id": "c1", "capability": "git_status",
                              "input": {"path": "/home/nit/x"}, "ok": True, "summary": "Git status completed",
                              "result_preview": "branch main; clean"})
        store.event(job_id, {"event": "model_call_finished", "stage": "direct"})  # non-capability event is ignored
        evidence = collect_attached_capabilities(store.get(job_id))
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["capability"], "git_status")
        self.assertEqual(evidence[0]["input"]["path"], "/home/nit/x")
        block = tool_evidence_prompt_block(evidence)
        self.assertIn("TOOL EVIDENCE GATHERED BY HOST", block)
        self.assertIn("branch main; clean", block)
        self.assertEqual(tool_evidence_prompt_block([]), "")

    def test_attached_capability_event_becomes_context(self):
        store = JobStore()
        job_id = store.create({"request": "x"})
        store.event(job_id, {"event": "capability_run_attached", "id": "c1", "capability": "python_eval", "summary": "ran", "result_preview": "42"})
        job = store.get(job_id)
        self.assertEqual(job["context_blocks"][0]["type"], "capability result")
        self.assertEqual(job["context_blocks"][0]["compact"]["summary"], "ran")

    def test_stale_running_jobs_are_marked_interrupted(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as temp:
            persistence = Persistence(Path(temp) / "test.sqlite3")
            first = JobStore(persistence)
            job_id = first.create({"request": "x"})
            first.update(job_id, status="running")
            second = JobStore(persistence)
            job = second.get(job_id)
            self.assertEqual(job["status"], "error")
            self.assertIn("server restarted", job["error"])

    def test_partial_project_summaries_are_recovered(self):
        store = JobStore()
        job_id = store.create({"request": "x"})
        store.update(job_id, route={"project_path": "/home/nit/project"})
        store.event(job_id, {"event": "project_discovered", "path": "/home/nit/project", "fingerprint": "abc123"})
        store.event(job_id, {"event": "project_file_read", "path": "app/page.tsx", "summary": "renders page"})
        summaries = store.partial_project_summaries("/home/nit/project", "abc123ffff")
        self.assertEqual(summaries["app/page.tsx"]["summary"], "renders page")

    def test_manual_memory_persistence(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as temp:
            persistence = Persistence(Path(temp) / "test.sqlite3")
            persistence.save_memory("user", "machine", "single gpu slot", ["local"])
            memories = persistence.load_memories("user")
            self.assertEqual(memories[0]["key"], "machine")
            self.assertEqual(memories[0]["tags"], ["local"])
            self.assertTrue(persistence.delete_memory(memories[0]["id"]))
            self.assertEqual(persistence.load_memories("user"), [])

    def test_project_memory_scope(self):
        with tempfile.TemporaryDirectory(dir=str(Path.home())) as temp:
            self.assertTrue(project_memory_scope(temp).startswith("project:"))

    def test_manual_run_settings_override_mode_preset(self):
        config = {
            "max_tasks": 1,
            "max_workers": 1,
            "max_rounds": 1,
            "planner_tokens": 80,
            "worker_tokens": 80,
            "verifier_tokens": 80,
            "compactor_tokens": 80,
            "synth_tokens": 80,
            "timeout": 300,
        }
        route = {"settings": {"max_tasks": 3, "planner_tokens": 520, "worker_tokens": 420}}
        settings = resolve_run_settings(config, route)
        self.assertEqual(settings["max_tasks"], 1)
        self.assertEqual(settings["planner_tokens"], 80)
        self.assertEqual(settings["worker_tokens"], 80)

    def test_metrics_planned_calls_include_started_calls(self):
        job = {
            "status": "running",
            "route": {"expected_calls": 1},
            "events": [
                {"event": "model_call_started"},
                {"event": "model_call_started"},
                {"event": "model_call_finished", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "elapsed_s": 1},
            ],
        }
        self.assertEqual(job_metrics(job)["calls_planned"], 2)

    def test_metrics_include_active_call_live_tokens(self):
        job = {
            "status": "running",
            "route": {"expected_calls": 2},
            "stream_stats": {"approx_completion_tokens": 7, "approx_tokens_per_second": 12.5},
            "events": [
                {"event": "model_call_started"},
                {"event": "model_call_finished", "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "elapsed_s": 1},
            ],
        }
        metrics = job_metrics(job)
        self.assertEqual(metrics["tokens_completion"], 12)
        self.assertEqual(metrics["tokens_total"], 15)
        self.assertEqual(metrics["tokens_per_second"], 12.5)

    def test_project_json_mapping_normalizes_lists(self):
        parsed = parsed_json_mapping('["a", "b"]', {"summary": "", "key_files": [], "data_flow": [], "risks_or_gaps": []})
        self.assertEqual(parsed["summary"], "a; b")
        self.assertEqual(parsed["key_files"], [])

    def test_conversation_history_sanitizer_bounds_roles_and_size(self):
        history = sanitize_conversation_history(
            [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "x" * 100},
            ],
            max_turns=3,
            max_chars=12,
        )
        self.assertEqual([item["role"] for item in history], ["user", "assistant", "user"])
        self.assertLessEqual(sum(len(item["content"]) for item in history), 12)


if __name__ == "__main__":
    unittest.main()
