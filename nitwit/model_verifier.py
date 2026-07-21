"""ModelVerifier: the real Verifier — asks the CPU Qwen3-4B whether a described success
condition is met by the current work. Lenient on parse failure (the tests criterion is the
hard gate, so a flaky judge shouldn't sink good work)."""
from __future__ import annotations

from orchestrator import extract_json

VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "schema": {
            "type": "object",
            "properties": {
                "pass": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["pass"],
            "additionalProperties": False,
        },
    },
}

VERIFIER_SYSTEM = (
    "You are a strict but fair verifier. Decide whether the described SUCCESS CONDITION is "
    "genuinely satisfied by the work shown (the repository files and the latest test output). "
    'Answer ONLY as JSON: {"pass": true|false, "reason": "<one sentence>"}. '
    "Pass only if the condition is really met; do not pass a stub, a placeholder, or work that "
    "merely looks plausible."
)


def build_verifier_messages(description: str, ctx) -> list[dict]:
    files = "\n\n".join(f"--- {p} ---\n{c[:8000]}" for p, c in (ctx.repo_files or {}).items()) or "(none)"
    user = (
        f"SUCCESS CONDITION:\n{description}\n\n"
        f"GOAL (for context):\n{ctx.goal}\n\n"
        f"LATEST TEST OUTPUT:\n{ctx.last_test_output or '(none)'}\n\n"
        f"REPOSITORY FILES:\n{files}"
    )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": user},
    ]


class ModelVerifier:
    def __init__(self, client, max_tokens: int = 700) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def judge(self, description: str, ctx) -> bool:
        messages = build_verifier_messages(description, ctx)
        response = self.client.chat(messages, temperature=0.0, max_tokens=self.max_tokens,
                                    response_format=VERDICT_FORMAT)
        try:
            parsed = extract_json(response.content)
        except ValueError:
            return True  # lenient: don't sink good work on a flaky judge
        if not isinstance(parsed, dict) or "pass" not in parsed:
            return True
        raw = parsed["pass"]
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "yes", "pass", "ok", "1")
        return bool(raw)
