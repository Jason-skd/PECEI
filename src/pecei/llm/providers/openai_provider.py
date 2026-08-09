"""OpenAI-compatible provider. Covers OpenAI and DeepSeek (via base_url)."""
from __future__ import annotations

from pecei.action import CompileError

from ..prompt import PLAN_TOOL_DESCRIPTION, PROGRAM_SCHEMA, load_system_prompt, render_user
from ..protocol import Directive, TurnInput, TurnOutput, parse_program

DEFAULT_MODEL = "gpt-4o-mini"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        import openai

        # api_key=None -> SDK resolves OPENAI_API_KEY env var.
        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        self._client = openai.OpenAI(**kw)
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.name = "deepseek" if base_url else "openai"

    def decide(self, turn: TurnInput) -> TurnOutput:
        tool = {
            "type": "function",
            "function": {"name": "plan", "description": PLAN_TOOL_DESCRIPTION, "parameters": PROGRAM_SCHEMA},
        }
        req = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": load_system_prompt()},
                {"role": "user", "content": render_user(turn)},
            ],
            "tools": [tool],
        }
        if turn.directive is Directive.PLAN:
            req["tool_choice"] = {"type": "function", "function": {"name": "plan"}}

        resp = self._client.chat.completions.create(**req)
        msg = resp.choices[0].message
        program: Program | None = None
        error: str | None = None
        if msg.tool_calls:
            call = msg.tool_calls[0]
            if call.function.name == "plan":
                try:
                    program = parse_program(call.function.arguments)
                except CompileError as e:           # malformed AST -> COMPILE_ERROR
                    error = str(e)
        return TurnOutput(
            program=program,
            reflection=msg.content,
            error=error,
            raw_request={**req, "provider": self.name},
            raw_response=resp.model_dump(),
        )
