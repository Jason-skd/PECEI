"""Anthropic provider (Claude). Structured output via tool-use.

Default model is Haiku 4.5 (cheap; grid-world turns are short). The recursive
Program AST is passed as a non-strict tool input_schema (strict structured-output
rejects recursion; non-strict tool-use accepts $defs/$ref).
"""
from __future__ import annotations

from pecei.action import CompileError

from ..prompt import PLAN_TOOL_DESCRIPTION, PROGRAM_SCHEMA, load_system_prompt, render_user
from ..protocol import Directive, TurnInput, TurnOutput, parse_program

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        base_url: str | None = None,
    ) -> None:
        import anthropic

        # api_key=None -> SDK resolves env var / `ant auth login` profile.
        # base_url=None -> SDK default; set to point at a relay / self-hosted endpoint.
        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        self._client = anthropic.Anthropic(**kw)
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens

    def decide(self, turn: TurnInput) -> TurnOutput:
        tool = {"name": "plan", "description": PLAN_TOOL_DESCRIPTION, "input_schema": PROGRAM_SCHEMA}
        req = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": load_system_prompt(),
            "messages": [{"role": "user", "content": render_user(turn)}],
            "tools": [tool],
        }
        if turn.directive is Directive.PLAN:
            req["tool_choice"] = {"type": "tool", "name": "plan"}

        resp = self._client.messages.create(**req)
        program: Program | None = None
        reflection: str | None = None
        error: str | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "plan":
                try:
                    program = parse_program(block.input)
                except CompileError as e:           # malformed AST -> COMPILE_ERROR
                    error = str(e)
            elif getattr(block, "type", None) == "text" and reflection is None:
                reflection = block.text
        return TurnOutput(
            program=program,
            reflection=reflection,
            error=error,
            raw_request={**req, "provider": self.name},
            raw_response=resp.to_dict(),
        )
