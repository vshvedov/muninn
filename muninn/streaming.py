"""Drive Agent.run_stream_events into a Textual Markdown widget."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.settings import ModelSettings
from textual.widgets import Markdown


LogFn = Callable[[dict[str, Any]], None]


async def run_and_stream(
    agent: Agent,
    prompt: str,
    md: Markdown,
    *,
    message_history: list[ModelMessage] | None = None,
    log: LogFn | None = None,
    model_settings: ModelSettings | None = None,
    label: str = "muninn",
) -> tuple[str, list[ModelMessage]]:
    """Stream agent output token-by-token into `md`.

    Returns (final_text, new_messages). `new_messages` is the message list to
    feed back into the agent on the next turn.

    Side effects:
    - Updates `md` via Textual's MarkdownStream (auto-batching for high-rate
      token streams).
    - Calls `log(record)` (if provided) for stream-event hooks: tool calls,
      tool results, final result. Per-token deltas are NOT logged (too noisy).
    """
    stream = Markdown.get_stream(md)
    final_text = ""
    final_messages: list[ModelMessage] = []
    log = log or (lambda _r: None)

    # Capture the input prompt so the JSONL is self-contained for re-analysis.
    log({
        "type": "agent_call_start",
        "label": label,
        "prompt": prompt,
        "history_len": len(message_history) if message_history else 0,
    })

    try:
        async for event in agent.run_stream_events(
            prompt,
            message_history=message_history,
            model_settings=model_settings,
        ):
            match event:
                case PartStartEvent(part=ToolCallPart(tool_name=name, args=args)):
                    await stream.write(f"\ncalling `{name}`(…)\n")
                    log({"type": "tool_call_started_stream", "label": label,
                         "name": name, "args": args})
                case PartDeltaEvent(delta=TextPartDelta(content_delta=t)):
                    final_text += t
                    await stream.write(t)
                case PartDeltaEvent(delta=ThinkingPartDelta()):
                    pass  # silently drop chain-of-thought
                case PartDeltaEvent(delta=ToolCallPartDelta()):
                    pass  # tool args streamed; rendered on PartStartEvent above
                case FunctionToolCallEvent():
                    pass  # already covered by PartStartEvent above
                case FunctionToolResultEvent(result=res):
                    log({"type": "tool_result_stream", "label": label,
                         "tool_call_id": getattr(res, "tool_call_id", None),
                         "content": str(getattr(res, "content", res))[:500]})
                case FinalResultEvent():
                    pass
                case AgentRunResultEvent(result=res):
                    final_messages = list(res.new_messages())
                    if not final_text and isinstance(res.output, str):
                        final_text = res.output
                        await stream.write(res.output)
                case _:
                    pass
    finally:
        await stream.stop()

    log({
        "type": "agent_call_complete",
        "label": label,
        "text_len": len(final_text),
        "output": final_text,  # full output text for re-analysis
    })
    return final_text, final_messages
