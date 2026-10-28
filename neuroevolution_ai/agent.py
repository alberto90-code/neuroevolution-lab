"""A bounded agent loop with explicit tool allowlisting and no code execution."""
from dataclasses import dataclass
from typing import Callable, Protocol

MAX_TASK_CHARS = 16_000
MAX_RESPONSE_CHARS = 8_192
MAX_PROMPT_CHARS = 20_000


class Model(Protocol):
    def respond(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Step:
    kind: str
    value: str


def run_agent(model: Model, task: str, tools: dict[str, Callable[[str], str]],
              max_steps: int = 5) -> tuple[str, list[Step]]:
    """Protocol: FINAL: answer or TOOL: name | argument.

    Only explicitly supplied tools may run. This function never executes model
    text as code, opens a network connection, or writes to disk. Input and model
    output limits prevent unbounded tool arguments and prompt growth.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be nonempty")
    if len(task) > MAX_TASK_CHARS:
        raise ValueError("task exceeds size limit")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 20:
        raise ValueError("max_steps must be between 1 and 20")
    transcript = [Step("task", task)]
    prompt = task
    for step_index in range(max_steps):
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("agent prompt exceeds size limit")
        response = model.respond(prompt)
        if not isinstance(response, str):
            raise TypeError("model response must be text")
        if len(response) > MAX_RESPONSE_CHARS:
            raise ValueError("model response exceeds size limit")
        if response.startswith("FINAL:"):
            answer = response[len("FINAL:"):].strip()
            if not answer:
                raise ValueError("empty final answer")
            transcript.append(Step("final", answer))
            return answer, transcript
        if not response.startswith("TOOL:"):
            raise ValueError("unsupported model response")
        payload = response[len("TOOL:"):].strip()
        name, sep, argument = payload.partition("|")
        name = name.strip()
        if not sep or name not in tools:
            raise ValueError("tool is not explicitly allowed")
        if not callable(tools[name]):
            raise TypeError("allowed tool must be callable")
        if step_index == max_steps - 1:
            raise RuntimeError("step budget exhausted")
        transcript.append(Step("tool", name))
        result = tools[name](argument.strip())
        if not isinstance(result, str):
            raise TypeError("tool output must be text")
        transcript.append(Step("observation", result[:4000]))
        prompt = f"Task: {task}\nTool {name} returned: {result[:4000]}\nRespond FINAL: or TOOL:"
    raise RuntimeError("step budget exhausted")
