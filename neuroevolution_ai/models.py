"""Offline model adapters with no external dependencies."""


class CallableModel:
    """Wrap a text generation callable as an agent model."""

    def __init__(self, generate):
        if not callable(generate):
            raise TypeError("generate must be callable")
        self.generate = generate

    def respond(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        response = self.generate(prompt)
        if not isinstance(response, str):
            raise TypeError("model output must be text")
        return response


class ReplayModel:
    """Use fixed outputs for repeatable agent tests."""

    def __init__(self, responses):
        self.responses = iter(responses)

    def respond(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        try:
            response = next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("replay responses exhausted") from exc
        if not isinstance(response, str):
            raise TypeError("model output must be text")
        return response
