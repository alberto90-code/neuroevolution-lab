"""Offline tests for the model adapter boundary."""
import unittest

from neuroevolution_ai.agent import run_agent
from neuroevolution_ai.models import CallableModel, ReplayModel


class ModelAdapterTests(unittest.TestCase):
    def test_callable_model(self):
        model = CallableModel(lambda prompt: "FINAL: " + prompt.upper())
        answer, _ = run_agent(model, "hello", {})
        self.assertEqual(answer, "HELLO")

    def test_replay_with_tool(self):
        model = ReplayModel(["TOOL: echo | hi", "FINAL: done"])
        answer, trace = run_agent(model, "task", {"echo": lambda arg: arg.upper()})
        self.assertEqual(answer, "done")
        self.assertEqual(trace[2].value, "HI")

    def test_replay_exhaustion(self):
        model = ReplayModel([])
        with self.assertRaisesRegex(RuntimeError, "replay responses exhausted"):
            model.respond("hello")

    def test_invalid_output(self):
        with self.assertRaises(TypeError):
            CallableModel(lambda _: None).respond("hello")
        with self.assertRaises(TypeError):
            ReplayModel([None]).respond("hello")

    def test_invalid_callable(self):
        with self.assertRaises(TypeError):
            CallableModel(None)

    def test_invalid_prompt(self):
        with self.assertRaises(TypeError):
            ReplayModel(["ok"]).respond(None)


if __name__ == "__main__":
    unittest.main()
