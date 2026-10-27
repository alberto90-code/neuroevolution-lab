"""Offline tests for the bounded agent core."""
import unittest
from neuroevolution_ai.agent import run_agent


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def respond(self, prompt):
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def test_final(self):
        answer, trace = run_agent(ScriptedModel(["FINAL: done"]), "task", {})
        self.assertEqual(answer, "done")
        self.assertEqual([s.kind for s in trace], ["task", "final"])

    def test_allowlisted_tool(self):
        answer, trace = run_agent(
            ScriptedModel(["TOOL: echo | hello", "FINAL: ok"]),
            "task", {"echo": lambda arg: arg.upper()})
        self.assertEqual(answer, "ok")
        self.assertEqual(trace[2].value, "HELLO")

    def test_reject_unknown_tool(self):
        with self.assertRaises(ValueError):
            run_agent(ScriptedModel(["TOOL: shell | rm -rf /"]), "task", {})

    def test_budget(self):
        with self.assertRaises(RuntimeError):
            run_agent(ScriptedModel(["TOOL: echo | x"] * 3),
                      "task", {"echo": lambda arg: arg}, max_steps=2)

    def test_bad_budget(self):
        with self.assertRaises(ValueError):
            run_agent(ScriptedModel([]), "task", {}, max_steps=True)


if __name__ == "__main__":
    unittest.main()
