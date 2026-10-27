"""Offline tests for the bounded agent core."""
import unittest
from neuroevolution_ai.agent import MAX_RESPONSE_CHARS, MAX_TASK_CHARS, run_agent


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

    def test_reject_noncallable_allowed_tool(self):
        with self.assertRaisesRegex(TypeError, "allowed tool must be callable"):
            run_agent(ScriptedModel(["TOOL: echo | hi"]), "task", {"echo": None})

    def test_budget(self):
        with self.assertRaises(RuntimeError):
            run_agent(ScriptedModel(["TOOL: echo | x"] * 3),
                      "task", {"echo": lambda arg: arg}, max_steps=2)

    def test_bad_budget(self):
        with self.assertRaises(ValueError):
            run_agent(ScriptedModel([]), "task", {}, max_steps=True)



    def test_reject_oversized_task(self):
        with self.assertRaisesRegex(ValueError, "task exceeds size limit"):
            run_agent(ScriptedModel([]), "x" * (MAX_TASK_CHARS + 1), {})

    def test_reject_oversized_model_response_before_tool_call(self):
        called = []
        response = "TOOL: echo | " + "x" * MAX_RESPONSE_CHARS
        with self.assertRaisesRegex(ValueError, "model response exceeds size limit"):
            run_agent(ScriptedModel([response]), "task", {"echo": called.append})
        self.assertEqual(called, [])

    def test_model_response_at_limit(self):
        response = "FINAL: " + "x" * (MAX_RESPONSE_CHARS - len("FINAL: "))
        answer, _ = run_agent(ScriptedModel([response]), "task", {})
        self.assertEqual(answer, "x" * (MAX_RESPONSE_CHARS - len("FINAL: ")))

if __name__ == "__main__":
    unittest.main()
