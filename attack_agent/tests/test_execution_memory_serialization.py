import unittest
from types import SimpleNamespace

from pyopenagi.agents.base_agent import BaseAgent
from pyopenagi.agents.react_agent_attack import build_execution_memory_messages


class ExecutionMemorySerializationTests(unittest.TestCase):
    def test_one_hundred_planner_retries_do_not_advance_execution_rounds(self):
        agent = BaseAgent.__new__(BaseAgent)
        agent.plan_max_fail_times = 101
        agent.messages = []
        agent.rounds = 0
        agent.start_time = None
        agent.request_waiting_times = []
        agent.request_turnaround_times = []
        agent.args = SimpleNamespace(llm_name="test-model")
        calls = {"count": 0}

        def get_response(*, query):
            calls["count"] += 1
            return (
                SimpleNamespace(response_message="workflow response"),
                [1.0],
                [2.0],
                [0.0],
                [1.0],
            )

        def check_workflow(_message):
            if calls["count"] <= 100:
                return None
            return [{"message": "inspect the host", "tool_use": []}]

        agent.get_response = get_response
        agent.check_workflow = check_workflow
        agent.set_start_time = lambda time: setattr(agent, "start_time", time)

        workflow = BaseAgent.automatic_workflow(agent)

        self.assertEqual(calls["count"], 101)
        self.assertEqual(agent.rounds, 0)
        self.assertEqual(workflow, [{"message": "inspect the host", "tool_use": []}])

    def test_planner_retries_do_not_shift_execution_memory(self):
        messages = [
            {"role": "system", "content": "agent instruction"},
            {"role": "user", "content": "workflow format instruction"},
            {"role": "assistant", "content": "retrieved memory prompt"},
            {"role": "user", "content": "The current task is: inspect hosts"},
            {"role": "user", "content": "Fail 1 times to generate a valid plan. I need to regenerate a plan"},
            {"role": "user", "content": "Fail 2 times to generate a valid plan. I need to regenerate a plan"},
            {"role": "assistant", "content": "[Thinking]: The workflow generated for the problem is [...]"},
            {"role": "user", "content": "At step 3, you need to inspect active hosts "},
            {"role": "assistant", "content": "[Action]: I will call HostInspector; [Observation]: host data"},
            {"role": "user", "content": "At step 4, you need to remediate the host "},
            {"role": "assistant", "content": "[Action]: I will call Remediator; [Observation]: remediation result"},
        ]

        stored = build_execution_memory_messages(messages)

        self.assertIn("user: At step 3, you need to inspect active hosts", stored)
        self.assertIn("assistant: [Action]: I will call HostInspector", stored)
        self.assertIn("user: At step 4, you need to remediate the host", stored)
        self.assertIn("assistant: [Action]: I will call Remediator", stored)
        self.assertNotIn("Fail 1 times", stored)
        self.assertNotIn("Fail 2 times", stored)
        self.assertNotIn("workflow generated", stored)
        self.assertNotIn("[Observation]", stored)

    def test_preserve_observation_memory_is_explicit(self):
        messages = [
            {"role": "user", "content": "At step 1, you need to inspect the host "},
            {"role": "assistant", "content": "[Action]: I will call HostInspector; [Observation]: host data"},
        ]

        stored = build_execution_memory_messages(
            messages,
            preserve_observation_memory=True,
        )

        self.assertIn("[Observation]: host data", stored)


if __name__ == "__main__":
    unittest.main()
