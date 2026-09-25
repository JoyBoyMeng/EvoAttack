import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from attack_agent import run_adaptive_attack
from attack_agent import run_final_policy_test
from attack_agent import run_repeated_adaptive_attack


class RuntimeReportingTests(unittest.TestCase):
    def test_training_runtime_is_the_last_event(self):
        output = io.StringIO()
        with patch.object(run_adaptive_attack, "_run_training"), redirect_stdout(output):
            run_adaptive_attack.main()

        self.assertIn("[AdaptiveAttack::OVERALL_RUNTIME]", output.getvalue())
        self.assertIn("overall_elapsed_seconds:", output.getvalue())
        self.assertIn("overall_elapsed_hms:", output.getvalue())

    def test_final_test_runtime_is_the_last_event(self):
        output = io.StringIO()
        with patch.object(run_final_policy_test, "_run_final_test"), redirect_stdout(output):
            run_final_policy_test.run_final_test()

        self.assertIn("[AdaptiveAttack::OVERALL_RUNTIME]", output.getvalue())
        self.assertIn("overall_elapsed_seconds:", output.getvalue())
        self.assertIn("overall_elapsed_hms:", output.getvalue())

    def test_repeated_training_runtime_is_last(self):
        output = io.StringIO()
        with patch.object(
            run_repeated_adaptive_attack,
            "_run_repeated_training",
        ), redirect_stdout(output):
            run_repeated_adaptive_attack.main()

        lines = [line for line in output.getvalue().splitlines() if line]
        self.assertIn("[RepeatedAdaptiveAttack] OVERALL_RUNTIME", lines)
        self.assertTrue(lines[-2].startswith("overall_elapsed_seconds:"))
        self.assertTrue(lines[-1].startswith("overall_elapsed_hms:"))


if __name__ == "__main__":
    unittest.main()
