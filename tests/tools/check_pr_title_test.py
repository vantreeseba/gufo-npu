import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/ci/check-pr-title.py"
spec = importlib.util.spec_from_file_location("check_pr_title", SCRIPT)
check_pr_title = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_pr_title)


class ValidateTest(unittest.TestCase):
    def test_accepts_conventional_titles(self):
        for title in (
            "feat(npu): fuse routed experts into one dispatch",
            "perf!: drop the sync path",
            "ci: publish branch images",
        ):
            self.assertIsNone(check_pr_title.validate(title), title)

    def test_rejects_others(self):
        for title in ("Add MoE kernel", "feat:no space", "feat(npu): a\nb"):
            self.assertIsNotNone(check_pr_title.validate(title), title)


if __name__ == "__main__":
    unittest.main()
