import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/bench/work_split.py"
spec = importlib.util.spec_from_file_location("work_split", SCRIPT)
work_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work_split)


def npu_bytes(work):
    return sum(b for (_, device), (_, b) in work.items() if device == "NPU")


class WorkSplitTest(unittest.TestCase):
    def test_decode_reads_the_decoders_q4nx_weights(self):
        # Bytes per token the NPU decoders read, as BENCHMARKS.md records
        # them: ~1.1 GB (2B), ~2.42 GB (4B), 1.69 GB (35B).
        for name, expected in (("Qwen3.5-2B", 1.09e9), ("Qwen3.5-4B", 2.42e9),
                               ("Qwen3.6-35B-A3B", 1.69e9)):
            c = work_split.shapes(work_split.MODELS[name])
            got = npu_bytes(work_split.decode_step(c, 128, False))
            self.assertAlmostEqual(got / expected, 1.0, delta=0.02, msg=name)

    def test_35b_prefill_dense_weights_match_the_resident_bfp16(self):
        # The 35B prefill keeps 1.44 GB of bfp16 dense matrices resident.
        c = work_split.shapes(work_split.MODELS["Qwen3.6-35B-A3B"])
        work = work_split.prefill(c, 1055)
        dense = work[("dense projections", "NPU")][1]
        self.assertAlmostEqual(dense / 1.44e9, 1.0, delta=0.03)

    def test_shares_add_up(self):
        for name, m in work_split.MODELS.items():
            c = work_split.shapes(m)
            for work in (work_split.decode_step(c, 1024, True),
                         work_split.prefill(c, m["prompts"][-1])):
                for i in (0, 1):
                    npu, cpu = work_split.totals(work, i)
                    total = sum(v[i] for v in work.values())
                    self.assertAlmostEqual(npu + cpu, total, delta=total * 1e-12)


if __name__ == "__main__":
    unittest.main()
