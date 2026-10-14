"""计算 IPC 不应混入 LSU/GSU 完成事件。"""
import unittest

from build_archive import compute_opcode


class ReportMetricsTest(unittest.TestCase):
    def test_memory_and_control_are_not_compute(self):
        for op in ('RV_VLDS','RV_VSTS','RV_VLDUS','RV_VGATHER2','RV_VSCATTER','RV_VLOOPv2','RV_SMEM_BAR','RV_PSET','RV_PLT'):
            with self.subTest(op=op):
                self.assertFalse(compute_opcode(op))

    def test_vector_math_and_move_are_compute(self):
        for op in ('RV_VADD','RV_VDIV','RV_VEXP','RV_VMOV','RV_VCI'):
            with self.subTest(op=op):
                self.assertTrue(compute_opcode(op))


if __name__=='__main__':unittest.main()
