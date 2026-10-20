"""Validation tooling must not mistake matching opcounts for matching dataflow."""
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE if (HERE / "explicit_spill_roundtrip.py").exists() else HERE.parent / "results/membar_results/scripts"
sys.path.insert(0, str(SCRIPTS))
from explicit_spill_roundtrip import address, affine, reconstruct
from compare_explicit_spill import semantic_trace


def row(op, detail, pc="0x100"):
    return dict(opcode=op, detail=detail, pc=pc)


class ExplicitSpillRoundtripTests(unittest.TestCase):
    def test_affine_rejects_unrepresentable_address_stream(self):
        self.assertEqual(affine([4, 260, 516]), (4, 256))
        with self.assertRaises(ValueError):
            affine([4, 260, 260])

    def test_vag_and_post_update_addresses(self):
        self.assertEqual(address(row("RV_VLD", " Sn[2]=0x100, Sn2[3]=0x1, Am[0]=0x200,"))[0], 0x10300)
        self.assertEqual(address(row("RV_VLDS", " Sn[2]=0x1200, Sm[4]=0x200, #p=1,"))[0], 0x1000)
        self.assertEqual(address(row("RV_VSTI", " Sn[95]=0x40000, #offset=247, #p=0,"))[0], 0x3fee0)
        self.assertEqual(address(row("RV_VSTS", " Sn[95]=0x40000, Sm[2]=0xefe0, Sm2[3]=0xffff, #p=0,"))[0], 0x3efe0)

    def test_transparent_register_renaming_but_not_dataflow_change(self):
        prefix = [row("RV_PSET", " Dtype: B32 Pd[0], #pattern=0,")]
        original = prefix + [row("RV_VLDI", " Vd[0], Sn[2]=0x0, #offset=0, dist:NORM, #p=0,"),
            row("RV_VMOV", " Vd[1], Vn[0],"),
            row("RV_VADD", " Vd[2], Vn[0], Vm[1], Pg[0],")]
        renamed = prefix + [row("RV_VLDS", " Vd[7], Sn[6]=0x0, Sm[4]=0x0, dist:NORAML, #p=0,"),
            row("RV_VADD", " Vd[9], Vn[7], Vm[7], Pg[0],")]
        signature = lambda rs: [r["signature"] for r in semantic_trace(rs)]
        self.assertEqual(signature(original), signature(renamed))
        renamed[1]["detail"] = " Vd[7], Sn[6]=0x100, Sm[4]=0x0, dist:NORAML, #p=0,"
        self.assertNotEqual(signature(original), signature(renamed))

    def test_undefined_predicate_cannot_pass_comparison(self):
        with self.assertRaises(ValueError):
            semantic_trace([row("RV_VDUPS", " Vd[0], Sn[2]=0x0, Sn2[3]=0x3f80, Pg[1],")])

    def test_same_opcounts_do_not_hide_wrong_register_dependency(self):
        prefix = [row("RV_PSET", " Dtype: B32 Pd[0], #pattern=0,"),
            row("RV_VLDI", " Vd[0], Sn[2]=0x0, #offset=0, dist:NORM, #p=0,"),
            row("RV_VLDI", " Vd[1], Sn[2]=0x100, #offset=0, dist:NORM, #p=0,")]
        left = semantic_trace(prefix + [row("RV_VSUB", " Vd[2], Vn[0], Vm[1], Pg[0],")])
        right = semantic_trace(prefix + [row("RV_VSUB", " Vd[2], Vn[1], Vm[0], Pg[0],")])
        self.assertNotEqual(left[-1]["signature"], right[-1]["signature"])

    def test_loop_count_does_not_consume_packed_sn2(self):
        lines = [
            "(PC: 0x100) (Binary: 0x0) (ID: 0) RV_PSET Dtype: B32 Pd[0], #pattern=0,",
            "(PC: 0x104) (Binary: 0x0) (ID: 1) RV_VLOOPv2 Sn[2]=0x2, Sn2[3]=0x5000, instr_num : 1,",
            "(PC: 0x108) (Binary: 0x0) (ID: 2) RV_VLDI Vd[0], Sn[4]=0x0, #offset=0, dist:NORM, #p=0,",
            "(PC: 0x108) (Binary: 0x0) (ID: 3) RV_VLDI Vd[0], Sn[4]=0x100, #offset=0, dist:NORM, #p=0,",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text("\n".join(lines))
            source, _ = reconstruct(path, (0,))
        self.assertIn("iteration < uint16_t(2)", source)
        self.assertIn("64 * iteration", source)


if __name__ == "__main__":
    unittest.main()
