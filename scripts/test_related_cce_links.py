import unittest

from build_spill_candidate_report import related_cce_links


class RelatedCceLinksTests(unittest.TestCase):
    def test_softmax_has_named_supplementary_source(self):
        link = related_cce_links(dict(case="softmax_three_loop"))
        self.assertIn('href="a6_softmax_fp16_row_scaling/runnable_a6/rows8/no_membar.cce"', link)
        self.assertIn("Giant-loop CCE", link)
        self.assertIn("不是本行周期数据的来源", link)

    def test_other_cases_are_unchanged(self):
        self.assertEqual(related_cce_links(dict(case="logit")), "")


if __name__ == "__main__":
    unittest.main()
