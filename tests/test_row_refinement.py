import unittest

import torch

from constraintflow.compiler.optimizations.row_refinement import block_plan, selected_patches_to_dense
from constraintflow.gbcsr.sparse_block import patches_to_dense


class BlockPlanTest(unittest.TestCase):
    def test_block_plan_pads_less_than_the_smallest_size(self):
        sizes = (8, 16, 32, 64, 128, 256, 500)  # 512 capped at a 500-wide layer
        # N8: 260 unstable rows ran as one 500-row block.
        self.assertEqual(block_plan(260, sizes), [256, 8])
        self.assertEqual(block_plan(467, sizes), [256, 128, 64, 16, 8])
        self.assertEqual(block_plan(500, sizes), [500])
        self.assertEqual(block_plan(0, sizes), [])
        for rows in range(1, 1200):
            plan = block_plan(rows, sizes)
            self.assertGreaterEqual(sum(plan), rows)
            self.assertLess(sum(plan) - rows, sizes[0])

    def test_unpacked_plan_covers_the_rest_with_one_block(self):
        sizes = (8, 16, 32, 64, 128, 256, 500)
        self.assertEqual(block_plan(260, sizes, pack=False), [500])
        self.assertEqual(block_plan(1100, sizes, pack=False), [500, 500, 128])
        self.assertEqual(block_plan(0, sizes, pack=False), [])


class SelectedPatchesTest(unittest.TestCase):
    def test_selected_patch_geometry_matches_dense(self):
        torch.manual_seed(9)
        for dims in [(3, 3, 4, 5, 2, 3, 2, 7, 8, 1, 2, 2, 1),
                     (1, 3, 4, 5, 2, 3, 2, 7, 8, 1, 2, 2, 1),
                     (1, 2, 3, 3, 2, 5, 5, 4, 4, 2, 2, 1, 1)]:
            b, oc, ox, oy, ic, kx, ky, *_ = dims
            pieces = torch.randn(b, oc * ox * oy, ic * kx * ky, dtype=torch.float64)
            indices = torch.tensor([oc * ox * oy - 1, 0, oy + 1, 0])
            got = selected_patches_to_dense(pieces.index_select(1, indices), indices, *dims)
            expected = patches_to_dense(pieces, *dims).index_select(1, indices)
            torch.testing.assert_close(got, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
