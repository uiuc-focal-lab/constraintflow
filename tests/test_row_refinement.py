import unittest

import torch

from constraintflow.compiler.optimizations.row_refinement import selected_patches_to_dense
from constraintflow.gbcsr.sparse_block import patches_to_dense


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
