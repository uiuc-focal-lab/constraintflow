"""Patch compositions agree with a dense reference on every code path.

KernelBlock @ PatchesBlock (forward conv on a patch-represented coefficient
matrix) and DenseBlock @ PatchesBlock (fully connected layer on one) may be
split by origin channel, and the former switches to the dense form once the
composed receptive field covers the input map. All variants must match
densify-then-multiply.
"""
import unittest

import torch
import torch.nn.functional as F

import constraintflow.lib.globals as g
import constraintflow.gbcsr.sparse_block as sb


def _patches(B, K1, Pox, c0, Pk, Ps, Pp, ix):
    block = torch.randn(B, K1 * Pox * Pox, c0 * Pk * Pk)
    total = torch.tensor([B, K1 * Pox * Pox, c0 * ix * ix])
    P = sb.PatchesBlock(block, total, ix, ix, Pox, Pox, Ps, Ps, Pp, Pp, Pk, Pk, c0, K1, [], -1)
    dense = sb.patches_to_dense(block, B, K1, Pox, Pox, c0, Pk, Pk, ix, ix, Pp, Pp, Ps, Ps)
    return P, dense


def _dense(block):
    if isinstance(block, sb.DenseBlock):
        return block.block
    dense = block.get_dense()
    return dense[0] if isinstance(dense, tuple) else dense


class PatchComposeTest(unittest.TestCase):
    def setUp(self):
        self.saved = sb.PATCH_COMPOSE_CHUNK_BYTES, g.compact_patches.get_flag()
        g.dummy_mode.reset_flag()
        g.reuse_mode.reset_flag()
        torch.manual_seed(0)

    def tearDown(self):
        sb.PATCH_COMPOSE_CHUNK_BYTES = self.saved[0]
        g.compact_patches.set_flag() if self.saved[1] else g.compact_patches.reset_flag()

    def kernel_compose(self, compact, chunk_bytes, B=1, K1=4, c0=3, K2=3, Pox=8, ix=16,
                       Pk=13, Ps=2, Pp=6, s=1):
        g.compact_patches.set_flag() if compact else g.compact_patches.reset_flag()
        sb.PATCH_COMPOSE_CHUNK_BYTES = chunk_bytes
        P, Pd = _patches(B, K1, Pox, c0, Pk, Ps, Pp, ix)
        ox = (Pox + 2 - 3) // s + 1
        W = torch.randn(K2, K1, 3, 3)
        KB = sb.KernelBlock(W, torch.tensor([B, K2 * ox * ox, K1 * Pox * Pox]),
                            Pox, Pox, ox, ox, s, s, 1, 1, [], -1)
        res, _ = KB.matmul_equal_dims(P, json_list=[], lhs_index=0, rhs_index=1)
        cols = Pd.transpose(1, 2).reshape(-1, K1, Pox, Pox)
        ref = F.conv2d(cols, W, stride=s, padding=1).reshape(B, c0 * ix * ix, -1).transpose(1, 2)
        return res, _dense(res).reshape(ref.shape), ref

    def test_kernel_compose_paths_match_dense(self):
        cases = {
            'dense switch': dict(compact=True, chunk_bytes=1 << 30),
            'patches': dict(compact=False, chunk_bytes=1 << 30),
            'patches, one channel per chunk': dict(compact=False, chunk_bytes=1),
            'small patch, chunked': dict(compact=True, chunk_bytes=1, Pk=5, Ps=1, Pp=2, ix=12, Pox=12),
            'stride 2': dict(compact=True, chunk_bytes=1 << 30, s=2),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                res, got, ref = self.kernel_compose(**kwargs)
                torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
        self.assertIsInstance(self.kernel_compose(True, 1 << 30)[0], sb.DenseBlock)
        self.assertIsInstance(self.kernel_compose(False, 1)[0], sb.PatchesBlock)

    def test_dense_compose_chunks_match(self):
        g.compact_patches.set_flag()
        B, K1, Pox, c0, Pk, M, ix = 1, 4, 12, 4, 5, 7, 12
        P, Pd = _patches(B, K1, Pox, c0, Pk, 1, 2, ix)
        C = torch.randn(B, M, K1 * Pox * Pox)
        ref = torch.matmul(C, Pd)
        for chunk in (1 << 30, 1):
            with self.subTest(chunk_bytes=chunk):
                sb.PATCH_COMPOSE_CHUNK_BYTES = chunk
                D = sb.DenseBlock(C, [], -1)
                res, _ = D.matmul_equal_dims(P, json_list=[], lhs_index=0, rhs_index=1)
                torch.testing.assert_close(_dense(res), ref, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
