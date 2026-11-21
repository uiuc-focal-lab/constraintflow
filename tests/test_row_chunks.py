import unittest
import torch
from constraintflow.compiler.optimizations.row_chunks import rewrite
from constraintflow.gbcsr.sparse_block import patches_to_dense, realized_pad

N, K = 64, 32


def program(stmt):
    """flow(x, w): over-budget statements on [1, N, K] rows (the second reads the first), then a per-neuron sum."""
    return f'def flow(x, w):\n    v = x * w\n    y = {stmt}\n    s = y.sum(dim=-1)\n    return s\n'


class PatchesToDenseTest(unittest.TestCase):
    # 8 channels x 2x2 positions = 32 rows, 1 input channel, kernel 3x3, input 4x4, padding 1, stride 2
    GEOMETRY = '1, {channels}, 2, 2, 1, 3, 3, 4, 4, 1, 1, 2, 2'

    def rewrite_case(self, budget, **kw):
        stmt = f'patches_to_dense(v, {self.GEOMETRY.format(channels=8)}).reshape([1, 32, 16])'
        source = f'def flow(x):\n    v = x * 2.0\n    y = {stmt} * 2.0\n    s = y.sum(dim=-1)\n    return s\n'
        torch.manual_seed(3)
        inputs = [torch.randn(1, 32, 9, dtype=torch.float64)]
        globs = {'torch': torch, 'patches_to_dense': patches_to_dense}
        return source, rewrite(source, dict(globs), inputs, budget=budget, report=lambda *_: None, **kw), inputs, globs

    def test_chunks_match_whole_call(self):
        source, out, inputs, globs = self.rewrite_case(1024)
        self.assertIn('for _r0 in range', out)
        self.assertIn('_n // 4', out)
        expected, actual = dict(globs), dict(globs)
        exec(source, expected)
        exec(out, actual)
        torch.testing.assert_close(actual['flow'](*inputs), expected['flow'](*inputs), rtol=1e-12, atol=1e-12)

    def test_positions_differ_between_rows(self):
        # Guards the test above: the wrong position's gather must change the result.
        x = torch.randn(1, 32, 9, dtype=torch.float64)
        whole = patches_to_dense(x, 1, 8, 2, 2, 1, 3, 3, 4, 4, 1, 1, 2, 2)
        shifted = patches_to_dense(x[:, 2:6], 1, 1, 2, 2, 1, 3, 3, 4, 4, 1, 1, 2, 2)
        self.assertFalse(torch.equal(whole[:, 2:6], shifted))

    def test_compiled_target_below_one_channel(self):
        # A budget whose target (1 row) is below one channel (4 positions) must still chunk whole channels.
        from unittest import mock
        source, out, inputs, globs = self.rewrite_case(64, compile_bodies=True)
        self.assertIn('_cf_chunks(32, 4)', out)
        expected, actual = dict(globs), dict(globs, SharedLhsMM=lambda: None)
        exec(source, expected)
        with mock.patch.object(torch, 'compile', lambda f, **kw: f):   # check the chunking eagerly
            exec(out, actual)
            torch.testing.assert_close(actual['flow'](*inputs), expected['flow'](*inputs), rtol=1e-12, atol=1e-12)


class RealizedPadTest(unittest.TestCase):
    def test_matches_f_pad(self):
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        for pad in ([1, 1, 1, 1], [2, 0], [1, 2, 3, 4]):
            torch.testing.assert_close(realized_pad(x, pad), torch.nn.functional.pad(x, pad), rtol=0, atol=0)

    def test_fake_shape_matches(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            self.assertEqual(realized_pad(torch.zeros(1, 2, 3, 4), [1, 2, 3, 4]).shape, (1, 2, 10, 7))

    def test_compiled_body_uses_it(self):
        source = 'def flow(x):\n    v = x * 2.0\n    p = torch.nn.functional.pad(v.view([1, 64, 4, 8]), [1, 1, 1, 1]).reshape([1, 64, 60])\n' \
                 '    s = p.sum(dim=-1)\n    return s\n'.replace('torch.nn.functional.pad', 'F.pad')
        source = source.replace('torch.nn.functional.pad', 'F.pad')
        inputs = [torch.randn(1, 64, 32, dtype=torch.float64)]
        out = rewrite(source, {'torch': torch, 'F': torch.nn.functional}, inputs, budget=4096, report=lambda *_: None, compile_bodies=True)
        body = out[out.index('def _cf_body_0'):out.index('_cf_body_0 = torch.compile')]
        self.assertIn('_cf_pad(', body)
        self.assertNotIn('F.pad(', body)


class RowChunksTest(unittest.TestCase):
    def run_case(self, stmt):
        torch.manual_seed(3)
        inputs = [torch.randn(1, N, K, dtype=torch.float64), torch.randn(1, N, K, dtype=torch.float64)]
        source = program(stmt)
        out = rewrite(source, {'torch': torch}, inputs, budget=4096, report=lambda *_: None)
        return source, out, inputs

    def check_same(self, stmt):
        source, out, inputs = self.run_case(stmt)
        self.assertIsNotNone(out)
        self.assertIn('for _r0 in range', out)
        expected, actual = {'torch': torch}, {'torch': torch}
        exec(source, expected)
        exec(out, actual)
        torch.testing.assert_close(actual['flow'](*inputs), expected['flow'](*inputs), rtol=1e-12, atol=1e-12)

    def check_refused(self, stmt, reason):
        with self.assertRaisesRegex(ValueError, reason):
            self.run_case(stmt)

    def test_row_local(self):
        self.check_same('torch.clamp(v, min=0.0).unsqueeze(3).squeeze(3) - w')

    def test_pad_after_rows(self):
        self.check_same('torch.nn.functional.pad(v.view([1, 64, 4, 8]), [1, 1, 1, 1]).reshape([1, 64, 60])')

    def test_split_rows_channels_positions(self):
        # 64 rows = 16 channels x 4 positions; the operand varies over positions only
        self.check_same('(v.reshape([1, 16, 4, 32]) * torch.arange(4.0, dtype=torch.float64).view([1, 1, 4, 1])).reshape([64, 32]).reshape([1, 64, 32])')

    def test_refuse_split_rows_channel_operand(self):
        self.check_refused('(v.reshape([1, 16, 4, 32]) * torch.arange(16.0, dtype=torch.float64).view([1, 16, 1, 1])).reshape([1, 64, 32])',
                           'channels of split rows')

    def test_refuse_pad_reaching_rows(self):
        self.check_refused('torch.nn.functional.pad(v, [1, 1, 0, 0])', 'pad at or before the row axis')

    def test_refuse_sum_over_rows(self):
        self.check_refused('v.sum(dim=1, keepdim=True) + x', 'sum over the row axis')

    def test_refuse_transpose(self):
        self.check_refused('v.transpose(1, 2).reshape([1, 64, 32])', 'transpose|keep the 64 rows')

    def test_refuse_rows_in_second_operand(self):
        self.check_refused('torch.matmul(w[0] @ w[0].t(), v)', 'rows outside its first operand')

    def test_refuse_reshape_merging_rows(self):
        self.check_refused('v.reshape([1, 1, 2048]).reshape([1, 64, 32])', 'keep the 64 rows')


class RowMixingTest(unittest.TestCase):
    # y = m @ v sums over v's N rows (the previous layer's, as wide as this one); z, over budget, is row-local in y.
    SOURCE = ('def flow(x, w, m, q):\n    v = x * w\n    y = torch.matmul(m, v)\n'
              '    z = (y.unsqueeze(3) * q).sum(dim=3)\n    s = z.sum(dim=-1)\n    return s\n')

    def run_case(self, budget):
        torch.manual_seed(3)
        inputs = [torch.randn(1, N, K, dtype=torch.float64), torch.randn(1, N, K, dtype=torch.float64),
                  torch.randn(N, N, dtype=torch.float64), torch.randn(1, 1, K, 64, dtype=torch.float64)]
        return rewrite(self.SOURCE, {'torch': torch}, inputs, budget=budget, report=lambda *_: None), inputs

    def test_mixing_statement_runs_before_the_loop(self):
        out, inputs = self.run_case(256 * 1024)
        self.assertLess(out.index('torch.matmul(m, v)'), out.index('for _r0 in range'))
        expected, actual = {'torch': torch}, {'torch': torch}
        exec(self.SOURCE, expected)
        exec(out, actual)
        torch.testing.assert_close(actual['flow'](*inputs), expected['flow'](*inputs), rtol=1e-12, atol=1e-12)

    def test_refuse_mixing_statement_over_budget(self):
        with self.assertRaisesRegex(ValueError, 'rows outside its first operand.*over the budget'):
            self.run_case(64 * 1024)


class ChannelFactorTest(unittest.TestCase):
    # y's 64 rows are 16 channels x 4 positions: a[c] * u[p], built from no row tensor; z is over budget.
    def run_case(self, factor):
        source = (f'def flow(a, u, q):\n    y = ({factor} * u).reshape([1, 64, 256])\n'
                  '    z = (y.unsqueeze(3) * q).sum(dim=3)\n    s = z.sum(dim=-1)\n    return s\n')
        torch.manual_seed(3)
        inputs = [torch.randn(16, 256, dtype=torch.float64), torch.randn(1, 1, 4, 256, dtype=torch.float64),
                  torch.randn(1, 1, 256, 64, dtype=torch.float64)]
        out = rewrite(source, {'torch': torch}, inputs, budget=512 * 1024, report=lambda *_: None)
        expected, actual = {'torch': torch}, {'torch': torch}
        exec(source, expected)
        exec(out, actual)
        torch.testing.assert_close(actual['flow'](*inputs), expected['flow'](*inputs), rtol=1e-12, atol=1e-12)
        return out

    def test_channel_factor_sliced_per_chunk(self):
        out = self.run_case('a.reshape([1, 16, 1, 256])')
        self.assertIn('_cf_channels(_cf_ch_', out)

    def test_mixed_channels_fall_back(self):
        # summing over the channels mixes them: computed whole before the loop, as before
        out = self.run_case('a.reshape([1, 16, 1, 256]).cumsum(1)')
        self.assertNotIn('_cf_channels(_cf_ch_', out)


if __name__ == '__main__':
    unittest.main()
