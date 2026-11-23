import unittest
from unittest.mock import patch
from constraintflow.compiler.ir import IrAssignment,IrVar
from constraintflow.compiler.optimizations.flow_shapes import probe
from constraintflow.lib.globals import device_mode
class FlowShapeModesTest(unittest.TestCase):
    def test_shapes_and_storage_aliases_match_full_profile(self):
        names=['a','b','c','d']
        stmts=[IrAssignment(IrVar(n,[]),IrVar('x',[])) for n in names]
        sources=['a = torch.ones_like(x)','b = a.view(3, 2)',
                 'c = (a + 1).sum(dim=1)','d = b.transpose(0, 1)']
        with patch.object(device_mode,'get_device',return_value='cpu'):
            full=probe(stmts,sources,{'x':([2,3],'Float')},1,{})
            basic=probe(stmts,sources,{'x':([2,3],'Float')},1,{},record_temporaries=False)
        self.assertIsNotNone(full);self.assertIsNotNone(basic)
        for i in range(4):
            for key in ('name','root','nbytes','storage_bytes','input','alloc','shape','dtype','stride','device'):
                self.assertEqual(full[i][key],basic[i][key])
        self.assertEqual(full[0]['root'],full[1]['root'])
        self.assertEqual(full[0]['root'],full[3]['root'])
        self.assertEqual(full[1]['alloc'],0)
        self.assertNotEqual(full[0]['root'],full[2]['root'])
        self.assertGreater(full[2]['temporary_peak'],0)
        self.assertEqual(full[3]['temporary_peak'],0)
        self.assertFalse(basic.temporaries_profiled)
        self.assertTrue(full.temporaries_profiled)
if __name__=='__main__':unittest.main()
