import os,sys,tempfile,unittest
import torch,torch.nn as nn
from constraintflow.lib import training
# Training builds: two programs load side by side, cli's bare `from main import run` still works, and
# TrainingBounds (build once, swap inputs) equals run() (parse every call) on new weights and labels.
@unittest.skipUnless(torch.cuda.is_available(),'needs CUDA')
class TrainingBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model=nn.Sequential(nn.Flatten(),nn.Linear(784,20),nn.ReLU(),nn.Linear(20,10)).cuda()
        cls.cache=tempfile.mkdtemp()
        cls.tb=training.TrainingBounds(cls.model,torch.zeros(2,1,28,28),'mnist',['IBP','CROWN-IBP'],[8],cls.cache,
                                       cudagraphs=False,data_root=training.CF_ROOT)
    def test_programs_are_separate_packages(self):
        ibp,crown=(self.tb.programs[m,8][2] for m in ('IBP','CROWN-IBP'))
        self.assertIsNot(ibp,crown)
        self.assertNotEqual(ibp.__module__,crown.__module__)
        self.assertNotIn('main',sys.modules)
    def test_cli_import_still_works(self):
        folder=training.program_dir(self.cache,self.tb.onnx_path,'IBP',8,'mnist',False)
        sys.path.insert(0,folder)
        try:
            from main import run
            self.assertTrue(callable(run))
        finally:
            sys.path.remove(folder)
            for m in ('main','transformers','jit_constants'):sys.modules.pop(m,None)
    def test_matches_run(self):
        x=torch.rand(8,1,28,28,device='cuda');y=torch.randint(0,10,(8,),device='cuda')
        lo,hi=(x-0.05).clamp(min=0),(x+0.05).clamp(max=1)
        with torch.no_grad():
            for p in self.model.parameters():p.add_(0.1*torch.randn_like(p))
            for m in ('IBP','CROWN-IBP'):
                main=training.load_program(training.program_dir(self.cache,self.tb.onnx_path,m,8,'mnist',False))
                ref=main.run(self.tb.onnx_path,8,0.0,x,y,'mnist',False,False,False,initializers=self.tb.params,input_box=(lo,hi))
                got=self.tb.compute_bounds(m,lo,hi,y)
                for a,b in zip(got,ref):torch.testing.assert_close(a,b,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
