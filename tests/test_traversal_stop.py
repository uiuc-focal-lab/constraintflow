"""Invariant-stop analysis and selective execution must preserve CF semantics."""
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import antlr4 as antlr
import torch
from constraintflow.ast_cflow import astcf as A, astBuilder, dslLexer, dslParser, astTC
from constraintflow.compiler.traversal_stop import transform
from constraintflow.compiler.optimizations.conditional_rows import refine_rows, plan, row_separable
from constraintflow.compiler.optimizations.row_refinement import warmup_row_kernels
from constraintflow.compiler.codeGen import CodeGen
from constraintflow.compiler.ir import IrBlock, IrTorchWhere, IrTorchMatmul, IrSimpleBinary, IrTorchSum, IrTransRetBasic
from constraintflow.compiler.optimizations.row_regions import assign, var


class Probe:
    """Evaluates IR expressions in a test environment for row-separability proofs."""
    def __init__(self, env):
        import operator
        self.env = dict(env, torch=torch, operator=operator)
        self.cg = CodeGen.__new__(CodeGen)
        self.cg.const_pool = {}

    def value(self, expr, **extra):
        return eval(self.cg.visit(expr), dict(self.env, **extra))

    def shape(self, expr):
        value = self.value(expr)
        return tuple(value.shape) if isinstance(value, torch.Tensor) else None

PROGRAM = Path(__file__).resolve().parents[1] / 'examples/compiler_examples/deeppoly_traversal_stop.cf'

def parse(source):
    lexer = dslLexer.dslLexer(antlr.InputStream(source))
    parser = dslParser.dslParser(antlr.CommonTokenStream(lexer))
    tree = parser.prog()
    assert parser.getNumberOfSyntaxErrors() == 0
    return astBuilder.ASTBuilder().visit(tree)

def walk(node):
    yield node
    if isinstance(node, A.ASTNode):
        for value in vars(node).values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)

class StopAnalysisTest(unittest.TestCase):
    def test_cf_typechecks_and_hoists_without_program_or_function_names(self):
        source = PROGRAM.read_text().replace('stop_row','arbitrary_predicate').replace('refine_lower','arbitrary_lower')
        ast = parse(source)
        astTC.ASTTC().visit(copy.deepcopy(ast))
        self.assertEqual(transform(ast), 2)
        astTC.ASTTC().visit(ast)
        guards = [x for x in walk(ast) if getattr(x,'invariant_stop',False)]
        self.assertEqual(len(guards),2)
        self.assertTrue(all(isinstance(x,A.TernaryNode) for x in guards))
        self.assertEqual(transform(ast),0)

    def test_visited_neuron_or_coefficient_is_not_row_invariant(self):
        for predicate in ('n[l] >= 0.0','c >= 0.0','(l >= 0.0) or (n[u] <= 0.0)'):
            source = PROGRAM.read_text()
            start=source.index('func stop_row(');end=source.index(';',start)
            source=source[:start]+f'func stop_row(Float l, Float u, Neuron n, Float c) = {predicate}'+source[end:]
            self.assertEqual(transform(parse(source)),0)

    def test_literal_stop_is_left_alone(self):
        # provesound examples write `true`/`false` in place of a stop function; that used to crash every compile.
        for literal in ('true', 'false'):
            source = PROGRAM.read_text().replace('stop_row(n[l], n[u])', literal)
            self.assertEqual(transform(parse(source)), 0)

    def test_helper_dependency_is_followed(self):
        for expr,expected in [('x[l] >= 0.0',0),('false',0)]:
            source=PROGRAM.read_text().replace('func stop_row(',f'func helper(Neuron x) = {expr};\nfunc stop_row(')
            start=source.index('func stop_row(');end=source.index(';',start)
            source=source[:start]+'func stop_row(Float l, Float u, Neuron n, Float c) = helper(n)'+source[end:]
            self.assertEqual(transform(parse(source)),expected)

    def test_arbitrary_captured_threshold_is_hoisted(self):
        source=PROGRAM.read_text()
        start=source.index('func stop_row(');end=source.index(';',start)
        source=source[:start]+'func stop_row(Float l, Float u, Neuron n, Float c) = l > 7.5'+source[end:]
        self.assertEqual(transform(parse(source)),2)

class RowRuntimeTest(unittest.TestCase):
    def test_masks_padding_batch_union_and_immutability(self):
        torch.manual_seed(3)
        values=(torch.randn(3,9),torch.randn(3,9))
        for stop in (torch.ones(3,9,dtype=torch.bool),torch.zeros(3,9,dtype=torch.bool),torch.rand(3,9)>.4):
            defaults=(torch.randn(3,9),torch.randn(3,9)); before=tuple(x.clone() for x in defaults);seen=[]
            def kernel(idx):
                seen.append(idx.clone());return tuple(x.index_select(1,idx) for x in values)
            result=refine_rows(stop,defaults,kernel,(),4)
            for got,dense,default,saved in zip(result,values,defaults,before):
                torch.testing.assert_close(got,torch.where(stop,default,dense),rtol=0,atol=0)
                self.assertTrue(torch.equal(default,saved))
            self.assertTrue(all(len(x)==4 for x in seen))
            if stop.all():self.assertEqual(seen,[])

    def test_union_rule_refines_selected_rows_for_every_example(self):
        # Row 1 is needed by example 0 only; under the union rule example 1 takes the refined value too.
        stop=torch.tensor([[True,False,True],[True,True,True]]); defaults=(torch.zeros(2,3),)
        def kernel(idx):return (torch.full((2,len(idx)),7.),)
        torch.testing.assert_close(refine_rows(stop,defaults,kernel,(),4)[0],torch.tensor([[0.,7.,0.],[0.,0.,0.]]))
        torch.testing.assert_close(refine_rows(stop,defaults,kernel,(),4,union=True)[0],torch.tensor([[0.,7.,0.],[0.,7.,0.]]))

    def test_several_chunk_sizes_pick_smallest_cover_and_warm_every_size(self):
        values=(torch.arange(300.)[None],)
        for needed,expected in [(1,[8]),(44,[64]),(129,[128,8]),(300,[128,128,64])]:
            stop=torch.ones(1,300,dtype=torch.bool); stop[0,:needed]=False; seen=[]
            def kernel(idx):
                seen.append(len(idx));return tuple(x.index_select(1,idx) for x in values)
            got=refine_rows(stop,(torch.zeros(1,300),),kernel,(),(8,16,32,64,128))
            self.assertEqual(seen,expected)
            torch.testing.assert_close(got[0][0,:needed],values[0][0,:needed])
        seen=[]
        warmup_row_kernels(refine_rows,torch.ones(1,50,dtype=torch.bool),(torch.zeros(1,50),),kernel,(),(8,32,128))
        self.assertEqual(seen,[8,32,50])

    def test_warmup_executes_skipped_kernel_and_discards_result(self):
        defaults=(torch.tensor([[1.,2.,3.]]),);seen=[]
        def kernel(idx):
            seen.append(idx.clone());return (torch.full((1,len(idx)),float('nan')),)
        got=warmup_row_kernels(refine_rows,torch.ones(1,3,dtype=torch.bool),defaults,kernel,(),128)
        self.assertEqual(len(seen),1)
        self.assertTrue(torch.equal(got[0],defaults[0]))

    def test_nonfinite_unselected_values_do_not_contaminate_results(self):
        def kernel(idx):return (torch.tensor([[float('nan'),5.]]).index_select(1,idx),)
        got=refine_rows(torch.tensor([[True,False]]),(torch.tensor([[2.,3.]]),),kernel,(),2)
        torch.testing.assert_close(got[0],torch.tensor([[2.,5.]]))


class RowAnalysisTest(unittest.TestCase):
    def test_reductions_and_contractions_cannot_mix_rows(self):
        probe=Probe({'x':torch.randn(2,5,7),'w':torch.randn(7,3),'other':torch.randn(2,7,5)})
        self.assertTrue(row_separable(IrTorchSum(var('x'),-1),{'x'},probe,5,2))
        self.assertFalse(row_separable(IrTorchSum(var('x'),1),{'x'},probe,5,2))
        self.assertTrue(row_separable(IrTorchMatmul(var('x'),var('w')),{'x'},probe,5,2))
        self.assertFalse(row_separable(IrTorchMatmul(var('other'),var('x')),{'x'},probe,5,2))

    def test_dependency_slicing_and_escaping_coefficients(self):
        probe=Probe({'weight':torch.randn(2,5,7),'bias':torch.randn(2,5),
                     'mask':torch.rand(2,5)>.5,'default':torch.randn(2,5)})
        stmts=[assign('coeff',IrSimpleBinary(var('weight'),2.0,'mul')),
               assign('dense',IrTorchSum(var('coeff'),-1)),
               assign('result',IrTorchWhere(var('mask'),var('default'),var('dense'))),
               IrTransRetBasic([var('result')])]
        for stmt in stmts[:-1]:probe.env[stmt.children[0].name]=probe.value(stmt.children[1])
        probe.shapes={name:tuple(v.shape) for name,v in probe.env.items() if isinstance(v,torch.Tensor)}
        node=SimpleNamespace(flow_block=IrBlock(stmts),flow_functional=True,flow_batch_size=2)
        regions,calls=plan(node,probe)
        self.assertEqual(sum(r.selected for r in regions),1)
        self.assertEqual(calls[1].function,'refine_rows')
        # An external consumer of coefficients prohibits dropping their dense computation.
        node.flow_block=IrBlock(stmts[:-1]+[IrTransRetBasic([var('result'),var('coeff')])])
        self.assertIsNone(plan(node,probe))


class CompilerIntegrationTest(unittest.TestCase):
    def test_compiled_fc_and_convolution_match_dense_cf_across_masks(self):
        torch.manual_seed(21)
        cases = [
            (torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                torch.nn.Linear(8, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)), (4,), 3),
            (torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(2, 2, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Flatten(), torch.nn.Linear(40, 2)), (1,4,5), 1),
        ]
        root = PROGRAM.parents[2]
        for model, shape, batch in cases:
            with self.subTest(shape=shape, batch=batch), tempfile.TemporaryDirectory() as tmp:
                network = Path(tmp,'network.onnx')
                torch.onnx.export(model.eval(), torch.zeros(1,*shape), network, dynamo=False)
                folder = Path(tmp,'program')
                dataset = 'box-'+'x'.join(map(str,shape))+'-2-2'
                env = dict(os.environ,PYTHONPATH=str(root))
                command = [sys.executable,'-m','constraintflow.cli','jit',str(PROGRAM),
                    '--network',str(network),'--dataset',dataset,'--device','cpu',
                    # Equality with the dense program holds under the program's own per-example stops.
                    '--batch-size',str(batch),'--fuse-affine-subst','--traversal-stop-per-example','--output-path',str(folder)]
                result = subprocess.run(command,cwd=tmp,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('[conditional-rows]',result.stdout)
                check = r"""
import sys, math
from pathlib import Path
import torch
sys.path.insert(0,sys.argv[1])
import main, transformers
from constraintflow.lib.globals import device_mode
device_mode.set_mode('cpu')
namespace=dict(vars(transformers))
exec(Path(sys.argv[1],'transformers_dense.py').read_text(),namespace)
shape=tuple(map(int,sys.argv[3].split(','))); batch=int(sys.argv[5])
for width in (0.,.01,.5,10.):
    x=torch.linspace(-2.,3.,batch*math.prod(shape)).reshape(batch,*shape)
    net,ae=main.build(sys.argv[2],batch,0,x,torch.zeros(batch,dtype=torch.long),
        sys.argv[4],False,False,False,input_box=(x-width,x+width))
    args=transformers.explode_inputs(ae,batch)
    for actual,expected in zip(transformers.flow(*args),namespace['flow'](*args)):
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
"""
                result=subprocess.run([sys.executable,'-c',check,str(folder),str(network),
                    ','.join(map(str,shape)),dataset,str(batch)],cwd=tmp,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()

