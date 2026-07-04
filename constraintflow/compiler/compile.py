import antlr4 as antlr
import os

from constraintflow.ast_cflow import dslLexer
from constraintflow.ast_cflow import dslParser
from constraintflow.ast_cflow import astBuilder
from constraintflow.ast_cflow import astTC
from constraintflow.compiler import convertToIr as c2r
from constraintflow.compiler import representations
from constraintflow.compiler import codeGen
from constraintflow.compiler.optimizations import polyOpt
from constraintflow.compiler.optimizations import symexpCount
from constraintflow.compiler.optimizations import loopInvariantCodeMotion
from constraintflow.compiler.optimizations import copyPropagation
from constraintflow.compiler.optimizations import dce
from constraintflow.compiler.optimizations import cse
from constraintflow.compiler.optimizations import rewrite
from constraintflow.compiler.optimizations import parallelizeLoops



optimizations_rewrite = [
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    polyOpt.poly_opt,
    cse.cse,
    dce.dce,
    rewrite.rewrite,
    cse.cse,
    copyPropagation.copy_proagate,
    dce.dce,
    dce.dce,
    dce.dce,
    loopInvariantCodeMotion.licm,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    symexpCount.correct_symexp_size,
    copyPropagation.copy_proagate,
    ]

# Same pipeline without the domain-specific rewrite pass (used for ablation N-R / N-R-S).
optimizations_no_rewrite = [
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    polyOpt.poly_opt,
    cse.cse,
    dce.dce,
    cse.cse,
    copyPropagation.copy_proagate,
    dce.dce,
    dce.dce,
    dce.dce,
    loopInvariantCodeMotion.licm,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    symexpCount.correct_symexp_size,
    copyPropagation.copy_proagate,
    ]


def compile(inputfile, output_path, no_rewrite=False):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()

    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)

    ir = c2r.ConvertToIr().visit(ast)
    representations.ssa(ir)

    optimizations = optimizations_no_rewrite if no_rewrite else optimizations_rewrite

    for opt in optimizations:
        opt(ir)
    representations.remove_phi(ir)
    if not os.environ.get('CF_NO_PARALLEL'):
        parallelizeLoops.parallelize_loops(ir)
    codeGen.CodeGen(output_path).visit(ir)

    return True