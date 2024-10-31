import sys
import antlr4 as antlr
import time

from ast_cflow import astcf as AST
from ast_cflow import dslLexer
from ast_cflow import dslParser
from ast_cflow import astBuilder
from ast_cflow import astTC
from compiler import convertToIr as c2r, representations
from compiler.optimizations import copyPropagation, polyOpt, uses, cse, dce, rewrite
from compiler import codeGen as codeGen


def genAST(inputfile):
    ttime = time.time()
    
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    
    print('Parsing', time.time()-ttime)
    ttime = time.time()
    
    ast = astBuilder.ASTBuilder().visit(tree)
    
    print("Building AST", time.time()-ttime)
    ttime = time.time()
    
    x = astTC.ASTTC().visit(ast)
    
    print("Type checking", time.time()-ttime)
    ttime = time.time()

    y = c2r.ConvertToIr().visit(ast)

    print("Creating IR", time.time()-ttime)
    ttime = time.time()

    representations.ssa(y)
    cse.cse(y)
    copyPropagation.copy_proagate(y)
    cse.cse(y)
    copyPropagation.copy_proagate(y)
    polyOpt.poly_opt(y)
    rewrite.rewrite(y)
    cse.cse(y)
    copyPropagation.copy_proagate(y)
    # dce.dce(y)
    # copy_propagation.copy_proagate(y)
    dce.dce(y)
    dce.dce(y)
    # cse.cse(y)

    # print("CSE", time.time()-ttime)
    # ttime = time.time()

    

    representations.remove_phi(y)
    z = codeGen.CodeGen('certifier').visit(y)
    print('Code Generation', time.time()-ttime)
    ttime = time.time()

if(len(sys.argv) <= 1):
    filename = 'certifier/testcases/deeppoly'
else:
    filename = sys.argv[1]
genAST(filename)
