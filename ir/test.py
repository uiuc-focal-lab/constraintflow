import sys
sys.path.append('../ast/')

import antlr4 as antlr

import astcf as AST
import dslLexer
import dslParser
import astBuilder
import astTC
import convert_to_ir_2 as c2r
import cse
import copy_propagation
import codeGen2 as codeGen
from print import *



import time
from multiprocessing import Process
from call_function_with_timeout import SetTimeout


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

    y = cse.cse(y)
    y = copy_propagation.copy_proagate(y)
    y = cse.cse(y)

    print("CSE", time.time()-ttime)
    ttime = time.time()

    # print('Printing')
    # pprint(y)
    # jkdf
    
    

    z = codeGen.CodeGen('../compiled_code').visit(y)
    print('Code Generation', time.time()-ttime)
    ttime = time.time()

if(len(sys.argv) <= 1):
    filename = '../compiled_code/testcases/deeppoly'
else:
    filename = sys.argv[1]
genAST(filename)
