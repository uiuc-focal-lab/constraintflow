import sys

import antlr4 as antlr

import astcf as AST
import dslLexer
import dslParser
import astBuilder
import astTC
import verify

import time
from multiprocessing import Process
from call_function_with_timeout import SetTimeout


def genAST(inputfile, nprev, nsymb, ncurr):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)

    x = astTC.ASTTC().visit(ast)

    v = verify.Verify()
    v.Nprev = nprev
    v.Nzono = nsymb
    v.Ncurr = ncurr
    v.visit(ast)

a = sys.argv[2].split(" ")
genAST(sys.argv[1],int(a[0]), int(a[1]), int(a[2]))
