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
    print("start", time.time())
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    print("lexer", time.time())
    tokens = antlr.CommonTokenStream(lexer)
    print("tokens", time.time())
    parser = dslParser.dslParser(tokens)
    print("parser", time.time())
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    print("visit", time.time())
    x = astTC.ASTTC().visit(ast)
    print("type check", time.time())
    v = verify.Verify()
    v.Nprev = nprev
    v.Nzono = nsymb
    v.Ncurr = ncurr
    v.visit(ast)
    print("verify", time.time())

a = sys.argv[2].split(" ")
genAST(sys.argv[1],int(a[0]), int(a[1]), int(a[2]))
