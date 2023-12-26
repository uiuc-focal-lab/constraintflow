from common.astinterpreter import *

import sys

import antlr4 as antlr

import astcf as AST
import dslLexer
import dslParser
import astBuilder
import astTC



def genAST(inputfile):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)

    x = astTC.ASTTC().visit(ast)

    newtrans = AstInterpret("newtransformer.py")
    newtrans.visit(ast)


genAST(sys.argv[1])
