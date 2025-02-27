import sys

import antlr4 as antlr

import astcf as AST
import dslLexer
import dslParser
import astBuilder
import astTC
import verify



def genAST(inputfile, nprev, nsymb):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)

    x = astTC.ASTTC().visit(ast)

    v = verify.Verify()
    v.Nprev = nprev
    v.Nsym = nsymb
    v.visit(ast)


genAST(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))