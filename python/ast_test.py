import antlr4 as antlr

import ast as AST
import dslLexer
import dslParser
import astBuilder
import astTC
import verify

def genAST(inputfile):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)

    #x = astTC.ASTTC().visit(ast)

    verify.Verify().visit(ast)


genAST("test_input.cf")