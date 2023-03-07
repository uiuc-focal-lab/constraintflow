import antlr4 as antlr

import ast as AST
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

    astTC.ASTTC().visit(ast)

genAST("test_input")
genAST("test_input2")
