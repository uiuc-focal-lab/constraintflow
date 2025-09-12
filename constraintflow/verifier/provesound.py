import antlr4 as antlr

from constraintflow.ast_cflow import dslLexer
from constraintflow.ast_cflow import dslParser
from constraintflow.ast_cflow import astBuilder
from constraintflow.ast_cflow import astTC

from constraintflow.verifier.src import verify 


def provesound(program, nprev=1, nsymb=1):
    lexer = dslLexer.dslLexer(antlr.FileStream(program))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)
    v = verify.Verify()
    v.Nprev = nprev
    v.Nsym = nsymb
    v.visit(ast)