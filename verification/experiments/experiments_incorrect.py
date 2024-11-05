import time
import sys
import antlr4 as antlr

from ast_cflow import dslLexer
from ast_cflow import dslParser
from ast_cflow import astBuilder
from ast_cflow import astTC
from verification.src import verify
from verification.experiments.buggy_ast import BuggyAst

def genAST(inputfile, nprev, nsymb):
    start_time = time.time()
    
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    lexer_time = time.time()
    print(f"Lexer {lexer_time - start_time :.5f}s")

    tokens = antlr.CommonTokenStream(lexer)
    tokens_time = time.time()
    print(f"Tokens {tokens_time - lexer_time :.5f}s")

    parser = dslParser.dslParser(tokens)
    parser_time = time.time()
    print(f"Parser {parser_time - tokens_time :.5f}s")

    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    ast_time = time.time()
    print(f"Ast Built {ast_time - parser_time :.5f}s")
    
    x = astTC.ASTTC().visit(ast)
    typechecking_time = time.time()
    print(f"Type Checking completed {typechecking_time - ast_time :.5f}s")

    b = BuggyAst()
    b.visit(ast)
    bug_time = time.time()
    print(f"Bug introduced randomly, {bug_time - typechecking_time : .5f}s")

    v = verify.Verify()
    v.Nprev = nprev
    v.Nzono = nsymb
    v.visit(ast)
    verify_time = time.time()
    print(f"Certifier Proved Unsound {verify_time - bug_time :.5f}s")

genAST(sys.argv[1],int(sys.argv[2]), int(sys.argv[3]))
