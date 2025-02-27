import antlr4 as antlr
import sys 
from tabulate import tabulate

from constraintflow import dslLexer
from constraintflow import dslParser
from constraintflow import astBuilder
from constraintflow import astTC

from provesound.src import verify

def run_verifier(inputfile, nprev, nsymb):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()
    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)
    v = verify.Verify()
    v.Nprev = nprev
    v.Nsym = nsymb
    ret_dict = v.visit(ast)
    return ret_dict

if __name__ == "__main__":
    certifier = sys.argv[1]
    nprev = int(sys.argv[2])
    nsym = int(sys.argv[3])
    ret_dict_correct = run_verifier(certifier, nprev, nsym)
    basicops = list(ret_dict_correct.keys())

    table = []
    row1 = []
    for b in basicops:
        row1.append(b)
        row1.append("")
    table.append(["Certifier"]+row1)
    heading = ['G', 'V']*len(basicops)
    table.append([" "]+heading)
    for c in [certifier]:
        row = [c]
        for b in basicops:
            row += [round(ret_dict_correct[b][1], 3), round(ret_dict_correct[b][0], 3)]
        table.append(row)
    print()
    print()
    print(tabulate(table))
