from verify import *
from z3 import *
from value import *
import ast as AST

n = Number()
prev1 = Vertex('prev1', [], n)
prev1.symmap["l"] = Real('Z1')
prev2 = Vertex('prev2', [], n)
prev2.symmap["l"] = Real('Z2')
prev3 = Vertex('prev3', [], n)
prev3.symmap["l"] = Real('Z3')
store = {}
store["prev"] = [prev1, prev2, prev3]
c0, c1, c2, c3 = Reals('c0 c1 c2 c3')
v1 = Vertex('v1', [], n)
v2 = Vertex('v2', [], n)
v3 = Vertex('v3', [], n)
polyexp = {v1: (c1, '+'), v2: (c2, '+'), v3: (c3, '+')}
p = PolyExpValue(polyexp, c0)
p2 = PolyExpValue(polyexp, c0)
store["e"] = p
store["e2"] = p2
prop = AST.SinglePropNode(AST.PropTermOpNode(AST.VarNode("e"), "-", AST.VarNode("e2")), "<=", AST.VarNode("e"))
e = Evaluate()
e.store = store
print(e.visit(prop))
