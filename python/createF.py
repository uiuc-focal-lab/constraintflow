import astVisitor
import ast as AST
from z3 import *
from value import *
from verify import *
from symbolicgraph import *

class IfListVal:

	def __init__(self, cond, left, right):
		self.cond = cond
		self.left = left
		self.right = right

class CreateF(astVisitor.ASTVisitor):

	def __init__(self):
		self.shape = {}
		self.F = {}
		self.theta = {}
		self.N = 3
		self.number = Number()

	def visitShapeDecl(self, node):
		for (t,e) in node.elements.arglist:
			self.shape[e.name] = t.name

		self.constraint = node.p

	def visitFunc(self, node):
		self.F[node.decl.name.name] = node

	def visitTransformer(self, node):
		self.theta[node.name] = node
		store = {}
		curr = Vertex('Curr')
		V = {}
		V[curr.name] = curr
		prev = []
		for i in range(self.N):
			p = Vertex('Prev'+str(i))
			prev.append((p.name, "Neuron"))
			V[p.name] = p

		store["curr"] = (curr.name, "Neuron")
		store["prev"] = prev
		s = SymbolicGraph(store, self.F, self.constraint, self.shape)
		s.V = V
		s.os.V = V

		for i in range(len(node.oplist.olist)):
			op = node.oplist.olist[i]
			s.visit(op.ret)
			vallist = None
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				vallist = self.visitTransRetBasic(op.ret, s)

			Cnew = []
			gv = getVars(self.constraint, self.shape)
			gv.visit(self.constraint)
			vars = gv.vars

			for v in s.V.keys():
				for var in vars.keys():
					if not var in s.V[v].symmap.keys():
						if(vars[var] == "Bool"):
							s.V[v].symmap[var] = (Bool(var + '_Y' + str(self.number.nextn())), "Bool")
						elif(vars[var] == "Int"):
							s.V[v].symmap[var] = (Int(var + '_Y' + str(self.number.nextn())), "Int")
						else:
							s.V[v].symmap[var] = (Real(var + '_Y' + str(self.number.nextn())), "Float")

				store["curr'"] = (v, "Neuron")
				Cnew.append(s.os.visit(self.constraint))
			del store["curr'"]

			currprime = Vertex('Currprime' + str(i))
			V[currprime.name] = currprime

			store["curr'"] = (currprime.name, "Neuron")
			if(op.op == "Affine"):
				exptemp = curr.symmap["bias"]
				for i in range(len(prev)):
					exptemp = Add(exptemp, Mult(curr.symmap["weight"][i], prev[i][0]))

				
			else:
				#(op.op == "Relu"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = Add(exptemp, prev[i])

			computation = s.os.convertToZ3(store["curr"]) == If(s.os.convertToZ3(exptemp) >= 0, s.os.convertToZ3(exptemp), 0) 
			leftC = And(And(And(And(s.os.C), And(Cnew)), computation), curr.name == currprime.name)

			self.applyTrans(leftC, vallist, s, currprime)

	def applyTrans(self, leftC, vallist, s, currprime):
		if(not isinstance(vallist, IfListVal)):
			for (elem, val) in zip(self.shape.keys(), vallist):
				currprime.symmap[elem] = val

			z3constraint = s.os.visit(self.constraint)

			solver = Solver()
			p = Not(Implies(leftC, z3constraint))
			solver.add(p)
			print("Constraint:")
			print(solver)
			if(not (solver.check() == unsat)):
				raise Exception("Transformer"+ " " + " not true")
		else:
			condz3 = s.os.convertToZ3(vallist.cond)
			self.applyTrans(And(leftC, condz3), vallist.left, s, currprime)
			self.applyTrans(And(leftC, Not(condz3)), vallist.right, s, currprime)

	def visitTransRetBasic(self, node, s):
		return s.os.visit(node.exprlist)

	def visitTransRetIf(self, node, s):
		cond = s.os.visit(node.cond)
		left = None
		right = None
		if(isinstance(node.tret, AST.TransRetIfNode)):
			left = self.visitTransRetIf(node.tret, s)
		else:
			left = self.visitTransRetBasic(node.tret, s)

		if(isinstance(node.fret, AST.TransRetIfNode)):
			right = self.visitTransRetIf(node.fret, s)
		else:
			right = self.visitTransRetBasic(node.fret, s)

		return IfListVal(cond, left, right)

	def visitFlow(self, node):
		pass

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitProg(self, node):
		self.visit(node.shape)
		self.visit(node.stmt)
