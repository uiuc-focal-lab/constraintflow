import astVisitor
import ast as AST
from z3 import *
from value import *
from symbolicos import *
from symbolicgraph import *

class IfListVal:

	def __init__(self, cond, left, right):
		self.cond = cond
		self.left = left
		self.right = right

class Verify(astVisitor.ASTVisitor):

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
		s.os.store = store

		for i in range(len(node.oplist.olist)):
			op = node.oplist.olist[i]
			Cnew = []

			#Define relationship between curr and prev
			if(op.op.op_name == "Affine"):
				if(not "bias" in curr.symmap.keys()):
					curr.symmap["bias"] = ((Real('bias_Y' + str(self.number.nextn())), "Float"))
				if(not "weight" in curr.symmap.keys()):
					curr.symmap["weight"] = [(Real('weight_Y' + str(i) + "_" + str(self.number.nextn())), "Float") for i in range(self.N)]
				exptemp = curr.symmap["bias"]
				for i in range(len(prev)):
					exptemp = ADD(exptemp, MULT(curr.symmap["weight"][i], prev[i]))
					
				exptemp = s.os.convertToZ3(exptemp)

			elif(op.op.op_name == "Relu"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])

				exptemp = If(s.os.convertToZ3(exptemp) >= 0, s.os.convertToZ3(exptemp), 0)

			else: #Maxpool
				mpool = Vertex("Maxpool")
				#Don't add to V bc you shouldn't need it and you don't want to assume the shape constraint holds for it
				for prevnode in prev:
					Cnew.append(prevnode[0] == mpool.name)
				exptemp = mpool.name

			s.currop = (curr.name == exptemp) #Would it be safe to just add this to s.C?

			s.visit(op.ret)
			vallist = None
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				vallist = self.visitTransRetBasic(op.ret, s)

			
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

				s.os.store["curr_new"] = (v, "Neuron")
				Cnew.append(s.os.visit(self.constraint))

			del s.os.store["curr_new"]

			currprime = Vertex('Currprime' + str(i))
			s.V[currprime.name] = currprime

			s.os.store["curr_new"] = (currprime.name, "Neuron")


			computation = s.os.convertToZ3(s.os.store["curr_new"]) == exptemp
			leftC = And(And(And(And(s.os.C), And(Cnew)), computation), currprime.name == curr.name)

			self.applyTrans(leftC, vallist, s, currprime)

	def applyTrans(self, leftC, vallist, s, currprime):
		if(not isinstance(vallist, IfListVal)):
			for (elem, val) in zip(self.shape.keys(), vallist):
				currprime.symmap[elem] = val

			z3constraint = s.os.visit(self.constraint)
			solver = Solver()
			p = Not(Implies(leftC, z3constraint))
			solver.add(p)
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