import astVisitor
import ast as AST

class TypeMismatchException(Exception):

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

class DuplicateVarException(Exception):

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

class UndefinedVarException(Exception):

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

class ArrowType():

	def __init__(self, tleft, tright):
		if(len(tleft) == 1):
			self.tleft = tleft[0]
		else:
			self.tleft = tleft
		self.tright = tright

	def __str__(self):
		return str(self.tleft) + ' -> ' + str(self.tright)

class ArrayType():

	def __init__(self, base):
		self.base = base

	def __str__(self):
		return str(self.base) + " list"

class TransformerType():

	def __init__(self):
		pass

	def __str__(self):
		return "Transformer"

class ASTTC(astVisitor.ASTVisitor):

	def __init__(self):
		self.vars = {}
		self.shape = []

	def isType(self, t1, t2):
		#May need to remove these two lines and figure out ['x'] to 'x' conversion elsewhere
		if(isinstance(t1, list) and len(t1) == 1):
			t1 = t1[0]
		if(isinstance(t2, list) and len(t2) == 1):
			t2 = t2[0]

		if(t1 == t2):
			return True

		if(t1 == "Int"):
			if(t2 == "Float"):
				return True
			if(t2 == "PolyExp"):
				return True
			if(t2 == "ZonoExp"):
				return True
		elif(t1 == "Float"):
			if(t2 == "PolyExp"):
				return True
			if(t2 == "ZonoExp"):
				return True
		elif(t1 == "Neuron"):
			if(t2 == "PolyExp"):
				return True

		if(isinstance(t1, list)):
			if(isinstance(t2, list)):
				if(len(t1) == len(t2)):
					for (type1, type2) in zip(t1, t2):
						if(not self.isType(type1, type2)):
							return False

					return True

		if(isinstance(t2, ArrowType)):
			if(isinstance(t1, ArrowType)):
				return self.isType(t1.tleft, t2.tleft) and self.isType(t1.tright, t2.tright)
			else:
				return self.isType(t1, t2.tright) #for True instead of Neuron -> Bool

		if(isinstance(t1, ArrayType)):
			if(isinstance(t2, ArrayType)):
				return self.isType(t1.base, t2.base)

		if(isinstance(t1, TransformerType) and isinstance(t2, TransformerType)):
			return True

		return False

	#Returns the LUB of the types or None
	def comparable(self, t1, t2):
		if(self.isType(t1, t2)):
			return t2
		elif(self.isType(t2, t1)):
			return t1
		elif(t1 == "Neuron" and self.isType(t2, "Float")):
			return "PolyExp"
		elif(t2 == "Neuron" and self.isType(t1, "Float")):
			return "PolyExp"
		else:
			return None

	def visitBaseType(self, node: AST.BaseTypeNode):
		return node.name

	def visitArrayType(self, node: AST.ArrayTypeNode):
		return ArrayType(node.base)

	def visitArgList(self, node: AST.ArgListNode):
		return [t[0].name for t in node.arglist]

	def visitExprList(self, node: AST.ExprListNode):
		listtype = []
		for e in node.exprlist:
			listtype.append(self.visit(e))

		return listtype

	def BinOphelper(self, node, ltype, rtype):
		accepted = ["Int", "Float", "Neuron", "PolyExp", "ZonoExp"]
		if(node.op == "+" or node.op == "-"):
			if(ltype in accepted and rtype in accepted):
				if(self.comparable(ltype, rtype)):
					return self.comparable(ltype, rtype)
				else:
					raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "*"):
			if(self.isType(ltype, "Float")):
				if(rtype in accepted):
					return self.comparable(ltype, rtype)
			elif(self.isType(rtype, "Float")):
				if(ltype in accepted):
					return self.comparable(rtype, ltype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "/"):
			if(self.isType(ltype, "Float") and self.isType(rtype, "Float")):
				return self.comparable(ltype, rtype)
			elif(self.isType(rtype, "Float")):
				if(ltype == "PolyExp" or ltype == "ZonoExp"):
					return ltype

		elif(node.op == "and" or node.op == "or"):
			if(ltype == "Bool" and rtype =="Bool"):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))
		else: #<=, >=, ==
			if(self.isType(ltype, "Float") and self.isType(rtype, "Float")):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))


	def visitBinOp(self, node: AST.BinOpNode):
		ltype = self.visit(node.left)
		rtype = self.visit(node.right)

		#Don't need the if statement before calling the function unless you define some operations on lists
		if(not isinstance(ltype, ArrayType) and not isinstance(rtype, ArrayType)):
			return self.BinOphelper(node, ltype, rtype)
		else:
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		
	def visitUnOp(self, node: AST.UnOpNode):
		t = self.visit(node.expr)
		if(node.op == "-"):
			if(t == "Int" or t =="Float"):
				return t
			else:
				raise TypeMismatchException("- not defined on " + str(t))
		elif(node.op == "~"):
			if(t == "Bool"):
				return t
			else:
				raise TypeMismatchException("~ not defined on " + str(t))
		else:
			assert False

	def visitNlistOp(self, node: AST.NlistOpNode):
		exptype = self.visit(node.expr)
		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException(node.op + " requires Neuron list as first argument")
		elif(not exptype.base == "Neuron"):
			raise TypeMismatchException(node.op + " requires Neuron list as first argument")

		if(not node.elem.name in [x[1] for x in self.shape]):
			raise TypeMismatchException(node.elem.name + " not found in shape declaration")
		
		if(node.op == "min" or node.op == "max"):
			return self.vars[node.elem.name]
		elif(node.op == "argmin" or node.op == "argmax"):
			return "Neuron"
		else:
			assert False

	def visitVar(self, node: AST.VarNode):
		if not node.name in self.vars.keys():
			raise UndefinedVarException(node.name + " is undefined")
		else:
			return self.vars[node.name]

	def visitInt(self, node: AST.ConstIntNode):
		return "Int"

	def visitFloat(self, node: AST.ConstFloatNode):
		return "Float"

	def visitBool(self, node: AST.ConstBoolNode):
		return "Bool"

	def visitCurr(self, node: AST.CurrNode):
		return "Neuron"

	def visitPrev(self, node: AST.PrevNode):
		return ArrayType("Neuron")

	def visitEpsilon(self, node: AST.EpsilonNode):
		return "Noise"

	def visitTernary(self, node: AST.TernaryNode):
		ctype = self.visit(node.cond)
		if(not self.isType(ctype, "Bool")):
			raise TypeMismatchException("Condition has to be boolean")

		ltype = self.visit(node.texpr)
		rtype = self.visit(node.fexpr)
		if(self.comparable(ltype, rtype)):
			return self.comparable(ltype, rtype)
		else:
			raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		etype = self.visit(node.expr)
		if(etype == "Neuron"):
			if(node.metadata.name == "layer"):
				return "Int"
			elif(node.metadata.name == "bias"):
				return "Float"
			elif(node.metadata.name == "weight"):
				return ArrayType("Float")
			else:
				assert False
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			if(node.metadata.name == "layer"):
				return ArrayType("Int")
			elif(node.metdata.name == "bias"):
				return ArrayType("Float")
			elif(node.metadata.name == "weight"):
				return ArrayType(ArrayType("Float"))
			else:
				assert False
		else:
			raise TypeMismatchException("To access metadata, expression must be Neuron or Neuron List")

	def visitGetElement(self, node: AST.GetElementNode):
		etype = self.visit(node.expr)
		if(not node.elem.name in self.vars.keys()):
			raise UndefinedVarException(node.elem.name + " is undefined")

		elemtype = self.vars[node.elem.name]
		if(not (elemtype, node.elem.name) in self.shape):
			raise TypeMismatchException(node.elem.name + " is not part of the shape")

		if(etype == "Neuron"):
			return elemtype
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			return ArrayType(elemtype)
		else:
			raise TypeMismatchException("To access shape variables, expression must be Neuron or Neuron List")


	def visitTraverse(self, node: AST.TraverseNode):
		exprtype = self.visit(node.expr)
		if(not exprtype == "PolyExp"):
			raise TypeMismatchException("Traverse must be called on PolyExp")

		prtype = self.visit(node.priority)
		sttype = self.visit(node.stop)
		if(not self.isType(prtype, ArrowType("Neuron", "Float"))):
			raise TypeMismatchException("Traverse priority function is wrong type")
		elif(not self.isType(sttype, ArrowType("Neuron", "Bool"))):
			raise TypeMismatchException("Traverse stopping function is wrong type")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Last argument to Traverse must be a function")
		left = ftype.tleft
		right = ftype.tright
		if(isinstance(left, list)):
			if(len(left) == 2):
				if(left[0] == "Neuron" and left[1] == "Float"):
					if(self.isType(right, "PolyExp")):
						self.visit(node.p)
						if(right == "Neuron"):
							return "PolyExp"
						else:
							return right

		raise TypeMismatchException("Last argument to traverse is the wrong type")


	def visitSum(self, node: AST.SumNode):
		exptype = self.visit(node.expr)

		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException("Expression passed to sum must be a list")

		base = exptype.base
		if(base == "Int" or base == "Float" or base == "PolyExp" or base == "ZonoExp"):
			return base
		elif (base == "Neuron"):
			return "PolyExp"
		else:
			raise TypeMismatchException(base + " list cannot be passed to sum")

	def visitSub(self, node: AST.SubNode):
		listtype = self.visit(node.listexpr)
		if(isinstance(listtype, ArrayType)):
			etype = self.visit(node.expr)
			if(self.isType(etype, listtype.base)):
				return listtype
			else:
				raise TypeMismatchException("Second argument of Sub should be of the same type as values in the list")
		else:
			raise TypeMismatchException("First argument of Sub should be a list")

	def visitMap(self, node: AST.MapNode):
		exptype = self.visit(node.expr)
		if(not exptype == "PolyExp"):
			raise TypeMismatchException("Left of Map function must be a PolyExp")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Argument to Map must be a function")
		left = ftype.tleft
		right = ftype.tright
		if(isinstance(left, list)):
			if(len(left) == 2):
				if(left[0] == "Neuron" and left[1] == "Float"):
					if(self.isType(right, "PolyExp")):
						if(right == "Neuron"):
							return "PolyExp"
						else:
							return right

		raise TypeMismatchException("Function argument to Map is the wrong type")


	def visitDot(self, node: AST.DotNode):
		tleft = self.visit(node.left)
		tright = self.visit(node.right)

		if(not isinstance(tleft, ArrayType)):
			raise TypeMismatchException("Left of dot product has to be Neuron list")
		elif(not tleft.base == "Neuron"):
			raise TypeMismatchException("Left of dot product has to be Neuron list")

		if(self.isType(tright, ArrayType("Float"))):
			return "PolyExp"
		else:
			raise TypeMismatchException("Right of dot product has to be Int or Float List")

	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		if(not name in self.vars.keys()):
			raise UndefinedVarException(name + " is not defined")

		ftype = self.vars[name]
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException(name + " is not a function")
		argstype = ftype.tleft
		exprstype = self.visit(node.arglist)
		if(self.isType(exprstype, argstype)):
			return ftype.tright
		else:
			raise TypeMismatchException("Arguments of " + name + " are not the correct type(s)")

	def visitShapeDecl(self, node: AST.ShapeDeclNode):
		for (t, e) in node.elements.arglist:
			if(e.name in self.vars.keys()):
				raise DuplicateVarException(str(e.name()) + " is already defined")
			else:
				self.vars[e.name] = t.name

			self.shape.append((t.name,e.name))

		self.visit(node.p)

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):
		rettype = self.visit(node.exprlist)
		if(not self.isType(rettype, [x[0] for x in self.shape])):
			raise TypeMismatchException("Expression in transformer is different from shape declaration")


	def visitTransRetIf(self, node: AST.TransRetIfNode):
		ctype = self.visit(node.cond)
		if(not self.isType(ctype, "Bool")):
			raise TypeMismatchException("Condition has to be boolean")

		self.visit(node.tret)
		self.visit(node.fret)

	def visitOpStmt(self, node: AST.OpStmtNode):
		self.visit(node.ret)
		
	def visitOpList(self, node: AST.OpListNode):
		for s in node.olist:
			self.visit(s)

	def visitTransformer(self, node: AST.TransformerNode):
		tname = node.name.name 
		if(tname in self.vars.keys()):
			raise DuplicateVarException(tname + " is already defined")

		self.visit(node.oplist)
		self.vars[tname] = TransformerType()

	def visitPropTermBasic(self, node: AST.PropTermBasicNode):
		return self.visit(node.term)

	def visitPropTermIn(self, node: AST.PropTermInNode):
		pass #Not finished

	def visitPropTermOut(self, node: AST.PropTermOutNode):
		pass #Not finished

	def visitPropTermOp(self, node: AST.PropTermOpNode):
		pass #Not finished
		'''
		left = self.visit(node.leftpt)
		right = self.visit(node.rightpt)
		if(self.comparable(left, right)):
			return self.comparable(left, right)
		else:
			raise TypeMismatchException(str(left) + " and " + str(right) + " are not comparable")
		'''

	def visitSingleProp(self, node: AST.SinglePropNode):
		pass #Not finished
		'''
		left = self.visit(node.leftpt)
		right = self.visit(node.rightpt)
		if(self.comparable(left, right)):
			return "Bool"
		else:
			raise TypeMismatchException(str(left) + " and " + str(right) + " are not comparable")
		'''

	def visitDoubleProp(self, node: AST.DoublePropNode):
		pass #Not finished
		'''
		l = self.visit(node.leftprop)
		r = self.visit(node.rightprop)
		#Shouldn't need the rest of this function
		if(r == "Bool" and l == "Bool"):
			return "Bool"
		else:
			raise TypeMismatchException("Both properties should be boolean expressions")
		'''

	def visitFunc(self, node: AST.FuncNode):
		argstype = self.visit(node.decl.arglist)
		fname = node.decl.name.name
		if(fname in self.vars.keys()):
			raise DuplicateVarException(fname + " is already defined")
		newvars = []
		oldvalues = {}
		for (t,e) in node.decl.arglist.arglist:
			if e.name in self.vars.keys():
				oldvalues[e.name] = self.vars[e.name]
			else:
				newvars.append(e.name)

			self.vars[e.name] = t.name

		exprtype = self.visit(node.expr)
		for v in newvars:
			del self.vars[v]

		for ov in oldvalues.keys():
			self.vars[ov] = oldvalues[ov]

		self.vars[fname] = ArrowType(argstype, exprtype)

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		prtype = self.visit(node.pfunc)
		sttype = self.visit(node.sfunc)
		if(not self.isType(prtype, ArrowType("Neuron", "Float"))):
			raise TypeMismatchException("Flow priority function is wrong type")
		elif(not self.isType(sttype, ArrowType("Neuron", "Bool"))):
			raise TypeMismatchException("Flow stopping function is wrong type")

		#Can, instead, call self.visit(node.trans) and check if the return value is a TransformerType()
		if not node.trans.name in self.vars.keys():
			raise UndefinedVarException(node.trans.name + " is undefined")
		elif (not self.isType(self.vars[node.trans.name], TransformerType())):
			raise TypeMismatchException(node.trans.name + " is not a Transformer")

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)