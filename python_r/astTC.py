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

	def __eq__(self, obj):
		return isinstance(obj, TransformerType)

class ASTTC(astVisitor.ASTVisitor):

	def __init__(self):
		self.Gamma = {}
		self.shape = []
		# self.currdefined = False #prev is also defined when curr is
		self.edges = {"Int": "Float", "Float" : ["PolyExp", "ZonoExp"], "Neuron":"PolyExp", "Noise":"ZonoExp", "ZonoExp":"Top", "PolyExp":"Top", "Bool":"Top", "Bot":["Bool", "Int", "Noise", "Neuron"]}
		self.metadata = {"layer":"Int", "weight": ArrayType("Float"), "bias": "Float", "serial":"Int", "local_serial": "Int"}
	

	def isSubType(self, t1, t2):
		if(t1 == t2):
			return True
		elif(t1 == "Top"):
			return False
		elif(isinstance(t1, list)):
			if(isinstance(t2, list)):
				if(len(t1) == len(t2)):
					for (type1, type2) in zip(t1, t2):
						if(not self.isSubType(type1, type2)):
							return False
					return True
		elif(isinstance(t2, ArrowType)):
			if(isinstance(t1, ArrowType)):
				return self.isSubType(t2.tleft, t1.tleft) and self.isSubType(t1.tright, t2.tright)
			else:
				return self.isSubType(t1, t2.tright) #for True instead of Neuron -> Bool

		elif(isinstance(t1, ArrayType)):
			if(isinstance(t2, ArrayType)):
				return self.isSubType(t1.base, t2.base)
		
		else:
			l = self.edges[t1]
			if isinstance(l, list):
				for t in l:
					if self.isSubType(t, t2):
						return True 
				return False 
			return self.isSubType(self.edges[t1], t2)

	#Returns the LUB of the types
	def bfs(self, s1, s2, s3 = None):
		if (s1==[]):
			return s2 
		else:
			t = s1[0]
			if s3:
				if t in s3:
					return t
			s1 = s1[1:]
			if (t=="Top"):
				return self.bfs(s1, s2, s3) 
			l = self.edges[t]
			if isinstance(l, list):
				s1 = s1 + self.edges[t]
				s2 = s2 + self.edges[t]
			else:
				s1 = s1 + [self.edges[t]]
				s2 = s2 + [self.edges[t]]
			return self.bfs(s1, s2, s3)

	def lub_type(self, t1, t2):
		if not(isinstance(t1, str) and isinstance(t2, str)):
			raise TypeMismatchException(str(t1) + " and " + str(t2) + " are not comparable")
		b = self.bfs([t1], [t1])
		return self.bfs([t2], [t2], b)

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

	def visitBinOp(self, node: AST.BinOpNode):
		ltype = self.visit(node.left)
		rtype = self.visit(node.right)
		accepted = ["Int", "Float", "Neuron", "PolyExp", "ZonoExp", "Noise"]
		if(node.op == "+" or node.op == "-"):
			if(ltype in accepted and rtype in accepted):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
				else:
					raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "*"):
			if(self.isSubType(ltype, "Float") or self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "/"):
			if(self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "and" or node.op == "or"):
			if(ltype == "Bool" and rtype =="Bool"):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))
		else: #<=, >=, ==
			if(self.isSubType(ltype, "Float") and self.isSubType(rtype, "Float")):
				return "Bool"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		
	def visitUnOp(self, node: AST.UnOpNode):
		t = self.visit(node.expr)
		if(node.op == "-"):
			if(t == "Int" or t =="Float"):
				return t
			else:
				raise TypeMismatchException("- not defined on " + str(t))
		elif(node.op == "!"):
			if(t == "Bool"):
				return t
			else:
				raise TypeMismatchException("! not defined on " + str(t))
		else:
			assert False

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		exptype = self.visit(node.expr)
		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException(node.op + " requires list as first argument")
		functype = self.visit(node.func)
		if(self.isSubType(functype, ArrowType([exptype.base, exptype.base], "Bool"))):
			return ArrayType(exptype.base)
		else:
			raise TypeMismatchException(node.op + " requires function from " + str(exptype.base) + " to Bool")

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		exptype = self.visit(node.expr)
		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException(node.op + " requires float or int list as first argument")
		elif(not self.isSubType(exptype.base,"Float")):
			raise TypeMismatchException(node.op + " requires Float or Int list as first argument")
		return exptype.base

	def visitMaxOp(self, node: AST.MaxOpNode):
		exp1type = self.visit(node.expr1)
		exp2type = self.visit(node.expr2)
		if(not self.isSubType(exp1type, "Float")) or (not self.isSubType(exp2type, "Float")):
			raise TypeMismatchException(node.op + " is not defined on " + str(exp1type) + " and " + str(exp2type))
		return self.lub_type(exp1type, exp2type)


	def visitVar(self, node: AST.VarNode):
		# if(node.name == "curr_new"):
		# 	return "Neuron"
		if not node.name in self.Gamma.keys():
			raise UndefinedVarException(node.name + " is undefined")
		else:
			return self.Gamma[node.name]

	def visitInt(self, node: AST.ConstIntNode):
		return "Int"

	def visitFloat(self, node: AST.ConstFloatNode):
		return "Float"

	def visitBool(self, node: AST.ConstBoolNode):
		return "Bool"

	# def visitCurr(self, node: AST.CurrNode):
	# 	if(self.currdefined):
	# 		return "Neuron"
	# 	else:
	# 		raise UndefinedVarException("Curr is not defined")

	# def visitPrev(self, node: AST.PrevNode):
	# 	if(self.currdefined):
	# 		return ArrayType("Neuron")
	# 	else:
	# 		raise UndefinedVarException("Prev is not defined")

	def visitEpsilon(self, node: AST.EpsilonNode):
		return "Noise"

	def visitTernary(self, node: AST.TernaryNode):
		ctype = self.visit(node.cond)
		if(not self.isSubType(ctype, "Bool")):
			raise TypeMismatchException("Condition has to be boolean")

		ltype = self.visit(node.texpr)
		rtype = self.visit(node.fexpr)
		if(self.lub_type(ltype, rtype) != "Top"):
			return self.lub_type(ltype, rtype)
		else:
			raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not lub_type")

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		etype = self.visit(node.expr)
		if(etype == "Neuron"):
			if(node.metadata.name in self.metadata.keys()):
				return self.metadata[node.metadata.name]
			else:
				assert False
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			if(node.metadata.name in self.metadata.keys()):
				return ArrayType(self.metadata[node.metdata.name])
			else:
				assert False
		else:
			raise TypeMismatchException("To access metadata, expression must be Neuron or Neuron List")

	def visitGetElement(self, node: AST.GetElementNode):
		etype = self.visit(node.expr)
		if(not node.elem.name in self.Gamma.keys()):
			raise UndefinedVarException(node.elem.name + " is undefined")

		elemtype = self.Gamma[node.elem.name]
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
		if(not self.isSubType(prtype, ArrowType("Neuron", "Float"))):
			raise TypeMismatchException("Traverse priority function is wrong type")
		elif(not self.isSubType(sttype, ArrowType("Neuron", "Bool"))):
			raise TypeMismatchException("Traverse stopping function is wrong type")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Last argument to Traverse must be a function")
		
		if(not self.isSubType(ftype, ArrowType(["Neuron", "Float"], "PolyExp"))):
			raise TypeMismatchException("Last argument to traverse is the wrong type")
		else:
			right = ftype.tright
			self.visit(node.p)
			if(right == "Neuron"):
				return "PolyExp"
			else:
				return right

		


	def visitListOp(self, node: AST.ListOpNode):
		exptype = self.visit(node.expr)

		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException("Expression passed to listOp must be a list")

		base = exptype.base
		if(node.op == "sum" or node.op == "avg"):
			if(base == "Int" or base == "Float" or base == "PolyExp" or base == "ZonoExp"):
				return base
			elif (base == "Neuron"):
				return "PolyExp"
			elif(base == "Noise"):
				return "ZonoExp"
			else:
				raise TypeMismatchException(base + " list cannot be passed to sum or avg")
		else:
			return "Int"

	# def visitSub(self, node: AST.SubNode):
	# 	listtype = self.visit(node.listexpr)
	# 	if(isinstance(listtype, ArrayType)):
	# 		etype = self.visit(node.expr)
	# 		if(self.isSubType(etype, listtype.base)):
	# 			return listtype
	# 		else:
	# 			raise TypeMismatchException("Second argument of Sub should be of the same type as values in the list")
	# 	else:
	# 		raise TypeMismatchException("First argument of Sub should be a list")

	def visitMap(self, node: AST.MapNode):
		exptype = self.visit(node.expr)
		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Argument to Map must be a function")

		if(self.isSubType(exptype,"PolyExp")):
			if(not self.isSubType(ftype, ArrowType(["Neuron", "Float"], "PolyExp"))):
				raise TypeMismatchException("Function argument to Map is the wrong type")
			right = ftype.tright
			if(right == "Neuron"):
				return "PolyExp"
			else:
				return right
		elif(self.isSubType(exptype,"ZonoExp")):
			if(not self.isSubType(ftype, ArrowType(["Noise", "Float"], "ZonoExp"))):
				raise TypeMismatchException("Function argument to Map is the wrong type")
			right = ftype.tright
			if(right == "Noise"):
				return "ZonoExp"
			else:
				return right
		elif isinstance(exptype, ArrayType):
			if(self.isSubType(exptype.base, ftype.tleft) and isinstance(ftype.tright, str)):
				return ArrayType(ftype.tright)
		else:
			raise TypeMismatchException("Left of Map function must be a PolyExp, ZonoExp or a list")

	def visitDot(self, node: AST.DotNode):
		tleft = self.visit(node.left)
		tright = self.visit(node.right)

		if(not isinstance(tleft, ArrayType) or not isinstance(tright, ArrayType)):
			raise TypeMismatchException("Left and right of dot product have to be list")
		
		if(self.isSubType(tleft.base, "Float") or self.isSubType(tright.base, "Float")):
			if(self.lub_type(tleft.base, tright.base) != "Top"):
				return self.lub_type(tleft.base, tright.base)

	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		if(not name in self.Gamma.keys()):
			raise UndefinedVarException(name + " is not defined")

		ftype = self.Gamma[name]
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException(name + " is not a function")
		argstype = ftype.tleft
		exprstype = self.visit(node.arglist)
		i = 0
		while i < len(exprstype):
			if(i == len(argstype)):
				raise TypeMismatchException("Arguments of " + name + " are not the correct type(s)")
			if(not self.isSubType(exprstype[i], argstype[i])):
				raise TypeMismatchException("Arguments of " + name + " are not the correct type(s)")
			i += 1
		if(i == len(argstype)):
			return ftype.tright
		else:
			return ArrowType(argstype[i:], ftype.tright)

	def visitShapeDecl(self, node: AST.ShapeDeclNode):
		for (t, e) in node.elements.arglist:
			if(e.name in self.Gamma.keys()):
				raise DuplicateVarException(str(e.name()) + " is already defined")
			else:
				self.Gamma[e.name] = t.name

			self.shape.append((t.name,e.name))

		self.Gamma["curr_new"] = "Neuron"
		self.visit(node.p)
		del self.Gamma["curr_new"]

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):
		rettype = self.visit(node.exprlist)
		if(not self.isSubType(rettype, [x[0] for x in self.shape])):
			raise TypeMismatchException("Expression in transformer is different from shape declaration")


	def visitTransRetIf(self, node: AST.TransRetIfNode):
		ctype = self.visit(node.cond)
		if(not self.isSubType(ctype, "Bool")):
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
		if(tname in self.Gamma.keys()):
			raise DuplicateVarException(tname + " is already defined")

		self.Gamma["curr"] = "Neuron"
		self.Gamma["prev"] = ArrayType("Neuron")
		self.visit(node.oplist)
		del self.Gamma["curr"]
		del self.Gamma["prev"]
		self.Gamma[tname] = TransformerType()

	def visitPropTermBasic(self, node: AST.PropTermBasicNode):
		return self.visit(node.term)

	def visitPropTermIn(self, node: AST.PropTermInNode):
		ntype = self.visit(node.n)
		ztype = self.visit(node.z)
		if(ntype == "Neuron" and ztype == "ZonoExp"):
			return "Bool"
		else:
			raise TypeMismatchException("in operator not defined for " + str(ntype) + " " + str(ztype))

	def visitPropTermOp(self, node: AST.PropTermOpNode):
		left = self.visit(node.leftpt)
		right = self.visit(node.rightpt)
		if(self.lub_type(left, right) != "Top"):
			return self.lub_type(left, right)
		else:
			raise TypeMismatchException(str(left) + " and " + str(right) + " are not lub_type")
		

	def visitSingleProp(self, node: AST.SinglePropNode):
		left = self.visit(node.leftpt)
		right = self.visit(node.rightpt)
		if(self.lub_type(left, right) != "Top"):
			return "Bool"
		else:
			raise TypeMismatchException(str(left) + " and " + str(right) + " are not lub_type")

	def visitDoubleProp(self, node: AST.DoublePropNode):
		l = self.visit(node.leftprop)
		r = self.visit(node.rightprop)
		#Shouldn't need the rest of this function bc of parsing
		if(r == "Bool" and l == "Bool"):
			return "Bool"
		else:
			raise TypeMismatchException("Both properties should be boolean expressions")

	def visitFunc(self, node: AST.FuncNode):
		argstype = self.visit(node.decl.arglist)
		fname = node.decl.name.name
		if(fname in self.Gamma.keys()):
			raise DuplicateVarException(fname + " is already defined")
		newGamma = []
		oldvalues = {}
		for (t,e) in node.decl.arglist.arglist:
			if e.name in self.Gamma.keys():
				oldvalues[e.name] = self.Gamma[e.name]
			else:
				newGamma.append(e.name)

			self.Gamma[e.name] = t.name

		exprtype = self.visit(node.expr)
		for v in newGamma:
			del self.Gamma[v]

		for ov in oldvalues.keys():
			self.Gamma[ov] = oldvalues[ov]

		self.Gamma[fname] = ArrowType(argstype, exprtype)

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		prtype = self.visit(node.pfunc)
		sttype = self.visit(node.sfunc)
		if(not self.isSubType(prtype, ArrowType("Neuron", "Float"))):
			raise TypeMismatchException("Flow priority function is wrong type")
		elif(not self.isSubType(sttype, ArrowType("Neuron", "Bool"))):
			raise TypeMismatchException("Flow stopping function is wrong type")

		#Can, instead, call self.visit(node.trans) and check if the return value is a TransformerType()
		if not node.trans.name in self.Gamma.keys():
			raise UndefinedVarException(node.trans.name + " is undefined")
		elif (not self.isSubType(self.Gamma[node.trans.name], TransformerType())):
			raise TypeMismatchException(node.trans.name + " is not a Transformer")

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)