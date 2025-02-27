from . import astVisitor
from . import astcf as AST

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
	
	def __eq__(self, obj):
		if not isinstance(obj, ArrayType):
			return False
		return self.base == obj.base

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
		self.edges = {"Int": "Float", "Float" : ["PolyExp", "SymExp"], "Neuron":"PolyExp", "Noise":"SymExp", "SymExp":"Top", "PolyExp":"Top", "Bool":"Ct", "Ct":"Top", "Bot":["Bool", "Int", "Noise", "Neuron"]}
		self.metadata = {"layer":"Int", "weight": ArrayType("Float"), "bias": "Float", "serial":"Int", "local_serial": "Int", "equations":ArrayType("PolyExp")}	
		self.hasE = False

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
		elif(isinstance(t1, ArrowType)):
			return False
		elif(isinstance(t1, ArrayType)):
			if(isinstance(t2, ArrayType)):
				return self.isSubType(t1.base, t2.base)
		elif(isinstance(t2, ArrayType)):
			return False
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
		node.type = node.name
		return node.name

	def visitArrayType(self, node: AST.ArrayTypeNode):
		node.type = ArrayType(node.base) 
		return ArrayType(node.base)

	def visitArgList(self, node: AST.ArgListNode):
		l = []
		for t in node.arglist:
			if isinstance(t[0], AST.ArrayTypeNode):
				l.append(ArrayType(t[0].base.name))
			else:
				l.append(t[0].name)
		node.type = l
		return l 
		# l = [t[0].name for t in node.arglist]
		# return l 
		# return [t[0].name for t in node.arglist]

	def visitExprList(self, node: AST.ExprListNode):
		listtype = []
		for e in node.exprlist:
			listtype.append(self.visit(e))
		node.type = listtype
		return listtype

	def get_binop(self, op, ltype, rtype):
		accepted = ["Int", "Float", "Neuron", "PolyExp", "SymExp", "Noise"]
		if (isinstance(ltype, ArrayType) and isinstance(rtype, ArrayType)):
			raise Exception('Not possible')
		if(op == "+" or op == "-"):
			# if((self.isSubType(ltype, "SymExp") and self.isSubType(rtype, "PolyExp")) or (self.isSubType(rtype, "SymExp") and self.isSubType(ltype, "PolyExp"))):
			# 	return 'PolyExp'
			if(ltype in accepted and rtype in accepted):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
				if((self.isSubType(ltype, "SymExp") and self.isSubType(rtype, "PolyExp")) or (self.isSubType(rtype, "SymExp") and self.isSubType(ltype, "PolyExp"))):
					return 'PolyExp'
				else:
					raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")
			else:
				raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(op == "*"):
			if(self.isSubType(ltype, "Float") or self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
			if((self.isSubType(ltype, "SymExp") and self.isSubType(rtype, "PolyExp")) or (self.isSubType(rtype, "SymExp") and self.isSubType(ltype, "PolyExp"))):
				return 'PolyExp'
			raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(op == "/"):
			if(self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					return self.lub_type(ltype, rtype)
			raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(op == "and" or op == "or"):
			if(self.isSubType(ltype, "Ct") and self.isSubType(rtype, "Ct")):
				return self.lub_type(ltype, rtype)
			else:
				raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))
		elif(op == "In"):
			if(self.isSubType(ltype, "PolyExp") and self.isSubType(rtype, "SymExp")):
				return "Ct"
			else:
				raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))
		else: #<=, >=, ==
			if(self.isSubType(ltype, "Float") and self.isSubType(rtype, "Float")):
				return "Bool"
			if(self.isSubType(ltype, "PolyExp") and self.isSubType(rtype, "PolyExp")):
				return "Ct"
			else:
				raise TypeMismatchException(op + " is not defined on " + str(ltype) + " and " + str(rtype))

	def visitBinOp(self, node: AST.BinOpNode):
		ltype = self.visit(node.left)
		rtype = self.visit(node.right)
		if (isinstance(ltype, ArrayType) and isinstance(rtype, ArrayType)):
			ttype = self.get_binop(node.op, ltype.base, rtype.base)
			node.type = ArrayType(ttype)
			return ArrayType(ttype)
		node.type = self.get_binop(node.op, ltype, rtype) 
		return self.get_binop(node.op, ltype, rtype)
		accepted = ["Int", "Float", "Neuron", "PolyExp", "SymExp", "Noise"]
		
		if(node.op == "+" or node.op == "-"):
			if(ltype in accepted and rtype in accepted):
				if(self.lub_type(ltype, rtype) != "Top"):
					node.type = self.lub_type(ltype, rtype) 
					return self.lub_type(ltype, rtype)
				else:
					raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not comparable")
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "*"):
			if(self.isSubType(ltype, "Float") or self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					node.type = self.lub_type(ltype, rtype)
					return self.lub_type(ltype, rtype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "/"):
			if(self.isSubType(rtype, "Float")):
				if(self.lub_type(ltype, rtype) != "Top"):
					node.type = self.lub_type(ltype, rtype)
					return self.lub_type(ltype, rtype)
			raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		elif(node.op == "and" or node.op == "or"):
			if(self.isSubType(ltype, "Ct") and self.isSubType(rtype, "Ct")):
				node.type = self.lub_type(ltype, rtype)
				return self.lub_type(ltype, rtype)
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))
		elif(node.op == "In"):
			if(self.isSubType(ltype, "PolyExp") and self.isSubType(rtype, "SymExp")):
				node.type = "Ct"
				return "Ct"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))
		else: #<=, >=, ==
			if(self.isSubType(ltype, "Float") and self.isSubType(rtype, "Float")):
				node.type = "Bool"
				return "Bool"
			if(self.isSubType(ltype, "PolyExp") and self.isSubType(rtype, "PolyExp")):
				node.type = "Ct"
				return "Ct"
			else:
				raise TypeMismatchException(node.op + " is not defined on " + str(ltype) + " and " + str(rtype))

		
	def visitUnOp(self, node: AST.UnOpNode):
		t = self.visit(node.expr)
		if(node.op == "-"):
			if(t == "Int" or t =="Float" or t == "PolyExp" or t =="SymExp"):
				node.type = t
				return t
			if(t == "Neuron"):
				node.type = "PolyExp"
				return "PolyExp"
			if(t == "Noise"):
				node.type = "SymExp"
				return "SymExp"
			else:
				raise TypeMismatchException("- not defined on " + str(t))
		elif(node.op == "!"):
			if self.isSubType(t, "Ct"):
				node.type = t
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
			node.type = ArrayType(exptype.base)
			return ArrayType(exptype.base)
		else:
			raise TypeMismatchException(node.op + " requires function from " + str(exptype.base) + " to Bool")

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		exptype = self.visit(node.expr)
		if isinstance(exptype, ArrayType):
			if self.isSubType(exptype.base,"Float"):
				node.type = exptype.base
				return exptype.base
		if isinstance(exptype, list):
			if len(exptype)==0:
				raise Exception('list empty')
			else:
				type = exptype[0]
				flag = False 
				if type == 'Float' or type == 'Int':
					flag = True 
				if isinstance(type, ArrayType):
					if type.base == 'Float' or type.base == 'Int':
						flag = True 
				if not flag:
					raise Exception('each element must be float or int')
			for e in exptype:
				if e != type:
					raise Exception('all elements of the list must be of the same type')
			node.type = type
			return type
		else:
			print(exptype)
			raise Exception('the argument to min or max must be a list')

	def visitMaxOp(self, node: AST.MaxOpNode):
		exp1type = self.visit(node.expr1)
		exp2type = self.visit(node.expr2)
		if(not self.isSubType(exp1type, "Float")) or (not self.isSubType(exp2type, "Float")):
			raise TypeMismatchException(node.op + " is not defined on " + str(exp1type) + " and " + str(exp2type))
		node.type = self.lub_type(exp1type, exp2type)
		return self.lub_type(exp1type, exp2type)


	def visitVar(self, node: AST.VarNode):
		# if(node.name == "curr_new"):
		# 	return "Neuron"
		if not node.name in self.Gamma.keys():
			raise UndefinedVarException(node.name + " is undefined")
		else:
			node.type = self.Gamma[node.name]
			return self.Gamma[node.name]

	def visitInt(self, node: AST.ConstIntNode):
		node.type = "Int"
		return "Int"

	def visitFloat(self, node: AST.ConstFloatNode):
		node.type = "Float"
		return "Float"

	def visitBool(self, node: AST.ConstBoolNode):
		node.type = "Bool"
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
		self.hasE = True
		node.type = "Noise"
		return "Noise"

	def visitTernary(self, node: AST.TernaryNode):
		ctype = self.visit(node.cond)
		if(not self.isSubType(ctype, "Bool")):
			raise TypeMismatchException("Condition has to be boolean")

		ltype = self.visit(node.texpr)
		rtype = self.visit(node.fexpr)
		if(self.lub_type(ltype, rtype) != "Top"):
			node.type = self.lub_type(ltype, rtype)
			return self.lub_type(ltype, rtype)
		else:
			raise TypeMismatchException(str(ltype) + " and " + str(rtype) + " are not lub_type")

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		etype = self.visit(node.expr)
		if(etype == "Neuron"):
			if(node.metadata.name in self.metadata.keys()):
				node.type = self.metadata[node.metadata.name] 
				return self.metadata[node.metadata.name]
			else:
				assert False
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			if(node.metadata.name in self.metadata.keys()):
				node.type = ArrayType(self.metadata[node.metdata.name])
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
			node.type = elemtype
			return elemtype
		if(etype == "Neuron List"):
			node.type = ArrayType(elemtype) 
			return ArrayType(elemtype)
		elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
			node.type = ArrayType(elemtype)
			return ArrayType(elemtype)
		else:
			raise TypeMismatchException("To access shape variables, expression must be Neuron or Neuron List")
		
	def visitGetElementAtIndex(self, node: AST.GetElementNode):
		etype = self.visit(node.expr)
		if isinstance(etype, ArrayType):
			node.type = etype.base
			return etype.base
		else:
			raise TypeMismatchException("To access element at an index, expression must be a list")
		# if(not node.elem.name in self.Gamma.keys()):
		# 	raise UndefinedVarException(node.elem.name + " is undefined")

		# elemtype = self.Gamma[node.elem.name]
		# if(not (elemtype, node.elem.name) in self.shape):
		# 	raise TypeMismatchException(node.elem.name + " is not part of the shape")

		# if(etype == "Neuron"):
		# 	return elemtype
		# elif(isinstance(etype, ArrayType) and etype.base == "Neuron"):
		# 	return ArrayType(elemtype)
		# else:
		# 	raise TypeMismatchException("To access shape variables, expression must be Neuron or Neuron List")


	def visitTraverse(self, node: AST.TraverseNode):
		exprtype = self.visit(node.expr)
		if(not exprtype == "PolyExp"):
			raise TypeMismatchException("Traverse must be called on PolyExp")

		prtype = self.visit(node.priority)
		sttype = self.visit(node.stop)
		if(not self.isSubType(prtype, ArrowType("Neuron", "Float"))):
			if(not self.isSubType(prtype, ArrowType(["Neuron", "Float"], "Float"))):
				raise TypeMismatchException("Traverse priority function is wrong type")
		elif(not self.isSubType(sttype, ArrowType("Neuron", "Bool"))):
			if(not self.isSubType(sttype, ArrowType(["Neuron", "Float"], "Bool"))):
				raise TypeMismatchException("Traverse stopping function is wrong type")

		ftype = self.visit(node.func)
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException("Last argument to Traverse must be a function")
		
		if(not self.isSubType(ftype, ArrowType(["Neuron", "Float"], "PolyExp"))):
			raise TypeMismatchException("Last argument to traverse is the wrong type")
		else:
			right = ftype.tright
			prop = self.visit(node.p)
			if self.isSubType(prop, "Ct"):
				if(right == "Neuron"):
					node.type = "PolyExp"
					return "PolyExp"
				else:
					node.type = right
					return right
			raise TypeMismatchException("Traverse property is the wrong type") 

		


	def visitListOp(self, node: AST.ListOpNode):
		exptype = self.visit(node.expr)
		if(not isinstance(exptype, ArrayType)):
			raise TypeMismatchException("Expression passed to listOp must be a list")

		base = exptype.base
		if(node.op == "sum" or node.op == "avg"):
			if(base == "Int" or base == "Float" or base == "PolyExp" or base == "SymExp"):
				node.type = base
				return base
			elif (base == "Neuron"):
				node.type = "PolyExp"
				return "PolyExp"
			elif(base == "Noise"):
				node.type = "SymExp"
				return "SymExp"
			else:
				raise TypeMismatchException(base + " list cannot be passed to sum or avg")
		else:
			node.type = "Int"
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
				node.type = "PolyExp"
				return "PolyExp"
			else:
				node.type = right
				return right
		elif(self.isSubType(exptype,"SymExp")):
			if(not self.isSubType(ftype, ArrowType(["Noise", "Float"], "SymExp"))):
				raise TypeMismatchException("Function argument to Map is the wrong type")
			right = ftype.tright
			if(right == "Noise"):
				node.type = "SymExp"
				return "SymExp"
			else:
				node.type = right
				return right
		else:
			raise TypeMismatchException("Left of Map function must be a PolyExp, SymExp or a list")

	def visitMapList(self, node: AST.MapNode):
		exptype = self.visit(node.expr)
		ftype = self.visit(node.func)
		if isinstance(exptype, ArrayType):
			if(self.isSubType(exptype.base, ftype.tleft) and isinstance(ftype.tright, str)):
				node.type = ArrayType(ftype.tright) 
				return ArrayType(ftype.tright)
		else:
			raise TypeMismatchException("Map list type mismatch")

	def visitDot(self, node: AST.DotNode):
		tleft = self.visit(node.left)
		tright = self.visit(node.right)

		if(not isinstance(tleft, ArrayType) or not isinstance(tright, ArrayType)):
			raise TypeMismatchException("Left and right of dot product have to be list")
		
		if(self.isSubType(tleft.base, "Float") or self.isSubType(tright.base, "Float")):
			if(self.lub_type(tleft.base, tright.base) != "Top"):
				node.type = self.lub_type(tleft.base, tright.base) 
				return self.lub_type(tleft.base, tright.base)

	def visitConcat(self, node: AST.DotNode):
		tleft = self.visit(node.left)
		tright = self.visit(node.right)

		if(not isinstance(tleft, ArrayType) or not isinstance(tright, ArrayType)):
			raise TypeMismatchException("Left and right of dot product have to be list")
		
		if(self.lub_type(tleft.base, tright.base) != "Top"):
			node.type = self.lub_type(tleft.base, tright.base)
			return self.lub_type(tleft.base, tright.base)

	def visitLp(self, node: AST.LpNode):
		tleft = self.visit(node.expr)
		tright = self.visit(node.constraints)

		if(not self.isSubType(tleft, "PolyExp") or not self.isSubType(tright, ArrayType("Ct"))):
			raise TypeMismatchException("Exp and constraint of LP are incorrect")
		
		node.type = "Float"
		return "Float" 
		
	def visitFuncCall(self, node: AST.FuncCallNode):
		name = node.name.name
		if(not name in self.Gamma.keys()):
			raise UndefinedVarException(name + " is not defined")

		ftype = self.Gamma[name]
		if(not isinstance(ftype, ArrowType)):
			raise TypeMismatchException(name + " is not a function")
		argstype = ftype.tleft
		if not isinstance(argstype, list):
			argstype = [argstype]
		exprstype = self.visit(node.arglist)
		i = 0
		while i < len(exprstype):
			if(i == len(argstype)):
				raise TypeMismatchException("Arguments of " + name + " are not the correct type(s)")
			if(not self.isSubType(exprstype[i], argstype[i])):
				print(i, exprstype[i], argstype[i])
				raise TypeMismatchException("Arguments of " + name + " are not the correct type(s)")
			i += 1
		if(i == len(argstype)):
			node.type = ftype.tright 
			return ftype.tright
		else:
			node.type = ArrowType(argstype[i:], ftype.tright)
			return ArrowType(argstype[i:], ftype.tright)

	def visitShapeDecl(self, node: AST.ShapeDeclNode):
		for (t, e) in node.elements.arglist:
			if(e.name in self.Gamma.keys()):
				raise DuplicateVarException(str(e.name()) + " is already defined")
			else:
				self.Gamma[e.name] = t.name

			self.shape.append((t.name,e.name))

		self.Gamma["curr_new"] = "Neuron"
		self.Gamma["curr"] = "Neuron"
		self.Gamma["prev"] = "Neuron"
		prop = self.visit(node.p)
		if not(self.isSubType(prop, "Ct")):
			if not(isinstance(prop, list)):
				raise TypeMismatchException("Shape property not correct type")
			for t in prop:
				if not self.isSubType(t, "Ct"):
					raise TypeMismatchException("Shape property not correct type")
		del self.Gamma["curr_new"]
		del self.Gamma["curr"]
		del self.Gamma["prev"]

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
		self.Gamma['curr'] = 'Neuron'
		if node.op.op_name == 'Affine':
			self.Gamma['prev'] = ArrayType('Neuron')
		elif node.op.op_name == 'Relu':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'Relu6':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'Abs':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_Abs':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'HardTanh':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'HardSigmoid':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_HardSigmoid':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_HardTanh':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_HardSwish':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'HardSwish':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_Relu':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_Relu6':
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'rev_Affine':
			self.Gamma['curr_list'] = ArrayType('Neuron')
			self.Gamma['prev'] = 'Neuron'
		elif node.op.op_name == 'Maxpool' or node.op.op_name == 'Minpool' or node.op.op_name == 'Avgpool':
			self.Gamma['prev'] = ArrayType('Neuron')
		elif node.op.op_name == 'rev_Maxpool':
			self.Gamma['prev'] = ArrayType('Neuron')
		elif node.op.op_name == 'Neuron_mult' or node.op.op_name == 'Neuron_add' or node.op.op_name == 'Neuron_max' or node.op.op_name == 'Neuron_min':
			self.Gamma['prev_0'] = 'Neuron'
			self.Gamma['prev_1'] = 'Neuron'
		elif node.op.op_name == 'rev_Neuron_mult' or node.op.op_name == 'rev_Neuron_add' or node.op.op_name == 'rev_Neuron_max' or node.op.op_name == 'rev_Neuron_min':
			self.Gamma['prev_0'] = 'Neuron'
			self.Gamma['prev_1'] = 'Neuron'
		elif node.op.op_name == 'Neuron_list_mult':
			self.Gamma['prev_0'] = ArrayType('Neuron')
			self.Gamma['prev_1'] = ArrayType('Neuron')
		# dshg
		self.visit(node.ret)
		if("curr_list" in self.Gamma):
			del self.Gamma["curr_list"]
		if("curr" in self.Gamma):
			del self.Gamma["curr"]
		if("prev" in self.Gamma):
			del self.Gamma["prev"]
		if("prev_0" in self.Gamma):
			del self.Gamma["prev_0"]
		if("prev_1" in self.Gamma):
			del self.Gamma["prev_1"]
		
	def visitOpList(self, node: AST.OpListNode):
		node.opsE = []
		for s in node.olist:
			self.visit(s)
			node.opsE.append(self.hasE)
			self.hasE = False

	def visitTransformer(self, node: AST.TransformerNode):
		tname = node.name.name 
		if(tname in self.Gamma.keys()):
			raise DuplicateVarException(tname + " is already defined")

		# for expr in node.arglist:
		# 	if(expr == "curr"):
		# 		self.Gamma["curr"] = "Neuron"
		# 	elif(expr == "prev"):
		# 		self.Gamma["prev"] = ArrayType("Neuron")
		# 	elif(expr == "curr_list"):
		# 		self.Gamma["curr_list"] = ArrayType("Neuron")
		# 	else:
		# 		raise TypeMismatchException("Arguments to transformer have to be prev, curr or curr_list")	

		self.visit(node.oplist)
		# if("curr_list" in self.Gamma):
		# 	del self.Gamma["curr_list"]
		# if("curr" in self.Gamma):
		# 	del self.Gamma["curr"]
		# if("prev" in self.Gamma):
		# 	del self.Gamma["prev"]
		self.Gamma[tname] = TransformerType()

	# def visitPropTermBasic(self, node: AST.PropTermBasicNode):
	# 	return self.visit(node.term)

	# def visitPropTermOp(self, node: AST.PropTermOpNode):
	# 	if(node.op == "In"):
	# 		ntype = self.visit(node.n)
	# 		ztype = self.visit(node.z)
	# 		if(ntype == "Neuron" and ztype == "SymExp"):
	# 			return "Bool"
	# 		else:
	# 			raise TypeMismatchException("in operator not defined for " + str(ntype) + " " + str(ztype))
	# 	else:

	# 		left = self.visit(node.leftpt)
	# 		right = self.visit(node.rightpt)
	# 		if(self.lub_type(left, right) != "Top"):
	# 			return self.lub_type(left, right)
	# 		else:
	# 			raise TypeMismatchException(str(left) + " and " + str(right) + " are not lub_type")
			

	# def visitSingleProp(self, node: AST.SinglePropNode):
	# 	if(node.op == "in"):
	# 		ntype = self.visit(node.leftpt)
	# 		ztype = self.visit(node.rightpt)
	# 		if(ntype == "Neuron" and ztype == "SymExp"):
	# 			return "Bool"
	# 		else:
	# 			raise TypeMismatchException("in operator not defined for " + str(ntype) + " " + str(ztype))
	# 	else:
	# 		left = self.visit(node.leftpt)
	# 		right = self.visit(node.rightpt)
	# 		if(self.lub_type(left, right) != "Top"):
	# 			return "Bool"
	# 		else:
	# 			print(node.op)
	# 			raise TypeMismatchException(str(left) + " and " + str(right) + " are not lub_type")

	# def visitDoubleProp(self, node: AST.DoublePropNode):
	# 	l = self.visit(node.leftprop)
	# 	r = self.visit(node.rightprop)
	# 	#Shouldn't need the rest of this function bc of parsing
	# 	if(r == "Bool" and l == "Bool"):
	# 		return "Bool"
	# 	else:
	# 		raise TypeMismatchException("Both properties should be boolean expressions")

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
		node.type = ArrowType(argstype, exprtype)

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node: AST.FlowNode):
		prtype = self.visit(node.pfunc)
		sttype = self.visit(node.sfunc)
		if(not self.isSubType(prtype, ArrowType("Neuron", "Float"))):
			if(not self.isSubType(prtype, ArrowType(["Neuron", "Float"], "Float"))):
				raise TypeMismatchException("Flow priority function is wrong type")
		elif(not self.isSubType(sttype, ArrowType("Neuron", "Bool"))):
			if(not self.isSubType(sttype, ArrowType(["Neuron", "Float"], "Bool"))):
				raise TypeMismatchException("Flow stopping function is wrong type")

		#Can, instead, call self.visit(node.trans) and check if the return value is a TransformerType()
		if not node.trans.name in self.Gamma.keys():
			raise UndefinedVarException(node.trans.name + " is undefined")
		elif (not self.isSubType(self.Gamma[node.trans.name], TransformerType())):
			raise TypeMismatchException(node.trans.name + " is not a Transformer")

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)
