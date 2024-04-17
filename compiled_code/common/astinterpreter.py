import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'verification/ast'))
import astVisitor
import astcf as AST

class AstInterpret(astVisitor.ASTVisitor):
	def __init__(self,filename):
		open(filename, "w").close()
		self.file = open(filename, "a")
		self.indent = ""
		self.file.write("from common.abs_elem import Abs_elem\n")
		self.file.write("from common.polyexp import PolyExp\n")
		self.file.write("from common.symexp import SymExp\n")
		self.file.write("from common.transformer import Transformer\n")
		self.file.write("import torch\n")
		self.file.write("import functools\n")
		self.file.write("\n");
		self.in_get = False

	def visitExprList(self, node: AST.ExprListNode):
		pass
			
	def visitBinOp(self, node: AST.BinOpNode):
		#Check if the expression is pw+b and replace with temp.copy()
		if(node.right == AST.GetMetadataNode(AST.VarNode("curr"), AST.MetadataNode("bias"))):
			if(node.left == AST.DotNode(AST.VarNode("prev"), AST.GetMetadataNode(AST.VarNode("curr"), AST.MetadataNode("weight")))):
				self.file.write("temp.copy()")
				return

		if(node.left.type == "PolyExp" or node.left.type == "ZonoExp" or node.left.type == "Neuron" or node.left.type == "Noise"): #Has to be add, mult, minus or division		
			self.file.write("(")
			self.visit(node.left)
			self.file.write(")")
			if(node.op == "+"):
				self.file.write(".add(")
			elif(node.op == "-"):
				self.file.write(".minus(")
			elif(node.op == "*"):
				self.file.write(".mult(")
			self.visit(node.right)
			self.file.write(")")
			return

		if(node.right.type == "PolyExp" or node.right.type == "ZonoExp" or node.right.type == "Neuron" or node.right.type == "Noise"): #Has to be add, mult, minus or division		
			self.file.write("(")
			self.visit(node.right)
			self.file.write(")")
			if(node.op == "+"):
				self.file.write(".add(")
			elif(node.op == "-"):
				self.file.write(".minus(")
			elif(node.op == "*"):
				self.file.write(".mult(")
			self.visit(node.left)
			self.file.write(")")
			return

		if(node.op == "*" or node.op == "/"):
			self.file.write("(")
			self.visit(node.left)
			self.file.write(")")
			self.file.write(" " + node.op + " ")
			self.file.write("(")
			self.visit(node.right)
			self.file.write(")")
		else:
			self.visit(node.left)
			self.file.write(" " + node.op + " ")
			self.visit(node.right)
		
	def visitUnOp(self, node: AST.UnOpNode):
		self.file.write(node.op)
		self.visit(node.expr)

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		#For max. Still have to implement argmin
		self.file.write("[i_argmax for i_argmax in ")
		self.visit(node.expr)
		self.file.write(" if ")
		self.visit(node.func)
		self.file.write("(i_argmax) == ")
		self.visit(node.func)
		self.file.write("(max(")
		self.visit(node.expr)
		self.file.write(", key=")
		self.visit(node.func)
		self.file.write("))]")


	def visitMaxOp(self, node: AST.MaxOpNode):
		self.file.write("max")
		self.visit(node.expr1)
		self.file.write(", ")
		self.visit(node.expr2)
		self.file.write(")")

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		self.file.write("max(")
		self.visit(node.expr)
		self.file.write(")")

	def visitVar(self, node: AST.VarNode):
		if(not self.in_get and node.type == "Neuron"):
			self.file.write("PolyExp(abs_elem.shapes).populate(0,"+node.name+")")
		else:
			self.file.write(node.name)

	def visitInt(self, node: AST.ConstIntNode):
		self.file.write(str(node.value))

	def visitFloat(self, node: AST.ConstFloatNode):
		self.file.write(str(node.value))

	def visitBool(self, node: AST.ConstBoolNode):
		self.file.write(str(node.value))

	def visitEpsilon(self, node: AST.EpsilonNode):
		self.file.write("SymExp().new_symbol().populate(coeff=1)")

	def visitTernary(self, node: AST.TernaryNode):
		self.visit(node.texpr)
		self.file.write(" if ")
		self.visit(node.cond)
		self.file.write(" else ")
		self.visit(node.fexpr)


	def visitTraverse(self, node: AST.TraverseNode):
		self.visit(node.expr)
		self.file.write(".traverse(abs_elem, neighbours, ")
		self.visit(node.stop)
		self.file.write(", ")
		self.visit(node.priority)
		self.file.write(", ")
		self.visit(node.func)
		self.file.write(")")


	def visitListOp(self, node: AST.ListOpNode):
		pass

	def visitMap(self, node: AST.MapNode):
		self.visit(node.expr)
		self.file.write(".map(abs_elem, neighbours, ")
		if(isinstance(node.func, AST.VarNode)):
			self.file.write(node.func.name)
		else:
			#node.func is a AST.FuncCallNode representing a partially applied function
			pass
		self.file.write(")")

	def visitDot(self, node: AST.DotNode):
		self.left_type = str(node.left.type).split(" ")[0]
		self.right_type = str(node.right.type).split(" ")[0]
		if(self.left_type == "PolyExp" or self.left_type == "ZonoExp"):
			in_list = True
			self.file.write("functools.reduce(lambda a_dot, b_dot: a_dot.add(b_dot),[i_dot[0].mult(i_dot[1]) for i_dot in zip(")
		elif(self.right_type == "PolyExp" or self.right_type == "ZonoExp"):
			in_list = True
			self.file.write("functools.reduce(lambda a_dot, b_dot: a_dot.add(b_dot),[i_dot[1].mult(i_dot[0]) for i_dot in zip(")
		else:
			in_list = False
			self.file.write("sum(i_dot[0] * i_dot[1] for i_dot in zip(")
		self.visit(node.left)
		self.file.write(",")
		self.visit(node.right)
		if(in_list):
			self.file.write(")])")
		else:
			self.file.write("))")
	
	def visitFuncCall(self, node: AST.FuncCallNode):
		self.in_get = True
		args = node.arglist.exprlist
		self.file.write(node.name.name)
		self.file.write("(")
		if(len(args) > 0):
			for arg in args:
				self.visit(arg)
				self.file.write(",")
		self.file.write("abs_elem, neighbours")
		self.file.write(")")
		self.in_get = False

	#Do we have to do a seperate analysis to know when to pass the weight and bias to defined functions?
	def visitGetMetadata(self, node: AST.GetMetadataNode):
		if(not isinstance(node.expr.type, str)):
			self.file.write("[")

			if(node.metadata.name == "layer"):
				self.file.write("i_metadata")
				self.file.write("[0]")
			elif(node.metadata.name == "weight"):
				# self.file.write("w[")
				# self.file.write("i_metadata")
				# self.file.write("]")
				self.file.write("w")
			elif(node.metadata.name == "bias"):
				self.file.write("b")

			self.file.write(" for i_metadata in ")
			self.in_get = True
			self.visit(node.expr)
			self.in_get = False
			self.file.write("]")
		else:
			self.in_get = True
			if(node.metadata.name == "layer"):
				self.visit(node.expr)
				self.file.write("[0]")
			elif(node.metadata.name == "weight"):
				self.file.write("w")
				# self.file.write("w[")
				# self.visit(node.expr)
				# self.file.write("]")
			elif(node.metadata.name == "bias"):
				self.file.write("b")
			self.in_get = False

	def visitGetElement(self, node):
		if(not isinstance(node.expr.type, str)):
			self.in_get = True
			self.file.write("[")

			self.file.write("abs_elem.get_elem(\'")
			self.visit(node.elem)
			self.file.write("\', ")
			self.file.write("i_element")
			self.file.write(")")

			self.file.write(" for i_element in ")
			self.visit(node.expr)
			self.in_get = False
			self.file.write("]")
		else:
			self.in_get = True
			self.file.write("abs_elem.get_elem(\'")
			self.visit(node.elem)
			self.file.write("\', ")
			self.visit(node.expr)
			self.file.write(")")
			self.in_get = False

	def visitShapeDecl(self, node: AST.ShapeDeclNode):
		shape_names = []
		shape_types = []
		for (t, e) in node.elements.arglist:
			shape_names.append(e.name)
			shape_types.append(t)

		self.shape_names = shape_names
		self.shape_types = shape_types

	def visitTransRetBasic(self, node: AST.TransRetBasicNode):
		retlist = node.exprlist.exprlist
		for i in range(len(self.shape_names)):
			self.file.write(self.indent + self.shape_names[i]+"_new = ")
			self.visit(retlist[i])
			self.file.write("\n")

	def visitTransRetIf(self, node: AST.TransRetIfNode):
		self.file.write(self.indent + "if ")
		self.visit(node.cond)
		self.file.write(":\n")
		self.indent = self.indent + "\t"
		self.visit(node.tret)
		self.indent = self.indent[:-1]
		self.file.write(self.indent + "else:\n")
		self.indent = self.indent + "\t"
		self.visit(node.fret)
		self.indent = self.indent[:-1]

	def visitOpStmt(self, node: AST.OpStmtNode):
		if node.op.op_name == 'Affine':
			self.file.write(self.indent + "def fc(self, abs_elem, neighbours, prev, curr, w, b):\n")
			self.indent = self.indent + "\t"
			self.file.write(self.indent + "temp = PolyExp(abs_elem.shapes)\n")
			self.file.write(self.indent + "temp.populate(b, prev, w)\n")
			self.visit(node.ret)

			#Convert any polyexps and symexps to constants
			self.file.write("\n")
			for (stype, sname) in zip(self.shape_types, self.shape_names):
				sname = sname + "_new"
				if(isinstance(stype, AST.BaseTypeNode)):
					if(stype.name == "Float" or stype.name == "Int"):
						self.file.write(self.indent + "if isinstance("+sname+",PolyExp) or isinstance("+sname+",SymExp):\n")
						self.file.write(self.indent + "\t" + sname + " = " + sname + ".get_const()\n")

			#convert any constants to polyexps or symexps
			self.file.write("\n")
			for (stype, sname) in zip(self.shape_types, self.shape_names):
				sname = sname + "_new"
				if(isinstance(stype, AST.BaseTypeNode)):
					if(stype.name == "PolyExp"):
						self.file.write(self.indent + "if isinstance("+sname+",float) or isinstance("+sname+",int):\n")
						self.file.write(self.indent + "\t" + sname + " = " + "PolyExp(abs_elem.shapes).populate(" + sname + ", [])\n")
					if(stype.name == "ZonoExp"):
						self.file.write(self.indent + "if isinstance("+sname+",float) or isinstance("+sname+",int):\n")
						self.file.write(self.indent + "\t" + sname + " = " + "SymExp().populate("+sname+")\n")

			self.file.write(self.indent + "return ")
			if(len(self.shape_names)>0):
				self.file.write(self.shape_names[0]+"_new")
				for i in range(1, len(self.shape_names)):
					self.file.write(","+self.shape_names[i]+"_new")
			self.file.write("\n")
			self.indent = self.indent[:-1]

		elif node.op.op_name == "Relu":
			self.file.write(self.indent + "def relu(self, abs_elem, neighbours, prev, curr):\n")
			self.indent = self.indent + "\t"
			self.visit(node.ret)

			#Convert any polyexps and symexps to constants
			self.file.write("\n")
			for (stype, sname) in zip(self.shape_types, self.shape_names):
				sname = sname + "_new"
				if(isinstance(stype, AST.BaseTypeNode)):
					if(stype.name == "Float" or stype.name == "Int"):
						self.file.write(self.indent + "if isinstance("+sname+",PolyExp) or isinstance("+sname+",SymExp):\n")
						self.file.write(self.indent + "\t" + sname + " = " + sname + ".get_const()\n")

			#convert any constants to polyexps or symexps
			self.file.write("\n")
			for (stype, sname) in zip(self.shape_types, self.shape_names):
				sname = sname + "_new"
				if(isinstance(stype, AST.BaseTypeNode)):
					if(stype.name == "PolyExp"):
						self.file.write(self.indent + "if isinstance("+sname+",float) or isinstance("+sname+",int):\n")
						self.file.write(self.indent + "\t" + sname + " = " + "PolyExp(abs_elem.shapes).populate(" + sname + ", [])\n")
					if(stype.name == "ZonoExp"):
						self.file.write(self.indent + "if isinstance("+sname+",float) or isinstance("+sname+",int):\n")
						self.file.write(self.indent + "\t" + sname + " = " + "SymExp().populate("+sname+")\n")

			self.file.write(self.indent + "return ")
			if(len(self.shape_names)>0):
				self.file.write(self.shape_names[0]+"_new")
				for i in range(1, len(self.shape_names)):
					self.file.write(","+self.shape_names[i]+"_new")
			self.file.write("\n")
			self.indent = self.indent[:-1]
		
	def visitOpList(self, node: AST.OpListNode):
		if(len(node.olist) > 0):
			self.visit(node.olist[0])
			for i in range(1,len(node.olist)):
				self.file.write("\n")
				self.visit(node.olist[i])

	def visitTransformer(self, node: AST.TransformerNode):
		self.file.write("\n")
		self.file.write("class Cflow" + node.name.name + "(Transformer):\n")
		self.indent = "\t"
		self.visit(node.oplist)
		self.indent = ""

	def visitFunc(self, node: AST.FuncNode):
		self.file.write("\n")
		self.file.write("def " + node.decl.name.name + "(")
		if(len(node.decl.arglist.arglist) > 0):
			self.file.write(node.decl.arglist.arglist[0][1].name)
			for i in range(1,len(node.decl.arglist.arglist)):
				self.file.write(", ")
				self.file.write(node.decl.arglist.arglist[i][1].name)
		self.file.write(", abs_elem, neighbours")
		self.file.write("):\n")
		self.indent = "\t"
		self.file.write(self.indent + "return ")
		self.visit(node.expr)
		self.file.write("\n")
		self.indent = ""
				

	def visitSeq(self, node: AST.SeqNode):
		self.visit(node.stmt1)
		self.visit(node.stmt2)

	def visitFlow(self, node):
		pass

	def visitProg(self, node: AST.ProgramNode):
		self.visit(node.shape)
		self.visit(node.stmt)