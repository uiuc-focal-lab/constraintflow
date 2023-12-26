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
		self.file.write("\n");

	def visitExprList(self, node: AST.ExprListNode):
		pass
			
	def visitBinOp(self, node: AST.BinOpNode):
		#Check if the expression is pw+b and replace with temp.copy()
		if(node.right == AST.GetMetadataNode(AST.VarNode("curr"), AST.MetadataNode("bias"))):
			if(node.left == AST.DotNode(AST.VarNode("prev"), AST.GetMetadataNode(AST.VarNode("curr"), AST.MetadataNode("weight")))):
				self.file.write("temp.copy()")
				return

		if(isinstance(node.right, AST.GetMetadataNode)):
			print(node.right.expr)
			print(node.right.metadata)

		self.visit(node.left)
		self.file.write(" " + node.op + " ")
		self.visit(node.right)
		
	def visitUnOp(self, node: AST.UnOpNode):
		self.file.write(node.op)
		self.visit(node.expr)

	def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
		pass

	def visitMaxOp(self, node: AST.MaxOpNode):
		pass

	def visitMaxOpList(self, node: AST.MaxOpListNode):
		pass

	def visitVar(self, node: AST.VarNode):
		self.file.write(node.name)

	def visitInt(self, node: AST.ConstIntNode):
		self.file.write(str(node.value))

	def visitFloat(self, node: AST.ConstFloatNode):
		self.file.write(str(node.value))

	def visitBool(self, node: AST.ConstBoolNode):
		self.file.write(str(node.value))

	def visitEpsilon(self, node: AST.EpsilonNode):
		self.file.write("SymExp()")

	def visitTernary(self, node: AST.TernaryNode):
		self.visit(node.texpr)
		self.file.write(" if ")
		self.visit(node.cond)
		self.file.write(" else ")
		self.visit(node.fexpr)


	def visitTraverse(self, node: AST.TraverseNode):
		pass


	def visitListOp(self, node: AST.ListOpNode):
		pass

	def visitMap(self, node: AST.MapNode):
		self.visit(node.expr)
		self.file.write(".map(abs_elem, ")
		if(isinstance(node.func, AST.VarNode)):
			self.file.write(node.func.name)
		else:
			#node.func is a AST.FuncCallNode representing a partially applied function
			pass
		self.file.write(")")

	def visitDot(self, node: AST.DotNode):
		pass
	
	def visitConcat(self, node):
		pass
	
	def visitFuncCall(self, node: AST.FuncCallNode):
		pass

	#Do we have to do a seperate analysis to know when to pass the weight and bias to defined functions?
	def visitGetMetadata(self, node: AST.GetMetadataNode):
		if(node.metadata.name == "layer"):
			self.visit(node.expr)
			self.file.write("[0]")
		elif(node.metadata.name == "weight"):
			self.file.write("w[")
			self.visit(node.expr)
			self.file.visit("]")
		elif(node.metadata.name == "bias"):
			self.file.write("b")

	def visitGetElement(self, node):
		self.file.write("abs_elem.get_elem(\'")
		self.visit(node.elem)
		self.file.write("\', ")
		self.visit(node.expr)
		self.file.write(")")

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
			self.file.write(self.indent + "def fc(self, abs_elem, prev, curr, w, b):\n")
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

			self.file.write(self.indent + "return ")
			if(len(self.shape_names)>0):
				self.file.write(self.shape_names[0]+"_new")
				for i in range(1, len(self.shape_names)):
					self.file.write(","+self.shape_names[i]+"_new")
			self.file.write("\n")
			self.indent = self.indent[:-1]

		elif node.op.op_name == "Relu":
			self.file.write(self.indent + "def relu(self, abs_elem, prev, curr):\n")
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
