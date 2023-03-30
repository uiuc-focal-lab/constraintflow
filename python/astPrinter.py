import astVisitor
import ast as AST

class ASTPrinter(astVisitor.ASTVisitor):

	def visitExprList(self, node: AST.ExprListNode):
		for e in node.exprlist:
			self.visit(e)
	
	def visitBinOp(self, node: AST.BinOpNode):
		print("Op: " + node.op)
		print("Left:")
		self.visit(node.left)
		print("Right:")
		self.visit(node.right)
		
	def visitUnOp(self, node: AST.UnOpNode):
		print("Op: " + node.op)
		self.visit(node.expr)

	def visitNlistOp(self, node: AST.NlistOpNode):
		print("Op: " + node.op)
		self.visit(node.expr)
		if(node.elem):
			self.visit(node.elem)

	def visitVar(self, node: AST.VarNode):
		print(node.name)

	def visitNeuron(self, node: AST.NeuronNode):
		print("Neuron")

	def visitInt(self, node: AST.ConstIntNode):
		print(node.value)

	def visitFloat(self, node: AST.ConstFloatNode):
		print(node.value)

	def visitBool(self, node: AST.ConstBoolNode):
		print(node.value)

	def visitCurr(self, node: AST.CurrNode):
		print("Curr")

	def visitPrev(self, node: AST.PrevNode):
		print("Prev")

	def visitEpsilon(self, node: AST.EpsilonNode):
		print("Epsilon")
	
	def visitTernary(self, node: AST.TernaryNode):
		print("Ternary Operator:")
		print("Condition:")
		self.visit(node.cond)
		print("True Expression:")
		self.visit(node.texpr)
		print("False Expression:")
		self.visit(node.fexpr)

	def visitTraverse(self, node: AST.TraverseNode):
		print("Traverse:")
		print("Expression:")
		self.visit(node.expr)
		print("Priority Function:")
		self.visit(node.priority)
		print("Stop Function:")
		self.visit(node.stop)
		print("Function:")
		self.visit(node.func)


	def visitSum(self, node: AST.SumNode):
		print("Sum:")
		self.visit(node.expr)

	# def visitSub(self, node: AST.SubNode):
	# 	print("Sub:")
	# 	self.visit(node.listexpr)
	# 	self.visit(node.expr)
	
	def visitMap(self, node: AST.MapNode):
		print("Map")
		print("Expression:")
		self.visit(node.expr)
		print("Function:")
		self.visit(node.func)

	def visitDot(self, node: AST.DotNode):
		print("Dot:")
		print("Left:")
		self.visit(node.left)
		print("Right:")
		self.visit(node.right)

	
	def visitFuncCall(self, node: AST.FuncCallNode):
		print("Function call: " + node.name.name)
		print("Arguments:")
		self.visit(F[name].expr)


	def visitSeq(self, node: AST.SeqNode):
		print("Sequence:")
		print("Stmt 1:")
		self.visit(node.stmt1)
		print("Stmt 2:")
		self.visit(node.stmt2)

	def visitFlow(self, node):
		print("Flow Statement")

	def visitProg(self, node: AST.ProgramNode):
		print("Program")
		self.visit(node.shape)
		self.visit(node.stmt)

	def visitGetMetadata(self, node: AST.GetMetadataNode):
		print("Get Metadata:")
		self.visit(node.expr)
		self.visit(node.metadata)

	def visitGetElement(self, node):
		print("Get element:")
		self.visit(node.expr)
		print(node.elem.name)		

	def visitPropTermBasic(self, node: AST.PropTermBasicNode):
		self.visit(node.term)

	def visitPropTermIn(self, node: AST.PropTermInNode):
		print("Constraint Term: in")
		self.visit(node.n)
		self.visit(node.z)

	def visitPropTermOp(self, node: AST.PropTermOpNode):
		print("Constraint Term: " + node.op)
		print("Left:")
		self.visit(node.leftpt)
		print("Right:")
		self.visit(node.rightpt)
		

	def visitSingleProp(self, node: AST.SinglePropNode):
		print("Constraint: " + node.op)
		print("Left:")
		self.visit(node.leftpt)
		print("Right:")
		self.visit(node.rightpt)

	def visitDoubleProp(self, node: AST.DoublePropNode):
		print("Constraint: " + node.op)
		print("Left:")
		self.visit(node.leftprop)
		print("Right:")
		self.visit(node.rightprop)