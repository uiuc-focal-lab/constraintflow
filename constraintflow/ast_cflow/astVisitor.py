from . import astcf as AST

class ASTVisitor:

	def visit(self, node):
		if isinstance(node, AST.BaseTypeNode):
			return self.visitBaseType(node)
		elif isinstance(node, AST.ArrayTypeNode):
			return self.visitArrayType(node)
		elif isinstance(node, AST.ArgListNode):
			return self.visitArgList(node)
		elif isinstance(node, AST.ExprListNode):
			return self.visitExprList(node)
		elif isinstance(node, AST.BinOpNode):
			return self.visitBinOp(node)
		elif isinstance(node, AST.UnOpNode):
			return self.visitUnOp(node)
		elif isinstance(node, AST.ArgmaxOpNode):
			return self.visitArgmaxOp(node)
		elif isinstance(node, AST.MaxOpListNode):
			return self.visitMaxOpList(node)
		elif isinstance(node, AST.MaxOpNode):
			return self.visitMaxOp(node)
		elif isinstance(node, AST.ListOpNode):
			return self.visitListOp(node)
		elif isinstance(node, AST.VarNode):
			return self.visitVar(node)
		elif isinstance(node, AST.NeuronNode):
			return self.visitNeuron(node)
		elif isinstance(node, AST.ConstIntNode):
			return self.visitInt(node)
		elif isinstance(node, AST.ConstFloatNode):
			return self.visitFloat(node)
		elif isinstance(node, AST.ConstBoolNode):
			return self.visitBool(node)
		# elif isinstance(node, AST.CurrNode):
		# 	return self.visitCurr(node)
		# elif isinstance(node, AST.PrevNode):
		# 	return self.visitPrev(node)
		elif isinstance(node, AST.EpsilonNode):
			return self.visitEpsilon(node)
		elif isinstance(node, AST.TernaryNode):
			return self.visitTernary(node)
		elif isinstance(node, AST.IfNode):
			return self.visitIf(node)
		elif isinstance(node, AST.GetMetadataNode):
			return self.visitGetMetadata(node)
		elif isinstance(node, AST.GetElementNode):
			return self.visitGetElement(node)
		# elif isinstance(node, AST.GetElementAtIndexNode):
		# 	return self.visitGetElementAtIndex(node)
		elif isinstance(node, AST.TraverseNode):
			return self.visitTraverse(node)
		elif isinstance(node, AST.MapNode):
			return self.visitMap(node)
		elif isinstance(node, AST.MapListNode):
			return self.visitMapList(node)
		elif isinstance(node, AST.DotNode):
			return self.visitDot(node)
		elif isinstance(node, AST.ConcatNode):
			return self.visitConcat(node)
		elif isinstance(node, AST.LpNode):
			return self.visitLp(node)
		elif isinstance(node, AST.FuncCallNode):
			return self.visitFuncCall(node)
		elif isinstance(node, AST.ShapeDeclNode):
			return self.visitShapeDecl(node)
		elif isinstance(node, AST.TransRetBasicNode):
			return self.visitTransRetBasic(node)
		elif isinstance(node, AST.TransRetIfNode):
			return self.visitTransRetIf(node)
		elif isinstance(node, AST.OpStmtNode):
			return self.visitOpStmt(node)
		elif isinstance(node, AST.OpListNode):
			return self.visitOpList(node)
		elif isinstance(node, AST.TransformerNode):
			return self.visitTransformer(node)
		# elif isinstance(node, AST.PropTermBasicNode):
		# 	return self.visitPropTermBasic(node)
		# elif isinstance(node, AST.PropTermInNode):
		# 	return self.visitPropTermIn(node)
		# elif isinstance(node, AST.PropTermOpNode):
		# 	return self.visitPropTermOp(node)
		# elif isinstance(node, AST.SinglePropNode):
		# 	return self.visitSingleProp(node)
		# elif isinstance(node, AST.DoublePropNode):
		# 	return self.visitDoubleProp(node)
		elif isinstance(node, AST.FuncNode):
			return self.visitFunc(node)
		elif isinstance(node, AST.SeqNode):
			return self.visitSeq(node)
		elif isinstance(node, AST.FlowNode):
			return self.visitFlow(node)
		elif isinstance(node, AST.ProgramNode):
			return self.visitProg(node)
		else:
			print("This is an error. This shouldn't happen")
			print(node)
			assert False