from . import astcf as AST

if __name__ is not None and "." in __name__:
    from .dslParser import dslParser
    from .dslVisitor import dslVisitor
else:
    from dslParser import dslParser
    from dslVisitor import dslVisitor


class ASTBuilder(dslVisitor):

    def __init__(self):
        self.epsilon_count = 0
        self.inconstraint = False

    def visitProg(self, ctx: dslParser.ProgContext):
        shape = self.visit(ctx.shape_decl())
        stmt = self.visit(ctx.statement())
        return AST.ProgramNode(shape, stmt)

    def visitShape_decl(self, ctx:dslParser.Shape_declContext):
        elements = self.visit(ctx.arglist())
        self.inconstraint = True
        p = self.visit(ctx.expr())
        self.inconstraint = False
        return AST.ShapeDeclNode(elements, p)

    def visitFlowstmt(self, ctx:dslParser.FlowstmtContext):
        direction = self.visit(ctx.direction())
        pfunc = self.visit(ctx.expr(0))
        sfunc = self.visit(ctx.expr(1))
        trans = AST.VarNode(ctx.VAR().getText())
        return AST.FlowNode(direction, pfunc, sfunc, trans)

    def visitFuncstmt(self, ctx:dslParser.FuncstmtContext):
        decl = self.visit(ctx.func_decl())
        expr = self.visit(ctx.expr())
        return AST.FuncNode(decl, expr)

    def visitSeqstmt(self, ctx:dslParser.SeqstmtContext):
        stmt1 = self.visit(ctx.statement(0))
        stmt2 = self.visit(ctx.statement(1))
        return AST.SeqNode(stmt1, stmt2)

    def visitTransstmt(self, ctx:dslParser.TransstmtContext):
        return self.visit(ctx.transformer())

    def visitFunc_decl(self, ctx:dslParser.Func_declContext):
        name = AST.VarNode(ctx.VAR().getText())
        arglist = self.visit(ctx.arglist())
        return AST.FuncDeclNode(name, arglist)

    def visitTransformer(self, ctx:dslParser.TransformerContext):
        op_list = self.visit(ctx.op_list())
        name = AST.VarNode(ctx.trans_decl().VAR().getText())
        # l = self.visit(ctx.trans_decl().expr_list())
        # expr_list = [v.name for v in l.exprlist]
        return AST.TransformerNode(name, op_list)

    #this O(n^2) to retain the order of the expressions
    def visitOp_list(self, ctx:dslParser.Op_listContext):
        oplist = ctx.op_list()
        stmt = self.visit(ctx.op_stmt())

        if(oplist):
            listNode = self.visit(oplist)
            newList = [stmt] + listNode.olist
            return AST.OpListNode(newList)
        else:
            return AST.OpListNode([stmt])

    def visitOp_stmt(self, ctx:dslParser.Op_stmtContext):
        op = self.visit(ctx.operator())
        ret = self.visit(ctx.trans_ret())
        return AST.OpStmtNode(op, ret)

    def visitOperator(self, ctx:dslParser.OperatorContext):
        value = ctx.getText()
        return AST.OperatorNode(value)

    def visitCondtrans(self, ctx:dslParser.CondtransContext):
        cond = self.visit(ctx.expr())
        texpr = self.visit(ctx.trans_ret(0))
        fexpr = self.visit(ctx.trans_ret(1))
        return AST.TransRetIfNode(cond, texpr, fexpr)

    def visitParentrans(self, ctx:dslParser.ParentransContext):
        return self.visit(ctx.trans_ret())

    def visitTrans(self, ctx:dslParser.TransContext):
        expr_list = self.visit(ctx.expr_list())
        return AST.TransRetBasicNode(expr_list)

    def visitTypes(self, ctx:dslParser.TypesContext):
        if(ctx.LIST()):
            tval = self.visit(ctx.types())
            return AST.ArrayTypeNode(tval)
        else:
            value = ctx.getText()
            return AST.BaseTypeNode(value)

    def visitArglist(self, ctx:dslParser.ArglistContext):
        args = ctx.arglist()
        type_node = self.visit(ctx.types())
        var = AST.VarNode(ctx.VAR().getText())

        if(args):
            listNode = self.visit(args)
            newList = [(type_node, var)] + listNode.arglist
            return AST.ArgListNode(newList)
        else:
            return AST.ArgListNode([(type_node, var)])

    def visitExpr_list(self, ctx:dslParser.Expr_listContext):
        expr = self.visit(ctx.expr())
        if(ctx.expr_list()):
            exprs = ctx.expr_list()
            listNode = self.visit(exprs)
            newList = [expr] + listNode.exprlist
            return AST.ExprListNode(newList)
        else:
            return AST.ExprListNode([expr])

    def visitExprs(self, ctx):
        expr1 = self.visit(ctx.expr())
        if(ctx.exprs()):
            expr_list = self.visit(ctx.exprs())
            return [expr1] + expr_list
        return [expr1]

    def visitDot(self, ctx:dslParser.DotContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        return AST.DotNode(left, right)

    def visitConcat(self, ctx:dslParser.ConcatContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        return AST.ConcatNode(left, right)
    
    def visitLp(self, ctx):
        op = ctx.lp_op().getText()
        e = self.visit(ctx.expr(0))
        c = self.visit(ctx.expr(1))
        return AST.LpNode(op, e, c)


    # def visitGetElement(self, ctx:dslParser.GetElementContext):
    #     op = ctx.list_op().getText()
    #     expr = self.visit(ctx.expr())
    #     if(op == "sum"):
    #         return AST.SumNode(expr)
    #     elif(op == "avg"):
    #         return AST.AvgNode(expr)
    #     elif(op == "len"):
    #         return AST.LenNode(expr)

    #def visitSum(self, ctx:dslParser.SumContext):
    #    expr = self.visit(ctx.expr())
    #    return AST.SumNode(expr)

    def visitFloat(self, ctx:dslParser.FloatContext):
        value = float(ctx.FloatConst().getText())
        return AST.ConstFloatNode(value)

    def visitCond(self, ctx:dslParser.CondContext):
        cond = self.visit(ctx.expr(0))
        texpr = self.visit(ctx.expr(1))
        fexpr = self.visit(ctx.expr(2))
        return AST.TernaryNode(cond, texpr, fexpr)
    
    def visitCond_if(self, ctx:dslParser.CondContext):
        cond = self.visit(ctx.expr(0))
        texpr = self.visit(ctx.expr(1))
        fexpr = self.visit(ctx.expr(2))
        return AST.IfNode(cond, texpr, fexpr)

    def visitEpsilon(self, ctx:dslParser.EpsilonContext):
        self.epsilon_count += 1
        return AST.EpsilonNode(self.epsilon_count)

    def visitNeg(self, ctx:dslParser.NegContext):
        expr = self.visit(ctx.expr())
        return AST.UnOpNode("-", expr)

    def visitNot(self, ctx:dslParser.NotContext):
        expr = self.visit(ctx.expr())
        return AST.UnOpNode("~", expr)
    
    def visitSigma(self, ctx:dslParser.SigmaContext):
        expr = self.visit(ctx.expr())
        return AST.UnOpNode("sigma", expr)

    def visitCurr(self, ctx:dslParser.CurrContext):
        if(self.inconstraint):
            return AST.VarNode("curr_new")
        else:
            return AST.VarNode("curr")

    def visitPrev(self, ctx:dslParser.PrevContext):
        return AST.VarNode("prev")
    
    def visitPrev_0(self, ctx:dslParser.Prev_0Context):
        return AST.VarNode("prev_0")
    
    def visitPrev_1(self, ctx:dslParser.Prev_1Context):
        return AST.VarNode("prev_1")
    
    def visitCurr_list(self, ctx:dslParser.Curr_listContext):
        return AST.VarNode("curr_list")
    
    def visitExprarray(self, ctx):
        return self.visit(ctx.expr_list())

    def visitMap(self, ctx:dslParser.MapContext):
        expr = self.visit(ctx.expr(0))
        func = self.visit(ctx.expr(1))
        if(not isinstance (func, AST.FuncCallNode)):
            func = AST.VarNode(ctx.expr(1).getText())
        return AST.MapNode(expr, func)

    def visitMap_list(self, ctx:dslParser.MapContext):
        expr = self.visit(ctx.expr(0))
        func = self.visit(ctx.expr(1))
        if(not isinstance (func, AST.FuncCallNode)):
            func = AST.VarNode(ctx.expr(1).getText())
        return AST.MapListNode(expr, func)

    def visitGetMetadata(self, ctx:dslParser.GetMetadataContext):
        expr = self.visit(ctx.expr())
        data = self.visit(ctx.metadata())
        # data = ctx.getText())
        return AST.GetMetadataNode(expr, data)
    
    # def visitGetElementAtIndex(self, ctx:dslParser.GetElementAtIndexContext):
    #     expr = self.visit(ctx.expr())
    #     index = int(ctx.IntConst().getText())
    #     return AST.GetElementAtIndexNode(expr, index)

    def visitVarExp(self, ctx:dslParser.VarExpContext):
        name = ctx.VAR().getText()
        return AST.VarNode(name)

    def visitFalse(self, ctx:dslParser.FalseContext):
        return AST.ConstBoolNode(False)

    def visitInt(self, ctx:dslParser.IntContext):
        value = int(ctx.IntConst().getText())
        return AST.ConstIntNode(value)

    def visitTraverse(self, ctx:dslParser.TraverseContext):
        expr = self.visit(ctx.expr(0))
        direction = self.visit(ctx.direction())
        priority = self.visit(ctx.expr(1))
        stop = self.visit(ctx.expr(2))
        func = self.visit(ctx.expr(3))
        self.inconstraint = True
        p = self.visit(ctx.expr(4))
        self.inconstraint = False
        return AST.TraverseNode(expr, direction, priority, stop, func, p)

    def visitBinopExp(self, ctx:dslParser.BinopExpContext):
        left = self.visit(ctx.expr(0))
        op = ctx.binop().getText()
        right = self.visit(ctx.expr(1)) 
        return AST.BinOpNode(left, op, right)

    def visitGetElement(self, ctx:dslParser.GetElementContext):
        expr = self.visit(ctx.expr())
        elem = AST.VarNode(ctx.VAR().getText())
        return AST.GetElementNode(expr, elem)

    def visitTrue(self, ctx:dslParser.TrueContext):
        return AST.ConstBoolNode(True)

    def visitParenExp(self, ctx:dslParser.ParenExpContext):
        return self.visit(ctx.expr())

    def visitArgmaxOp(self, ctx:dslParser.ArgmaxOpContext):
        op = ctx.argmax_op().getText()
        expr = self.visit(ctx.expr(0))
        func = self.visit(ctx.expr(1))
        return AST.ArgmaxOpNode(op, expr, func)

    def visitMaxOpList(self, ctx:dslParser.MaxOpListContext):
        op = ctx.max_op().getText()
        expr = self.visit(ctx.expr())
        return AST.MaxOpListNode(op, expr)

    def visitMaxOp(self, ctx:dslParser.MaxOpContext):
        op = ctx.max_op().getText()
        expr1 = self.visit(ctx.expr(0))
        expr2 = self.visit(ctx.expr(1))
        return AST.MaxOpNode(op, expr1, expr2)

    def visitListOp(self, ctx:dslParser.ListOpContext):
        op = ctx.list_op().getText()
        expr = self.visit(ctx.expr())
        return AST.ListOpNode(op, expr)

    def visitFuncCall(self, ctx:dslParser.FuncCallContext):
        name = AST.VarNode(ctx.VAR().getText())
        arglist = self.visit(ctx.expr_list())
        return AST.FuncCallNode(name, arglist)

    def visitCurry(self, ctx):
        name = AST.VarNode(ctx.VAR().getText())
        arglist = self.visit(ctx.exprs())
        return AST.FuncCallNode(name, AST.ExprListNode(arglist))

    def visitMetadata(self, ctx:dslParser.MetadataContext):
        return AST.MetadataNode(ctx.getText())

    def visitDirection(self, ctx:dslParser.DirectionContext):
        return AST.DirectionNode(ctx.getText())

    # def visitPtop(self, ctx:dslParser.PtopContext):
    #     leftpt = self.visit(ctx.pt(0))
    #     rightpt = self.visit(ctx.pt(1))
    #     isplus = ctx.PLUS()
    #     if(isplus):
    #         return AST.PropTermOpNode(leftpt, isplus.getText(), rightpt)
    #     else:
    #         return AST.PropTermOpNode(leftpt, ctx.MINUS().getText(), rightpt)

    # def visitPtparen(self, ctx):
    #     return self.visit(ctx.pt())

    # def visitPtbasic(self, ctx:dslParser.PtbasicContext):
    #     if(ctx.IntConst()):
    #         intNode = AST.ConstIntNode(int(ctx.IntConst().getText()))
    #         return AST.PropTermBasicNode(intNode)
    #     elif(ctx.FloatConst()):
    #         floatNode = AST.ConstFloatNode(float(ctx.FloatConst().getText()))
    #         return AST.PropTermBasicNode(floatNode)
    #     # elif(ctx.TRUE()):
    #     #     boolNode = AST.ConstBoolNode(True)
    #     #     return AST.PropTermBasicNode(boolNode)
    #     # elif(ctx.FALSE()):
    #     #     boolNode = AST.ConstBoolNode(False)
    #     #     return AST.PropTermBasicNode(boolNode)
    #     elif(ctx.pt() and ctx.VAR()):
    #         expr = self.visit(ctx.pt())
    #         elem = AST.VarNode(ctx.VAR().getText())
    #         return AST.GetElementNode(expr, elem)
    #     elif(ctx.pt() and ctx.metadata()):
    #         expr = self.visit(ctx.pt())
    #         metadata =  AST.MetadataNode(ctx.metadata().getText())
    #         return AST.GetMetadataNode(expr, metadata)
    #     elif(ctx.CURR()):
    #         return AST.VarNode("curr_new")
    #         #return AST.CurrNode()
    #     elif(ctx.VAR()):
    #         return AST.VarNode(ctx.VAR().getText())

    # def visitPtin(self, ctx:dslParser.PtinContext):
    #     n = self.visit(ctx.pt(0))
    #     z = self.visit(ctx.pt(1))
    #     return AST.SinglePropNode(n, "in", z)

    # def visitPropsingle(self, ctx:dslParser.PropsingleContext):
    #     if(ctx.TRUE()):
    #         boolNode = AST.ConstBoolNode(True)
    #         return AST.PropTermBasicNode(boolNode)
    #     elif(ctx.FALSE()):
    #         boolNode = AST.ConstBoolNode(False)
    #         return AST.PropTermBasicNode(boolNode)
    #     else:
    #         leftpt = self.visit(ctx.pt(0))
    #         rightpt = self.visit(ctx.pt(1))
    #         op = ctx.children[1].getText()
    #         return AST.SinglePropNode(leftpt, op, rightpt)

    # def visitPropparen(self, ctx:dslParser.PropparenContext):
    #     return self.visit(ctx.prop())

    # def visitPropdouble(self, ctx:dslParser.PropdoubleContext):
    #     leftprop = self.visit(ctx.prop(0))
    #     rightprop = self.visit(ctx.prop(1))
    #     isand = ctx.AND()
    #     if(isand):
    #         return AST.DoublePropNode(leftprop, isand.getText(), rightprop)
    #     else:
    #         return AST.DoublePropNode(leftprop, ctx.OR().getText(), rightprop)

