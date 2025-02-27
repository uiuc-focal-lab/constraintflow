# Generated from dsl.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .dslParser import dslParser
else:
    from dslParser import dslParser

# This class defines a complete generic visitor for a parse tree produced by dslParser.

class dslVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by dslParser#prog.
    def visitProg(self, ctx:dslParser.ProgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#shape_decl.
    def visitShape_decl(self, ctx:dslParser.Shape_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#flowstmt.
    def visitFlowstmt(self, ctx:dslParser.FlowstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#funcstmt.
    def visitFuncstmt(self, ctx:dslParser.FuncstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#seqstmt.
    def visitSeqstmt(self, ctx:dslParser.SeqstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#transstmt.
    def visitTransstmt(self, ctx:dslParser.TransstmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#func_decl.
    def visitFunc_decl(self, ctx:dslParser.Func_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#transformer.
    def visitTransformer(self, ctx:dslParser.TransformerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#op_list.
    def visitOp_list(self, ctx:dslParser.Op_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#op_stmt.
    def visitOp_stmt(self, ctx:dslParser.Op_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#trans_decl.
    def visitTrans_decl(self, ctx:dslParser.Trans_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#operator.
    def visitOperator(self, ctx:dslParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#condtrans.
    def visitCondtrans(self, ctx:dslParser.CondtransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#parentrans.
    def visitParentrans(self, ctx:dslParser.ParentransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#trans.
    def visitTrans(self, ctx:dslParser.TransContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#types.
    def visitTypes(self, ctx:dslParser.TypesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#arglist.
    def visitArglist(self, ctx:dslParser.ArglistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#expr_list.
    def visitExpr_list(self, ctx:dslParser.Expr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#exprs.
    def visitExprs(self, ctx:dslParser.ExprsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#lp.
    def visitLp(self, ctx:dslParser.LpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#argmaxOp.
    def visitArgmaxOp(self, ctx:dslParser.ArgmaxOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#prev.
    def visitPrev(self, ctx:dslParser.PrevContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#maxOp.
    def visitMaxOp(self, ctx:dslParser.MaxOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#dot.
    def visitDot(self, ctx:dslParser.DotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#map_list.
    def visitMap_list(self, ctx:dslParser.Map_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#float.
    def visitFloat(self, ctx:dslParser.FloatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#cond.
    def visitCond(self, ctx:dslParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#epsilon.
    def visitEpsilon(self, ctx:dslParser.EpsilonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#varExp.
    def visitVarExp(self, ctx:dslParser.VarExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#neg.
    def visitNeg(self, ctx:dslParser.NegContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#not.
    def visitNot(self, ctx:dslParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#listOp.
    def visitListOp(self, ctx:dslParser.ListOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#curr_list.
    def visitCurr_list(self, ctx:dslParser.Curr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#curr.
    def visitCurr(self, ctx:dslParser.CurrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#maxOpList.
    def visitMaxOpList(self, ctx:dslParser.MaxOpListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#map.
    def visitMap(self, ctx:dslParser.MapContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#exprarray.
    def visitExprarray(self, ctx:dslParser.ExprarrayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#getMetadata.
    def visitGetMetadata(self, ctx:dslParser.GetMetadataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#false.
    def visitFalse(self, ctx:dslParser.FalseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#concat.
    def visitConcat(self, ctx:dslParser.ConcatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#curry.
    def visitCurry(self, ctx:dslParser.CurryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#int.
    def visitInt(self, ctx:dslParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#prev_0.
    def visitPrev_0(self, ctx:dslParser.Prev_0Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#prev_1.
    def visitPrev_1(self, ctx:dslParser.Prev_1Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#traverse.
    def visitTraverse(self, ctx:dslParser.TraverseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#binopExp.
    def visitBinopExp(self, ctx:dslParser.BinopExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#getElement.
    def visitGetElement(self, ctx:dslParser.GetElementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#true.
    def visitTrue(self, ctx:dslParser.TrueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#parenExp.
    def visitParenExp(self, ctx:dslParser.ParenExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#funcCall.
    def visitFuncCall(self, ctx:dslParser.FuncCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#argmax_op.
    def visitArgmax_op(self, ctx:dslParser.Argmax_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#lp_op.
    def visitLp_op(self, ctx:dslParser.Lp_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#max_op.
    def visitMax_op(self, ctx:dslParser.Max_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#list_op.
    def visitList_op(self, ctx:dslParser.List_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#binop.
    def visitBinop(self, ctx:dslParser.BinopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#metadata.
    def visitMetadata(self, ctx:dslParser.MetadataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by dslParser#direction.
    def visitDirection(self, ctx:dslParser.DirectionContext):
        return self.visitChildren(ctx)



del dslParser