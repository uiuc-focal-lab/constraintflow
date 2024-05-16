# Generated from dsl.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .dslParser import dslParser
else:
    from dslParser import dslParser

# This class defines a complete listener for a parse tree produced by dslParser.
class dslListener(ParseTreeListener):

    # Enter a parse tree produced by dslParser#prog.
    def enterProg(self, ctx:dslParser.ProgContext):
        pass

    # Exit a parse tree produced by dslParser#prog.
    def exitProg(self, ctx:dslParser.ProgContext):
        pass


    # Enter a parse tree produced by dslParser#shape_decl.
    def enterShape_decl(self, ctx:dslParser.Shape_declContext):
        pass

    # Exit a parse tree produced by dslParser#shape_decl.
    def exitShape_decl(self, ctx:dslParser.Shape_declContext):
        pass


    # Enter a parse tree produced by dslParser#flowstmt.
    def enterFlowstmt(self, ctx:dslParser.FlowstmtContext):
        pass

    # Exit a parse tree produced by dslParser#flowstmt.
    def exitFlowstmt(self, ctx:dslParser.FlowstmtContext):
        pass


    # Enter a parse tree produced by dslParser#funcstmt.
    def enterFuncstmt(self, ctx:dslParser.FuncstmtContext):
        pass

    # Exit a parse tree produced by dslParser#funcstmt.
    def exitFuncstmt(self, ctx:dslParser.FuncstmtContext):
        pass


    # Enter a parse tree produced by dslParser#seqstmt.
    def enterSeqstmt(self, ctx:dslParser.SeqstmtContext):
        pass

    # Exit a parse tree produced by dslParser#seqstmt.
    def exitSeqstmt(self, ctx:dslParser.SeqstmtContext):
        pass


    # Enter a parse tree produced by dslParser#transstmt.
    def enterTransstmt(self, ctx:dslParser.TransstmtContext):
        pass

    # Exit a parse tree produced by dslParser#transstmt.
    def exitTransstmt(self, ctx:dslParser.TransstmtContext):
        pass


    # Enter a parse tree produced by dslParser#func_decl.
    def enterFunc_decl(self, ctx:dslParser.Func_declContext):
        pass

    # Exit a parse tree produced by dslParser#func_decl.
    def exitFunc_decl(self, ctx:dslParser.Func_declContext):
        pass


    # Enter a parse tree produced by dslParser#transformer.
    def enterTransformer(self, ctx:dslParser.TransformerContext):
        pass

    # Exit a parse tree produced by dslParser#transformer.
    def exitTransformer(self, ctx:dslParser.TransformerContext):
        pass


    # Enter a parse tree produced by dslParser#op_list.
    def enterOp_list(self, ctx:dslParser.Op_listContext):
        pass

    # Exit a parse tree produced by dslParser#op_list.
    def exitOp_list(self, ctx:dslParser.Op_listContext):
        pass


    # Enter a parse tree produced by dslParser#op_stmt.
    def enterOp_stmt(self, ctx:dslParser.Op_stmtContext):
        pass

    # Exit a parse tree produced by dslParser#op_stmt.
    def exitOp_stmt(self, ctx:dslParser.Op_stmtContext):
        pass


    # Enter a parse tree produced by dslParser#trans_decl.
    def enterTrans_decl(self, ctx:dslParser.Trans_declContext):
        pass

    # Exit a parse tree produced by dslParser#trans_decl.
    def exitTrans_decl(self, ctx:dslParser.Trans_declContext):
        pass


    # Enter a parse tree produced by dslParser#operator.
    def enterOperator(self, ctx:dslParser.OperatorContext):
        pass

    # Exit a parse tree produced by dslParser#operator.
    def exitOperator(self, ctx:dslParser.OperatorContext):
        pass


    # Enter a parse tree produced by dslParser#condtrans.
    def enterCondtrans(self, ctx:dslParser.CondtransContext):
        pass

    # Exit a parse tree produced by dslParser#condtrans.
    def exitCondtrans(self, ctx:dslParser.CondtransContext):
        pass


    # Enter a parse tree produced by dslParser#parentrans.
    def enterParentrans(self, ctx:dslParser.ParentransContext):
        pass

    # Exit a parse tree produced by dslParser#parentrans.
    def exitParentrans(self, ctx:dslParser.ParentransContext):
        pass


    # Enter a parse tree produced by dslParser#trans.
    def enterTrans(self, ctx:dslParser.TransContext):
        pass

    # Exit a parse tree produced by dslParser#trans.
    def exitTrans(self, ctx:dslParser.TransContext):
        pass


    # Enter a parse tree produced by dslParser#types.
    def enterTypes(self, ctx:dslParser.TypesContext):
        pass

    # Exit a parse tree produced by dslParser#types.
    def exitTypes(self, ctx:dslParser.TypesContext):
        pass


    # Enter a parse tree produced by dslParser#arglist.
    def enterArglist(self, ctx:dslParser.ArglistContext):
        pass

    # Exit a parse tree produced by dslParser#arglist.
    def exitArglist(self, ctx:dslParser.ArglistContext):
        pass


    # Enter a parse tree produced by dslParser#expr_list.
    def enterExpr_list(self, ctx:dslParser.Expr_listContext):
        pass

    # Exit a parse tree produced by dslParser#expr_list.
    def exitExpr_list(self, ctx:dslParser.Expr_listContext):
        pass


    # Enter a parse tree produced by dslParser#exprs.
    def enterExprs(self, ctx:dslParser.ExprsContext):
        pass

    # Exit a parse tree produced by dslParser#exprs.
    def exitExprs(self, ctx:dslParser.ExprsContext):
        pass


    # Enter a parse tree produced by dslParser#lp.
    def enterLp(self, ctx:dslParser.LpContext):
        pass

    # Exit a parse tree produced by dslParser#lp.
    def exitLp(self, ctx:dslParser.LpContext):
        pass


    # Enter a parse tree produced by dslParser#argmaxOp.
    def enterArgmaxOp(self, ctx:dslParser.ArgmaxOpContext):
        pass

    # Exit a parse tree produced by dslParser#argmaxOp.
    def exitArgmaxOp(self, ctx:dslParser.ArgmaxOpContext):
        pass


    # Enter a parse tree produced by dslParser#prev.
    def enterPrev(self, ctx:dslParser.PrevContext):
        pass

    # Exit a parse tree produced by dslParser#prev.
    def exitPrev(self, ctx:dslParser.PrevContext):
        pass


    # Enter a parse tree produced by dslParser#maxOp.
    def enterMaxOp(self, ctx:dslParser.MaxOpContext):
        pass

    # Exit a parse tree produced by dslParser#maxOp.
    def exitMaxOp(self, ctx:dslParser.MaxOpContext):
        pass


    # Enter a parse tree produced by dslParser#dot.
    def enterDot(self, ctx:dslParser.DotContext):
        pass

    # Exit a parse tree produced by dslParser#dot.
    def exitDot(self, ctx:dslParser.DotContext):
        pass


    # Enter a parse tree produced by dslParser#map_list.
    def enterMap_list(self, ctx:dslParser.Map_listContext):
        pass

    # Exit a parse tree produced by dslParser#map_list.
    def exitMap_list(self, ctx:dslParser.Map_listContext):
        pass


    # Enter a parse tree produced by dslParser#float.
    def enterFloat(self, ctx:dslParser.FloatContext):
        pass

    # Exit a parse tree produced by dslParser#float.
    def exitFloat(self, ctx:dslParser.FloatContext):
        pass


    # Enter a parse tree produced by dslParser#cond.
    def enterCond(self, ctx:dslParser.CondContext):
        pass

    # Exit a parse tree produced by dslParser#cond.
    def exitCond(self, ctx:dslParser.CondContext):
        pass


    # Enter a parse tree produced by dslParser#epsilon.
    def enterEpsilon(self, ctx:dslParser.EpsilonContext):
        pass

    # Exit a parse tree produced by dslParser#epsilon.
    def exitEpsilon(self, ctx:dslParser.EpsilonContext):
        pass


    # Enter a parse tree produced by dslParser#varExp.
    def enterVarExp(self, ctx:dslParser.VarExpContext):
        pass

    # Exit a parse tree produced by dslParser#varExp.
    def exitVarExp(self, ctx:dslParser.VarExpContext):
        pass


    # Enter a parse tree produced by dslParser#neg.
    def enterNeg(self, ctx:dslParser.NegContext):
        pass

    # Exit a parse tree produced by dslParser#neg.
    def exitNeg(self, ctx:dslParser.NegContext):
        pass


    # Enter a parse tree produced by dslParser#not.
    def enterNot(self, ctx:dslParser.NotContext):
        pass

    # Exit a parse tree produced by dslParser#not.
    def exitNot(self, ctx:dslParser.NotContext):
        pass


    # Enter a parse tree produced by dslParser#listOp.
    def enterListOp(self, ctx:dslParser.ListOpContext):
        pass

    # Exit a parse tree produced by dslParser#listOp.
    def exitListOp(self, ctx:dslParser.ListOpContext):
        pass


    # Enter a parse tree produced by dslParser#curr_list.
    def enterCurr_list(self, ctx:dslParser.Curr_listContext):
        pass

    # Exit a parse tree produced by dslParser#curr_list.
    def exitCurr_list(self, ctx:dslParser.Curr_listContext):
        pass


    # Enter a parse tree produced by dslParser#curr.
    def enterCurr(self, ctx:dslParser.CurrContext):
        pass

    # Exit a parse tree produced by dslParser#curr.
    def exitCurr(self, ctx:dslParser.CurrContext):
        pass


    # Enter a parse tree produced by dslParser#maxOpList.
    def enterMaxOpList(self, ctx:dslParser.MaxOpListContext):
        pass

    # Exit a parse tree produced by dslParser#maxOpList.
    def exitMaxOpList(self, ctx:dslParser.MaxOpListContext):
        pass


    # Enter a parse tree produced by dslParser#map.
    def enterMap(self, ctx:dslParser.MapContext):
        pass

    # Exit a parse tree produced by dslParser#map.
    def exitMap(self, ctx:dslParser.MapContext):
        pass


    # Enter a parse tree produced by dslParser#exprarray.
    def enterExprarray(self, ctx:dslParser.ExprarrayContext):
        pass

    # Exit a parse tree produced by dslParser#exprarray.
    def exitExprarray(self, ctx:dslParser.ExprarrayContext):
        pass


    # Enter a parse tree produced by dslParser#getMetadata.
    def enterGetMetadata(self, ctx:dslParser.GetMetadataContext):
        pass

    # Exit a parse tree produced by dslParser#getMetadata.
    def exitGetMetadata(self, ctx:dslParser.GetMetadataContext):
        pass


    # Enter a parse tree produced by dslParser#false.
    def enterFalse(self, ctx:dslParser.FalseContext):
        pass

    # Exit a parse tree produced by dslParser#false.
    def exitFalse(self, ctx:dslParser.FalseContext):
        pass


    # Enter a parse tree produced by dslParser#concat.
    def enterConcat(self, ctx:dslParser.ConcatContext):
        pass

    # Exit a parse tree produced by dslParser#concat.
    def exitConcat(self, ctx:dslParser.ConcatContext):
        pass


    # Enter a parse tree produced by dslParser#curry.
    def enterCurry(self, ctx:dslParser.CurryContext):
        pass

    # Exit a parse tree produced by dslParser#curry.
    def exitCurry(self, ctx:dslParser.CurryContext):
        pass


    # Enter a parse tree produced by dslParser#int.
    def enterInt(self, ctx:dslParser.IntContext):
        pass

    # Exit a parse tree produced by dslParser#int.
    def exitInt(self, ctx:dslParser.IntContext):
        pass


    # Enter a parse tree produced by dslParser#prev_0.
    def enterPrev_0(self, ctx:dslParser.Prev_0Context):
        pass

    # Exit a parse tree produced by dslParser#prev_0.
    def exitPrev_0(self, ctx:dslParser.Prev_0Context):
        pass


    # Enter a parse tree produced by dslParser#prev_1.
    def enterPrev_1(self, ctx:dslParser.Prev_1Context):
        pass

    # Exit a parse tree produced by dslParser#prev_1.
    def exitPrev_1(self, ctx:dslParser.Prev_1Context):
        pass


    # Enter a parse tree produced by dslParser#traverse.
    def enterTraverse(self, ctx:dslParser.TraverseContext):
        pass

    # Exit a parse tree produced by dslParser#traverse.
    def exitTraverse(self, ctx:dslParser.TraverseContext):
        pass


    # Enter a parse tree produced by dslParser#binopExp.
    def enterBinopExp(self, ctx:dslParser.BinopExpContext):
        pass

    # Exit a parse tree produced by dslParser#binopExp.
    def exitBinopExp(self, ctx:dslParser.BinopExpContext):
        pass


    # Enter a parse tree produced by dslParser#getElement.
    def enterGetElement(self, ctx:dslParser.GetElementContext):
        pass

    # Exit a parse tree produced by dslParser#getElement.
    def exitGetElement(self, ctx:dslParser.GetElementContext):
        pass


    # Enter a parse tree produced by dslParser#true.
    def enterTrue(self, ctx:dslParser.TrueContext):
        pass

    # Exit a parse tree produced by dslParser#true.
    def exitTrue(self, ctx:dslParser.TrueContext):
        pass


    # Enter a parse tree produced by dslParser#parenExp.
    def enterParenExp(self, ctx:dslParser.ParenExpContext):
        pass

    # Exit a parse tree produced by dslParser#parenExp.
    def exitParenExp(self, ctx:dslParser.ParenExpContext):
        pass


    # Enter a parse tree produced by dslParser#funcCall.
    def enterFuncCall(self, ctx:dslParser.FuncCallContext):
        pass

    # Exit a parse tree produced by dslParser#funcCall.
    def exitFuncCall(self, ctx:dslParser.FuncCallContext):
        pass


    # Enter a parse tree produced by dslParser#argmax_op.
    def enterArgmax_op(self, ctx:dslParser.Argmax_opContext):
        pass

    # Exit a parse tree produced by dslParser#argmax_op.
    def exitArgmax_op(self, ctx:dslParser.Argmax_opContext):
        pass


    # Enter a parse tree produced by dslParser#lp_op.
    def enterLp_op(self, ctx:dslParser.Lp_opContext):
        pass

    # Exit a parse tree produced by dslParser#lp_op.
    def exitLp_op(self, ctx:dslParser.Lp_opContext):
        pass


    # Enter a parse tree produced by dslParser#max_op.
    def enterMax_op(self, ctx:dslParser.Max_opContext):
        pass

    # Exit a parse tree produced by dslParser#max_op.
    def exitMax_op(self, ctx:dslParser.Max_opContext):
        pass


    # Enter a parse tree produced by dslParser#list_op.
    def enterList_op(self, ctx:dslParser.List_opContext):
        pass

    # Exit a parse tree produced by dslParser#list_op.
    def exitList_op(self, ctx:dslParser.List_opContext):
        pass


    # Enter a parse tree produced by dslParser#binop.
    def enterBinop(self, ctx:dslParser.BinopContext):
        pass

    # Exit a parse tree produced by dslParser#binop.
    def exitBinop(self, ctx:dslParser.BinopContext):
        pass


    # Enter a parse tree produced by dslParser#metadata.
    def enterMetadata(self, ctx:dslParser.MetadataContext):
        pass

    # Exit a parse tree produced by dslParser#metadata.
    def exitMetadata(self, ctx:dslParser.MetadataContext):
        pass


    # Enter a parse tree produced by dslParser#direction.
    def enterDirection(self, ctx:dslParser.DirectionContext):
        pass

    # Exit a parse tree produced by dslParser#direction.
    def exitDirection(self, ctx:dslParser.DirectionContext):
        pass


