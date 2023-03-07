# Generated from dsl.g4 by ANTLR 4.12.0
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


    # Enter a parse tree produced by dslParser#sub.
    def enterSub(self, ctx:dslParser.SubContext):
        pass

    # Exit a parse tree produced by dslParser#sub.
    def exitSub(self, ctx:dslParser.SubContext):
        pass


    # Enter a parse tree produced by dslParser#getMetadata.
    def enterGetMetadata(self, ctx:dslParser.GetMetadataContext):
        pass

    # Exit a parse tree produced by dslParser#getMetadata.
    def exitGetMetadata(self, ctx:dslParser.GetMetadataContext):
        pass


    # Enter a parse tree produced by dslParser#prev.
    def enterPrev(self, ctx:dslParser.PrevContext):
        pass

    # Exit a parse tree produced by dslParser#prev.
    def exitPrev(self, ctx:dslParser.PrevContext):
        pass


    # Enter a parse tree produced by dslParser#false.
    def enterFalse(self, ctx:dslParser.FalseContext):
        pass

    # Exit a parse tree produced by dslParser#false.
    def exitFalse(self, ctx:dslParser.FalseContext):
        pass


    # Enter a parse tree produced by dslParser#dot.
    def enterDot(self, ctx:dslParser.DotContext):
        pass

    # Exit a parse tree produced by dslParser#dot.
    def exitDot(self, ctx:dslParser.DotContext):
        pass


    # Enter a parse tree produced by dslParser#sum.
    def enterSum(self, ctx:dslParser.SumContext):
        pass

    # Exit a parse tree produced by dslParser#sum.
    def exitSum(self, ctx:dslParser.SumContext):
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


    # Enter a parse tree produced by dslParser#int.
    def enterInt(self, ctx:dslParser.IntContext):
        pass

    # Exit a parse tree produced by dslParser#int.
    def exitInt(self, ctx:dslParser.IntContext):
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


    # Enter a parse tree produced by dslParser#traverse.
    def enterTraverse(self, ctx:dslParser.TraverseContext):
        pass

    # Exit a parse tree produced by dslParser#traverse.
    def exitTraverse(self, ctx:dslParser.TraverseContext):
        pass


    # Enter a parse tree produced by dslParser#not.
    def enterNot(self, ctx:dslParser.NotContext):
        pass

    # Exit a parse tree produced by dslParser#not.
    def exitNot(self, ctx:dslParser.NotContext):
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


    # Enter a parse tree produced by dslParser#nlistOp.
    def enterNlistOp(self, ctx:dslParser.NlistOpContext):
        pass

    # Exit a parse tree produced by dslParser#nlistOp.
    def exitNlistOp(self, ctx:dslParser.NlistOpContext):
        pass


    # Enter a parse tree produced by dslParser#funcCall.
    def enterFuncCall(self, ctx:dslParser.FuncCallContext):
        pass

    # Exit a parse tree produced by dslParser#funcCall.
    def exitFuncCall(self, ctx:dslParser.FuncCallContext):
        pass


    # Enter a parse tree produced by dslParser#curr.
    def enterCurr(self, ctx:dslParser.CurrContext):
        pass

    # Exit a parse tree produced by dslParser#curr.
    def exitCurr(self, ctx:dslParser.CurrContext):
        pass


    # Enter a parse tree produced by dslParser#map.
    def enterMap(self, ctx:dslParser.MapContext):
        pass

    # Exit a parse tree produced by dslParser#map.
    def exitMap(self, ctx:dslParser.MapContext):
        pass


    # Enter a parse tree produced by dslParser#func_op.
    def enterFunc_op(self, ctx:dslParser.Func_opContext):
        pass

    # Exit a parse tree produced by dslParser#func_op.
    def exitFunc_op(self, ctx:dslParser.Func_opContext):
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


    # Enter a parse tree produced by dslParser#ptop.
    def enterPtop(self, ctx:dslParser.PtopContext):
        pass

    # Exit a parse tree produced by dslParser#ptop.
    def exitPtop(self, ctx:dslParser.PtopContext):
        pass


    # Enter a parse tree produced by dslParser#ptbasic.
    def enterPtbasic(self, ctx:dslParser.PtbasicContext):
        pass

    # Exit a parse tree produced by dslParser#ptbasic.
    def exitPtbasic(self, ctx:dslParser.PtbasicContext):
        pass


    # Enter a parse tree produced by dslParser#ptout.
    def enterPtout(self, ctx:dslParser.PtoutContext):
        pass

    # Exit a parse tree produced by dslParser#ptout.
    def exitPtout(self, ctx:dslParser.PtoutContext):
        pass


    # Enter a parse tree produced by dslParser#ptin.
    def enterPtin(self, ctx:dslParser.PtinContext):
        pass

    # Exit a parse tree produced by dslParser#ptin.
    def exitPtin(self, ctx:dslParser.PtinContext):
        pass


    # Enter a parse tree produced by dslParser#propsingle.
    def enterPropsingle(self, ctx:dslParser.PropsingleContext):
        pass

    # Exit a parse tree produced by dslParser#propsingle.
    def exitPropsingle(self, ctx:dslParser.PropsingleContext):
        pass


    # Enter a parse tree produced by dslParser#propparen.
    def enterPropparen(self, ctx:dslParser.PropparenContext):
        pass

    # Exit a parse tree produced by dslParser#propparen.
    def exitPropparen(self, ctx:dslParser.PropparenContext):
        pass


    # Enter a parse tree produced by dslParser#propdouble.
    def enterPropdouble(self, ctx:dslParser.PropdoubleContext):
        pass

    # Exit a parse tree produced by dslParser#propdouble.
    def exitPropdouble(self, ctx:dslParser.PropdoubleContext):
        pass



del dslParser