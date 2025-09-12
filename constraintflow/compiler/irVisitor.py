from constraintflow.compiler import ir as IR 

class IRVisitor:
    def visit(self, node):
        if isinstance(node, IR.IrConst):
            return self.visitIrConst(node)
        
        elif isinstance(node, int):
            return self.visitInt(node)
        
        elif isinstance(node, IR.IrVar):
            return self.visitIrVar(node)
        
        elif isinstance(node, IR.IrEpsilon):
            return self.visitIrEpsilon(node)
        
        elif isinstance(node, IR.IrPhi):
            return self.visitIrPhi(node)
        
        elif isinstance(node, IR.IrConvertBoolToFloat):
            return self.visitIrConvertBoolToFloat(node)
        
        elif isinstance(node, IR.IrClamp):
            return self.visitIrClamp(node)
        
        elif isinstance(node, IR.IrGetDefaultStop):
            return self.visitIrGetDefaultStop(node)
        
        elif isinstance(node, IR.IrGetPriorityLList):
            return self.visitIrGetPriorityLList(node)
        
        elif isinstance(node, IR.IrGetPolyexpStop):
            return self.visitIrGetPolyexpStop(node)
        
        elif isinstance(node, IR.IrGetPolyexpNotStop):
            return self.visitIrGetPolyexpNotStop(node)
        
        elif isinstance(node, IR.IrRepeat):
            return self.visitIrRepeat(node)
        
        elif isinstance(node, IR.IrAddDimension):
            return self.visitIrAddDimension(node)
        
        elif isinstance(node, IR.IrRemoveDimension):
            return self.visitIrRemoveDimension(node)
        
        elif isinstance(node, IR.IrAddDimensionConst):
            return self.visitIrAddDimensionConst(node)
        
        elif isinstance(node, IR.IrBinaryOp):
            return self.visitIrBinaryOp(node)
        
        elif isinstance(node, IR.IrUnaryOp):
            return self.visitIrUnaryOp(node)
        
        elif isinstance(node, IR.IrMult):
            return self.visitIrMult(node)
        
        elif isinstance(node, IR.IrInnerProduct):
            return self.visitIrInnerProduct(node)
        
        elif isinstance(node, IR.IrDot):
            return self.visitIrDot(node)
        
        elif isinstance(node, IR.IrTernary):
            return self.visitIrTernary(node)
        
        elif isinstance(node, IR.IrCombineToPoly):
            return self.visitIrCombineToPoly(node)
        
        elif isinstance(node, IR.IrCombineToSym):
            return self.visitIrCombineToSym(node)
        
        elif isinstance(node, IR.IrExtractPolyCoeff):
            return self.visitIrExtractPolyCoeff(node)
        
        elif isinstance(node, IR.IrExtractSymCoeff):
            return self.visitIrExtractSymCoeff(node)
        
        elif isinstance(node, IR.IrExtractPolyConst):
            return self.visitIrExtractPolyConst(node)
        
        elif isinstance(node, IR.IrExtractSymConst):
            return self.visitIrExtractSymConst(node)
        
        elif isinstance(node, IR.IrConvertNeuronToPoly):
            return self.visitIrConvertNeuronToPoly(node)
        
        elif isinstance(node, IR.IrConvertConstToPoly):
            return self.visitIrConvertConstToPoly(node)
        
        elif isinstance(node, IR.IrConvertConstToSym):
            return self.visitIrConvertConstToSym(node)
        
        elif isinstance(node, IR.IrAccess):
            return self.visitIrAccess(node)
        
        elif isinstance(node, IR.IrReduce):
            return self.visitIrReduce(node)
        
        elif isinstance(node, IR.IrMapCoeff):
            return self.visitIrMapCoeff(node)
        
        elif isinstance(node, IR.IrMapNeuron):
            return self.visitIrMapNeuron(node)
        
        elif isinstance(node, IR.IrExpandSymExp):
            return self.visitIrExpandSymExp(node)
        
        elif isinstance(node, IR.IrAssignment):
            return self.visitIrAssignment(node)
        
        elif isinstance(node, IR.IrIte):
            return self.visitIrIte(node)
        
        elif isinstance(node, IR.IrBreak):
            return self.visitIrBreak(node)
        
        elif isinstance(node, IR.IrTransRetBasic):
            return self.visitIrTransRetBasic(node)
        
        elif isinstance(node, IR.IrWhile):
            return self.visitIrWhile(node)
        
        elif isinstance(node, IR.IrBlock):
            return self.visitIrBlock(node)
        
        elif isinstance(node, IR.IrBreak):
            return self.visitIrBreak(node)
        
        # elif isinstance(node, IR.IrCustomCodeGen):
        #     return self.visitIrCustomCodeGen(node)
        
        # elif isinstance(node, IR.IrTransRetIf):
        #     return self.visitIrTransRetIf(node)
        
        # elif isinstance(node, IR.IrTraverse):
        #     return self.visitIrTraverse(node)
        
        elif isinstance(node, IR.IrOpStmt):
            return self.visitIrOpStmt(node)
        
        elif isinstance(node, IR.IrFlow):
            return self.visitIrFlow(node)
        
        elif isinstance(node, IR.IrProgram):
            return self.visitIrProgram(node)
        
        elif isinstance(node, str):
            return self.visitIrStr(node)

        else:
            print("This is an error. This shouldn't happen")
            print(type(node))
            assert False