import ir_ast_stack2 as IR 

class IRVisitor:
    def visit(self, node):
        if isinstance(node, IR.IrConst):
            return self.visitIrConst(node)
        
        elif isinstance(node, IR.IrVar):
            return self.visitIrVar(node)
        
        elif isinstance(node, IR.IrPhi):
            return self.visitIrPhi(node)
        
        elif isinstance(node, IR.IrConvertBoolToFloat):
            return self.visitIrConvertBoolToFloat(node)
        
        elif isinstance(node, IR.IrRepeat):
            return self.visitIrRepeat(node)
        
        elif isinstance(node, IR.IrAddDimension):
            return self.visitIrAddDimension(node)
        
        elif isinstance(node, IR.IrAddDimensionConst):
            return self.visitIrAddDimensionConst(node)
        
        elif isinstance(node, IR.IrBinaryOp):
            return self.visitIrBinaryOp(node)
        
        elif isinstance(node, IR.IrMult):
            return self.visitIrMult(node)
        
        elif isinstance(node, IR.IrDot):
            return self.visitIrDot(node)
        
        elif isinstance(node, IR.IrTernary):
            return self.visitIrTernary(node)
        
        elif isinstance(node, IR.IrCombineToPoly):
            return self.visitIrCombineToPoly(node)
        
        elif isinstance(node, IR.IrExtractPolyCoeff):
            return self.visitIrExtractPolyCoeff(node)
        
        elif isinstance(node, IR.IrExtractPolyConst):
            return self.visitIrExtractPolyConst(node)
        
        elif isinstance(node, IR.IrConvertNeuronToPoly):
            return self.visitIrConvertNeuronToPoly(node)
        
        elif isinstance(node, IR.IrConvertConstToPoly):
            return self.visitIrConvertConstToPoly(node)
        
        elif isinstance(node, IR.IrAccess):
            return self.visitIrAccess(node)
        
        elif isinstance(node, IR.IrReduce):
            return self.visitIrReduce(node)
        
        elif isinstance(node, IR.IrMapCoeff):
            return self.visitIrMapCoeff(node)
        
        elif isinstance(node, IR.IrMapNeuron):
            return self.visitIrMapNeuron(node)
        
        elif isinstance(node, IR.IrSymbolic):
            return self.visitIrSymbolic(node)
        
        elif isinstance(node, IR.IrAssignment):
            return self.visitIrAssignment(node)
        
        elif isinstance(node, IR.IrTransRetBasic):
            return self.visitIrTransRetBasic(node)
        
        elif isinstance(node, IR.IrWhile):
            return self.visitIrWhile(node)
        
        elif isinstance(node, IR.IrCustomCodeGen):
            return self.visitIrCustomCodeGen(node)
        
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

        else:
            print("This is an error. This shouldn't happen")
            print(node)
            assert False