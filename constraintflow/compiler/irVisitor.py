import torch
from constraintflow.compiler import ir as IR 

class IRVisitor:
    def visit(self, node):
        if isinstance(node, IR.IrRowGather):
            return self.visitIrRowGather(node)
        if isinstance(node, IR.IrSelectedPatchesToDense):
            return self.visitIrSelectedPatchesToDense(node)
        if isinstance(node, IR.IrConst):
            return self.visitIrConst(node)
        
        elif isinstance(node, int):
            return self.visitInt(node)
        
        elif isinstance(node, IR.IrVar):
            return self.visitIrVar(node)
        
        elif isinstance(node, IR.IrEpsilon):
            return self.visitIrEpsilon(node)

        elif isinstance(node, IR.IrNewEps):
            return self.visitIrNewEps(node)

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

        elif isinstance(node, IR.IrConcatStitch):
            return self.visitIrConcatStitch(node)

        elif isinstance(node, IR.IrConcatStitchMat):
            return self.visitIrConcatStitchMat(node)

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

        # For the del statements
        elif isinstance(node, IR.IrDel):
            return self.visitIrDel(node)
        
        elif isinstance(node, IR.IrTransRetBasic):
            return self.visitIrTransRetBasic(node)
        
        elif isinstance(node, IR.IrWhile):
            return self.visitIrWhile(node)
        
        elif isinstance(node, IR.IrParallelBlock):
            return self.visitIrParallelBlock(node)

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

        elif isinstance(node, IR.IrSparseTensor):
            return self.visitIrSparseTensor(node)
        
        elif isinstance(node, IR.IrGetSubBlockCustomRange):
            return self.visitIrGetSubBlockCustomRange(node)
        
        elif isinstance(node, IR.IrEmptyList):
            return self.visitIrEmptyList(node)
        
        elif isinstance(node, IR.IrAppendList):
            return self.visitIrAppendList(node)
        
        elif isinstance(node, IR.IrConstBlock):
            return self.visitIrConstBlock(node)

        elif isinstance(node, IR.IrDenseBlock):
            return self.visitIrDenseBlock(node)

        elif isinstance(node, IR.IrPatchesBlock):
            return self.visitIrPatchesBlock(node)

        elif isinstance(node, IR.IrKernelBlock):
            return self.visitIrKernelBlock(node)

        elif isinstance(node, IR.IrRepeatBlock):
            return self.visitIrRepeatBlock(node)

        elif isinstance(node, IR.IrTorchDiagonal):
            return self.visitIrTorchDiagonal(node)

        elif isinstance(node, IR.IrTorchPermute):
            return self.visitIrTorchPermute(node)

        elif isinstance(node, IR.IrTorchTranspose):
            return self.visitIrTorchTranspose(node)

        elif isinstance(node, IR.IrTorchMatmul):
            return self.visitIrTorchMatmul(node)
        elif isinstance(node, IR.IrTorchWhere):
            return self.visitIrTorchWhere(node)

        elif isinstance(node, IR.IrTorchUnsqueeze):
            return self.visitIrTorchUnsqueeze(node)

        elif isinstance(node, IR.IrTorchSqueeze):
            return self.visitIrTorchSqueeze(node)

        elif isinstance(node, IR.IrTorchReshape):
            return self.visitIrTorchReshape(node)

        elif isinstance(node, IR.IrTorchView):
            return self.visitIrTorchView(node)

        elif isinstance(node, IR.IrTorchRepeat):
            return self.visitIrTorchRepeat(node)

        elif isinstance(node, IR.IrTorchExpand):
            return self.visitIrTorchExpand(node)
        elif isinstance(node, IR.IrTorchPad):
            return self.visitIrTorchPad(node)

        elif isinstance(node, IR.IrTorchSum):
            return self.visitIrTorchSum(node)

        elif isinstance(node, IR.IrTorchZeros):
            return self.visitIrTorchZeros(node)

        elif isinstance(node, IR.IrTorchEye):
            return self.visitIrTorchEye(node)

        elif isinstance(node, IR.IrTorchFloat):
            return self.visitIrTorchFloat(node)

        elif isinstance(node, IR.IrTorchDiagEmbed):
            return self.visitIrTorchDiagEmbed(node)

        elif isinstance(node, IR.IrTorchStride):
            return self.visitIrTorchStride(node)

        elif isinstance(node, IR.IrTorchAsStrided):
            return self.visitIrTorchAsStrided(node)
        elif isinstance(node, IR.IrPatchesToDense):
            return self.visitIrPatchesToDense(node)
        elif isinstance(node, IR.IrTensorScatter):
            return self.visitIrTensorScatter(node)

        elif isinstance(node, IR.IrTorchSlice):
            return self.visitIrTorchSlice(node)

        elif isinstance(node, IR.IrFConv2d):
            return self.visitIrFConv2d(node)

        elif isinstance(node, IR.IrFConvTranspose2d):
            return self.visitIrFConvTranspose2d(node)

        elif isinstance(node, IR.IrFUnfold):
            return self.visitIrFUnfold(node)

        elif isinstance(node, IR.IrFFold):
            return self.visitIrFFold(node)

        elif isinstance(node, IR.IrTorchEinsum):
            return self.visitIrTorchEinsum(node)

        elif isinstance(node, IR.IrTorchCat):
            return self.visitIrTorchCat(node)

        elif isinstance(node, IR.IrAssignToView):
            return self.visitIrAssignToView(node)

        elif isinstance(node, IR.IrDiagonalBlock):
            return self.visitIrDiagonalBlock(node)
        
        elif isinstance(node, IR.IrBlockBinaryOp):
            return self.visitIrBlockBinaryOp(node)
        
        elif isinstance(node, IR.IrBlockWhereBlock):
            return self.visitIrBlockWhereBlock(node)
        
        elif isinstance(node, IR.IrBlockUnaryOp):
            return self.visitIrBlockUnaryOp(node)
        
        elif isinstance(node, IR.IrSimpleUnary):
            return self.visitIrSimpleUnary(node)

        elif isinstance(node, IR.IrLambda):
            return self.visitIrLambda(node)
        
        elif isinstance(node, IR.IrBlockInnerProduct):
            return self.visitIrBlockInnerProduct(node)
        
        elif isinstance(node, torch.Tensor):
            return self.visitTorchTensor(node)
        
        elif isinstance(node, type):
            return self.visitType(node)
        
        elif isinstance(node, float):
            return self.visitFloat(node)
        
        elif isinstance(node, int):
            return self.visitInt(node)
        
        elif isinstance(node, list):
            return self.visitList(node)
        
        elif isinstance(node, IR.IrListExtract):
            return self.visitIrListExtract(node)
        
        elif isinstance(node, IR.IrBlockExtract):
            return self.visitIrBlockExtract(node)

        elif isinstance(node, IR.IrBlockCopy):
            return self.visitIrBlockCopy(node)
        
        elif isinstance(node, IR.IrGetSparseTensorBlocks):
            return self.visitIrGetSparseTensorBlocks(node)
        
        elif isinstance(node, IR.IrGetAbsElemSparseDKey):
            return self.visitIrGetAbsElemSparseDKey(node)
        
        elif isinstance(node, IR.IrGetPolyExpSparseConst):
            return self.visitIrGetPolyExpSparseConst(node)

        elif isinstance(node, IR.IrGetPolyExpSparseMat):
            return self.visitIrGetPolyExpSparseMat(node)

        elif isinstance(node, IR.IrGetSymExpSparseConst):
            return self.visitIrGetSymExpSparseConst(node)

        elif isinstance(node, IR.IrGetSymExpSparseMat):
            return self.visitIrGetSymExpSparseMat(node)

        elif isinstance(node, IR.IrGetKthLayerNetworkParam):
            return self.visitIrGetKthLayerNetworkParam(node)

        elif isinstance(node, IR.IrSimpleBinary):
            return self.visitIrSimpleBinary(node)
        
        elif isinstance(node, IR.IrBlockClamp):
            return self.visitIrBlockClamp(node)

        elif isinstance(node, IR.IrBlockSqueeze):
            return self.visitIrBlockSqueeze(node)

        elif isinstance(node, IR.IrTensorOnes):
            return self.visitIrTensorOnes(node)
        
        elif isinstance(node, IR.IrTensorRepeat):
            return self.visitIrTensorRepeat(node)
        
        elif isinstance(node, IR.IrTensorClamp):
            return self.visitIrTensorClamp(node)

        elif isinstance(node, IR.IrPolyExpMat):
            return self.visitIrPolyExpMat(node)
        
        elif isinstance(node, IR.IrPolyExpNotStopFloat):
            return self.visitIrPolyExpNotStopFloat(node)
        
        elif isinstance(node, IR.IrBlockPolyexpStop):
            return self.visitIrBlockPolyexpStop(node)
        
        elif isinstance(node, IR.IrBlockPolyexpNotStop):
            return self.visitIrBlockPolyexpNotStop(node)

        elif isinstance(node, IR.IrBlockAny):
            return self.visitIrBlockAny(node)

        elif isinstance(node, IR.IrBlockAll):
            return self.visitIrBlockAll(node)

        elif isinstance(node, IR.IrBlockGetDims):
            return self.visitIrBlockGetDims(node)

        elif isinstance(node, IR.IrObjectLookup):
            return self.visitIrObjectLookup(node)

        elif isinstance(node, IR.IrBlockCreateSimilar):
            return self.visitIrBlockCreateSimilar(node)

        elif isinstance(node, IR.IrSetBlockTotalShapeLastDim):
            return self.visitIrSetBlockTotalShapeLastDim(node)

        elif isinstance(node, IR.IrAssignToBlock):
            return self.visitIrAssignToBlock(node)

        elif isinstance(node, IR.IrBlockGetSubBlockCustomRange):
            return self.visitIrBlockGetSubBlockCustomRange(node)

        else:
            print("This is an error. This shouldn't happen")
            print(type(node))
            assert False

    def visitIrParallelBlock(self, node):
        pass
