from constraintflow.ast_cflow import astcf as AST
from constraintflow.ast_cflow import astVisitor
from constraintflow.compiler.ir import *
from constraintflow.compiler import representations
from constraintflow.compiler.globals import *

class FuncDef:
    def __init__(self, retIr, argIrs):
        self.retIr = retIr
        self.argIrs = argIrs

class ConvertToIr(astVisitor.ASTVisitor):
    def __init__(self):
        self.counter = 0
        self.store = {}
        self.fstore = {}
        self.ast_fstore = {}
        self.tstore = {}
        self.shape = {}
        self.var = -1

    def get_var(self):
        self.var+=1
        return 'var_'+str(self.var)

    def visitInt(self, ast_node: AST.ConstIntNode):
        return IrConst(ast_node.value, 'Int'), [] 
    
    def visitFloat(self, ast_node: AST.ConstFloatNode):
        return IrConst(ast_node.value, 'Float'), []
    
    def visitBool(self, ast_node: AST.ConstBoolNode):
        return IrConst(ast_node.value, 'Bool'), []
    
    def visitVar(self, ast_node: AST.VarNode):
        return self.store[ast_node.name], []
    
    def visitEpsilon(self, ast_node: AST.EpsilonNode):
        irEps = IrEpsilon()
        return irEps, []
    
    def visitUnOp(self, ast_node):
        exprIr, exprSeqIr = self.visit(ast_node.expr)
        return IrUnaryOp(exprIr, ast_node.op), exprSeqIr
    
    def visitMaxOp(self, ast_node: AST.MaxOpNode):
        lhsIr, lhsSeqIr = self.visit(ast_node.expr1)
        rhsIr, rhsSeqIr = self.visit(ast_node.expr2)
        return IrBinaryOp(lhsIr, rhsIr, ast_node.op), lhsSeqIr + rhsSeqIr
        

    def visitBinOp(self, ast_node: AST.BinOpNode, lhsIr=None, rhsIr=None, ast_node_type=None, op=None):
        seqIr = []
        if lhsIr==None:
            lhsIr, lhsSeqIr = self.visit(ast_node.left)
            rhsIr, rhsSeqIr = self.visit(ast_node.right)
            seqIr = lhsSeqIr + rhsSeqIr
            op = ast_node.op
            ast_node_type = ast_node.type
        if ast_node_type == 'PolyExp':
            if(lhsIr.irMetadata[-1].type == "Neuron"):
                lhsIr = IrConvertNeuronToPoly(lhsIr)
            if(rhsIr.irMetadata[-1].type == "Neuron"):
                rhsIr = IrConvertNeuronToPoly(rhsIr)


            if lhsIr.irMetadata[-1].type == 'PolyExp':
                new_lhs_var = IrVar(self.get_var(), lhsIr.irMetadata)
                new_lhs_assignment = IrAssignment(new_lhs_var, lhsIr)
                seqIr.append(new_lhs_assignment)
                lhsIr = new_lhs_var
            if rhsIr.irMetadata[-1].type == 'PolyExp':
                new_rhs_var = IrVar(self.get_var(), rhsIr.irMetadata)
                new_rhs_assignment = IrAssignment(new_rhs_var, rhsIr)
                seqIr.append(new_rhs_assignment)
                rhsIr = new_rhs_var

            
            if(op in ["*", "/"] ):
                
                    

                lhsIrMetadata = lhsIr.irMetadata
                rhsIrMetadata = rhsIr.irMetadata

                if(lhsIrMetadata[-1].type == "Float" or lhsIrMetadata[-1].type == "Int"):
                    set_debug_flag1()
                    rhsIr_coeff = IrExtractPolyCoeff(rhsIr)
                    new_lhsIr = IrMult(lhsIr, rhsIr_coeff, op)
                    reset_debug_flag1()
                    new_rhsIr = IrMult(lhsIr, IrExtractPolyConst(rhsIr), op)
                    
                elif(rhsIrMetadata[-1].type == "Float" or rhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrMult(IrExtractPolyCoeff(lhsIr), rhsIr, op)
                    new_rhsIr = IrMult(IrExtractPolyConst(lhsIr), rhsIr, op)
                return IrCombineToPoly(new_lhsIr, new_rhsIr), seqIr
            else:
                

                lhsIrMetadata = lhsIr.irMetadata
                rhsIrMetadata = rhsIr.irMetadata

                if(lhsIrMetadata[-1].type == "Float" or lhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrExtractPolyCoeff(rhsIr)
                    new_rhsIr = IrBinaryOp(lhsIr, IrExtractPolyConst(rhsIr), op)

                elif(rhsIrMetadata[-1].type == "Float" or rhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrExtractPolyCoeff(lhsIr)
                    new_rhsIr = IrBinaryOp(IrExtractPolyConst(lhsIr), rhsIr, op)
                    
                else:
                    new_lhsIr = IrBinaryOp(IrExtractPolyCoeff(lhsIr), IrExtractPolyCoeff(rhsIr), op)
                    new_rhsIr = IrBinaryOp(IrExtractPolyConst(lhsIr), IrExtractPolyConst(rhsIr), op)
                
                return IrCombineToPoly(new_lhsIr, new_rhsIr), seqIr
            
        elif ast_node_type == 'SymExp':

            if lhsIr.irMetadata[-1].type == 'SymExp':
                new_lhs_var = IrVar(self.get_var(), lhsIr.irMetadata)
                new_lhs_assignment = IrAssignment(new_lhs_var, lhsIr)
                seqIr.append(new_lhs_assignment)
                lhsIr = new_lhs_var
            if rhsIr.irMetadata[-1].type == 'SymExp':
                new_rhs_var = IrVar(self.get_var(), rhsIr.irMetadata)
                new_rhs_assignment = IrAssignment(new_rhs_var, rhsIr)
                seqIr.append(new_rhs_assignment)
                rhsIr = new_rhs_var

            
            if(op in ["*", "/"] ):
                

                lhsIrMetadata = lhsIr.irMetadata
                rhsIrMetadata = rhsIr.irMetadata

                if(lhsIrMetadata[-1].type == "Float" or lhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrMult(lhsIr, IrExtractSymCoeff(rhsIr), op)
                    new_rhsIr = IrMult(lhsIr, IrExtractSymConst(rhsIr), op)
                    
                elif(rhsIrMetadata[-1].type == "Float" or rhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrMult(IrExtractSymCoeff(lhsIr), rhsIr, op)
                    new_rhsIr = IrMult(IrExtractSymConst(lhsIr), rhsIr, op)
                return IrCombineToSym(new_lhsIr, new_rhsIr), seqIr
            else:
                

                lhsIrMetadata = lhsIr.irMetadata
                rhsIrMetadata = rhsIr.irMetadata

                if(lhsIrMetadata[-1].type == "Float" or lhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrExtractSymCoeff(rhsIr)
                    new_rhsIr = IrBinaryOp(lhsIr, IrExtractSymConst(rhsIr), op)

                elif(rhsIrMetadata[-1].type == "Float" or rhsIrMetadata[-1].type == "Int"):
                    new_lhsIr = IrExtractSymCoeff(lhsIr)
                    constIr = IrExtractSymConst(lhsIr)
                    new_rhsIr = IrBinaryOp(constIr, rhsIr, op)
                    
                else:
                    new_lhsIr = IrBinaryOp(IrExtractSymCoeff(lhsIr), IrExtractSymCoeff(rhsIr), op)
                    new_rhsIr = IrBinaryOp(IrExtractSymConst(lhsIr), IrExtractSymConst(rhsIr), op)
                
                return IrCombineToSym(new_lhsIr, new_rhsIr), seqIr
            
        else:
            if(op in ["*", '/']):
                return IrMult(lhsIr, rhsIr, op), seqIr
            else:
                return IrBinaryOp(lhsIr, rhsIr, op), seqIr
    
    def get_conditional_args(self, ast_node, output_node = IrTernary) :
        condIr, condSeqIr = self.visit(ast_node.cond)
        lhsIr, lhsSeqIr = self.visit(ast_node.left)
        rhsIr, rhsSeqIr = self.visit(ast_node.right)
        seqIr = condSeqIr + lhsSeqIr + rhsSeqIr
        if ast_node.type == 'PolyExp':
            if (lhsIr.irMetadata[-1].type == "PolyExp"):
                if (rhsIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(lhsIr.irMetadata)
                    copied_irMetadata[-1].type = rhsIr.irMetadata[-1].type
                    rhsIr = IrAddDimensionConst(rhsIr, copied_irMetadata)
                if (condIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(lhsIr.irMetadata)
                    copied_irMetadata[-1].type = condIr.irMetadata[-1].type
                    condIr = IrAddDimensionConst(condIr, copied_irMetadata)
            
            elif (rhsIr.irMetadata[-1].type == "PolyExp"):
                if (lhsIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(rhsIr.irMetadata)
                    copied_irMetadata[-1].type = lhsIr.irMetadata[-1].type
                    lhsIr = IrAddDimensionConst(lhsIr, copied_irMetadata)
                if (condIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(rhsIr.irMetadata)
                    copied_irMetadata[-1].type = condIr.irMetadata[-1].type
                    condIr = IrAddDimensionConst(condIr, copied_irMetadata)

            if(lhsIr.irMetadata[-1].type == "Neuron"):
                lhsIr = IrConvertNeuronToPoly(lhsIr)
            elif lhsIr.irMetadata[-1].type == 'Float' or lhsIr.irMetadata[-1].type == 'Int':
                lhsIr = IrConvertConstToPoly(lhsIr)

            if(rhsIr.irMetadata[-1].type == "Neuron"):
                rhsIr = IrConvertNeuronToPoly(rhsIr)
            elif rhsIr.irMetadata[-1].type == 'Float' or rhsIr.irMetadata[-1].type == 'Int':
                rhsIr = IrConvertConstToPoly(rhsIr)

            lhs_coeff = IrExtractPolyCoeff(lhsIr)
            lhs_const = IrExtractPolyConst(lhsIr)

            rhs_coeff = IrExtractPolyCoeff(rhsIr)
            rhs_const = IrExtractPolyConst(rhsIr)

            if output_node == IrTernary:
                new_lhsIr = output_node(condIr, lhs_coeff, rhs_coeff)
                new_rhsIr = output_node(condIr, lhs_const, rhs_const)
                return IrCombineToPoly(new_lhsIr, new_rhsIr), seqIr
            else:
                new_var_coeff = IrVar(self.get_var(), lhs_coeff.irMetadata)
                lhs_coeff_output = IrAssignment(new_var_coeff, lhs_coeff)
                lhsSeqIr.append(lhs_coeff_output)

                new_var_const = IrVar(self.get_var(), lhs_const.irMetadata)
                lhs_const_output = IrAssignment(new_var_const, lhs_const)
                lhsSeqIr.append(lhs_const_output)

                rhs_coeff_output = IrAssignment(new_var_coeff, rhs_coeff)
                rhsSeqIr.append(rhs_coeff_output)

                rhs_const_output = IrAssignment(new_var_const, rhs_const)
                rhsSeqIr.append(rhs_const_output)

                new_out = IrIte(condIr, lhsSeqIr, rhsSeqIr)
                return IrCombineToPoly(new_var_coeff, new_var_const), condSeqIr + [new_out]
        elif ast_node.type == 'SymExp':

            if (lhsIr.irMetadata[-1].type == "SymExp"):
                if (rhsIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(lhsIr.irMetadata)
                    copied_irMetadata[-1].type = rhsIr.irMetadata[-1].type
                    rhsIr = IrAddDimensionConst(rhsIr, copied_irMetadata)
                if (condIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(lhsIr.irMetadata)
                    copied_irMetadata[-1].type = condIr.irMetadata[-1].type
                    condIr = IrAddDimensionConst(condIr, copied_irMetadata)
            
            elif (rhsIr.irMetadata[-1].type == "SymExp"):
                if (lhsIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(rhsIr.irMetadata)
                    copied_irMetadata[-1].type = lhsIr.irMetadata[-1].type
                    lhsIr = IrAddDimensionConst(lhsIr, copied_irMetadata)
                if (condIr.irMetadata[-1].isConst):
                    copied_irMetadata = copy_metadata(rhsIr.irMetadata)
                    copied_irMetadata[-1].type = condIr.irMetadata[-1].type
                    condIr = IrAddDimensionConst(condIr, copied_irMetadata)



            if lhsIr.irMetadata[-1].type == 'Float' or lhsIr.irMetadata[-1].type == 'Int':
                lhsIr = IrConvertConstToSym(lhsIr)

            elif rhsIr.irMetadata[-1].type == 'Float' or rhsIr.irMetadata[-1].type == 'Int':
                rhsIr = IrConvertConstToSym(rhsIr)
            
            lhs_coeff = IrExtractSymCoeff(lhsIr)
            lhs_const = IrExtractSymConst(lhsIr)

            rhs_coeff = IrExtractSymCoeff(rhsIr)
            rhs_const = IrExtractSymConst(rhsIr)
            
            if output_node == IrTernary:
                new_lhsIr = output_node(condIr, lhs_coeff, rhs_coeff)
                new_rhsIr = output_node(condIr, lhs_const, rhs_const)
                return IrCombineToSym(new_lhsIr, new_rhsIr), seqIr
            else:
                new_var_coeff = IrVar(self.get_var(), lhs_coeff.irMetadata)
                lhs_coeff_output = IrAssignment(new_var_coeff, lhs_coeff)
                lhsSeqIr.append(lhs_coeff_output)

                new_var_const = IrVar(self.get_var(), lhs_const.irMetadata)
                lhs_const_output = IrAssignment(new_var_const, lhs_const)
                lhsSeqIr.append(lhs_const_output)

                rhs_coeff_output = IrAssignment(new_var_coeff, rhs_coeff)
                rhsSeqIr.append(rhs_coeff_output)

                rhs_const_output = IrAssignment(new_var_const, rhs_const)
                rhsSeqIr.append(rhs_const_output)

                new_out = IrIte(condIr, lhsSeqIr, rhsSeqIr)
                return IrCombineToSym(new_var_coeff, new_var_const), condSeqIr + [new_out]
        else:
            if output_node == IrTernary:
                return output_node(condIr, lhsIr, rhsIr), seqIr
            else:
                new_var = IrVar(self.get_var(), lhsIr.irMetadata)
                lhs_output = IrAssignment(new_var, lhsIr)
                lhsSeqIr.append(lhs_output)

                rhs_output = IrAssignment(new_var, rhsIr)
                rhsSeqIr.append(rhs_output)

                new_out = IrIte(condIr, lhsSeqIr, rhsSeqIr)
                return new_var, condSeqIr + [new_out]

    def visitTernary(self, ast_node: AST.TernaryNode):
        return self.get_conditional_args(ast_node, IrTernary)
    
    def visitIf(self, ast_node: AST.IfNode):
        return self.get_conditional_args(ast_node, IrIte)
        
    

    def visitDot(self, ast_node):
        lhsIr, lhsSeqIr = self.visit(ast_node.left)
        rhsIr, rhsSeqIr = self.visit(ast_node.right)
        assert(len(lhsIr.irMetadata)==1)
        assert(len(rhsIr.irMetadata)==1)
        if lhsIr.irMetadata[-1].type == 'PolyExp':
            coeff = IrExtractPolyCoeff(lhsIr)
            const = IrExtractPolyConst(lhsIr)
            return IrCombineToPoly(IrDot(coeff, rhsIr), IrDot(const, rhsIr)), lhsSeqIr + rhsSeqIr
        elif lhsIr.irMetadata[-1].type == 'Neuron':
            return IrDot(lhsIr, rhsIr), lhsSeqIr + rhsSeqIr
        elif lhsIr.irMetadata[-1].type == 'SymExp':
            coeff = IrRemoveDimension(IrExtractSymCoeff(lhsIr),1)
            const = IrRemoveDimension(IrExtractSymConst(lhsIr),1)
            assert(rhsIr.irMetadata[-1].type=='Float')
            new_irMetadataElement = rhsIr.irMetadata[-1].copy()
            new_irMetadataElement.shape.append(1)
            new_irMetadataElement.broadcast.append(mult_metadata(coeff.irMetadata[-1].shape[-1], coeff.irMetadata[-1].broadcast[-1]))

            return IrCombineToSym(IrDot(rhsIr, coeff), IrDot(rhsIr, const)), lhsSeqIr + rhsSeqIr
        else:
            print(lhsIr.irMetadata[-1].type)
            print(rhsIr.irMetadata[-1].type)
            assert False

        return IrDot(lhsIr, rhsIr), lhsSeqIr + rhsSeqIr
    
    def visitGetElement(self, ast_node: AST.GetElementNode):
        exprIr, exprSeqIr = self.visit(ast_node.expr)
        return IrAccess(exprIr, ast_node.elem.name, self.shape[ast_node.elem.name], False), exprSeqIr

    def visitGetMetadata(self, ast_node: AST.GetMetadataNode):
        exprIr, exprSeqIr = self.visit(ast_node.expr)
        return IrAccess(exprIr, ast_node.metadata.name, None, True), exprSeqIr
    
    def visitFuncCall(self, ast_node):
        seqIr = []
        argIrs = []
        for i in range(len(ast_node.arglist.exprlist)):
            exprIr, exprSeqIr = self.visit(ast_node.arglist.exprlist[i])
            seqIr += exprSeqIr
            if (not isinstance(exprIr, IrVar)) and (not isinstance(exprIr, IrConst)):
                new_var = IrVar(self.get_var(), exprIr.irMetadata)
                new_var_assignment = IrAssignment(new_var, exprIr)
                argIrs.append(new_var)
                seqIr.append(new_var_assignment)
            else:
                argIrs.append(exprIr)
        # argIrs = [self.visit(expr) for expr in ast_node.arglist.exprlist]
        original_store = copy.deepcopy(self.store)
        for i, (type, arg) in enumerate(self.ast_fstore[ast_node.name.name].decl.arglist.arglist):
            self.store[arg.name] = argIrs[i]
        retIr, retSeqIr = self.visit(self.ast_fstore[ast_node.name.name].expr)
        self.store = original_store
        return retIr, seqIr + retSeqIr 

    def visitMap(self, ast_node, inputExpIr=None, inputExpSeqIr = None, name=None, reduce=True, dims=None):
        if inputExpIr == None:
            inputExpIr, inputExpSeqIr = self.visit(ast_node.expr)
            name = ast_node.func
        seqIr = inputExpSeqIr
        if (not isinstance(inputExpIr, IrVar)) and (not isinstance(inputExpIr, IrConst)):
            new_inputExp_var = IrVar(self.get_var(), inputExpIr.irMetadata)
            new_inputExp_var_assignment = IrAssignment(new_inputExp_var, inputExpIr)
            inputExpIr = new_inputExp_var
            seqIr += [new_inputExp_var_assignment]
            
        if inputExpIr.irMetadata[-1].type=='Neuron':
            inputExpIr = IrConvertNeuronToPoly(inputExpIr)
        elif inputExpIr.irMetadata[-1].type=='Noise':
            assert(False)
            # inputExpIr = IrConvertNoiseToSym(inputExpIr)
        coeffIr = IrMapCoeff(inputExpIr, reduce)
        if inputExpIr.irMetadata[-1].type == 'PolyExp':
            if dims:
                neuronIr = IrMapNeuron(inputExpIr, reduce, True, dims)
            else:
                neuronIr = IrMapNeuron(inputExpIr, reduce, False)
            constIr = IrExtractPolyConst(inputExpIr)
        elif inputExpIr.irMetadata[-1].type == 'SymExp':
            neuronIr = IrMapNoise(inputExpIr)
            constIr = IrExtractSymConst(inputExpIr)
        original_store = copy.deepcopy(self.store)
        if isinstance(name.name, str):
            func_ast_node = self.ast_fstore[name.name]
        else:
            func_ast_node = self.ast_fstore[name.name.name]
            #partial function call

            funSeqIr = []
            argIrs = []
            for i in range(len(name.arglist.exprlist)):
                exprIr, exprSeqIr = self.visit(name.arglist.exprlist[i])
                funSeqIr += exprSeqIr
                argIrs.append(exprIr)
            funArgs = func_ast_node.decl.arglist.arglist
            for i, arg in enumerate(argIrs):
                newargIrMetadata = copy.deepcopy(arg).irMetadata
                newargIrMetadata[-1].shape.append(1)
                newargIrMetadata[-1].broadcast.append(IrAst.poly_size)
                if arg.irMetadata[-1].isConst:
                    newarg = IrAddDimensionConst(arg, newargIrMetadata)
                    newarg.irMetadata[-1].isConst = True
                else:
                    newarg = IrAddDimension(arg, newargIrMetadata[-1])
                if(reduce):
                    newarg.irMetadata[-1].shape  = newargIrMetadata[-1].shape[0:-1]
                    newarg.irMetadata[-1].broadcast  = newargIrMetadata[-1].broadcast[0:-1]
                    newarg.irMetadata.append(IrMetadataElement([1], newarg.irMetadata[-1].type, [IrAst.poly_size], newarg.irMetadata[-1].isConst))
                self.store[funArgs[i][1].name] = newarg

        for (i, (type_node, var_node)) in enumerate(func_ast_node.decl.arglist.arglist):
            if i > len(func_ast_node.decl.arglist.arglist) - 3:
                if type_node == AST.BaseTypeNode('Neuron'):
                    self.store[var_node.name] = neuronIr
                elif type_node == AST.BaseTypeNode('Noise'):
                    self.store[var_node.name] = neuronIr
                else:
                    self.store[var_node.name] = coeffIr
        funcIr, funcSeqIr = self.visit(func_ast_node.expr)
        self.store = original_store
        seqIr += funcSeqIr
        if reduce:
            if funcIr.irMetadata[-1].type == 'PolyExp':
                redCoeffIr = IrReduce(IrExtractPolyCoeff(funcIr))
                redConstIr = IrReduce(IrExtractPolyConst(funcIr))
                redIr = IrCombineToPoly(redCoeffIr, redConstIr)
            elif funcIr.irMetadata[-1].type == 'SymExp':
                redCoeffIr = IrReduce(IrExtractSymCoeff(funcIr))
                redConstIr = IrReduce(IrExtractSymConst(funcIr))
                redIr = IrCombineToSym(redCoeffIr, redConstIr)
            else:
                redIr = IrReduce(funcIr)
            binopIr, binopSeqIr = self.visitBinOp(None, redIr, constIr, redIr.irMetadata[-1].type, '+')
            return binopIr, seqIr + binopSeqIr
        else:
            return funcIr, seqIr


    def visitTraverse(self, ast_node):
        self.counter += 1
        exprIr, exprSeqIr = self.visit(ast_node.expr)
        rhs = IrUnaryOp(IrExtractPolyCoeff(exprIr), 'get_shape_1')
        trav_size_var = IrVar('trav_size', rhs.irMetadata)
        trav_size_assignment = IrAssignment(trav_size_var, rhs)
        exprSeqIr.append(trav_size_assignment)

        new_name = 'trav_exp'+str(self.counter)

        if is_expanded_metadata(exprIr.irMetadata):
            polyExpIr = IrVar(new_name, exprIr.irMetadata)
            temp = IrAssignment(polyExpIr, exprIr)
        else:
            coeffIr, constIr = IrExtractPolyCoeff(exprIr), IrExtractPolyConst(exprIr)
            new_coeffIr = IrRepeat(coeffIr, expand_irMetadata(coeffIr.irMetadata))
            new_constIr = IrRepeat(constIr, expand_irMetadata(constIr.irMetadata))
            repeatedIr = IrCombineToPoly(new_coeffIr, new_constIr)
            polyExpIr = IrVar(new_name, repeatedIr.irMetadata)
            temp = IrAssignment(polyExpIr, repeatedIr)

        exprSeqIr.append(temp)

        dims_var_name = 'trav_exp_dims'+str(self.counter)
        dims_var = IrVar(dims_var_name, [IrMetadataElement([1], 'Int', [1], True)])
        temp = IrAssignment(dims_var, IrBinaryOp(IrUnaryOp(IrExtractPolyCoeff(polyExpIr), 'get_dims'), IrConst(1, 'Int'), '-'))
        exprSeqIr.append(temp)
        


        stopIr, stopSeqIr = self.visitMap(None, polyExpIr, [], ast_node.stop, False, dims_var)
        varStopIr = IrVar('vertices_stop'+str(self.counter), stopIr.irMetadata)
        tempStop = IrAssignment(varStopIr, stopIr)
        stopSeqIr.append(tempStop)


        custom_stop_assignments = []
        
        rhs = IrBinaryOp(IrExtractPolyCoeff(polyExpIr), IrConst(0.0, 'Float'), '!=')
        new_var_vertices = IrVar('vertices'+str(self.counter), rhs.irMetadata)
        new_assignment = IrAssignment(new_var_vertices, rhs)
        custom_stop_assignments.append(new_assignment)

        
        rhs = IrGetDefaultStop(polyExpIr)
        new_var = IrVar('vertices_stop_default'+str(self.counter), rhs.irMetadata)
        new_assignment = IrAssignment(new_var, rhs)
        custom_stop_assignments.append(new_assignment)

        varStopTempIr = varStopIr
        rhs = IrBinaryOp(varStopIr, new_var, 'or')
        varStopIr = IrVar('vertices_stop_temp'+str(self.counter), rhs.irMetadata)
        new_assignment = IrAssignment(varStopIr, IrBinaryOp(varStopTempIr, new_var, 'or'))
        custom_stop_assignments.append(new_assignment)

        rhs = IrBinaryOp(new_var_vertices, IrUnaryOp(varStopIr, 'not'), 'and')
        new_assignment = IrAssignment(new_var_vertices, rhs)
        custom_stop_assignments.append(new_assignment)



        cond = IrUnaryOp(IrUnaryOp(new_var_vertices, 'any'), 'not')
        lhs = [IrBreak()]
        rhs = []
        ifStatement = IrIte(cond, lhs, rhs)
        custom_stop_assignments.append(ifStatement)
        stopSeqIr += custom_stop_assignments

        priorityIr, prioritySeqIr = self.visitMap(None, polyExpIr, [], ast_node.priority, False, dims_var)
        varPriorityIr = IrVar('vertices_priority'+str(self.counter), priorityIr.irMetadata)
        tempPriority = IrAssignment(varPriorityIr, priorityIr)
        prioritySeqIr.append(tempPriority)

        rhs = IrGetPriorityLList(varPriorityIr, new_var_vertices)
        new_var = IrVar('priority_vertices'+str(self.counter), rhs.irMetadata)
        new_assignment = IrAssignment(new_var, rhs)
        prioritySeqIr.append(new_assignment)

        new_stop = IrVar('vertices_final_stop'+str(self.counter), varStopIr.irMetadata)
        rhs = IrBinaryOp(varStopIr, IrUnaryOp(new_var, 'not'), 'or')
        new_assignment = IrAssignment(new_stop, rhs)
        prioritySeqIr.append(new_assignment)

        polyexp_stop = IrVar('polyexp_stop'+str(self.counter), polyExpIr.irMetadata)
        rhs = IrGetPolyexpStop(polyExpIr, new_stop)
        new_assignment = IrAssignment(polyexp_stop, rhs)
        prioritySeqIr.append(new_assignment)

        polyexp_not_stop = IrVar('polyexp_not_stop'+str(self.counter), polyExpIr.irMetadata)
        rhs = IrGetPolyexpNotStop(polyExpIr, new_stop)
        new_assignment = IrAssignment(polyexp_not_stop, rhs)
        prioritySeqIr.append(new_assignment)

        funcIr, funcSeqIr = self.visitMap(None, polyexp_not_stop, [], ast_node.func, True, dims_var)
        tempFunc = IrAssignment(polyExpIr, funcIr)
        funcSeqIr.append(tempFunc)


        new_const = IrBinaryOp(IrExtractPolyConst(polyExpIr), IrExtractPolyConst(polyexp_stop), '+')
        new_coeff = IrBinaryOp(IrExtractPolyCoeff(polyExpIr), IrExtractPolyCoeff(polyexp_stop), '+')
        new_assignment = IrAssignment(polyExpIr, IrCombineToPoly(new_coeff, new_const))
        funcSeqIr.append(new_assignment)

        rhs = IrUnaryOp(IrExtractPolyCoeff(polyExpIr), 'get_shape_1')
        trav_size_assignment = IrAssignment(trav_size_var, rhs)

        seqIr = [trav_size_assignment] + stopSeqIr + prioritySeqIr + funcSeqIr
        
        return polyExpIr, exprSeqIr + [IrWhile(IrConst(True, 'Bool'), seqIr)]

    def visitFunc(self, ast_node):
        self.ast_fstore[ast_node.decl.name.name] = ast_node

    def visitShapeDecl(self, ast_node):
        for i, (type, var) in enumerate(ast_node.elements.arglist):
            self.shape[var.name] = type.name 
        
    def visitSeq(self, ast_node):
        lhsIr = self.visit(ast_node.stmt1)
        rhsIr = self.visit(ast_node.stmt2)
        if not isinstance(lhsIr, list):
            if lhsIr == None:
                lhsIr = []
            else:
                lhsIr = [lhsIr]
        if not isinstance(rhsIr, list):
            if rhsIr == None:
                rhsIr = []
            else:
                rhsIr = [rhsIr]
        return lhsIr + rhsIr
    
    def visitTransRetBasic(self, ast_node):
        retlist = []
        seqIr = []
        for i in range(len(self.shape.keys())):
            exprIr, exprSeqIr = self.visit(ast_node.exprlist.exprlist[i])
            seqIr += exprSeqIr

            if exprIr.irMetadata[-1].type != self.shape[list(self.shape.keys())[i]]:
                if exprIr.irMetadata[-1].isConst:
                    exprIr = IrAddDimensionConst(exprIr, [IrMetadataElement([IrAst.batch_size, IrAst.curr_size], exprIr.irMetadata[-1].type, [1,1], False)])
                if self.shape[list(self.shape.keys())[i]] == 'PolyExp':
                    exprIr = IrConvertConstToPoly(exprIr)
                elif self.shape[list(self.shape.keys())[i]] == 'SymExp':
                    exprIr = IrConvertConstToSym(exprIr)
                
            if is_expanded_metadata(exprIr.irMetadata):
                varIr = IrVar(list(self.shape.keys())[i]+'_new', exprIr.irMetadata)
                temp = IrAssignment(varIr, exprIr)
            else:
                if exprIr.irMetadata[-1].type == 'PolyExp':
                    coeffIr, constIr = IrExtractPolyCoeff(exprIr), IrExtractPolyConst(exprIr)               
                    new_coeffIr = IrRepeat(coeffIr, expand_irMetadata(coeffIr.irMetadata))
                    new_constIr = IrRepeat(constIr, expand_irMetadata(constIr.irMetadata))
                    repeatedIr = IrCombineToPoly(new_coeffIr, new_constIr)
                    varIr = IrVar(list(self.shape.keys())[i]+'_new', repeatedIr.irMetadata)
                    temp = IrAssignment(varIr, repeatedIr)
                else:
                    varIr = IrVar(list(self.shape.keys())[i]+'_new', expand_irMetadata(exprIr.irMetadata))
                    temp = IrAssignment(varIr, IrRepeat(exprIr, expand_irMetadata(exprIr.irMetadata)))

            seqIr.append(temp)
            retlist.append(varIr)
        seqIr.append(IrTransRetBasic(retlist))
        return seqIr

    def merge_condition(self, ast_node):
        if isinstance(ast_node, AST.TransRetBasicNode):
            return ast_node 
        lhs = self.merge_condition(ast_node.left)
        rhs = self.merge_condition(ast_node.right)
        new_exprs = []
        for i in range(len(lhs.exprlist.exprlist)):
            expr = AST.TernaryNode(ast_node.cond, lhs.exprlist.exprlist[i], rhs.exprlist.exprlist[i])
            new_type = lhs.exprlist.exprlist[i].type
            if(lhs.exprlist.exprlist[i].type != rhs.exprlist.exprlist[i].type):
                if (lhs.exprlist.exprlist[i].type in ['Int', 'Float']) and (rhs.exprlist.exprlist[i].type in ['Int', 'Float']):
                    new_type = 'Float'
                elif (lhs.exprlist.exprlist[i].type in ['Neuron', 'Float', 'Int', 'PolyExp']) and (rhs.exprlist.exprlist[i].type in ['Neuron', 'Float', 'Int', 'PolyExp']):
                    new_type = 'PolyExp'
                elif (lhs.exprlist.exprlist[i].type in ['Noise', 'Float', 'Int', 'SymExp']) and (rhs.exprlist.exprlist[i].type in ['Noise', 'Float', 'Int', 'SymExp']):
                    new_type = 'SymExp'
                else:
                    assert(False)
            expr.type = new_type
            new_exprs.append(copy.deepcopy(expr))
        new_ast_node = AST.TransRetBasicNode(AST.ExprListNode(new_exprs))
        return new_ast_node

    def visitTransRetIf(self, ast_node):
        condIr = self.visit(ast_node.cond)
        leftIrs = self.visit(ast_node.left)
        rightIrs = self.visit(ast_node.right)
        
        new_var = IrVar(self.get_var(), IrUnaryOp(condIr[0], 'any').irMetadata)
        new_assignment = IrAssignment(new_var, IrUnaryOp(condIr[0], 'any'))
        return condIr[1] + [new_assignment, IrIte(new_var, leftIrs, rightIrs)]
    
    def visitOpStmt(self, ast_node):
        original_store = copy.deepcopy(self.store)
        if ast_node.op.op_name == 'Relu' or ast_node.op.op_name == 'Abs' or ast_node.op.op_name == 'HardSigmoid' or ast_node.op.op_name == 'HardTanh' or ast_node.op.op_name == 'Sigmoid':
            self.store['curr'] = IrVar('curr', [IrMetadataElement([1, IrAst.curr_size], 'Neuron', [IrAst.batch_size, 1], False)])
            self.store['prev'] = IrVar('prev', [IrMetadataElement([1, IrAst.curr_size], 'Neuron', [IrAst.batch_size, 1], False)])
        elif ast_node.op.op_name == 'Affine':
            self.store['curr'] = IrVar('curr', [IrMetadataElement([1, IrAst.curr_size], 'Neuron', [IrAst.batch_size, 1], False)])
            self.store['prev'] = IrVar('prev', [IrMetadataElement([1, 1, IrAst.prev_size], 'Neuron', [IrAst.batch_size, IrAst.curr_size, 1], False)])
        else:
            raise Exception('Not Implemented')
        temp = self.visit(ast_node.ret)
        
        cfg = representations.create_cfg(temp)
        retIr = IrOpStmt(ast_node.op.op_name, cfg)
        self.store = original_store
        
        return retIr
    
    def visitOpList(self, ast_node):
        return [self.visit(opStmt) for opStmt in ast_node.olist]
    
    def visitTransformer(self, ast_node):
        self.tstore[ast_node.name.name] = self.visit(ast_node.oplist)
    
    def visitFlow(self, ast_node):
        self.counter += 1

        neuronIr = IrVar('flow_var'+str(self.counter), [IrMetadataElement([Int('flow'+str(self.counter))], 'Neuron', [1], False)])
        self.counter += 1
        original_store = copy.deepcopy(self.store)
        
        stop_ast_node = self.ast_fstore[ast_node.sfunc.name]
        assert(len(stop_ast_node.decl.arglist.arglist) == 1)
        for (type_node, var_node) in stop_ast_node.decl.arglist.arglist:
            if type_node == AST.BaseTypeNode('Neuron'):
                self.store[var_node.name] = neuronIr 
        stopIr = self.visit(stop_ast_node.expr)
        self.fstore[ast_node.sfunc.name] = (stopIr, [(var_node.name, 'Neuron')])

        priority_ast_node = self.ast_fstore[ast_node.pfunc.name]
        assert(len(priority_ast_node.decl.arglist.arglist) == 1)
        for (type_node, var_node) in priority_ast_node.decl.arglist.arglist:
            if type_node == AST.BaseTypeNode('Neuron'):
                self.store[var_node.name] = neuronIr 
        priorityIr = self.visit(priority_ast_node.expr)
        self.fstore[ast_node.pfunc.name] = (priorityIr, [(var_node.name, 'Neuron')])

        self.store = original_store

        return IrFlow(ast_node.sfunc.name, ast_node.pfunc.name, ast_node.trans.name, ast_node.direction.value)

    def visitProg(self, ast_node):
        _ = IrVar('_', [])
        self.visit(ast_node.shape)
        irNodes = self.visit(ast_node.stmt)
        return IrProgram(self.shape, self.tstore, self.fstore, irNodes)