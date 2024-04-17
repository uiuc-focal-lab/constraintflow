import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../ast'))

import astcf as AST
import astVisitor
from ir_ast_stack2 import *

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
                    new_lhsIr = IrMult(lhsIr, IrExtractPolyCoeff(rhsIr), op)
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
                    new_rhsIr = IrBinaryOp(rhsIr, IrExtractPolyConst(lhsIr), op)
                    
                else:
                    new_lhsIr = IrBinaryOp(IrExtractPolyCoeff(lhsIr), IrExtractPolyCoeff(rhsIr), op)
                    new_rhsIr = IrBinaryOp(IrExtractPolyConst(lhsIr), IrExtractPolyConst(rhsIr), op)
                
                return IrCombineToPoly(new_lhsIr, new_rhsIr), seqIr
        else:
            if(op in ["*", '/']):
                return IrMult(lhsIr, rhsIr, op), seqIr
            else:
                return IrBinaryOp(lhsIr, rhsIr, op), seqIr
    
    def visitTernary(self, ast_node: AST.TernaryNode):
        condIr, condSeqIr = self.visit(ast_node.cond)
        lhsIr, lhsSeqIr = self.visit(ast_node.left)
        rhsIr, rhsSeqIr = self.visit(ast_node.right)
        seqIr = condSeqIr + lhsSeqIr + rhsSeqIr
        if ast_node.type == 'PolyExp':
            if(lhsIr.irMetadata[-1].type == "Neuron"):
                lhsIr = IrConvertNeuronToPoly(lhsIr)
            elif lhsIr.irMetadata[-1].type == 'Float' or lhsIr.irMetadata[-1].type == 'Int':
                lhsIr = IrConvertConstToPoly(lhsIr)

            if(rhsIr.irMetadata[-1].type == "Neuron"):
                rhsIr = IrConvertNeuronToPoly(rhsIr)
            elif rhsIr.irMetadata[-1].type == 'Float' or rhsIr.irMetadata[-1].type == 'Int':
                rhsIr = IrConvertConstToPoly(rhsIr)

            if (not isinstance(lhsIr, IrVar)) and (not isinstance(lhsIr, IrConst)):
                new_lhs_var = IrVar(self.get_var(), lhsIr.irMetadata)
                lhs_assignment = IrAssignment(new_lhs_var, lhsIr)
                seqIr.append(lhs_assignment)
                lhsIr = new_lhs_var
            if (not isinstance(rhsIr, IrVar)) and (not isinstance(rhsIr, IrConst)):
                new_rhs_var = IrVar(self.get_var(), rhsIr.irMetadata)
                rhs_assignment = IrAssignment(new_rhs_var, rhsIr)
                seqIr.append(rhs_assignment)
                rhsIr = new_rhs_var
            



            lhs_coeff = IrExtractPolyCoeff(lhsIr)
            lhs_const = IrExtractPolyConst(lhsIr)

            rhs_coeff = IrExtractPolyCoeff(rhsIr)
            rhs_const = IrExtractPolyConst(rhsIr)

            new_lhsIr = IrTernary(condIr, lhs_coeff, rhs_coeff)
            new_rhsIr = IrTernary(condIr, lhs_const, rhs_const)
            return IrCombineToPoly(new_lhsIr, new_rhsIr), seqIr
        else:
            return  IrTernary(condIr, lhsIr, rhsIr), seqIr

    def visitDot(self, ast_node):
        lhsIr, lhsSeqIr = self.visit(ast_node.left)
        rhsIr, rhsSeqIr = self.visit(ast_node.right)
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
    
    def visitMap(self, ast_node, polyExpIr=None, polyExpSeqIr = None, name=None, reduce=True):
        if polyExpIr == None:
            polyExpIr, polyExpSeqIr = self.visit(ast_node.expr)
            name = ast_node.func
        seqIr = polyExpSeqIr
        if (not isinstance(polyExpIr, IrVar)) and (not isinstance(polyExpIr, IrConst)):
            new_polyExp_var = IrVar(self.get_var(), polyExpIr.irMetadata)
            new_polyExp_var_assignment = IrAssignment(new_polyExp_var, polyExpIr)
            polyExpIr = new_polyExp_var
            seqIr += [new_polyExp_var_assignment]
        coeffIr = IrMapCoeff(polyExpIr)
        neuronIr = IrMapNeuron(polyExpIr)
        constIr = IrExtractPolyConst(polyExpIr)
        original_store = copy.deepcopy(self.store)
        func_ast_node = self.ast_fstore[name.name]
        assert(len(func_ast_node.decl.arglist.arglist) == 2)
        for (type_node, var_node) in func_ast_node.decl.arglist.arglist:
            if type_node == AST.BaseTypeNode('Neuron'):
                self.store[var_node.name] = neuronIr
            else:
                self.store[var_node.name] = coeffIr
        funcIr, funcSeqIr = self.visit(func_ast_node.expr)
        self.store = original_store
        seqIr += funcSeqIr
        if reduce:
            redIr = IrReduce(funcIr)
            binopIr, binopSeqIr = self.visitBinOp(None, redIr, constIr, redIr.irMetadata[-1].type, '+')
            return binopIr, seqIr + binopSeqIr
        else:
            return funcIr, seqIr
    
    def visitTraverse(self, ast_node):
        self.counter += 1
        exprIr, exprSeqIr = self.visit(ast_node.expr)
        varIr = IrVar('trav_exp'+str(self.counter), exprIr.irMetadata)
        temp = IrAssignment(varIr, exprIr)
        exprSeqIr.append(temp)

        polyExpIr = IrSymbolic(varIr.name, [IrMetadataElement([Int('trav_size')], 'PolyExp', [1], False)])

        stopIr, stopSeqIr = self.visitMap(None, polyExpIr, [], ast_node.stop, False)
        varStopIr = IrVar('vertices_stop', stopIr.irMetadata)
        tempStop = IrAssignment(varStopIr, stopIr)
        stopSeqIr.append(tempStop)
        stopSeqIr.append(IrCustomCodeGen('stop', varIr.name))

        priorityIr, prioritySeqIr = self.visitMap(None, polyExpIr, [], ast_node.priority, False)
        varPriorityIr = IrVar('vertices_priority', priorityIr.irMetadata)
        tempPriority = IrAssignment(varPriorityIr, priorityIr)
        prioritySeqIr.append(tempPriority)
        prioritySeqIr.append(IrCustomCodeGen('priority', varIr.name))

        funcIr, funcSeqIr = self.visitMap(None, polyExpIr, [], ast_node.func, True)
        tempFunc = IrAssignment(varIr, funcIr)
        funcSeqIr.append(tempFunc)

        trav_size_assignment = IrCustomCodeGen('trav_size', varIr.name)

        seqIr = [trav_size_assignment] + stopSeqIr + prioritySeqIr + funcSeqIr
        
        return varIr, exprSeqIr + [IrWhile(IrConst(True, 'Bool'), seqIr)]

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

    # def visitTransRetBasic(self, ast_node):
    #     return IrTransRetBasic([self.visit(expr) for expr in ast_node.exprlist.exprlist])
    
    def visitTransRetBasic(self, ast_node):
        retlist = []
        seqIr = []
        for i in range(len(self.shape.keys())):
            exprIr, exprSeqIr = self.visit(ast_node.exprlist.exprlist[i])
            seqIr += exprSeqIr
            varIr = IrVar(list(self.shape.keys())[i]+'_new', exprIr.irMetadata)
            temp = IrAssignment(varIr, exprIr)
            seqIr.append(temp)
            retlist.append(copy.deepcopy(varIr))
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
                else:
                    assert(False)
            expr.type = new_type
            new_exprs.append(copy.deepcopy(expr))
        new_ast_node = AST.TransRetBasicNode(AST.ExprListNode(new_exprs))
        return new_ast_node

    def visitTransRetIf(self, ast_node):
        ast_node = self.merge_condition(ast_node)
        return self.visit(ast_node)
        # condIr = self.visit(ast_node.cond)
        # leftIrs = self.visit(ast_node.left)
        # rightIrs = self.visit(ast_node.right)
        # return IrTransRetIf(condIr, leftIrs, rightIrs)
    
    def visitOpStmt(self, ast_node):
        original_store = copy.deepcopy(self.store)
        if ast_node.op.op_name == 'Relu':
            self.store['curr'] = IrVar('curr', [IrMetadataElement([CURR_SIZE], 'Neuron', [1], False)])
            self.store['prev'] = IrVar('prev', [IrMetadataElement([CURR_SIZE], 'Neuron', [1], False)])
        elif ast_node.op.op_name == 'Affine':
            self.store['curr'] = IrVar('curr', [IrMetadataElement([CURR_SIZE], 'Neuron', [1], False)])
            self.store['prev'] = IrVar('prev', [IrMetadataElement([1, PREV_SIZE], 'Neuron', [CURR_SIZE, 1], False)])
        else:
            raise Exception('Not Implemented')
        retIr = IrOpStmt(ast_node.op.op_name, self.visit(ast_node.ret))
        self.store = original_store
        return retIr
    
    def visitOpList(self, ast_node):
        return [self.visit(opStmt) for opStmt in ast_node.olist]
    
    def visitTransformer(self, ast_node):
        self.tstore[ast_node.name.name] = self.visit(ast_node.oplist)
    
    def visitFlow(self, ast_node):
        self.counter += 1

        neuronIr = IrSymbolic('flow_var'+str(self.counter), [IrMetadataElement([Int('flow'+str(self.counter))], 'Neuron', [1], False)])
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
        self.visit(ast_node.shape)
        irNodes = self.visit(ast_node.stmt)
        return IrProgram(self.shape, self.tstore, self.fstore, irNodes)