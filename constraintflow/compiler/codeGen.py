from . import irVisitor 
import copy
from .ir import * 

class CodeGen(irVisitor.IRVisitor):
    def __init__(self,folder):
        self.folder = folder 
        if self.folder.endswith('/'):
            self.folder = self.folder[:-1]
        self.main_file = self.folder + '/main.py'
        self.transformers_file = self.folder + '/transformers.py'
        self.shape = None
        open(self.main_file, "w").close()
        open(self.transformers_file, "w").close()

        self.file = open(self.main_file, "a")
        self.indent = 0

        self.write("import sys")
        self.write("import os")
        self.write("from constraintflow.lib.spec import *")
        self.write("from constraintflow.lib.flow_sparse import Flow")
        self.write("from constraintflow.lib.abs_elem import Abs_elem_sparse")
        self.write("from constraintflow.lib.symexp import *")
        self.write("from transformers import *")
        self.write("\n")
        self.write("def run(network_file, batch_size, eps, dataset_X, dataset_y, dataset, train, print_intermediate_results, no_sparsity):")
        
        self.indent += 1
        self.visited = set()

    def write(self, str, flag=True):
        self.file.write('\t'*self.indent + str)
        if flag:
            self.file.write('\n')

    def write_expr(self, str, flag=True):
        self.file.write(str)
        if flag:
            self.file.write('\n')


    def open(self, file):
        self.file.close()
        self.file = open(file, "a")

    def visitIrProgram(self, node):
        self.shape = node.shape
        temp_shape = copy.deepcopy(self.shape)
        temp_shape['llist'] = 'bool'
        temp_dict = "{"
        key = 'llist'
        i = 0

        temp_dict += "\'" + key + "\' : " + key + ', '
        for i, key in enumerate(node.shape.keys()):
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
        temp_dict += '}'


        self.write("network, l, u, L, U, Z, llist = get_network_and_input_spec(network_file, batch_size, dataset_X, dataset_y, dataset, eps=eps, train=train, no_sparsity=no_sparsity)")
        self.write("abs_elem = Abs_elem_sparse(" + temp_dict + ", " + str(temp_shape) + ", network, batch_size=batch_size, no_sparsity=no_sparsity)")
        



        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.indent = 0
        self.write('import torch')
        self.write('from constraintflow.lib.polyexp import PolyExpSparse')
        self.write('from constraintflow.lib.symexp import *')
        self.write('from constraintflow.gbcsr.sparse_tensor import SparseTensor')
        self.write('from constraintflow.lib.llist import Llist')
        self.write('from constraintflow.gbcsr.tensor_ops import *')

        for i, transformer_name in enumerate(node.tstore.keys()):
            self.write('class ' + transformer_name + ':')
            self.indent += 1

            transformerIr = node.tstore[transformer_name]
            for j, opStmtIr in enumerate(transformerIr):
                self.write('def ' + opStmtIr.op + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):')
                self.indent += 1
                
                cfg = opStmtIr.cfg
                self.visit(cfg.ir[cfg.entry_node])
                self.indent -= 1
                self.write('', True)
            
            self.indent -=1

        self.open(self.main_file)
        for i in range(len(node.irNodes)):
            self.visit(node.irNodes[i])

    def visitIrBlock(self, node):
        if not node in self.visited:
            self.visited.add(node)
            ir_list = node.children
            for counter, i in enumerate(ir_list):
                self.visit(i)
            if node.inner_jump != None:
                if len(node.inner_jump)==3:
                    cond = self.visit(node.inner_jump[0])
                    self.write('if(' + str(cond) + '):')
                    self.indent += 1
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                    self.write('else:')
                    self.indent += 1
                    self.visit(node.inner_jump[2])
                    self.indent -= 1
                elif not isinstance(node.inner_jump[1], IrWhileBlock):
                    cond = self.visit(node.inner_jump[0])
                    self.write('if(' + str(cond) + '):')
                    self.indent += 1
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                else:
                    cond = self.visit(node.inner_jump[0])
                    self.write('while(' + str(cond) + '):')
                    self.indent += 1
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
            if node.jump != None:
                self.visit(node.jump[1])

    def visitIrAssignment(self, node):
        var = str(self.visit(node.children[0]))
        expr = str(self.visit(node.children[1]))
        self.write(var + ' = ' + expr)

    def visitIrBreak(self, node):
        self.write('break')

    def visitIrTransRetBasic(self, node):
        exprs = []
        for i in range(len(node.children)):
            expr = self.visit(node.children[i])
            exprs.append(expr)
        ret_expr = 'return '
        for i in range(len(node.children)-1):
            ret_expr += exprs[i]
            ret_expr += ', '
        ret_expr += exprs[-1]
        self.write(ret_expr)

    def visitIrTransRetIf(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1
        else:
            self.write('pass')
        self.write('return')

    def visitIrIte(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1

    def visitIrWhile(self, node):
        cond = self.visit(node.children[0])
        self.write('while(' + str(cond) + '):')
        self.indent += 1
        for ir in node.children[1:]:
            self.visit(ir)
        self.indent -= 1
    

    def visitIrStr(self, node):
        return node
    
    def visitIrConst(self, node):
        return str(node.const)
    
    def visitInt(self, node):
        return str(node)

    def visitIrVar(self, node):
        if node.name == 'sym_size':
            return 'SymExpSparse.count'
        return node.name
    
    def visitIrEpsilon(self, node):
        # num = self.visit(node.num)
        shape = '['
        for i in range(len(node.irMetadata)):
            for j in range(len(node.irMetadata[i].shape)):
                shape += self.visit(node.irMetadata[i].shape[j]) + ","
        shape += ']'
        return 'get_new_eps(abs_elem.network, torch.tensor(' + shape + '))' 
    
    def visitIrPhi(self, node):
        s = 'phi(['
        for i in range(len(node.children)):
            s += self.visit(node.children[i])
            if i != len(node.children)-1:
                s += ', '
        s += '])'
        return s
    
    def visitIrConvertBoolToFloat(self, node):
        return 'convert_to_float(' + self.visit(node.children[0]) + ')'

    def visitIrRepeat(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        repeat_dims = 'torch.tensor([' + repeat_dims + '])'
        ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ')'
        return ret

    def visitIrAddDimension(self, node):
        [inputIr] = node.children
        size = 0
        for i in range(len(node.irMetadata)-1):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
        size += len(inputIr.irMetadata[-1].shape)
        ret = self.visit(inputIr)
        for i in range(len(node.irMetadata[-1].shape) - len(inputIr.irMetadata[-1].shape)):
            ret += '.unsqueeze(' + str(size) + ')'
            size += 1
        return ret
    
    def visitIrRemoveDimension(self, node):
        [inputIr] = node.children
        ret = self.visit(inputIr) + '.squeeze('+str(node.numDim) + ')'
        return ret

    def visitIrAddDimensionConst(self, node):
        assert(isinstance(node, IrAddDimensionConst))
        inputIr = node.children[0]
        size = len(node.children)-1
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        if inputIr.irMetadata[-1].isConst:
            ret = 'SparseTensor([], [], 0, torch.tensor([]), dense_const=' + str(self.visit(inputIr)) + ', type= type(' + str(self.visit(inputIr)) + '))'
        else:
            ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(torch.tensor([' + repeat_dims + ']))'
        return ret
    
    def visitIrBinaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == 'max':
            op_name = 'cf_max'
            flag = True
        elif node.op == 'min':
            op_name = 'cf_min'
            flag = True
        elif node.op == '+':
            op_name = 'operator.add'
        elif node.op == '-':
            op_name = 'operator.sub'
        elif node.op == '<=':
            op_name = 'operator.le'
        elif node.op == '<':
            op_name = 'operator.lt'
        elif node.op == '>=':
            op_name = 'operator.ge'
        elif node.op == '>':
            op_name = 'operator.gt'
        elif node.op == '==':
            op_name = 'operator.eq'
        elif node.op == '!=':
            op_name = 'operator.ne'
        elif node.op == 'and':
            op_name = 'operator.and_'
        elif node.op == 'or':
            op_name = 'operator.or_'
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        if flag:
            return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'
        else:
            return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ')'
    
    def visitIrUnaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == '-':
            op_name = 'operator.neg'
        elif node.op == 'not':
            op_name = 'operator.not_'
        elif node.op == 'sigma':
            op_name = f"'sigma'"
        elif node.op == 'any':
            op_name = 'any'
            flag = True
        elif node.op == 'all':
            op_name = 'all'
            flag = True
        elif node.op == 'get_dims':
            op_name = 'get_dims'
            flag = True
        elif node.op == 'get_shape_1':
            op_name = 'get_shape_1'
            flag = True
        elif node.op == 'get_shape_0':
            op_name = 'get_shape_0'
            flag = True
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [inputIr] = node.children
        if flag:
            return op_name + '(' + self.visit(inputIr) + ')'
        else:
            return 'unary(' + self.visit(inputIr) + ', ' + op_name + ')'

    def visitIrGetDefaultStop(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'get_default_stop([' + repeat_dims + '], abs_elem, batch_size, curr_size, poly_size)'
    
    def visitIrGetPriorityLList(self, node):
        return 'get_max_priority(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    def visitIrGetPolyexpStop(self, node):
        return 'filter_trav_exp_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    def visitIrGetPolyexpNotStop(self, node):
        return 'filter_trav_exp_not_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    def visitIrConvertToTensor(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'convert_to_tensor(' + self.visit(node.children[0]) + ', [' + repeat_dims +  '])'

    def visitIrMult(self, node):
        op_name = None 
        if node.op == '*':
            op_name = 'operator.mul'
        elif node.op == '/':
            op_name = 'operator.truediv'
        else:
            op_name = node.op
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        return 'binary' + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ')'
    
    def visitIrInnerProduct(self, node):
        op_name = 'inner_prod'
        
        [lhsIr, rhsIr] = node.children
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'

    def visitIrDot(self, node):
        [lhsIr, rhsIr] = node.children
        if lhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(lhsIr) + '.dot(' + self.visit(rhsIr) + ', abs_elem.get_poly_size())'
        elif rhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(rhsIr) + '.dot(' + self.visit(lhsIr) + ', abs_elem.get_poly_size())'
        elif lhsIr.irMetadata[-1].type == 'Float':
            return 'inner_prod(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'
        else:
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        [condIr, lhsIr, rhsIr] = node.children
        return 'where(' + self.visit(condIr) + ', ' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'

    def visitIrClamp(self, node):
        [inputIr, const] = node.children
        min_true = node.min_true 
        return 'clamp(' + self.visit(inputIr) + ', ' + str(min_true) + ', ' + str(const) + ')'

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'poly_size'
        return 'PolyExpSparse(abs_elem.network, ' + self.visit(coeffIr) + ' , ' + self.visit(constIr) + ')'

    def visitIrCombineToSym(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'


    def visitIrExtractPolyCoeff(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_mat(abs_elem)'
    
    def visitIrExtractSymCoeff(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count)'

    def visitIrExtractPolyConst(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_const()'
    
    def visitIrExtractSymConst(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_const()'

    def visitIrConvertNeuronToPoly(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.convert_to_poly(abs_elem)'
    
    def visitIrConvertConstToPoly(self, node):
        [inputIr, rows] = node.children
        cols = 'poly_size'
        return 'PolyExpSparse(abs_elem.network, 0.0, ' + self.visit(inputIr) + ')'
        
    def visitIrConvertConstToSym(self, node):
        [inputIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + 'None, ' + self.visit(inputIr) + ')'
    
    def visitIrExpandSymExp(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.expand_symexp_mat(SymExpSparse.count)'

    def visitIrAccess(self, node):
        [lhsIr] = node.children
        if not node.isMetadata:
            return 'abs_elem.get_elem(\'' + node.elem + '\', ' + self.visit(lhsIr) + ')'
        else:
            return self.visit(lhsIr) + '.get_metadata(\'' + node.elem + '\', batch_size)'
        
    def visitIrReduce(self, node):
        [inputIr] = node.children
        size = node.reduce_dim
        return self.visit(inputIr) + '.sum(dim=' + str(size) + ')'

    def visitIrMapCoeff(self, node):
        [inputIr] = node.children
        if inputIr.irMetadata[-1].type == 'PolyExp':    
            return self.visit(inputIr) + '.get_mat(abs_elem)'
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count)'

    def visitIrMapNeuron(self, node):
        if node.dims:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '), None, None,' + "torch.nonzero(abs_elem.d['llist']).flatten().tolist())"
        else:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '.mat.dims-1), None, None,' + "torch.nonzero(abs_elem.d['llist']).flatten().tolist())"

    def visitIrSymbolic(self, node):
        return node.name
    
    def visitIrFlow(self, node):
        self.indent += 1
        self.write('flow = Flow(abs_elem, ' + str(node.transformer) + '(), network, print_intermediate_results, no_sparsity)')
        self.write('return flow.flow()')
        self.indent -= 1
