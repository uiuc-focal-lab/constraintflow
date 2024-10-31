from . import irVisitor 
import copy
from .ir import * 

class CodeGen(irVisitor.IRVisitor):
    def __init__(self,folder):
        self.folder = folder 
        self.main_file = folder + '/output/main.py'
        self.transformers_file = folder + '/output/transformers.py'
        self.functions_file = folder + '/output/functions.py'
        self.shape = None
        open(self.main_file, "w").close()
        open(self.transformers_file, "w").close()
        open(self.functions_file, "w").close()

        self.file = open(self.main_file, "a")
        self.indent = 0
        self.write("import torch")
        # self.write("import functools")
        # self.write("import sys")
        # self.write("import pickle")
        self.write("\n")

        self.write("from certifier.lib.spec import *")
        self.write("from certifier.output.flow_sparse import Flow")
        self.write("from certifier.lib.abs_elem import Abs_elem_sparse")
        self.write("from certifier.lib.utils import *")
        # self.write("from certifier.lib.network import LayerType")
        self.write("from certifier.output.transformers import *")

        self.write("\n")
        self.write("def run(network_file, batch_size, eps):")
        self.indent += 1
        # self.write("batch_size = int(sys.argv[2])")
        self.write("network = get_net(network_file)")
        # self.write("input_filename = sys.argv[2]")
        # self.write("with open(input_filename, 'rb') as file:")
        # self.write("\tinput_spec = pickle.load(file)")


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
        shapeDecl = key + ' = torch.tensor([True] + [False]*network.num_layers)'
        self.write(shapeDecl)
        shapeDecl = 'shapes = [layer.shape for layer in network]'
        self.write(shapeDecl)
        shapeDecl = "l, u = get_input_spec(shapes=shapes, n=0, transformer='ibp', eps=eps)"
        self.write(shapeDecl)
        shapeDecl = 'L = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)'
        self.write(shapeDecl)
        shapeDecl = 'U = PolyExpSparse(network, SparseTensorBlock([], [], 3, torch.tensor([batch_size, network.size, network.size])), 0)'
        self.write(shapeDecl)
        # shapeDecl = key + ' = torch.tensor([True, False, False, False, False, False, False])'
        temp_dict += "\'" + key + "\' : " + key + ', '
        self.write(shapeDecl)
        for i, key in enumerate(node.shape.keys()):
            # if self.shape[key] == 'PolyExp':
            #     shapeDecl = key + ' = input_spec[' + str(i+1) + '].convert_to_polyexp_sparse(network, batch_size)'
            # else:
            #     shapeDecl = key + ' = input_spec[' + str(i+1) + ']'
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
            # self.write(shapeDecl)
        temp_dict += '}'

        my_str = "l = convert_to_sparse(l, float(\'-inf\'), network.size, batch_size)"
        self.write(my_str)
        my_str = "u = convert_to_sparse(u, float(\'inf\'), network.size, batch_size)"
        self.write(my_str)
        my_str = "L.const = copy.deepcopy(l)"
        # my_str = "L.const = l"
        self.write(my_str)
        my_str = "U.const = copy.deepcopy(u)"
        # my_str = "U.const = u"
        self.write(my_str)

        my_str = "abs_elem = Abs_elem_sparse(" + temp_dict + ", " + str(temp_shape) + ", network, batch_size=batch_size)"
        self.write(my_str)

        self.open(self.functions_file)


        # TODO - GENERATE STOP AND PRIORITY FUNCTIONS

        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.indent = 0
        self.write('import torch')
        self.write('import copy')
        self.write('from certifier.lib.polyexp import PolyExpSparse, SymExp')
        self.write('from certifier.lib.sparse_tensor import SparseTensorBlock')
        self.write('from certifier.lib.nlist import Llist')
        self.write('from certifier.lib.tensor_ops import *')

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

    def visitIrBreak(self, node):
        self.write('break')

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
            return 'SymExp.count'
        return node.name
    
    def visitIrEpsilon(self, node):
        num = self.visit(node.num)
        return 'SymExp(' + num + ", SymExp.count, torch.zeros(" + num + ", SymExp.count), torch.zeros(" + num + "), SymExp.count, SymExp.count).add_eps(" + num + ")"
    
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
            ret = 'SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=' + str(self.visit(inputIr)) + ', type= type(' + str(self.visit(inputIr)) + '))'
            # ret = 'torch.tensor(' + str(self.visit(inputIr)) + ')'
        else:
            ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(torch.tensor([' + repeat_dims + ']))'
        return ret
    
    def visitIrBinaryOp(self, node):
        op_name = None 
        if node.op == 'max':
            op_name = 'cf_max'
        elif node.op == 'min':
            op_name = 'cf_min'
        elif node.op == '+':
            op_name = 'plus'
        elif node.op == '-':
            op_name = 'minus'
        elif node.op == '<=':
            op_name = 'le'
        elif node.op == '<':
            op_name = 'lt'
        elif node.op == '>=':
            op_name = 'ge'
        elif node.op == '>':
            op_name = 'gt'
        elif node.op == '==':
            op_name = 'eq'
        elif node.op == '!=':
            op_name = 'ne'
        elif node.op == 'and':
            op_name = 'conj'
        elif node.op == 'or':
            op_name = 'disj'
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'
    
    def visitIrUnaryOp(self, node):
        op_name = None 
        if node.op == '-':
            op_name = 'neg'
        elif node.op == 'not':
            op_name = 'boolNeg'
        elif node.op == 'any':
            op_name = 'any'
        elif node.op == 'get_shape_1':
            op_name = 'get_shape_1'
        elif node.op == 'get_shape_0':
            op_name = 'get_shape_0'
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [inputIr] = node.children
        return op_name + '(' + self.visit(inputIr) + ')'
    
    def visitIrGetDefaultStop(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'get_default_stop([' + repeat_dims + '])'
    
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
            op_name = 'mult'
        elif node.op == '/':
            op_name = 'divide'
        else:
            op_name = node.op
        
        [lhsIr, rhsIr] = node.children
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'
    
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

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'poly_size'
        # return 'PolyExpSparse(abs_elem.network, copy.deepcopy(' + self.visit(coeffIr) + ') , copy.deepcopy(' + self.visit(constIr) + '))'
        return 'PolyExpSparse(abs_elem.network, ' + self.visit(coeffIr) + ' , ' + self.visit(constIr) + ')'

    def visitIrCombineToSym(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'SymExp.count'
        rows = self.visit(rows)
        return 'SymExp(' + rows + ', ' + cols + ', ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ', 0, SymExp.count)'


    def visitIrExtractPolyCoeff(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_mat(abs_elem)'
    
    def visitIrExtractSymCoeff(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_mat(SymExp.count)'

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
        cols = 'SymExp.count'
        rows = self.visit(rows)
        return 'SymExp(' + self.visit(rows) + ', ' + cols + ', ' + 'torch.zeros(' + rows + ', ' + cols + '), ' + self.visit(inputIr) + ', 0, SymExp.count)'

    def visitIrAccess(self, node):
        [lhsIr] = node.children
        if not node.isMetadata:
            return 'abs_elem.get_elem_new(\'' + node.elem + '\', ' + self.visit(lhsIr) + ')'
        else:
            return self.visit(lhsIr) + '.get_metadata(\'' + node.elem + '\')'
        
    def visitIrReduce(self, node):
        [inputIr] = node.children
        size = 0
        for i in range(len(inputIr.irMetadata)-1):
            for j in range(len(inputIr.irMetadata[i].broadcast)):
                size += 1
        return self.visit(inputIr) + '.sum(dim=' + str(size) + ')'

    def visitIrMapCoeff(self, node):
        [inputIr] = node.children
        if inputIr.irMetadata[-1].type == 'PolyExp':    
            return self.visit(inputIr) + '.get_mat(abs_elem)'
        return self.visit(inputIr) + '.get_mat(SymExp.count)'

    def visitIrMapNeuron(self, node):
        [inputIr] = node.children
        return 'Llist(abs_elem.network, [1]*(' + self.visit(inputIr) + '.mat.dims-1), None, None,' + "torch.nonzero(abs_elem.d['llist']).flatten().tolist())"

    def visitIrSymbolic(self, node):
        return node.name
    
    def visitIrFlow(self, node):
        self.indent += 1
        self.write('flow = Flow(abs_elem, ' + str(node.transformer) + '(), network, None)')
        self.write('flow.flow()')
        self.indent -= 1
