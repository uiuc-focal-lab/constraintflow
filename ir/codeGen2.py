from . import irVisitor 
import copy
from .ir_ast_stack2 import * 

class CodeGen(irVisitor.IRVisitor):
    def __init__(self,folder):
        self.folder = folder 
        self.main_file = folder + '/main_compiled2.py'
        self.transformers_file = folder + '/transformers_compiled2.py'
        self.functions_file = folder + '/functions_compiled.py'
        self.shape = None
        open(self.main_file, "w").close()
        open(self.transformers_file, "w").close()
        open(self.functions_file, "w").close()

        self.file = open(self.main_file, "a")
        self.indent = 0
        self.write("import torch")
        self.write("import functools")
        self.write("import sys")
        self.write("import pickle")
        self.write("\n")

        self.write("from specs.spec import *")
        self.write("from certifier_sparse import Certifier")
        self.write("from common.abs_elem import Abs_elem_sparse")
        self.write("from common.polyexp import Network_graph")
        self.write("from specs.network import LayerType")
        self.write("from transformers_compiled2 import *")

        self.write("\n")
        self.write("network_file = sys.argv[1]")
        self.write("network = get_net(network_file)")
        self.write("input_filename = sys.argv[2]")
        self.write("with open(input_filename, 'rb') as file:")
        self.write("\tinput_spec = pickle.load(file)")

        self.write("shapes = [network.input_shape]")
        self.write("for layer in network:")
        self.indent += 1
        self.write("shapes.append(layer.shape)")
        self.indent -= 1
        self.write("network = Network_graph(shapes, network)")

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
        shapeDecl = key + ' = torch.tensor([True, False, False, False, False, False, False])'
        temp_dict += "\'" + key + "\' : " + key + ', '
        self.write(shapeDecl)
        for i, key in enumerate(node.shape.keys()):
            if self.shape[key] == 'PolyExp':
                shapeDecl = key + ' = input_spec[' + str(i+1) + '].convert_to_polyexp_sparse(network)'
            else:
                shapeDecl = key + ' = input_spec[' + str(i+1) + ']'
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
            self.write(shapeDecl)
        temp_dict += '}'

        my_str = "l = convert_to_sparse(l, float(\'-inf\'))"
        self.write(my_str)
        my_str = "u = convert_to_sparse(u, float(\'inf\'))"
        self.write(my_str)
        my_str = "L.const = copy.deepcopy(l)"
        self.write(my_str)
        my_str = "U.const = copy.deepcopy(u)"
        self.write(my_str)

        my_str = "abs_elem = Abs_elem_sparse(" + temp_dict + ", " + str(temp_shape) + ", network)"
        self.write(my_str)

        self.open(self.functions_file)


        # TODO - GENERATE STOP AND PRIORITY FUNCTIONS

        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.write('import torch')
        self.write('import copy')
        self.write('from common.polyexp import PolyExpSparse, SymExp')
        self.write('from common.sparse_tensor import SparseTensorBlock')
        self.write('from common.nlist import Llist')
        self.write('from utils import *')

        for i, transformer_name in enumerate(node.tstore.keys()):
            self.write('class ' + transformer_name + ':')
            self.indent += 1

            transformerIr = node.tstore[transformer_name]
            for j, opStmtIr in enumerate(transformerIr):
                self.write('def ' + opStmtIr.op + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):')
                self.indent += 1
                
                cfg = opStmtIr.cfg
                self.visit(cfg.ir[cfg.entry_node])
                # for ir in opStmtIr.children:
                #     self.visit(ir)

                # self.visit(opStmtIr.inputIr)
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
                print(node.jump)
                self.visit(node.jump[1])

    def visitIrBreak(self, node):
        self.write('break')

    def visitIrAssignment(self, node):
        var = str(self.visit(node.children[0]))
        expr = str(self.visit(node.children[1]))
        # print(var)
        # print(expr)
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

    # def visitIrCustomCodeGen(self, node):
    #     if node.documentation == 'stop':
    #         self.write('vertices = ' + self.visit(node.children[0]) + '.mat != 0')
    #         self.write('vertices_stop = convert_to_tensor(vertices_stop, poly_size) != True')
    #         self.write('vertices_stop_default = torch.zeros(vertices_stop.size())')
    #         self.write('vertices_stop_default[0:input_size] = 1')
    #         self.write('vertices_stop_default = vertices_stop_default.bool()')
    #         self.write('vertices_stop = vertices_stop | vertices_stop_default')
    #         self.write('vertices = vertices & (~ vertices_stop)')
    #         self.write('if not(vertices.any()):')
    #         self.write('\tbreak')
    #     elif node.documentation == 'priority':
    #         print('CODE GEN FOR PRIORITY NOT IMPLEMENTED')
    #     elif node.documentation == 'trav_size':
    #         self.write('trav_size = ' + self.visit(node.children[0]) + '.const.shape[0]')
    #     else:
    #         raise Exception('NOT IMPLEMENTED')
        
    

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
        # inputIr = node.children[0]
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        # for i in range(len(inputIr.irMetadata)):
        #     for j in range(len(inputIr.irMetadata[i].broadcast)):
        #         repeat_dims += ' ' + str(inputIr.irMetadata[i].broadcast[j]) + ','
        # repeat_dims = repeat_dims[:-1]
        repeat_dims = 'torch.tensor([' + repeat_dims + '])'
        ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ')'
        # ret = self.visit(node.children[0])
        # ret += '.repeat(' + repeat_dims + ')'
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
        # repeat_dims = node.children[1:]
        size = len(node.children)-1
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        # print(inputIr.irMetadata)
        if inputIr.irMetadata[-1].isConst:
            ret = 'torch.tensor(' + str(self.visit(inputIr)) + ')'
        else:
            ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(' + repeat_dims + ')'
        return ret

    # def visitIrBinaryPolyExpOp(self, node):
    #     if node.op == '+':
    #         op_name = 'sum'
    #     elif node.op == '-':
    #         op_name = 'diff'
    #     elif node.op == '*':
    #         op_name = 'mult'
    #     elif node.op == '/':
    #         op_name = 'divide'
    #     else:
    #         raise Exception('NOT IMPLMENTED')
    #     [lhsIr, rhsIr] = node.children   
    #     return self.visit(lhsIr) + '.' + op_name + '(' + self.visit(rhsIr) + ')' 
    
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
        # def get_shape(irMetadata):
        #     shape = []
        #     for i in irMetadata:
        #         shape += i.shape
        #     return shape
        
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'get_default_stop([' + repeat_dims + '])'
    
    def visitIrConvertToTensor(self, node):
        # def get_shape(irMetadata):
        #     shape = []
        #     for i in irMetadata:
        #         shape += i.shape
        #     return shape
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
            print(lhsIr.irMetadata[-1].type)
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        [condIr, lhsIr, rhsIr] = node.children
        return 'where(' + self.visit(condIr) + ', ' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr, rows] = node.children
        # rows = str(constIr.irMetadata[-1].shape[0])
        cols = 'poly_size'
        return 'PolyExpSparse(abs_elem.network, copy.deepcopy(' + self.visit(coeffIr) + ') , copy.deepcopy(' + self.visit(constIr) + '))'
        # return 'PolyExpSparse(abs_elem.network, ' + self.visit(rows) + ', copy.deepcopy(' + self.visit(coeffIr) + '.dense_layers), copy.deepcopy(' + self.visit(coeffIr) + '.mats), copy.deepcopy(' + self.visit(constIr) + '.const))'
        # return 'PolyExp(' + self.visit(rows) + ', ' + cols + ', ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'
        # return 'PolyExpNew(poly_size, ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'

    def visitIrCombineToSym(self, node):
        [coeffIr, constIr, rows] = node.children
        # rows = str(constIr.irMetadata[-1].shape[0])
        cols = 'SymExp.count'
        rows = self.visit(rows)
        return 'SymExp(' + rows + ', ' + cols + ', ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ', 0, SymExp.count)'
        # return 'PolyExpNew(poly_size, ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'


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
        # rows = str(inputIr.irMetadata[-1].shape[0])
        cols = 'SymExp.count'
        rows = self.visit(rows)
        return 'SymExp(' + self.visit(rows) + ', ' + cols + ', ' + 'torch.zeros(' + rows + ', ' + cols + '), ' + self.visit(inputIr) + ', 0, SymExp.count)'
        # return 'PolyExpNew(poly_size, None, ' + str(self.visit(inputIr)) + ')'

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
        return 'Llist(abs_elem.network, [1]*(' + self.visit(inputIr) + '.mat.dims-1), None, None,' + self.visit(inputIr) + '.get_dense_layers())'
        # return self.visit(inputIr) + '.get_dense_layers()'
        # return 'abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))'

    def visitIrSymbolic(self, node):
        return node.name
    
    def visitIrFlow(self, node):
        self.write('certifier = Certifier(abs_elem, ' + str(node.transformer) + '(), network, None)')
        self.write('certifier.flow()')