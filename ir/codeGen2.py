import irVisitor 
import copy
from ir_ast_stack2 import * 

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
        self.write("from certifier import Certifier")
        self.write("from common.abs_elem import Abs_elem")
        self.write("from common.transformer import *")
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
        temp_shape['t'] = 'bool'
        temp_dict = "{"
        key = 't'
        i = 0
        shapeDecl = key + ' = input_spec[' + str(i) + ']'
        temp_dict += "\'" + key + "\' : " + key + ', '
        self.write(shapeDecl)
        for i, key in enumerate(node.shape.keys()):
            shapeDecl = key + ' = input_spec[' + str(i+1) + ']'
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
            self.write(shapeDecl)
        temp_dict += '}'

        my_str = "abs_elem = Abs_elem(" + temp_dict + ", " + str(temp_shape) + ", shapes)"
        self.write(my_str)

        self.open(self.functions_file)


        # TODO - GENERATE STOP AND PRIORITY FUNCTIONS

        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.write('import torch')
        self.write('from common.polyexp import PolyExpNew, Nlist')
        self.write('from utils import *')

        for i, transformer_name in enumerate(node.tstore.keys()):
            self.write('class ' + transformer_name + ':')
            self.indent += 1

            transformerIr = node.tstore[transformer_name]
            for j, opStmtIr in enumerate(transformerIr):
                self.write('def ' + opStmtIr.op + '(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):')
                self.indent += 1
                
                for ir in opStmtIr.children:
                    self.visit(ir)

                # self.visit(opStmtIr.inputIr)
                self.indent -= 1
                self.write('', True)
            
            self.indent -=1

        self.open(self.main_file)
        for i in range(len(node.irNodes)):
            self.visit(node.irNodes[i])

    def visitIrAssignment(self, node):
        var = str(self.visit(node.children[0]))
        expr = str(self.visit(node.children[1]))
        # print(var)
        # print(expr)
        self.write(var + ' = ' + expr)

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

    def visitIrWhile(self, node):
        cond = self.visit(node.children[0])
        self.write('while(' + str(cond) + '):')
        self.indent += 1
        for ir in node.children[1:]:
            self.visit(ir)
        self.indent -= 1

    def visitIrCustomCodeGen(self, node):
        if node.documentation == 'stop':
            self.write('vertices = ' + self.visit(node.children[0]) + '.mat != 0')
            self.write('vertices_stop = convert_to_tensor(vertices_stop, poly_size) != True')
            self.write('vertices_stop_default = torch.zeros(vertices_stop.size())')
            self.write('vertices_stop_default[0:input_size] = 1')
            self.write('vertices_stop_default = vertices_stop_default.bool()')
            self.write('vertices_stop = vertices_stop | vertices_stop_default')
            self.write('vertices = vertices & (~ vertices_stop)')
            self.write('if not(vertices.any()):')
            self.write('\tbreak')
        elif node.documentation == 'priority':
            print('CODE GEN FOR PRIORITY NOT IMPLEMENTED')
        elif node.documentation == 'trav_size':
            self.write('trav_size = ' + self.visit(node.children[0]) + '.const.shape[0]')
        else:
            raise Exception('NOT IMPLEMENTED')
        
    

    def visitIrConst(self, node):
        return node.const

    def visitIrVar(self, node):
        return node.name
    
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
        [inputIr] = node.children
        repeat_dims = ''
        for i in range(len(inputIr.irMetadata)):
            for j in range(len(inputIr.irMetadata[i].broadcast)):
                repeat_dims += ' ' + str(inputIr.irMetadata[i].broadcast[j]) + ','
        repeat_dims = repeat_dims[:-1]
        ret = self.visit(node.children[0])
        ret += '.repeat(' + repeat_dims + ')'
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

    def visitIrAddDimensionConst(self, node):
        assert(isinstance(node, IrAddDimensionConst))
        [inputIr] = node.children
        size = 0
        repeat_dims = ''
        for i in range(len(node.irMetadata)):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
                repeat_dims += ' ' + str(node.irMetadata[i].broadcast[j]) + '*' + str(node.irMetadata[i].shape[j]) + ','
        repeat_dims = repeat_dims[:-1]
        if inputIr.irMetadata[-1].isConst:
            ret = 'torch.tensor(' + str(self.visit(inputIr)) + ')'
        else:
            ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(' + repeat_dims + ')'
        return ret

    def visitIrBinaryOp(self, node):
        op_name = None 
        if node.op == '+':
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
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'

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

    def visitIrDot(self, node):
        [lhsIr, rhsIr] = node.children
        if lhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(lhsIr) + '.dot(' + self.visit(rhsIr) + ')'
        elif rhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(rhsIr) + '.dot(' + self.visit(lhsIr) + ')'
        else:
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        [condIr, lhsIr, rhsIr] = node.children
        return 'torch.where(' + self.visit(condIr) + ', ' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ')'

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr] = node.children
        return 'PolyExpNew(poly_size, ' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'

    def visitIrExtractPolyCoeff(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_mat(abs_elem)'

    def visitIrExtractPolyConst(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.get_const()'

    def visitIrConvertNeuronToPoly(self, node):
        [inputIr] = node.children
        return self.visit(inputIr) + '.convert_to_poly()'
    
    def visitIrConvertConstToPoly(self, node):
        [inputIr] = node.children
        return 'PolyExpNew(poly_size, None, ' + str(self.visit(inputIr)) + ')'

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
        return self.visit(inputIr) + '.get_mat(abs_elem)'

    def visitIrMapNeuron(self, node):
        return 'abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))'

    def visitIrSymbolic(self, node):
        return node.name
    
    def visitIrFlow(self, node):
        self.write('certifier = Certifier(abs_elem, ' + str(node.transformer) + '(), network, None)')
        self.write('certifier.flow()')