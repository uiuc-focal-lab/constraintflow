import irVisitor 
import copy
# import ir_ast_stack as IR 

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
                
                for ir in opStmtIr.inputIr:
                    self.visit(ir)

                # self.visit(opStmtIr.inputIr)
                self.indent -= 1
                self.write('', True)
            
            self.indent -=1

        self.open(self.main_file)
        for i in range(len(node.irNodes)):
            self.visit(node.irNodes[i])

    def visitIrAssignment(self, node):
        self.write(node.varIr.name + ' = ', False)
        self.visit(node.inputIr)
        self.write('', True)

    def visitIrTransRetBasic(self, node):
        self.write('return ', False)
        for i, shape in enumerate(self.shape.keys()):
            self.visit(node.exprIrs[i])
            if i<len(self.shape.keys())-1:
                self.write_expr(', ', False)
        self.write('', True)

    def visitIrTransRetIf(self, node):
        # SHOULD NEVER REACH HERE
        pass
        # THE FOLLOWING IS WRONG. WHAT IF THE CONDITION IS A TENSOR OF TRUE ANF FALSE INSTEAD OF JUST ONE TRUE AND FALSE?
        # self.write('if ', False)
        # self.visit(node.condIr)
        # self.write_expr(':')
        # self.indent += 1 
        # self.visit(node.lhsIrs)
        # self.indent -= 1
        # self.write('else:')
        # self.indent += 1
        # self.visit(node.rhsIrs)
        # self.indent -= 1

    def visitIrConst(self, node):
        self.write_expr(str(node.const), False)

    def visitIrVar(self, node):
        self.write_expr(node.name, False)

    def visitIrRepeat(self, node):
        repeat_dims = ''
        for i in range(len(node.inputIr.irMetadata)):
            for j in range(len(node.inputIr.irMetadata[i].broadcast)):
                repeat_dims += ' ' + str(node.inputIr.irMetadata[i].broadcast[j]) + ','
        repeat_dims = repeat_dims[:-1]
        self.visit(node.inputIr)
        self.write_expr('.repeat(' + repeat_dims + ')', False)

    def visitIrAddDimension(self, node):
        size = 0
        for i in range(len(node.irMetadata)-1):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
        size += len(node.inputIr.irMetadata[-1].shape)
        self.visit(node.inputIr)
        for i in range(len(node.irMetadata[-1].shape) - len(node.inputIr.irMetadata[-1].shape)):
            self.write_expr('.unsqueeze(' + str(size) + ')', False)
            size += 1

    def visitIrAddDimensionConst(self, node):
        size = 0
        repeat_dims = ''
        for i in range(len(node.irMetadata)):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
                repeat_dims += ' ' + str(node.irMetadata[i].broadcast[j]) + '*' + str(node.irMetadata[i].shape[j]) + ','
        repeat_dims = repeat_dims[:-1]
        if node.inputIr.irMetadata[-1].isConst:
            self.write_expr('torch.tensor(', False)
            self.visit(node.inputIr)
            self.write_expr(')', False)
        else:
            self.visit(node.inputIr)
        for i in range(size):
            self.write_expr('.unsqueeze(' + str(i) + ')', False)
        self.write_expr('.repeat(' + repeat_dims + ')', False)
        # self.visit(node.inputIr)

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
            op_name = node.op
        
        self.write_expr(op_name + '(', False)
        self.visit(node.lhsIr)
        self.write_expr(', ', False)
        self.visit(node.rhsIr)
        self.write_expr(')', False)

    def visitIrMult(self, node):
        op_name = None 
        if node.op == '*':
            op_name = 'mult'
        elif node.op == '/':
            op_name = 'divide'
        else:
            op_name = node.op
        
        self.write_expr(op_name + '(', False)
        self.visit(node.lhsIr)
        self.write_expr(', ', False)
        self.visit(node.rhsIr)
        self.write_expr(')', False)

    def visitIrDot(self, node):
        if node.lhsIr.irMetadata[-1].type == 'Neuron':
            self.visit(node.lhsIr)
            self.write_expr('.dot(', False)
            self.visit(node.rhsIr)
            self.write_expr(')', False)
        elif node.rhsIr.irMetadata[-1].type == 'Neuron':
            self.visit(node.rhsIr)
            self.write_expr('.dot(', False)
            self.visit(node.lhsIr)
            self.write_expr(')', False)
        else:
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        self.write_expr('torch.where(', False)
        self.visit(node.condIr)
        self.write_expr(', ', False)
        self.visit(node.lhsIr)
        self.write_expr(', ', False)
        self.visit(node.rhsIr)
        self.write_expr(')', False)

    def visitIrCombineToPoly(self, node):
        self.write_expr('PolyExpNew(poly_size, ', False)
        self.visit(node.coeffIr)
        self.write_expr(', ', False)
        self.visit(node.constIr)
        self.write_expr(')', False)

    def visitIrExtractPolyCoeff(self, node):
        self.visit(node.inputIr)
        self.write_expr('.get_mat(abs_elem)', False)

    def visitIrExtractPolyConst(self, node):
        self.visit(node.inputIr)
        self.write_expr('.get_const()', False)

    def visitIrConvertNeuronToPoly(self, node):
        self.visit(node.inputIr)
        self.write_expr('.convert_to_poly()', False)
        # raise Exception('NOT IMPLEMENTED')
    
    def visitIrConvertConstToPoly(self, node):
        self.write_expr('PolyExpNew(poly_size, None, ', False)
        self.visit(node.inputIr)
        self.write_expr(')', False)

    def visitIrAccess(self, node):
        if not node.isMetadata:
            self.write_expr('abs_elem.get_elem_new(\'' + node.elem + '\', ', False)
            self.visit(node.lhsIr)
            self.write_expr(')', False)

        else:
            self.visit(node.lhsIr)
            self.write_expr('.get_metadata(\'' + node.elem + '\')', False)
        
    def visitIrReduce(self, node):
        size = 0
        for i in range(len(node.inputIr.irMetadata)-1):
            for j in range(len(node.inputIr.irMetadata[i].broadcast)):
                size += 1
        self.visit(node.inputIr)
        self.write_expr('.sum(dim=' + str(size) + ')', False)

    def visitIrMapCoeff(self, node):
        self.visit(node.inputIr)
        self.write_expr('.get_mat(abs_elem)', False)

    def visitIrMapNeuron(self, node):
        self.write_expr('abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))', False)

    def visitIrSymbolic(self, node):
        self.write_expr('trav_exp', False)
        # raise Exception('DO NOT REACH HERE')
    
    def visitIrTraverse(self, node):
        # self.write_expr('trav_exp = ', False)
        # self.visit(node.exprIr)
        self.write('while True:')
        self.indent += 1
        for ir in node.funcSeqIr:
            self.visit(ir)
        self.write('trav_size = trav_exp.const.shape[0]')
        for ir in node.stopSeqIr:
            self.visit(ir)
        self.write('vertices = trav_exp.mat != 0')
        self.write('vertices_stop = convert_to_tensor(vertices_stop, poly_size) != True')
        self.write('vertices_stop_default = torch.zeros(vertices_stop.size())')
        self.write('vertices_stop_default[0:input_size] = 1')
        self.write('vertices_stop_default = vertices_stop_default.bool()')
        self.write('vertices_stop = vertices_stop | vertices_stop_default')
        self.write('vertices = vertices & vertices_stop')
        self.write('if not(vertices.any()):')
        self.write('\tbreak')
        self.write('trav_exp = ', False)
        self.visit(node.funcIr)
        self.indent -= 1
        self.write('', True)

        # raise Exception('NOT IMPLEMENTED')
    
    def visitIrFlow(self, node):
        self.write('certifier = Certifier(abs_elem, ' + str(node.transformer) + '(), network, None)')
        self.write('certifier.flow()')