class ASTNode:

    def __init__(self):
        self.node_name = "ASTNode"

class TypeNode(ASTNode):

    def __init__(self):
        super().__init__()
        self.node_name = "Type"

class BaseTypeNode(TypeNode):
    #name: Int, Float, Bool, Neuron, PolyExp, SymExp

    def __init__(self, name):
        super().__init__()
        self.node_name = "Base Type"
        self.name = name

    def __eq__(self,obj):
        if isinstance(obj, BaseTypeNode):
            return self.name==obj.name 
        return False 

class ArrayTypeNode(TypeNode):

    def __init__(self, base):
        super().__init__()
        self.node_name = "Array Type"
        self.base = base
        self.name = self.base.name + ' List'
    
    def __eq__(self,obj):
        if isinstance(obj, ArrayTypeNode):
            return self.name==obj.name 
        return False 

class ArgListNode(ASTNode):
    #arglist is a list of (TypeNode, VarNode)
    def __init__(self, arglist):
        super().__init__()
        self.node_name = "ArgList"
        self.arglist = arglist

class ExprListNode(ASTNode):

    def __init__(self, exprlist):
        super().__init__()
        self.node_name = "ExprList"
        self.exprlist = exprlist

class ExprNode(ASTNode):

    def __init__(self):
        super().__init__()
        self.node_name = "Expr"

class BinOpNode(ExprNode):
    #op: +,-,*,/,and,or,>=,<=,==
    def __init__(self, left, op, right):
        super().__init__()
        self.node_name = "BinOp"
        self.left = left
        self.op = op
        self.right = right

class UnOpNode(ExprNode):
    #op: -,~
    def __init__(self, op, expr):
        super().__init__()
        self.node_name = "UnOp"
        self.op = op
        self.expr = expr

class MetadataNode(ASTNode):
    #Weight, Bias, Layer
    def __init__(self, name):
        super().__init__()
        self.node_name = "Metadata"
        self.name = name

    def __eq__(self, obj):
        if(isinstance(obj, MetadataNode)):
            return self.name == obj.name
        return False

class ArgmaxOpNode(ExprNode):

    #op: min, max, argmin, argmax
    def __init__(self, op, expr, func):
        super().__init__()
        self.node_name = "ArgmaxOp"
        self.op = op
        self.expr = expr
        self.func = func

class MaxOpListNode(ExprNode):

    #op: min, max, argmin, argmax
    def __init__(self, op, expr):
        super().__init__()
        self.node_name = "MaxOpList"
        self.op = op
        self.expr = expr

class MaxOpNode(ExprNode):

    #op: min/max
    def __init__(self, op, expr1, expr2):
        super().__init__()
        self.node_name = "Max of 2 elements"
        self.op = op
        self.expr1 = expr1
        self.expr2 = expr2


class VarNode(ExprNode):

    def __init__(self, name):
        super().__init__()
        self.node_name = "Var"
        self.name = name

    def __eq__(self, obj):
        if(isinstance(obj, VarNode)):
            return self.name == obj.name
        return False

class NeuronNode(ExprNode):

    def __init__(self, w, b, l):
        self.elements = {}
        self.weight = w
        self.bias = b
        self.l = l

class ConstIntNode(ExprNode):

    def __init__(self, value):
        super().__init__()
        self.node_name = "Int"
        self.value = value

class ConstFloatNode(ExprNode):

    def __init__(self, value):
        super().__init__()
        self.node_name = "Float"
        self.value = value

class ConstBoolNode(ExprNode):

    def __init__(self, value):
        super().__init__()
        self.node_name = "Bool"
        self.value = value

# class CurrNode(ExprNode):

#     def __init__(self):
#         super().__init__()
#         self.node_name = "Curr"

# class PrevNode(ExprNode):

#     def __init__(self):
#         super().__init__()
#         self.node_name = "Prev"

class EpsilonNode(ExprNode):

    def __init__(self, identifier):
        super().__init__()
        self.node_name = "Epsilon"
        self.identifier = identifier

class TernaryNode(ExprNode):

    def __init__(self, cond, texpr, fexpr):
        super().__init__()
        self.node_name = "Ternary"
        self.cond = cond
        self.texpr = texpr
        self.fexpr = fexpr
        self.left = texpr
        self.right = fexpr

class GetMetadataNode(ExprNode):

    def __init__(self, expr, metadata):
        super().__init__()
        self.node_name = "Access Metadata"
        self.expr = expr
        self.metadata = metadata

    def __eq__(self, obj):
        if(isinstance(obj, GetMetadataNode)):
            return self.expr == obj.expr and self.metadata == obj.metadata

class GetElementNode(ExprNode):

    def __init__(self, expr, elem):
        super().__init__()
        self.node_name = "Access Element"
        self.expr = expr
        self.elem = elem

# class GetElementAtIndexNode(ExprNode):

#     def __init__(self, expr, index):
#         super().__init__()
#         self.node_name = "Access Element at Index"
#         self.expr = expr
#         self.index = index 

class TraverseNode(ExprNode):

    def __init__(self, expr, direction, priority, stop, func, p):
        super().__init__()
        self.node_name = "Traverse"
        self.expr = expr
        self.direction = direction
        self.priority = priority
        self.stop = stop
        self.func = func
        self.p = p

class DirectionNode(ASTNode):

    #value should be "forward" or "backward"
    def __init__(self, value):
        super().__init__()
        self.node_name = "Direction"
        self.value = value

class ListOpNode(ExprNode):

    def __init__(self, op, expr):
        super().__init__()
        self.op = op
        self.expr = expr

class MapNode(ExprNode):

    def __init__(self, expr, func):
        super().__init__()
        self.node_name = "Map"
        self.expr = expr
        self.func = func

class MapListNode(ExprNode):

    def __init__(self, expr, func):
        super().__init__()
        self.node_name = "MapList"
        self.expr = expr
        self.func = func

class DotNode(ExprNode):

    def __init__(self, left, right):
        super().__init__()
        self.node_name = "Dot"
        self.left = left
        self.right = right

    def __eq__(self, obj):
        if(isinstance(obj, DotNode)):
            return self.left == obj.left and self.right == obj.right
        return False

class ConcatNode(ExprNode):

    def __init__(self, left, right):
        super().__init__()
        self.node_name = "Concat"
        self.left = left
        self.right = right

class LpNode(ExprNode):

    def __init__(self, op, expr, constraints):
        self.op = op
        self.expr = expr
        self.constraints = constraints

class FuncCallNode(ExprNode):

    def __init__(self, name, arglist):
        super().__init__()
        self.node_name = "Function Call"
        self.name = name
        self.arglist = arglist #ExprListNode


class ShapeDeclNode(ASTNode):

    #elements is an ArgListNode
    #p is a property
    def __init__(self, elements, p):
        super().__init__()
        self.node_name = "Shape Declaration"
        self.elements = elements
        self.p = p

class FuncDeclNode(ASTNode):

    def __init__(self, name, arglist):
        super().__init__()
        self.node_name = "Function Declaration"
        self.name = name
        self.arglist = arglist

class OperatorNode(ASTNode):
    #Relu, Affine, Maxpool, Sigmoid, Tanh

    def __init__(self, op_name):
        super().__init__()
        self.node_name = "Operator"
        self.op_name = op_name

class TransRetNode(ASTNode):

    def __init__(self):
        super().__init__()
        self.node_name = "TransRet"

class TransRetBasicNode(TransRetNode):

    def __init__(self, exprlist):
        super().__init__()
        self.node_name = "TransRetBasic"
        self.exprlist = exprlist

class TransRetIfNode(TransRetNode):

    def __init__(self, cond, tret, fret):
        super().__init__()
        self.node_name = "TransRetIf"
        self.cond = cond
        self.tret = tret
        self.fret = fret

        self.left = tret
        self.right = fret

class OpStmtNode(ASTNode):

    def __init__(self, op, ret):
        super().__init__()
        self.node_name = "OpStmt"
        self.op = op
        self.ret = ret

#supposed to represent Relu -> x; Affine -> y;
class OpListNode(ASTNode):

    #olist should be list of OpStmtNodes
    def __init__(self, olist):
        super().__init__()
        self.node_name = "OpList"
        self.olist = olist

class StatementNode(ASTNode):

    def __init__(self):
        super().__init__()
        self.node_name = "Statement"

class TransformerNode(StatementNode):
    #can have a seperate TransDeclNode class but it's not necessary
    def __init__(self, name, oplist):
        super().__init__()
        self.node_name = "Transformer"
        self.name = name
        # self.arglist = arglist
        self.oplist = oplist

# class PropTermNode(ASTNode):

#     def __init__(self):
#         super().__init__()
#         self.node_name = "Property Term"

# class PropTermBasicNode(PropTermNode):

#     #term can be Int, Float, Var or Metadata
#     def __init__(self, term):
#         super().__init__()
#         self.node_name = "Basic Property Term"
#         self.term = term

# class PropTermInNode(PropTermNode):

#     def __init__(self, n, z):
#         super().__init__()
#         self.node_name = "In"
#         self.n = n
#         self.z = z
    

# class PropTermOpNode(PropTermNode):
#     #op: +,-
#     def __init__(self, leftpt, op, rightpt):
#         super().__init__()
#         self.node_name = "Binop Property Term"
#         self.leftpt = leftpt
#         self.rightpt = rightpt
#         self.op = op

# class PropNode(ASTNode):

#     def __init__(self):
#         super().__init__()
#         self.node_name = "Property"

# class SinglePropNode(PropNode):
#     #op: <,<=,>,>=,==
#     def __init__(self, leftpt, op, rightpt):
#         super().__init__()
#         self.node_name = "Single Property"
#         self.leftpt = leftpt
#         self.op = op
#         self.rightpt = rightpt

# class DoublePropNode(PropNode):
#     #op: and, or
#     def __init__(self, leftprop, op, rightprop):
#         super().__init__()
#         self.node_name = "Double Property"
#         self.leftprop = leftprop
#         self.op = op
#         self.rightprop = rightprop

class FuncNode(StatementNode):

    def __init__(self, decl, expr):
        super().__init__()
        self.node_name = "Function"
        self.decl = decl
        self.expr = expr

class SeqNode(StatementNode):

    def __init__(self, stmt1, stmt2):
        super().__init__()
        self.node_name = "Seq"
        self.stmt1 = stmt1
        self.stmt2 = stmt2

class FlowNode(StatementNode):

    def __init__(self, direction, pfunc, sfunc, trans):
        super().__init__()
        self.node_name = "Flow"
        self.direction = direction
        self.pfunc = pfunc
        self.sfunc = sfunc
        self.trans = trans

class ProgramNode(ASTNode):

    #shape is ShapeDeclNode
    def __init__(self, shape, stmt):
        super().__init__()
        self.node_name = "Program"
        self.shape = shape
        self.stmt = stmt