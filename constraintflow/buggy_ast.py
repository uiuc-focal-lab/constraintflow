from constraintflow import astcf as AST
from constraintflow import astTC
from constraintflow.astVisitor import ASTVisitor
import random



class BuggyAst(ASTVisitor):
    def __init__(self):
        pass 

    def visitShapeDecl(self, node):
        self.Gamma = dict()
        for (t, e) in node.elements.arglist:
            if t.name not in self.Gamma:
                self.Gamma[t.name] = [e.name]
            else:
                self.Gamma[t.name].append(e.name)

    def visitInt(self, node):
        node.value += random.randint(-1,1)

    def visitFloat(self, node):
        node.value += random.randint(-1,1)

    def visitBinOp(self, node):
        if node.op in ['+', '-']:
            index = random.randint(0, 1)
            node.op = ['+', '-'][index]
        elif node.op in ['*', '/']:
            index = random.randint(0, 2)
            node.op = ['+', '-', '*'][index]
    
    def visitGetElement(self, node):
        etype = node.type
        if isinstance(etype, astTC.ArrayType):
            index = random.randint(0, len(self.Gamma[etype.base])-1)
            node.elem.name = self.Gamma[etype.base][index]
        else:
            index = random.randint(0, len(self.Gamma[etype])-1)
            node.elem.name = self.Gamma[etype][index]

    def visitExprList(self, node: AST.ExprListNode):
        for e in node.exprlist:
            self.visit(e)
            
    def visitUnOp(self, node: AST.UnOpNode):
        self.visit(node.expr)

    def visitArgmaxOp(self, node: AST.ArgmaxOpNode):
        self.visit(node.expr)
        self.visit(node.func)

    def visitMaxOp(self, node: AST.MaxOpNode):
        self.visit(node.expr1)
        self.visit(node.expr2)

    def visitMaxOpList(self, node: AST.MaxOpListNode):
        self.visit(node.expr)

    def visitLp(self, node: AST.LpNode):
        self.visit(node.expr)
        self.visit(node.constraints)
    
    def visitMapList(self, node: AST.MapNode):
        self.visit(node.expr)

    def visitVar(self, node: AST.VarNode):
        pass

    def visitNeuron(self, node: AST.NeuronNode):
        pass

    def visitBool(self, node: AST.ConstBoolNode):
        pass

    def visitEpsilon(self, node: AST.EpsilonNode):
        pass

    def visitTernary(self, node: AST.TernaryNode):
        self.visit(node.cond)
        self.visit(node.texpr)
        self.visit(node.fexpr)


    def visitTraverse(self, node: AST.TraverseNode):
        self.visit(node.expr)
        self.visit(node.priority)
        self.visit(node.stop)
        self.visit(node.func)


    def visitListOp(self, node: AST.ListOpNode):
        self.visit(node.expr)

    def visitMap(self, node):
        self.visit(node.expr)

    def visitDot(self, node: AST.DotNode):
        self.visit(node.left)
        self.visit(node.right)

    def visitConcat(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visitFuncCall(self, node):
        self.visit(node.arglist)

    def visitGetMetadata(self, node: AST.GetMetadataNode):
        self.visit(node.expr)

    def visitTransRetBasic(self, node):
        self.visit(node.exprlist)

    def visitTransRetIf(self, node):
        self.visit(node.cond)
        self.visit(node.tret)
        self.visit(node.fret)

    def visitOpStmt(self, node):
        self.visit(node.ret)

    def visitOpList(self, node):
        for s in node.olist:
            self.visit(s)

    def visitTransformer(self, node):
        self.visit(node.oplist)

    def visitFunc(self, node):
        self.visit(node.expr)

    def visitSeq(self, node):
        self.visit(node.stmt1)
        self.visit(node.stmt2)

    def visitFlow(self, node):
        pass 

    def visitProg(self, node: AST.ProgramNode):
        self.visit(node.shape)
        self.visit(node.stmt)