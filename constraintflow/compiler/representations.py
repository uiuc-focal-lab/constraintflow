from constraintflow.compiler.ir import *

import networkx as nx
import matplotlib.pyplot as plt
import graphviz

def get_z3_vars(z3_expr):
    if isinstance(z3_expr, z3.z3.ArithRef):
        if len(z3_expr.children())>0:
            vars = set()
            for child in z3_expr.children():
                vars = vars.union(get_z3_vars(child))
            return vars
        else:
            return {z3_expr}
    else:
        return set()

class Graph:
    def __init__(self):
        self.nodes = []
        self.predecessors = {}
        self.successors = {}
        self.ir = {}
        self.counter = 0
        self.entry_node = -1

    def get_block_id(self, block):
        for node in self.nodes:
            if self.ir[node] == block:
                return node
    
    def add(self, node = None):
        if node == None:
            node = self.counter 
            self.counter += 1
        else:
            if self.counter <= node:
                self.counter = node+1
        if node not in self.nodes:
            self.nodes.append(node)
            self.predecessors[node] = []
            self.successors[node] = []
        if self.entry_node < 0:
            self.entry_node = node
        return node
    
    def add_edge(self, i, j):
        if i not in self.nodes:
            self.nodes.append(i)
            self.predecessors[i] = []
            self.successors[i] = []
        if j not in self.nodes:
            self.nodes.append(j)
            self.predecessors[j] = []
            self.successors[j] = []
        if self.counter <= max(i, j):
            self.counter = max(i, j) + 1
        self.successors[i].append(j)
        self.predecessors[j].append(i)

    def topological_sort(self):
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node)
            for neighbor in self.successors[node]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)
        return stack[::-1]
    
    def get_vars(self):
        def get_vars_expr(expr, l):
            if isinstance(expr, int):
                pass
            elif isinstance(expr, IrVar):
                l.add(expr.name)
            else:
                for child in expr.children:
                    get_vars_expr(child, l)
        def get_vars_block(block, l):
            ir_list = block.children
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    get_vars_expr(ir_list[i].children[0], l)
                    get_vars_expr(ir_list[i].children[1], l)
                elif isinstance(ir_list[i], IrTransRetBasic):
                    for child in ir_list[i].children:
                        get_vars_expr(child, l)
        vars = set()
        for node in self.nodes:
            block = self.ir[node]
            get_vars_block(block, vars)
        return vars
    
    def get_vars_metadata(self):
        def get_vars_expr(expr, l):
            if isinstance(expr, IrVar):
                if expr.name not in l:
                    l[expr.name] = expr
        def get_vars_block(block, l):
            ir_list = block.children
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    get_vars_expr(ir_list[i].children[0], l)
        vars = dict()
        for node in self.nodes:
            block = self.ir[node]
            get_vars_block(block, vars)
        return vars
    
    def get_nodes_modifying_vars(self, vars):
        def get_nodes_modifying_vars_block(node, block, l):
            ir_list = block.children
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    l[ir_list[i].children[0].name].append(node)
        vars_modifying_nodes = {var:[] for var in vars}
        for node in self.nodes:
            block = self.ir[node]
            get_nodes_modifying_vars_block(node, block, vars_modifying_nodes)
        return vars_modifying_nodes


    def print(self):
        dot = graphviz.Digraph()
        for i in self.nodes:
            dot.node(str(i))
            for j in self.successors[i]:
                dot.edge(str(i), str(j))
        dot.render('graph', view=True)

    def find_node(self, ir):
        for i in self.nodes:
            if self.ir[i].identifier==ir.identifier:
                return i
        return -1


def create_cfg(ir_list):
    cfg = Graph()
    def create_cfg_rec(ir_list, cfg, currBlock, currNode, whileBlock, exitWhileBlock):
        end = 0
        while(end < len(ir_list)):
            if isinstance(ir_list[end], IrWhile):
                new_children = ir_list[:end]
                currBlock.update_parent_child(new_children)
                if whileBlock != None:
                    if currBlock not in whileBlock.loopBody:
                        whileBlock.loopBody.append(currBlock)
                cond = ir_list[end].children[0]
                node_2 = cfg.add()

                block_2 = IrWhileBlock(ir_list[end].children[0])
                cfg.ir[node_2] = block_2
                currBlock.inner_jump = [cond, block_2]

                node_3 = cfg.add()
                block_3 = IrBlock()
                cfg.ir[node_3] = block_3
                

                cond = IrUnaryOp(cond, 'not')
                currBlock.jump = [cond, block_3]

                break_node = create_cfg_rec(ir_list[end].children[1:], cfg, block_2, node_2, block_2, block_3)
                block_2.loopBody[-1].loopBack = [block_2.condIr, block_2]
                
                ret_node = create_cfg_rec(ir_list[end+1:], cfg, block_3, node_3, whileBlock, exitWhileBlock)

                cfg.successors[currNode].append(node_2)

                cfg.predecessors[node_2].append(currNode)
                node = cfg.find_node(block_2.loopBody[-1])
                cfg.predecessors[node_2].append(node)
                cfg.successors[node].append(node_2)



                return ret_node
            
            elif isinstance(ir_list[end], IrIte):
                new_children = ir_list[:end]
                currBlock.update_parent_child(new_children)
                if whileBlock != None:
                    if currBlock not in whileBlock.loopBody:
                        whileBlock.loopBody.append(currBlock)
                cond_if = ir_list[end].condIr 
                cond_else = IrUnaryOp(cond_if, 'not')
                block_if = IrBlock()
                node_if = cfg.add()
                cfg.ir[node_if] = block_if

                
                
                end_if_node = create_cfg_rec(ir_list[end].lhsIrs, cfg, block_if, node_if, whileBlock, exitWhileBlock)

                if len(ir_list[end].rhsIrs) > 0:
                    block_else = IrBlock()
                    node_else = cfg.add()
                    cfg.ir[node_else] = block_else
                    end_else_node = create_cfg_rec(ir_list[end].rhsIrs, cfg, block_else, node_else, whileBlock, exitWhileBlock)
                    currBlock.inner_jump = [cond_if, block_if, block_else]
                else:
                    currBlock.inner_jump = [cond_if, block_if]
                
                node_3 = cfg.add()
                block_3 = IrBlock()
                cfg.ir[node_3] = block_3
                create_cfg_rec(ir_list[end+1:], cfg, block_3, node_3, whileBlock, exitWhileBlock)
                currBlock.jump = [cond_else, block_3]



                cfg.successors[currNode].append(node_if)
                cfg.successors[end_if_node].append(node_3)
                
                cfg.predecessors[node_3].append(end_if_node)
                cfg.predecessors[node_if].append(currNode)
                
                

                if len(ir_list[end].rhsIrs) > 0:
                    cfg.successors[currNode].append(node_else)
                    cfg.successors[end_else_node].append(node_3)

                    cfg.predecessors[node_3].append(end_else_node)
                    cfg.predecessors[node_else].append(currNode)
                else:
                    cfg.successors[currNode].append(node_3)
                    cfg.predecessors[node_3].append(currNode)
                
                return node_3
            
            elif isinstance(ir_list[end], IrBreak):
                exitWhileNode = cfg.find_node(exitWhileBlock)
                cfg.successors[currNode].append(exitWhileNode)
                cfg.predecessors[exitWhileNode].append(currNode)
                new_children = ir_list[:end+1]
                currBlock.update_parent_child(new_children)
                if whileBlock != None:
                    if currBlock not in whileBlock.loopBody:
                        whileBlock.loopBody.append(currBlock)

                return currNode

            else:
                end += 1
                if end >= len(ir_list):
                    new_children = ir_list[:end]
                    currBlock.update_parent_child(new_children)
                    if whileBlock != None:
                        if currBlock not in whileBlock.loopBody:
                            whileBlock.loopBody.append(currBlock)
                    return currNode
    rootBlock = IrBlock([])
    whileBlock = None
    rootNode = cfg.add()
    cfg.ir[rootNode] = rootBlock
    create_cfg_rec(ir_list, cfg, rootBlock, rootNode, whileBlock, whileBlock)
    return cfg
    



def find_dominators(cfg):
    dominators = {node: set(cfg.nodes) for node in cfg.nodes}
    dominators[cfg.entry_node] = {cfg.entry_node}
    
    changed = True
    while changed:
        changed = False
        for node in cfg.nodes:
            if node == cfg.entry_node:
                continue
            new_dominators = dominators[list(cfg.predecessors[node])[0]]
            for predecessor in cfg.predecessors[node]:
                new_dominators = new_dominators.intersection(dominators[predecessor])
            new_dominators.add(node)
            if dominators[node] != new_dominators:
                dominators[node] = new_dominators
                changed = True
        
    return dominators

def construct_dominator_tree(cfg):
    dominators = find_dominators(cfg)
    dominator_tree = Graph()
    for n in dominators.keys():
        dominator_tree.add(n)
        for m in dominators[n]:
            if m==n:
                continue
            idom = True
            for p in dominators.keys():
                p_sdom_n = (p in dominators[n]) and (p != n)
                p_dom_m = (p in dominators[m])
                idom = idom and (not(p_sdom_n) or p_dom_m)
            if idom:
                dominator_tree.add_edge(m, n)
                break 
    return dominator_tree 


def compute_dominance_frontier(cfg, dominator_tree):
    dominance_frontier = {}
    topological_sort = dominator_tree.topological_sort()

    for node in reversed(topological_sort):
        dominance_frontier[node] = set()

        for successor in cfg.successors[node]:
            if not successor in dominator_tree.successors[node]:
                dominance_frontier[node].add(successor)

        for child in dominator_tree.successors[node]:
            for frontier_node in dominance_frontier[child]:
                if not frontier_node in dominator_tree.successors[node]:
                    dominance_frontier[node].add(frontier_node)

    return dominance_frontier

            
def get_phi_nodes(cfg, df):
    vars = cfg.get_vars()
    nodes_modifying_vars = cfg.get_nodes_modifying_vars(vars)
    phi_nodes = {var:[] for var in vars}
    for var in vars:
        if len(nodes_modifying_vars[var]) < 2:
            continue
        hasAlready = set()
        everOnWorklist = set()
        worklist = set()
        for x in nodes_modifying_vars[var]:
            everOnWorklist.add(x)
            worklist.add(x)
        while worklist:
            x = worklist.pop()
            for y in df[x]:
                if y not in hasAlready:
                    phi_nodes[var].append(y)
                    hasAlready.add(y)
                    if y not in everOnWorklist:
                        everOnWorklist.add(y)
                        worklist.add(y)
    phi_nodes_ret = {var:phi_nodes[var] for var in phi_nodes.keys() if len(phi_nodes[var])>0}
    return phi_nodes_ret

def replace(expr, old_var, new_var):
    if isinstance(expr, IrVar):
        if expr.name == old_var:
            return new_var
    for i in range(len(expr.children)):
        new_child = replace(expr.children[i], old_var, new_var)
        expr.children[i] = new_child
    return expr

def convert_to_ssa(cfg, dtree, phi_nodes, metadata):
    vars = {var:[] for var in phi_nodes.keys()}
    counter = {var: 0 for var in phi_nodes.keys()}
    stack = {var: [] for var in phi_nodes.keys()}
    original_vars = {}
    entry_node = cfg.entry_node
    vars_list = copy.deepcopy(list(vars.keys()))
    vars_list.sort(reverse=True)
    for var in vars_list:
        for node in phi_nodes[var]:
            block = cfg.ir[node]
            phi_node = IrPhi(var, [], metadata[var].irMetadata)
            new_assignment = IrAssignment(metadata[var], phi_node)
            new_children = block.children 
            new_children.insert(0, new_assignment)
            block.update_parent_child(new_children)
            
    def get_new_name(irVar, node, phi=False):
        var = irVar.name
        if var in counter.keys():
            i = counter[var]
            counter[var] += 1
            if phi:
                new_name = 'phi_' + var + '_' + str(node) + '_' + str(i)
            else:
                new_name = var + '_' + str(node) + '_' + str(i)
            new_var = IrVar(new_name, metadata[var].irMetadata)
            stack[var].append(new_var)
            original_vars[new_var.name] = var
            return new_var
        else:
            return irVar 
    def rename_metadata(irMetadata, expr=None):
        flag = False
        for j, irMetadataElement in enumerate(irMetadata):
            for i in range(len(irMetadataElement.shape)):
                if isinstance(irMetadataElement.shape[i], int):
                    continue
                irMetadataElement.shape[i] = rename_expr(irMetadataElement.shape[i])
            for i in range(len(irMetadataElement.broadcast)):
                if isinstance(irMetadataElement.broadcast[i], int):
                    continue
                irMetadataElement.broadcast[i] = rename_expr(irMetadataElement.broadcast[i])
                    
        return flag
    def rename_expr(expr):
        if isinstance(expr, int):
            return expr
        rename_metadata(expr.irMetadata, expr)
        if isinstance(expr, IrVar):
            if expr.name in stack.keys():
                return stack[expr.name][-1]
        new_children = []
        for i in range(len(expr.children)):
            new_child = rename_expr(expr.children[i])
            new_children.append(new_child)
        expr.update_parent_child(new_children)
        return expr
    def rename(node):
        block = cfg.ir[node]
        ir_list = block.children
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if isinstance(ir_list[j].children[1], IrPhi):
                    new_name = get_new_name(ir_list[j].children[0], node, True)
                    new_children = [new_name, ir_list[j].children[1]]
                    ir_list[j].update_parent_child(new_children)   
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                rename_metadata(ir_list[j].irMetadata)
                if not isinstance(ir_list[j].children[1], IrPhi):
                    new_rhs = rename_expr(ir_list[j].children[1])
                    new_lhs = get_new_name(ir_list[j].children[0], node, False)
                    ir_list[j].update_parent_child([new_lhs, new_rhs])
            elif isinstance(ir_list[j], IrTransRetBasic):
                for i in range(len(ir_list[j].children)):
                    new_child = rename_expr(ir_list[j].children[i])
                    ir_list[j].children[i] = new_child
        if block.inner_jump is not None:
            new_cond = rename_expr(block.inner_jump[0])
            block.inner_jump[0] = new_cond
        if block.jump is not None:
            new_cond = rename_expr(block.jump[0])
            block.jump[0] = new_cond
        for successor in cfg.successors[node]:
            successor_block = cfg.ir[successor]
            ir_list_successor = successor_block.children
            for j in range(len(ir_list_successor)):
                if not isinstance(ir_list_successor[j], IrAssignment):
                    continue 
                if not isinstance(ir_list_successor[j].children[1], IrPhi):
                    continue 
                var = ir_list_successor[j].children[1].original_name
                if len(stack[var])>0:
                    new_children = ir_list_successor[j].children[1].children + [stack[var][-1]]
                    ir_list_successor[j].children[1].update_parent_child(new_children)
                    ir_list_successor[j].children[1].parent_nodes.append(node)
        for successor in dtree.successors[node]:
            rename(successor)
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if not isinstance(ir_list[j].children[1], IrPhi):
                    if ir_list[j].children[0].name in original_vars.keys():
                        if len(stack[original_vars[ir_list[j].children[0].name]]) > 0:
                            stack[original_vars[ir_list[j].children[0].name]] = stack[original_vars[ir_list[j].children[0].name]][:-1]
    def prune(node):
        block = cfg.ir[node]
        ir_list = block.children
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if isinstance(ir_list[j].children[1], IrPhi):
                    s = set(ir_list[j].children[1].children)
                    if len(s)<=1:
                        new_children = [ir_list[j].children[0], list(s)[0]]
                        ir_list[j].update_parent_child(new_children)
    rename(entry_node)

def remove_cfg_block(block, node, cfg):
    ir_list = block.children
    remove_list = []
    for count, i in enumerate(ir_list):
        if isinstance(i, IrAssignment):
            if isinstance(i.children[1], IrPhi):
                remove_list.append(count)
                for j in range(len(i.children[1].children)):
                    predecessor = cfg.ir[i.children[1].parent_nodes[j]]
                    # predecessor = cfg.ir[cfg.predecessors[node][j]]
                    var = i.children[1].children[j]
                    if var==None:
                        continue
                    new_assignment = IrAssignment(i.children[0], var)
                    new_children = predecessor.children
                    if len(new_children)>0 and not isinstance(new_children[-1], IrBreak):
                        new_children.append(new_assignment)
                    else:
                        new_children.insert(-1, new_assignment)
                    predecessor.update_parent_child(new_children)
                for predecessors_node in cfg.predecessors[node]:
                    if predecessors_node not in i.children[1].parent_nodes:
                        predecessor = cfg.ir[predecessors_node]
                        var = IrConst(0, 'Float')
                        new_assignment = IrAssignment(i.children[0], var)
                        new_children = predecessor.children
                        if not isinstance(new_children[-1], IrBreak):
                            new_children.append(new_assignment)  
                        else:
                            new_children.insert(-1, new_assignment)
                        predecessor.update_parent_child(new_children)
    for i in range(len(remove_list)-1, -1, -1):
        del ir_list[remove_list[i]]

def remove_phi(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            for node in cfg.nodes:
                block = cfg.ir[node]
                remove_cfg_block(block, node, cfg)


def ssa(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            dtree = construct_dominator_tree(cfg)
            df = compute_dominance_frontier(cfg, dtree)
            phi_nodes = get_phi_nodes(cfg, df)
            vars_metadata = cfg.get_vars_metadata()
            convert_to_ssa(cfg, dtree, phi_nodes, vars_metadata)