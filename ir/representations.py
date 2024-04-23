from ir_ast_stack2 import *
import networkx as nx
import matplotlib.pyplot as plt
import graphviz

class Graph:
    def __init__(self):
        self.nodes = []
        self.predecessors = {}
        self.successors = {}
        self.ir = {}
        self.counter = 0
        self.entry_node = -1

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
        print(stack[::-1])
        return stack[::-1]
    
    def get_vars(self):
        def get_vars_expr(expr, l):
            if isinstance(expr, IrVar) or isinstance(expr, IrSymbolic):
                l.add(expr.name)
            else:
                for child in expr.children:
                    get_vars_expr(child, l)
        def get_vars_block(ir_list, l):
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    get_vars_expr(ir_list[i].children[0], l)
                    get_vars_expr(ir_list[i].children[1], l)
                elif isinstance(ir_list[i], IrTransRetBasic):
                    for child in ir_list[i].children:
                        get_vars_expr(child, l)
                elif isinstance(ir_list[i], IrWhile):
                    get_vars_expr(ir_list[i].children[0], l)
        vars = set()
        for node in self.nodes:
            ir_list = self.ir[node]
            get_vars_block(ir_list, vars)
        return vars
    
    def get_vars_metadata(self):
        def get_vars_expr(expr, l):
            if isinstance(expr, IrVar) or isinstance(expr, IrSymbolic):
                if expr.name not in l:
                    l[expr.name] = expr
        def get_vars_block(ir_list, l):
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    get_vars_expr(ir_list[i].children[0], l)
        vars = dict()
        for node in self.nodes:
            ir_list = self.ir[node]
            get_vars_block(ir_list, vars)
        return vars
    
    def get_nodes_modifying_vars(self, vars):
        def get_nodes_modifying_vars_block(node, ir_list, l):
            for i in range(len(ir_list)):
                if isinstance(ir_list[i], IrAssignment):
                    l[ir_list[i].children[0].name].append(node)
        vars_modifying_nodes = {var:[] for var in vars}
        for node in self.nodes:
            ir_list = self.ir[node]
            get_nodes_modifying_vars_block(node, ir_list, vars_modifying_nodes)
        return vars_modifying_nodes


    def print(self):
        dot = graphviz.Digraph()
        for i in self.nodes:
            dot.node(str(i))
            for j in self.successors[i]:
                dot.edge(str(i), str(j))
        dot.render('graph', view=True)


def create_cfg(ir_list):
    cfg = Graph()
    def create_cfg_rec(ir_list, cfg):
        end = 0
        while(True):
            if isinstance(ir_list[end], IrWhile):
                node_1 = cfg.add()
                cfg.ir[node_1] = ir_list[:end]
                node_2 = create_cfg_rec(ir_list[end].children[1:], cfg)
                node_3 = create_cfg_rec(ir_list[end+1:], cfg) 

                cfg.successors[node_1].append(node_2)
                cfg.successors[node_1].append(node_3)

                cfg.predecessors[node_2].append(node_1)
                cfg.predecessors[node_2].append(node_2)
                cfg.successors[node_2].append(node_2)
                cfg.successors[node_2].append(node_3)

                cfg.predecessors[node_3].append(node_1)
                cfg.predecessors[node_3].append(node_2)

                return node_1 
            
            else:
                end += 1
                if end >= len(ir_list):
                    node = cfg.add()
                    cfg.ir[node] = ir_list
                    return node
    create_cfg_rec(ir_list, cfg)
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
    # print(dominators)
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
    # print(vars)
    print(nodes_modifying_vars)
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
    phi_nodes_ret = {var:phi_nodes[var] for var in phi_nodes.keys() if len(phi_nodes[var])>1}
    print(phi_nodes_ret)
    # del phi_nodes_ret['vertices_priority']
    # del phi_nodes_ret['vertices_stop']
    # del phi_nodes_ret['trav_exp2']
    print(phi_nodes_ret)
    return phi_nodes_ret

def replace(expr, old_var, new_var):
    if isinstance(expr, IrSymbolic) or isinstance(expr, IrVar):
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
            print(var, node)
            ir_list = cfg.ir[node]
            phi_node = IrPhi(var, [], metadata[var].irMetadata)
            new_assignment = IrAssignment(metadata[var], phi_node)
            # print('!!!!!!!!!!', node, len(ir_list[0].parents))
            parent_ir = ir_list[0].parents[0]
            index = parent_ir.children.index(ir_list[0])
            new_children = parent_ir.children 
            new_children.insert(index, new_assignment)
            parent_ir.update_parent_child(new_children)
            ir_list.insert(0, new_assignment)
            print(node, len(parent_ir.children), len(ir_list))
            # print(new_assignment.parents)
            # print(type(parent_ir))
            
    def get_new_name(irVar, node, phi=False):
        var = irVar.name
        if var in counter.keys():
            i = counter[var]
            counter[var] += 1
            if phi:
                new_name = 'phi_' + var + '_' + str(node) + '_' + str(i)
            else:
                new_name = var + '_' + str(node) + '_' + str(i)
            new_var = IrVar(new_name, metadata[var])
            stack[var].append(new_var)
            original_vars[new_var.name] = var
            return new_var
        else:
            return irVar 
    def rename_expr(expr):
        if isinstance(expr, IrSymbolic) or isinstance(expr, IrVar):
            if expr.name in stack.keys():
                return stack[expr.name][-1]
        new_children = []
        for i in range(len(expr.children)):
            new_child = rename_expr(expr.children[i])
            new_children.append(new_child)
        expr.update_parent_child(new_children)
        return expr
    def rename(node):
        ir_list = cfg.ir[node]
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if isinstance(ir_list[j].children[1], IrPhi):
                    new_name = get_new_name(ir_list[j].children[0], node, True)
                    new_children = [new_name, ir_list[j].children[1]]
                    ir_list[j].update_parent_child(new_children)        
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrCustomCodeGen):
                new_children = [rename_expr(ir_list[j].children[0])]
                ir_list[j].update_parent_child(new_children)
            if isinstance(ir_list[j], IrAssignment):
                if not isinstance(ir_list[j].children[1], IrPhi):
                    new_lhs = get_new_name(ir_list[j].children[0], node, False)
                    new_rhs = rename_expr(ir_list[j].children[1])
                    ir_list[j].update_parent_child([new_lhs, new_rhs])

        for successor in cfg.successors[node]:
            ir_list = cfg.ir[successor]
            for j in range(len(ir_list)):
                if not isinstance(ir_list[j], IrAssignment):
                    continue 
                if not isinstance(ir_list[j].children[1], IrPhi):
                    continue 
                var = ir_list[j].children[1].original_name
                if len(stack[var])>0:
                    new_children = ir_list[j].children[1].children + [stack[var][-1]]
                    ir_list[j].children[1].update_parent_child(new_children)
        for successor in dtree.successors[node]:
            rename(successor)
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if not isinstance(ir_list[j].children[1], IrPhi):
                    if ir_list[j].children[0].name in original_vars.keys():
                        if len(stack[original_vars[ir_list[j].children[0].name]]) > 0:
                            stack[original_vars[ir_list[j].children[0].name]] = stack[original_vars[ir_list[j].children[0].name]][:-1]
    def prune(node):
        ir_list = cfg.ir[node]
        for j in range(len(ir_list)):
            if isinstance(ir_list[j], IrAssignment):
                if isinstance(ir_list[j].children[1], IrPhi):
                    s = set(ir_list[j].children[1].children)
                    if len(s)<=1:
                        new_children = [ir_list[j].children[0], list(s)[0]]
                        ir_list[j].update_parent_child(new_children)
    rename(entry_node)
    for node in cfg.nodes:
        prune(node)




def ssa(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = create_cfg(ir.tstore[transformer][i].children)
            dtree = construct_dominator_tree(cfg)
            df = compute_dominance_frontier(cfg, dtree)
            phi_nodes = get_phi_nodes(cfg, df)
            vars_metadata = cfg.get_vars_metadata()
            convert_to_ssa(cfg, dtree, phi_nodes, vars_metadata)
    return ir    

