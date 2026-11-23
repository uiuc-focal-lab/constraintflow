from constraintflow.compiler.ir import *
from constraintflow.compiler import representations


counter = 0

def get_var():
    global counter 
    counter += 1
    return 'cse_var_' + str(counter)

def compare(x):
    return x[0]

def _key(expr, memo):
    """Structural key, coarser than __eq__ so it never separates equal nodes."""
    k = memo.get(id(expr))
    if k is not None:
        return k
    if isinstance(expr, (int, float, str)) or expr is None:
        k = ('lit', repr(expr))
    elif isinstance(expr, IrVar):
        k = ('var', expr.name)
    else:
        k = (type(expr).__name__, len(expr.children),
             tuple(_key(c, memo) for c in expr.children))
    memo[id(expr)] = k
    return k


class _Table:
    """Buckets equal expressions together; __eq__ still decides within a bucket.

    IrAst.__hash__ is 0, so a plain dict degrades to one bucket and every lookup
    becomes a linear scan of deep comparisons -- quadratic on a large block.
    """

    def __init__(self):
        self.buckets = {}
        self.memo = {}
        self.ordered = set()

    def find_or_add(self, expr):
        bucket = self.buckets.setdefault(_key(expr, self.memo), [])
        for rep in bucket:
            if rep == expr:
                return rep, False
        bucket.append(expr)
        return expr, True


def check_expr_visit(expr, node, visited_expr, visited_order, table):
    if isinstance(expr, IrConst) or isinstance(expr, IrVar) or isinstance(expr, IrPhi) or isinstance(expr, (int, float, str)) or expr is None:
        return
    rep, is_new = table.find_or_add(expr)
    if is_new:
        visited_expr[id(rep)] = {node}
        for i in range(len(expr.children)):
            check_expr_visit(expr.children[i], node, visited_expr, visited_order, table)
    else:
        if id(rep) not in table.ordered:
            table.ordered.add(id(rep))
            visited_order.append(rep)
        visited_expr[id(rep)].add(node)

def cse_block(block, node, visited_expr, visited_order, table):
    ir_list = block.children
    for ir in ir_list:
        if isinstance(ir, IrAssignment):
            check_expr_visit(ir.children[1], node, visited_expr, visited_order, table)
        elif isinstance(ir, IrTransRetBasic):
            for j in range(len(ir.children)):
                check_expr_visit(ir.children[j], node, visited_expr, visited_order, table)

def replace_all_occurrences_expr(expr, sub_expr, var):
    replaced = False
    if expr == sub_expr:
        return var, True
    if isinstance(expr, (int, float, str)) or expr is None:
        return expr, False
    for i in range(len(expr.children)):
        new_child, replaced_temp = replace_all_occurrences_expr(expr.children[i], sub_expr, var)
        expr.children[i] = new_child
        replaced = replaced or replaced_temp
    return expr, replaced

def replace_all_occurrences_block(block, new_assignment):
    ir_list = block.children
    replaced = False
    # index = []
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            new_expr, replaced_temp = replace_all_occurrences_expr(ir_list[i].children[1], new_assignment.children[1], new_assignment.children[0])
            replaced = replaced or replaced_temp
            new_children = [ir_list[i].children[0], new_expr]
            ir_list[i].update_parent_child(new_children)
        elif isinstance(ir_list[i], IrTransRetBasic):
            new_children = []
            for j in range(len(ir_list[i].children)):
                new_expr, replaced_temp = replace_all_occurrences_expr(ir_list[i].children[j], new_assignment.children[1], new_assignment.children[0])
                replaced = replaced or replaced_temp
                new_children.append(new_expr)
            ir_list[i].update_parent_child(new_children)
    if block.inner_jump != None:
        block.inner_jump[0], replaced_temp = replace_all_occurrences_expr(block.inner_jump[0], new_assignment.children[1], new_assignment.children[0])
        replaced = replaced or replaced_temp
    if block.jump != None:
        block.jump[0], replaced_temp = replace_all_occurrences_expr(block.jump[0], new_assignment.children[1], new_assignment.children[0])
        replaced = replaced or replaced_temp
    return replaced

def check_ancestor(dtree, ancestor, nodes):
    if ancestor in nodes:
        nodes.remove(ancestor)
    for child in dtree.successors[ancestor]:
        check_ancestor(dtree, child, nodes)
    if len(nodes)>0:
        return False 
    return True

def compute_ancestor(occurrences, dtree, node):
    for child in dtree.successors[node]:
        temp = copy.deepcopy(occurrences)
        if check_ancestor(dtree, child, temp):
            return compute_ancestor(occurrences, dtree, child)
    return node
    

def check_occurrence(ir, var):
    if ir == var:
        return True
    if isinstance(ir, (int, float, str)) or ir is None:
        return False
    occurrs = False
    for i in range(len(ir.children)):
        occurrs = occurrs or check_occurrence(ir.children[i], var)
    return occurrs

def _loop_body_nodes(cfg):
    nodes = set()
    for node in cfg.nodes:
        block = cfg.ir[node]
        if isinstance(block, IrWhileBlock):
            nodes.update(cfg.get_block_id(inner) for inner in block.loopBody)
    return nodes

def _clear_inside_while(node):
    if isinstance(node, IrAst):
        node.inside_while = False
        for child in node.children:
            _clear_inside_while(child)

def add_assignment(assignment, occurrences, cfg, dtree):
    node = compute_ancestor(occurrences, dtree, cfg.entry_node)
    if node not in _loop_body_nodes(cfg):
        # Hoisted out of a traversal loop: its jit capture must not be keyed by
        # (or reference) a loop iteration that does not exist at this point.
        _clear_inside_while(assignment.children[1])
    ir_list = cfg.ir[node].children
    index = -1
    if node in occurrences:
        for l in ir_list:
            if check_occurrence(l, assignment.children[0]):
                index = ir_list.index(l)
                break
    else:
        index = len(ir_list)
    ir_list.insert(index, assignment)
            
def create_new_assignments(visited_order, visited_expr, cfg, dtree):
    for i in range(len(visited_order)-1, -1, -1):
        original_expr = visited_order[i]
        new_var = IrVar(get_var(), original_expr.irMetadata)
        new_assignment = IrAssignment(new_var, original_expr)
        new_var.defs = new_assignment
        occurrences = []
        for node in cfg.nodes:
            block = cfg.ir[node]
            replaced = replace_all_occurrences_block(block, new_assignment)
            if replaced:
                occurrences.append(node)
        visited_expr[id(original_expr)] = occurrences

        add_assignment(new_assignment, occurrences, cfg, dtree)

def cse_cfg(cfg, dtree):
    visited_expr = {}
    visited_order = []
    table = _Table()
    for node in cfg.nodes:
        cse_block(cfg.ir[node], node, visited_expr, visited_order, table)
    
    create_new_assignments(visited_order, visited_expr, cfg, dtree)
    

def cse(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            # if i==0:
            #     continue
            cfg = ir.tstore[transformer][i].cfg
            dtree = representations.construct_dominator_tree(cfg)
            cse_cfg(cfg, dtree)
