import networkx as nx

def find_dominators(graph, entry_node):
    dominators = {node: set(graph.nodes) for node in graph.nodes}
    dominators[entry_node] = {entry_node}
    
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node == entry_node:
                continue
            new_dominators = dominators[list(graph.predecessors(node))[0]]
            for predecessor in graph.predecessors(node):
                new_dominators = new_dominators.intersection(dominators[predecessor])
            new_dominators.add(node)
            if dominators[node] != new_dominators:
                dominators[node] = new_dominators
                changed = True
    return dominators

def construct_dominator_tree(graph, dominators):
    dominator_tree = nx.DiGraph()
    for n in dominators.keys():
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

    # Perform bottom-up traversal of dominator tree
    for node in reversed(list(nx.topological_sort(dominator_tree))):
        dominance_frontier[node] = set()

        # Local dominance frontier
        for successor in cfg.successors(node):
            if not dominator_tree.has_edge(node, successor):
                dominance_frontier[node].add(successor)

        # Upward dominance frontier
        for child in dominator_tree.successors(node):
            for frontier_node in dominance_frontier[child]:
                if not dominator_tree.has_edge(node, frontier_node):
                    dominance_frontier[node].add(frontier_node)

    return dominance_frontier

def convert_to_ssa(cfg):
    ssa_mapping = {}
    counter = {}

    def rename_variable(variable):
        if variable not in counter:
            counter[variable] = 0
        counter[variable] += 1
        return f"{variable}_{counter[variable]}"

    def visit(node):
        for instruction in cfg[node]:
            if instruction.opcode == 'assign':
                operand = instruction.operands[0]
                if operand not in ssa_mapping:
                    ssa_mapping[operand] = rename_variable(operand)
                instruction.operands[0] = ssa_mapping[operand]
            elif instruction.opcode == 'phi':
                for i, operand in enumerate(instruction.operands):
                    if operand not in ssa_mapping:
                        ssa_mapping[operand] = rename_variable(operand)
                    instruction.operands[i] = ssa_mapping[operand]

    # Perform topological sort to ensure proper order of SSA conversion
    topological_order = list(nx.topological_sort(cfg))

    # Visit each node in the topological order
    for node in topological_order:
        visit(node)

    return cfg

# Example usage:
# ssa_cfg = convert_to_ssa(cfg)



cfg = nx.DiGraph()
cfg.add_edges_from([(1, 2), (1, 7), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (6, 2), (6, 7)])
entry_node = 1

# Find dominators for each node in the CFG
dominators = find_dominators(cfg, entry_node)
print(dominators)

# Construct dominator tree from dominators
dominator_tree = construct_dominator_tree(cfg, dominators)
print(dominator_tree)

dominance_frontier = compute_dominance_frontier(cfg, dominator_tree)
# Print the dominator tree
print("Dominance Frontier:")
print(dominance_frontier)
