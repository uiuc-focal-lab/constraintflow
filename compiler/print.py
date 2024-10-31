from .ir_ast_stack2 import *
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
types = {}

def add(node):
    global G 
    global types
    G.add_node(node.identifier)
    types[node.identifier] = node
    # print(node.identifier)
    # if node.identifier == 4:
    #     print(types[4])

def add_edge(node1, node2):
    global G 
    if node1.identifier not in G.nodes:
        raise Exception('NO NODE', node1)
    if node2.identifier not in G.nodes:
        raise Exception('NO NODE', node2)
    G.add_edge(node1.identifier, node2.identifier)


def create_graph(node):
    global G
    global types
    if isinstance(node, IrProgram):
        for i, transformer_name in enumerate(node.tstore.keys()):
            transformerIr = node.tstore[transformer_name]
            for opStmtIr in transformerIr:
                create_graph(opStmtIr)
                break
                
                
    elif(isinstance(node, IrOpStmt)):
        add(node)
        for i, child in enumerate(node.children):
            add(child)
            if i==0:
                # G.add_edge(node.identifier, child.identifier)
                add_edge(node, child)
            else:
                # G.add_edge(node.children[i-1].identifier, child.identifier)
                add_edge(node.children[i-1], child)
            create_graph(child)

    elif(isinstance(node, IrWhile)):
        add(node)
        children = node.children[1:]
        for i, child in enumerate(children):
            add(child)
            if i==0:
                # G.add_edge(node.identifier, child.identifier)
                add_edge(node, child)
            else:
                # G.add_edge(node.children[i-1].identifier, child.identifier)
                add_edge(children[i-1], child)
            create_graph(child)

    elif(isinstance(node, IrStatement)):
        add(node)
        for child in node.children:
            create_graph(child)
            # G.add_edge(node.identifier, child.identifier)
            add_edge(node, child)

    elif(isinstance(node, IrExpression)):
        add(node)
        for child in node.children:
            create_graph(child)
            # G.add_edge(node.identifier, child.identifier)
            add_edge(node, child)


def pprint(node):
    create_graph(node)
    # plt.subplot(121)
    # nx.draw(G, with_labels=True)
    # plt.show()
    # print(G.nodes)
    # print(G.edges)
    # print([j for (i, j) in G.edges if i==168])
    parents = {}
    children = {}
    for i,j in G.edges:
        if j not in parents:
            parents[j] = [i]
        else:
            parents[j].append(i)
        
        if i not in children:
            children[i] = [j]
        else:
            children[i].append(j)

    # for i in G.nodes:
    #     if isinstance(types[i], IrAddDimensionConst):
    #         print('###############')
    #         print(types[i].children)
    # jfd

    multiple_parents = [i for i in parents.keys() if len(parents[i])>1]
    # print([types[i] for i in multiple_parents])
    # for i in multiple_parents:
    #     print(types[i], [types[j] for j in parents[i]])