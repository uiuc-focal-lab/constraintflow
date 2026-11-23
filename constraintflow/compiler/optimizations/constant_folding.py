from constraintflow.compiler.ir import *


def collect_static_list_elements(lhs):
    if isinstance(lhs, IrEmptyList):
        return []
    if isinstance(lhs, list):
        return list(lhs)
    if isinstance(lhs, IrAppendList):
        elements = collect_static_list_elements(lhs.children[0])
        if elements is not None:
            elements.append(lhs.children[1])
            return elements
    return None


def fold_append_list(lhs, rhs):
    elements = collect_static_list_elements(lhs)
    if elements is None:
        return None
    elements.append(rhs)
    return elements


def fold_expr(expr):
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, list):
        return [fold_expr(child) for child in expr]
    if isinstance(expr, IrEmptyList):
        return expr

    new_children = [fold_expr(child) for child in expr.children]
    expr.update_parent_child(new_children)

    if isinstance(expr, IrAppendList):
        # print(expr)
        # print(expr.children[0])
        # print(expr.children[1])
        # rljh
        folded = fold_append_list(expr.children[0], expr.children[1])
        if folded is not None:
            # lsdh
            return folded

    if isinstance(expr, IrListExtract):
        list_ir = expr.children[0]
        index = expr.children[1]
        if isinstance(list_ir, list) and isinstance(index, int):
            return list_ir[index]

    return expr


def fold_block(block):
    ir_list = block.children
    for i in range(len(ir_list)):
        if isinstance(ir_list[i], IrAssignment):
            new_expr = fold_expr(ir_list[i].children[1])
            ir_list[i].update_parent_child([ir_list[i].children[0], new_expr])
        elif isinstance(ir_list[i], IrTransRetBasic):
            new_children = [fold_expr(child) for child in ir_list[i].children]
            ir_list[i].update_parent_child(new_children)
    if block.inner_jump is not None:
        block.inner_jump[0] = fold_expr(block.inner_jump[0])
    if block.jump is not None:
        block.jump[0] = fold_expr(block.jump[0])


def fold_cfg(cfg):
    for node in cfg.nodes:
        fold_block(cfg.ir[node])


def constant_fold(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformer_ir = ir.tstore[transformer][i]
            if transformer_ir.layerwise_cfgs is not None:
                for cfg in transformer_ir.layerwise_cfgs.values():
                    fold_cfg(cfg)
            else:
                fold_cfg(transformer_ir.cfg)
    return ir
