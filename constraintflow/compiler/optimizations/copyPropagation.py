from constraintflow.compiler.ir import *
from constraintflow.compiler import representations



def replace_all_occurrences_metadata(irMetadata, old_var, new_var):
    for j, irMetadataElement in enumerate(irMetadata):
        for i in range(len(irMetadataElement.shape)):
            if isinstance(irMetadataElement.shape[i], IrAst):
                replace_all_occurrences_expr(irMetadataElement.shape[i], old_var, new_var)
        for i in range(len(irMetadataElement.broadcast)):
            if isinstance(irMetadataElement.broadcast[i], IrAst):
                replace_all_occurrences_expr(irMetadataElement.broadcast[i], old_var, new_var)
                

def replace_all_occurrences_expr(expr, old_var, new_var):
    if not (isinstance(expr, (int, float, str)) or expr is None):
        replace_all_occurrences_metadata(expr.irMetadata, old_var, new_var)
        if expr == old_var:
            return new_var
        for i in range(len(expr.children)):
            new_child = replace_all_occurrences_expr(expr.children[i], old_var, new_var)
            expr.children[i] = new_child
    return expr
    

def replace_all_occurrences(old_var, new_var, cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        ir_list = block.children
        for i in range(len(ir_list)):
            new_children = []
            for child in ir_list[i].children:
                new_children.append(replace_all_occurrences_expr(child, old_var, new_var))
            ir_list[i].update_parent_child(new_children)
        if block.inner_jump != None:
            block.inner_jump[0] = replace_all_occurrences_expr(block.inner_jump[0], old_var, new_var)
        if block.jump != None:
            block.jump[0] = replace_all_occurrences_expr(block.jump[0], old_var, new_var)

def _target(var, mapping):
    """Follow a copy chain to its end; mirrors applying the copies in order."""
    seen = set()
    while var.name in mapping and var.name not in seen:
        seen.add(var.name)
        var = mapping[var.name]
    return var


def _copy_map(ir_list):
    """Every `x = y` in the block, resolved through the copies before it."""
    mapping, removed = {}, []
    for i, stmt in enumerate(ir_list):
        if isinstance(stmt, IrAssignment) and isinstance(stmt.children[1], IrVar):
            mapping[stmt.children[0].name] = _target(stmt.children[1], mapping)
            removed.append(i)
    return mapping, removed


def _rewrite_metadata(irMetadata, mapping):
    for irMetadataElement in irMetadata:
        for seq in (irMetadataElement.shape, irMetadataElement.broadcast):
            for i in range(len(seq)):
                if isinstance(seq[i], IrAst):
                    _rewrite_expr(seq[i], mapping)


def _rewrite_expr(expr, mapping):
    if isinstance(expr, (int, float, str)) or expr is None:
        return expr
    _rewrite_metadata(expr.irMetadata, mapping)
    if isinstance(expr, IrVar) and expr.name in mapping:
        return mapping[expr.name]
    for i in range(len(expr.children)):
        expr.children[i] = _rewrite_expr(expr.children[i], mapping)
    return expr


def _rewrite_cfg(cfg, mapping):
    """One walk applying every copy at once; per-copy walks were quadratic."""
    for node in cfg.nodes:
        block = cfg.ir[node]
        for stmt in block.children:
            stmt.update_parent_child(
                [_rewrite_expr(child, mapping) for child in stmt.children])
        if block.inner_jump != None:
            block.inner_jump[0] = _rewrite_expr(block.inner_jump[0], mapping)
        if block.jump != None:
            block.jump[0] = _rewrite_expr(block.jump[0], mapping)


def cp_block(block, cfg):
    ir_list = block.children
    mapping, to_be_removed = _copy_map(ir_list)
    if mapping:
        _rewrite_cfg(cfg, mapping)
    for i in range(len(to_be_removed)-1, -1, -1):
        del ir_list[to_be_removed[i]]
    return ir_list

def cp_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        cp_block(block, cfg)

def copy_proagate(ir):
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            cp_cfg(cfg)
    return ir