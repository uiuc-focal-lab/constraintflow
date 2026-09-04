from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses


# A polyexp/symexp variable whose every use reads only the coefficient half
# leaves its constant half dead. DCE works per assignment and cannot see this,
# because the constant is still a child of the IrCombineTo* node -- so the
# producer keeps a use and survives. Replacing that child with a literal zero
# drops the use and lets the ordinary DCE below collect the producer.
COEFF_ONLY_CONSUMERS = (IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff, IrMapNeuron)


def uses_are_coeff_only(defn):
    result = True

    def scan(expr, parent):
        nonlocal result
        if not result or isinstance(expr, int) or expr is None:
            return
        if isinstance(expr, IrVar):
            if expr.defs is defn and not isinstance(parent, COEFF_ONLY_CONSUMERS):
                result = False
            return
        for child in getattr(expr, 'children', []):
            scan(child, expr)

    for user in defn.uses:
        if isinstance(user, IrAssignment):
            scan(user.children[1], None)
        elif isinstance(user, (list, tuple)):
            for e in user:
                scan(e, None)
        else:
            for child in getattr(user, 'children', []):
                scan(child, None)
    return result


def kill_dead_poly_const_block(block):
    for stmt in block.children:
        if not isinstance(stmt, IrAssignment):
            continue
        rhs = stmt.children[1]
        if not isinstance(rhs, (IrCombineToPoly, IrCombineToSym)):
            continue
        if isinstance(rhs.children[1], IrConst):
            continue
        if uses_are_coeff_only(stmt):
            rhs.update_parent_child([rhs.children[0], IrConst(0.0, 'Float'), rhs.children[2]])


def dce_block(block):
    ir_list = block.children
    for i in range(len(ir_list)-1,-1,-1):
        if isinstance(ir_list[i], IrAssignment):
            if len(ir_list[i].uses) == 0:
                del ir_list[i]

def dce_cfg(cfg):
    for node in cfg.nodes:
        block = cfg.ir[node]
        dce_block(block)

def dce(ir):
    uses.populate_uses_defs(ir)
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            for node in cfg.nodes:
                kill_dead_poly_const_block(cfg.ir[node])
    uses.populate_uses_defs(ir)
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            cfg = ir.tstore[transformer][i].cfg
            dce_cfg(cfg)