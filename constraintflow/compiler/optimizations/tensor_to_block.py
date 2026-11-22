import json
import os
import torch 

from constraintflow.compiler.ir import *
from constraintflow.compiler.optimizations import uses, fuse_affine_subst, subexp_inlining
#FEAT: per-change toggles for the DeepPoly-efficient passes
from constraintflow.compiler.optimizations import tree_walk
from constraintflow.compiler.optimizations.subexp_inlining import get_vars_expr_occurrences
from constraintflow.gbcsr.sparse_tensor import get_operator_func
from constraintflow.lib.globals import load_capture, capture_exists, program_add_rule
from constraintflow.lib.network import LayerType
import constraintflow.lib.globals as globals


counter = -1
def get_var():
    global counter 
    counter += 1
    return 'ttb_var_' + str(counter)

def get_var_while(name, while_iteration):
    return name + '_iter_' + str(while_iteration)


def _collect_ir_ast_nodes(obj, visited_objects, ast_nodes):
    obj_id = id(obj)
    if obj_id in visited_objects:
        return
    visited_objects.add(obj_id)

    if isinstance(obj, IrAst):
        ast_nodes[obj_id] = obj

    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_ir_ast_nodes(key, visited_objects, ast_nodes)
            _collect_ir_ast_nodes(value, visited_objects, ast_nodes)
        return

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_ir_ast_nodes(item, visited_objects, ast_nodes)
        return

    if hasattr(obj, '__dict__'):
        for value in vars(obj).values():
            _collect_ir_ast_nodes(value, visited_objects, ast_nodes)


def deepcopy_cfg_with_fresh_identifiers(cfg):
    cfg_copy = copy.deepcopy(cfg)

    visited_objects = set()
    ast_nodes = {}
    _collect_ir_ast_nodes(cfg_copy, visited_objects, ast_nodes)

    # Preserve relative order by old identifiers while assigning new global ids.
    ordered_nodes = sorted(ast_nodes.values(), key=lambda node: node.identifier)

    next_identifier = IrAst.counter + 1
    for node in ordered_nodes:
        node.identifier = next_identifier
        next_identifier += 1

    IrAst.counter = next_identifier - 1
    return cfg_copy


def get_profiled_branch(layer_index, block_id):
    if block_id is None:
        return None

    rel_path = f"jit_branch/branch_{layer_index}_{block_id}.json"
    if not capture_exists(rel_path):
        return None

    json_obj = load_capture(rel_path)

    taken = json_obj["taken"]
    if taken in ("then", "else"):
        return taken
    return None

# Worklist Algorithm to find live nodes (essentially branches that are taken based on the profiling data that we collect during simulacrum mode)
def get_live_nodes(cfg, layer_index):
    live_nodes = set()
    worklist = [cfg.entry_node]

    while worklist:
        node = worklist.pop()
        if node in live_nodes or node not in cfg.ir:
            continue

        live_nodes.add(node)
        block = cfg.ir[node]
        taken = None
        if block.inner_jump is not None and len(block.inner_jump) == 3:
            taken = get_profiled_branch(layer_index, block.block_id)

        if taken == "then":
            successors = [cfg.get_block_id(block.inner_jump[1])]
        elif taken == "else":
            successors = [cfg.get_block_id(block.inner_jump[2])]
        else:
            successors = cfg.successors[node]

        for successor in successors:
            if successor not in live_nodes:
                worklist.append(successor)

    return live_nodes


def convert_to_ir_ttb(expr, layer_index, while_iteration):
    # targets = ()
    # targets = (IrUnaryOp)
    # targets= (IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp, IrDot, IrTernary, IrUnaryOp, IrGetDefaultStop)
    
    targets = (
        IrBinaryOp, IrMult, IrInnerProduct, IrRepeat, IrClamp,
        IrDot, IrTernary, IrUnaryOp, IrGetDefaultStop,
        IrGetPriorityLList, IrGetPolyexpStop, IrGetPolyexpNotStop,
        IrAddDimension, IrAddDimensionConst, IrRemoveDimension, IrAccess,
        IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff,
        IrReduce, IrEpsilon, IrConvertNeuronToPoly,
        IrConcatStitch, IrConcatStitchMat, IrUpdateReplay
        # IrGetAbsElemSparseDKey, # IrGetPolyExpSparseConst,
        # IrGetPolyExpSparseMat
    )
    # targets = ()
    if not isinstance(expr, targets):
        return expr, []

    if isinstance(expr, IrUnaryOp) and expr.op in ('get_shape_1', 'get_shape_0'):
        return expr, []
    binary_instance = expr.ttb_counter

    if while_iteration is None and getattr(expr, "inside_while", False):
        while_iteration = -1
    
    
    
    if isinstance(expr, IrUnaryOp):
        if expr.op == 'any':
            filename = f"jit_any/any_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif expr.op == 'all':
            filename = f"jit_all/all_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif expr.op == 'get_dims':
            filename = f"jit_get_dims/get_dims_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        else:
            filename = f"jit_unary/unary_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrMult) or isinstance(expr, IrBinaryOp):
        filename = f"jit_binary/binary_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrInnerProduct):
        filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrRepeat):
        filename = f"jit_repeat/repeat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrClamp):
        filename = f"jit_clamp/clamp_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrAddDimension):
        filename = f"jit_unsqueeze/unsqueeze_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrAddDimensionConst):
        filename = f"jit_add_dim_const/add_dim_const_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrRemoveDimension):
        filename = f"jit_squeeze/squeeze_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrTernary):
        filename = f"jit_where/where_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrDot):
        [lhsIr, rhsIr] = expr.children
        if lhsIr.irMetadata[-1].type == 'Float':
            filename = f"jit_matmul/matmul_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        elif lhsIr.irMetadata[-1].type == 'Neuron' or rhsIr.irMetadata[-1].type == 'Neuron':
            filename = f"jit_Llist_dot/Llist_dot_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
        else:
            return expr, []
    elif isinstance(expr, IrGetDefaultStop):
        filename = f"jit_defaultstop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPriorityLList):
        filename = f"jit_priority/priority_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    
    elif isinstance(expr, IrGetPolyexpStop):
        filename = f"jit_polyexp_stop/stop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrGetPolyexpNotStop):
        filename = f"jit_polyexp_not_stop/notstop_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrReduce):
        filename = f"jit_sum/sum_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrEpsilon):
        filename = f"jit_new_eps/new_eps_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json"
    elif isinstance(expr, IrAccess) and (not expr.isMetadata):
        filename = f'jit_Abs_elem_sparse_get_elem/Abs_elem_sparse_get_elem_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrAccess) and expr.isMetadata:
        filename = f'jit_llist_get_metadata/llist_get_metadata_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrExtractPolyCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrExtractSymCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrMapCoeff):
        filename = f'jit_poly_exp_sparse_get_mat/poly_exp_sparse_get_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrConvertNeuronToPoly):
        filename = f'jit_convert_to_poly/convert_to_poly_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrConcatStitch):
        filename = f'jit_concat/concat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'
    elif isinstance(expr, IrConcatStitchMat):
        filename = f'jit_concat/concat_mat_{layer_index}_{binary_instance}_{expr.inside_while}_{expr.while_number}_{while_iteration}.json'

    # The update tape travels on the node itself; every other kind is loaded by name.
    json_list = expr.records if isinstance(expr, IrUpdateReplay) else load_capture(filename)
    if isinstance(expr, IrTernary):
        cond = expr.children[0]
        lhs = expr.children[1]
        rhs = expr.children[2]
    elif isinstance(expr, IrUnaryOp):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrGetDefaultStop):
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrEpsilon):
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrGetPriorityLList):
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrGetPolyexpStop):
        cond = None
        lhs_input = expr.children[0]
        rhs_input = expr.children[1]
        lhs = IrPolyExpMat(lhs_input)
        # The tape now carries the bool-to-float conversion, so bind the raw stop.
        rhs = rhs_input
    elif isinstance(expr, IrGetPolyexpNotStop):
        cond = None
        lhs_input = expr.children[0]
        rhs_input = expr.children[1]
        lhs = IrPolyExpMat(lhs_input)
        rhs = rhs_input
    elif isinstance(expr, IrAddDimension) or isinstance(expr, IrRemoveDimension) or isinstance(expr, IrReduce):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrAddDimensionConst):
        # The constant is baked into each record's dense_const, so the tape never refers back to
        # an operand -- there is nothing to bind here.
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrConvertNeuronToPoly):
        # Built from the llist alone; the tape refers to no operand.
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, (IrConcatStitch, IrConcatStitchMat)):
        # Everything (which block ids to extract, the parent layout) is baked
        # into the captured tape; the tape refers to no operand.
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrUpdateReplay):
        # The tape names its operand `abs_shape_<n>`, resolved in the noop handler.
        cond = None
        lhs = None
        rhs = None
    elif isinstance(expr, IrAccess) and (not expr.isMetadata):
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, IrAccess) and expr.isMetadata:
        cond = None
        lhs = expr.children[0]
        rhs = None
    elif isinstance(expr, (IrExtractPolyCoeff, IrExtractSymCoeff, IrMapCoeff)):
        cond = None
        lhs = expr.children[0]
        rhs = None
    else:
        cond = None
        lhs = expr.children[0]
        rhs = expr.children[1]
    if isinstance(expr, IrDot):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrAccess):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrEpsilon):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrAddDimensionConst):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrConvertNeuronToPoly):
        irMetadata = expr.irMetadata
    elif isinstance(expr, (IrConcatStitch, IrConcatStitchMat)):
        irMetadata = expr.irMetadata
    elif isinstance(expr, IrUpdateReplay):
        irMetadata = expr.irMetadata
    elif isinstance(rhs, IrAst):
        irMetadata = rhs.irMetadata
    elif isinstance(lhs, IrAst):
        irMetadata = lhs.irMetadata
    else:
        irMetadata = None
    new_assignments = []
    output_vars = []

    for json_index, json_obj in enumerate(json_list):
        new_name = get_var()
        new_var = IrVar(new_name, copy_metadata(irMetadata) if irMetadata is not None else None)
        if json_obj["method"] == "noop":
            if json_obj["input"] == "cond":
                output = cond
            elif json_obj["input"] == "rhs":
                output = rhs
            elif json_obj["input"] == "lhs":
                output = lhs
            elif 'json_list_' in json_obj["input"]:
                output = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"].startswith("abs_shape_"):
                output = expr.children[0]
            else:
                # print(json_obj["input"])
                raise Exception("NOT IMPLEMENTED")
        
        elif json_obj["method"] == "SparseTensorConstructor":
            if isinstance(json_obj["blocks"], list):
                blocksIr = []
            elif "json_list_" in json_obj["blocks"]:
                blocksIr = output_vars[int(json_obj["blocks"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED SparseTensorConstructor blocks ref")

            total_sizeIr = torch.tensor(json_obj["total_size"], dtype=torch.int64)

            end_indices = [torch.tensor(end_index, dtype=torch.int64) for end_index in json_obj["end_indices"]]

            output = IrSparseTensor(
                [torch.tensor(json_obj["start_indices"][i], dtype=torch.int64) for i in range(len(json_obj["start_indices"]))],
                blocksIr,
                json_obj["dims"],
                total_sizeIr,
                end_indices=end_indices,
                type=getattr(builtins, json_obj["type"]),
                dense_const=json_obj["dense_const"],
                delete_indices=json_obj["delete_indices"],
            )


        elif json_obj["method"] == "initialise":
            output = json_obj["value"]
            if output == "[]":
                output = IrEmptyList()
            elif isinstance(output, bool):
                output = IrConst(output, 'Bool')
            else:
                raise Exception(f"NOT IMPLEMENTED output: {output}")

        elif json_obj["method"] == "bool_value":
            output = IrConst(json_obj["value"], 'Bool')
        elif json_obj["method"] == "unary_block":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockUnaryOp(inputIr, json_obj["op"])

        elif json_obj["method"] == "simple_unary":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            op_ref = json_obj.get("op", json_obj.get("operation"))
            if isinstance(op_ref, str) and "json_list_" in op_ref:
                op_idx = int(op_ref.split("_")[-1])
                # Inline lambdas directly rather than referencing their ttb_var;
                # see the apply_lambda note below for why the reference would be
                # dropped by subexp_inlining.
                if json_list[op_idx]["method"] == "lambda":
                    op_ref = IrLambda(json_list[op_idx]["op"])
                else:
                    op_ref = output_vars[op_idx]
            output = IrSimpleUnary(inputIr, op_ref)

        elif json_obj["method"] == "apply_lambda":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if not (isinstance(json_obj["op"], str) and "json_list_" in json_obj["op"]):
                raise Exception("NOT IMPLEMENTED")
            lambda_obj = json_list[int(json_obj["op"].split("_")[-1])]
            if lambda_obj["method"] != "lambda":
                raise Exception("NOT IMPLEMENTED")
            # Inline the lambda directly as the op instead of referencing the
            # ttb_var it was assigned to. 
            output = IrSimpleUnary(inputIr, IrLambda(lambda_obj["op"]))

        elif json_obj["method"] == "torch_sigmoid":
            input_ref = json_obj.get("block", json_obj.get("input"))
            if "json_list_" in input_ref:
                inputIr = output_vars[int(input_ref.split("_")[-1])]
            elif input_ref == "lhs":
                inputIr = lhs
            elif input_ref == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrSimpleUnary(inputIr, json_obj.get("op", "sigma"))

        elif json_obj["method"] == "lambda":
            output = IrLambda(json_obj["op"])

            
        elif json_obj["method"] == "append_list":
            if 'json_list_' in json_obj["list"]:
                list1 = output_vars[int(json_obj["list"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if 'json_list_' in json_obj["value"]:
                val = output_vars[int(json_obj["value"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrAppendList(list1, val)

        elif json_obj["method"] == "index_lookup":
            if "json_list_" in json_obj["input"]:
                listIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            if isinstance(json_obj["index"], int):
                indexIr = json_obj["index"]
            elif "json_list_" in json_obj["index"]:
                indexIr = output_vars[int(json_obj["index"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            output = IrListExtract(listIr, indexIr)

        elif json_obj["method"] == "extract_block":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:                
                raise Exception("NOT IMPLEMENTED")
            if "index" in json_obj:
                if isinstance(json_obj["index"], int):
                    indexIr = json_obj["index"]
                elif "json_list_" in json_obj["index"]:
                    indexIr = output_vars[int(json_obj["index"].split("_")[-1])]
                else:                
                    raise Exception("NOT IMPLEMENTED")
            elif "block_id" in json_obj:
                indexIr = json_obj["block_id"]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockExtract(inputIr, indexIr)

        elif json_obj["method"] == "block_copy":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockCopy(inputIr)
            
        elif json_obj["method"] == "binary":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrBlockBinaryOp(lhsIr, rhsIr, get_operator_func(json_obj["op"]))

        elif json_obj["method"] == "torch_binary":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrSimpleBinary(lhsIr, rhsIr, json_obj["op"])

        elif json_obj["method"] == "torch_where":
            condIr = output_vars[int(json_obj["cond"].split("_")[-1])]
            lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            output = IrTorchWhere(condIr, lhsIr, rhsIr)

        elif json_obj["method"] == "DenseBlockConstructor":
            if "json_list_" not in str(json_obj["block"]):
                raise Exception("NOT IMPLEMENTED DenseBlockConstructor block ref")
            blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            output = IrDenseBlock(
                blockIr,
                total_shape="torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                batch_size=json_obj["batch_size"],
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "DiagonalBlockConstructor":
            if "json_list_" not in str(json_obj["block"]):
                raise Exception("NOT IMPLEMENTED DiagonalBlockConstructor block ref")
            blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            output = IrDiagonalBlock(
                blockIr,
                "torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                json_obj["diag_index"],
                batch_size=json_obj["batch_size"],
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "RepeatBlockConstructor":
            if "json_list_" not in str(json_obj["block"]):
                raise Exception("NOT IMPLEMENTED RepeatBlockConstructor block ref")
            blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            output = IrRepeatBlock(
                blockIr,
                "torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                repeat_dims="torch.tensor(" + str(json_obj["repeat_dims"]) + ")",
                only_one_repeat=json_obj["only_one_repeat"],
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "KernelBlockConstructor":
            if "json_list_" not in str(json_obj["block"]):
                raise Exception("NOT IMPLEMENTED KernelBlockConstructor block ref")
            blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            output = IrKernelBlock(
                blockIr,
                "torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                json_obj["ix"],
                json_obj["iy"],
                json_obj["ox"],
                json_obj["oy"],
                json_obj["sx"],
                json_obj["sy"],
                json_obj["px"],
                json_obj["py"],
                kx=json_obj["kx"],
                ky=json_obj["ky"],
                num_channels=json_obj["num_channels"],
                num_kernels=json_obj["num_kernels"],
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "PatchesBlockConstructor":
            if "json_list_" not in str(json_obj["block"]):
                raise Exception("NOT IMPLEMENTED PatchesBlockConstructor block ref")
            blockIr = output_vars[int(str(json_obj["block"]).split("_")[-1])]
            output = IrPatchesBlock(
                blockIr,
                "torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                json_obj["ix"],
                json_obj["iy"],
                json_obj["ox"],
                json_obj["oy"],
                json_obj["sx"],
                json_obj["sy"],
                json_obj["px"],
                json_obj["py"],
                json_obj["kx"],
                json_obj["ky"],
                json_obj["num_channels"],
                json_obj["num_kernels"],
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "ConstBlockConstructor":
            # A bare literal payload is kept as a plain Python value.
            if isinstance(json_obj["block"], str) and "json_list_" in json_obj["block"]:
                blockIr = output_vars[int(json_obj["block"].split("_")[-1])]
            else:
                blockIr = json_obj["block"]
            output = IrConstBlock(
                blockIr,
                "torch.tensor(" + str(json_obj["total_shape"]) + ", dtype=torch.int64)",
                sparse_block_type=json_obj["SparseBlockType"],
            )

        elif json_obj["method"] == "matmul_unequal_dims":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            output = IrBlockInnerProduct(lhsIr, rhsIr, type='unequal_dims')

        elif json_obj["method"] == "matmul_equal_dims":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            output = IrBlockInnerProduct(lhsIr, rhsIr, type='equal_dims')
        
        elif json_obj["method"] == "block_clamp":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockClamp(inputIr, json_obj["const"], json_obj["min_true"])

        elif json_obj["method"] == "block_squeeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrBlockSqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "tensor_ones":
            #FIX FOR: records without a batch dim (llist's layer-size ones) were batched when repeat_dims[0] == 1.
            if isinstance(json_obj["repeat_dims"], list) and len(json_obj["repeat_dims"]) > 0 and json_obj["repeat_dims"][0] == 1 and json_obj.get("batch_dim", True):
                output = IrTensorOnes("[batch_size" + "".join([", " + str(json_obj["repeat_dims"][i]) for i in range(1, len(json_obj["repeat_dims"]))]) + "]")
            else:
                output = IrTensorOnes(torch.tensor(json_obj["repeat_dims"], dtype=torch.int64))

        # torch_ones: sparse_block DummyBlock.get_dense_const_block (size = total_shape).
        elif json_obj["method"] in ("torch_ones", "torch.ones"):
            size = json_obj.get("size")
            if size is None:
                size = json_obj.get("repeat_dims")
            if size is None:
                size = json_obj.get("input")
                if isinstance(size, list) and len(size) == 1 and isinstance(size[0], list):
                    size = size[0]
            #FIX FOR: records without a batch dim (symexp's neuron-count ones) were batched when size[0] == 1.
            if isinstance(size, list) and len(size) > 0 and size[0] == 1 and json_obj.get("batch_dim", True):
                output = IrTensorOnes("[batch_size" + "".join([", " + str(size[i]) for i in range(1, len(size))]) + "]")
            else:
                output = IrTensorOnes(torch.tensor(size, dtype=torch.int64))

        elif json_obj["method"] in ("torch_mul", "torch.mul"):
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrSimpleBinary(lhsIr, rhsIr, "mul")

        elif json_obj["method"] == "torch_matmul":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrTorchMatmul(lhsIr, rhsIr)

        elif json_obj["method"] == "torch_unsqueeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchUnsqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "torch_squeeze":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchSqueeze(inputIr, json_obj["index"])

        elif json_obj["method"] == "torch_reshape":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchReshape(inputIr, shape)

        elif json_obj["method"] == "torch_view":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchView(inputIr, shape)

        elif json_obj["method"] == "torch_repeat":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchRepeat(inputIr, json_obj["repeats"])

        elif json_obj["method"] == "torch_pad":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchPad(inputIr, json_obj["pad"])

        elif json_obj["method"] == "torch_expand":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            shape = json_obj["shape"]
            if isinstance(shape, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    shape = shape.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchExpand(inputIr, shape)

        elif json_obj["method"] == "torch_sum":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchSum(inputIr, json_obj["dim"])

        elif json_obj["method"] == "torch_zeros":
            size = json_obj["size"]
            if isinstance(size, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    size = size.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            device = json_obj.get("device", None)
            if isinstance(device, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    device = device.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            dtype = json_obj.get("dtype", None)
            if isinstance(dtype, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    dtype = dtype.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchZeros(size, device=device, dtype=dtype)

        elif json_obj["method"] == "torch_eye":
            size = json_obj["size"]
            if isinstance(size, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    size = size.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            device = json_obj.get("device", None)
            if isinstance(device, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    device = device.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            dtype = json_obj.get("dtype", None)
            if isinstance(dtype, str):
                for ref_idx in range(len(output_vars) - 1, -1, -1):
                    dtype = dtype.replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchEye(size, device=device, dtype=dtype)

        elif json_obj["method"] == "torch_float":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchFloat(inputIr)

        elif json_obj["method"] == "torch_diag_embed":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchDiagEmbed(inputIr)

        elif json_obj["method"] == "torch_stride":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchStride(inputIr)

        elif json_obj["method"] == "torch_as_strided":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            size = json_obj["size"]
            stride = json_obj["stride"]
            output = IrTorchAsStrided(inputIr, size, stride)

        elif json_obj["method"] == "patches_to_dense":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrPatchesToDense(inputIr, json_obj["geometry"])

        elif json_obj["method"] == "tensor_scatter":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            output = IrTensorScatter(inputIr, valueIr, json_obj["dim"], json_obj["index"])

        elif json_obj["method"] == "torch_slice":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            index = json_obj["index"]
            for item_idx in range(len(index)):
                if isinstance(index[item_idx], str):
                    for ref_idx in range(len(output_vars) - 1, -1, -1):
                        index[item_idx] = index[item_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
                elif isinstance(index[item_idx], list):
                    for range_idx in range(len(index[item_idx])):
                        if isinstance(index[item_idx][range_idx], str):
                            for ref_idx in range(len(output_vars) - 1, -1, -1):
                                index[item_idx][range_idx] = index[item_idx][range_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
            output = IrTorchSlice(inputIr, index)

        elif json_obj["method"] == "F.conv2d":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["weight"]:
                weightIr = output_vars[int(json_obj["weight"].split("_")[-1])]
            elif json_obj["weight"] == "lhs":
                weightIr = lhs
            elif json_obj["weight"] == "rhs":
                weightIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFConv2d(inputIr, weightIr, json_obj["stride"], json_obj["padding"])

        elif json_obj["method"] == "F.conv_transpose2d":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["weight"]:
                weightIr = output_vars[int(json_obj["weight"].split("_")[-1])]
            elif json_obj["weight"] == "lhs":
                weightIr = lhs
            elif json_obj["weight"] == "rhs":
                weightIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFConvTranspose2d(
                inputIr,
                weightIr,
                stride=json_obj["stride"],
                padding=json_obj["padding"],
                output_padding=json_obj["output_padding"],
            )

        elif json_obj["method"] == "F.unfold":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFUnfold(inputIr, json_obj["kernel_size"], json_obj["padding"], json_obj["stride"])

        elif json_obj["method"] == "F.fold":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrFFold(inputIr, json_obj["output_size"], json_obj["kernel_size"], json_obj["stride"], json_obj.get("padding"))

        elif json_obj["method"] == "torch_einsum":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchEinsum(json_obj["equation"], [lhsIr, rhsIr])

        elif json_obj["method"] == "torch_cat":
            inputIrs = [output_vars[int(ref.split("_")[-1])] for ref in json_obj["inputs"]]
            output = IrTorchCat(inputIrs, json_obj["dim"])

        elif json_obj["method"] == "torch_diagonal":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTorchDiagonal(inputIr, json_obj["dim1"], json_obj["dim2"])

        elif json_obj["method"] == "torch_permute":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            if "permutation" in json_obj:
                permutation = json_obj["permutation"]
            else:
                permutation = json_obj["perm"]
            output = IrTorchPermute(inputIr, permutation)

        elif json_obj["method"] == "torch_transpose":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            elif json_obj["input"] == "lhs":
                inputIr = lhs
            elif json_obj["input"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            dim0, dim1 = json_obj["dims"]
            output = IrTorchTranspose(inputIr, dim0, dim1)

        elif json_obj["method"] == "tensor_repeat":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                inputIr = lhs
            elif json_obj["lhs"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            repeat_dims = torch.tensor(json_obj["repeat_dims"], dtype=torch.int64)
            output = IrTensorRepeat(inputIr, repeat_dims)

        elif json_obj["method"] == "tensor_clamp":
            if "json_list_" in json_obj["lhs"]:
                inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                inputIr = lhs
            elif json_obj["lhs"] == "rhs":
                inputIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrTensorClamp(inputIr, json_obj['const'], json_obj['min_true'])

        elif json_obj["method"] == "simple_binary":
            if "json_list_" in json_obj["lhs"]:
                lhsIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
            elif json_obj["lhs"] == "lhs":
                lhsIr = lhs
            elif json_obj["lhs"] == "rhs":
                lhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            
            if "json_list_" in json_obj["rhs"]:
                rhsIr = output_vars[int(json_obj["rhs"].split("_")[-1])]
            elif json_obj["rhs"] == "lhs":
                rhsIr = lhs
            elif json_obj["rhs"] == "rhs":
                rhsIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrSimpleBinary(lhsIr, rhsIr, json_obj['op'])
        
        elif json_obj["method"] == "sp_where_block":
            condIr = output_vars[int(json_obj["x"].split("_")[-1])]
            lhsIr  = output_vars[int(json_obj["y"].split("_")[-1])]
            rhsIr  = output_vars[int(json_obj["z"].split("_")[-1])]
            output = IrBlockWhereBlock(condIr, lhsIr, rhsIr)

        elif json_obj["method"] == "new_eps":
            matIr   = output_vars[int(json_obj["mat"].split("_")[-1])]
            constIr = output_vars[int(json_obj["const"].split("_")[-1])]
            output = IrNewEps(matIr, constIr)

        elif json_obj["method"] == "scalar_const":
            output = IrConst(json_obj["value"], 'Float')

        elif json_obj["method"] == "create_similar":
            matIr = output_vars[int(json_obj["mat"].split("_")[-1])]
            if "const" in json_obj:
                constIr = output_vars[int(json_obj["const"].split("_")[-1])]
                output = IrBlockPolyexpNotStop(expr.children[0], matIr, constIr)
            else:
                output = IrBlockPolyexpStop(expr.children[0], matIr)

        elif json_obj["method"] == "any":
            output = IrBlockAny(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "any_torch":
            output = IrBlockAny(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "all":
            output = IrBlockAll(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "all_torch":
            output = IrBlockAll(output_vars[int(json_obj["input"].split("_")[-1])])

        elif json_obj["method"] == "get_dims":
            output = IrBlockGetDims(output_vars[int(json_obj["input"].split("_")[-1])])
        
        elif json_obj["method"] == "object_lookup":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrObjectLookup(inputIr, json_obj["object"])

        elif json_obj["method"] in ("sparse_block_extract", "extract_sparse_block"):
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrObjectLookup(inputIr, "block")

        elif json_obj["method"] == "block_create_similar":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["arg"]:
                argIr = output_vars[int(json_obj["arg"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            output = IrBlockCreateSimilar(inputIr, argIr)

        elif json_obj["method"] == "block_set_total_shape_last_dim":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if isinstance(json_obj["value"], int):
                valueIr = json_obj["value"]
            elif "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            new_assignment = IrAssignment(new_var, inputIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            new_assignments.append(IrSetBlockTotalShapeLastDim(new_var, valueIr))

            if len(output_vars) != json_obj["output"] + 1:
                print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
            assert(len(output_vars) == json_obj["output"] + 1)
            continue

        # elif json_obj["method"] == "get_sub_block_custom_range_block":
        #     if "json_list_" in json_obj["lhs"]:
        #         inputIr = output_vars[int(json_obj["lhs"].split("_")[-1])]
        #     else:
        #         raise Exception("NOT IMPLEMENTED")

        #     output = IrBlockGetSubBlockCustomRange(
        #         inputIr,
        #         torch.tensor(json_obj["start_index"], dtype=torch.int64),
        #         torch.tensor(json_obj["end_index"], dtype=torch.int64),
        #         torch.tensor(json_obj["block_start_index"], dtype=torch.int64),
        #     )
        elif json_obj["method"] == "torch_any":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrBlockAny(inputIr)

        elif json_obj["method"] == "const_any":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrSimpleBinary(inputIr, IrConst(True, 'Bool'), 'eq')

        elif json_obj["method"] == "torch_clamp":
            inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            output = IrTensorClamp(inputIr, json_obj["const"], json_obj["min_true"])

        elif json_obj["method"] == "assign_to_view":
            if "json_list_" in json_obj["input"]:
                inputIr = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")

            if "base" in json_obj and "json_list_" in json_obj["base"]:
                baseIr = output_vars[int(json_obj["base"].split("_")[-1])]
            else:
                baseIr = inputIr

            if "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            elif json_obj["value"] == "lhs":
                valueIr = lhs
            elif json_obj["value"] == "rhs":
                valueIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            index = json_obj["index"]
            for item_idx in range(len(index)):
                if isinstance(index[item_idx], str):
                    for ref_idx in range(len(output_vars) - 1, -1, -1):
                        index[item_idx] = index[item_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)
                elif isinstance(index[item_idx], list):
                    for range_idx in range(len(index[item_idx])):
                        if isinstance(index[item_idx][range_idx], str):
                            for ref_idx in range(len(output_vars) - 1, -1, -1):
                                index[item_idx][range_idx] = index[item_idx][range_idx].replace("json_list_" + str(ref_idx), output_vars[ref_idx].name)

            new_assignment = IrAssignment(new_var, baseIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            new_assignments.append(IrAssignToView(inputIr, index, valueIr))

            if len(output_vars) != json_obj["output"] + 1:
                print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
            assert(len(output_vars) == json_obj["output"] + 1)
            continue

        elif json_obj["method"] == "assign_to_block":
            if "json_list_" in json_obj["assign_to"]:
                assignToIr = output_vars[int(json_obj["assign_to"].split("_")[-1])]
            elif json_obj["assign_to"] == "lhs":
                assignToIr = lhs
            elif json_obj["assign_to"] == "rhs":
                assignToIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            if "json_list_" in json_obj["value"]:
                valueIr = output_vars[int(json_obj["value"].split("_")[-1])]
            elif json_obj["value"] == "lhs":
                valueIr = lhs
            elif json_obj["value"] == "rhs":
                valueIr = rhs
            else:
                raise Exception("NOT IMPLEMENTED")

            new_assignments.append(IrAssignToBlock(assignToIr, valueIr))
            new_assignment = IrAssignment(new_var, assignToIr)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "PolyExpSparseConstructor":
            if "json_list_" in json_obj["mat"]:
                mat_ir = output_vars[int(json_obj["mat"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if "json_list_" in json_obj["const"]:
                const_ir = output_vars[int(json_obj["const"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if irMetadata is None:
                raise Exception("PolyExpSparse replay requires PolyExp irMetadata")
            mat_metadata = copy_metadata(irMetadata)
            mat_metadata[-1].type = 'Float'
            mat_metadata[-1].shape.append(IrAst.poly_size)
            mat_metadata[-1].broadcast.append(1)
            mat_ir.irMetadata = mat_metadata
            const_metadata = copy_metadata(irMetadata)
            const_metadata[-1].type = 'Float'
            const_ir.irMetadata = const_metadata
            output = IrCombineToPoly(mat_ir, const_ir)
            new_assignment = IrAssignment(new_var, output)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "SymExpSparse":
            if "json_list_" in json_obj["mat"]:
                mat_ir = output_vars[int(json_obj["mat"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if "json_list_" in json_obj["const"]:
                const_ir = output_vars[int(json_obj["const"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            if irMetadata is None:
                raise Exception("SymExpSparse replay requires SymExp irMetadata")
            mat_metadata = copy_metadata(irMetadata)
            mat_metadata[-1].type = 'Float'
            mat_metadata[-1].shape.append(IrAst.sym_size)
            mat_metadata[-1].broadcast.append(1)
            mat_ir.irMetadata = mat_metadata
            const_metadata = copy_metadata(irMetadata)
            const_metadata[-1].type = 'Float'
            const_ir.irMetadata = const_metadata
            output = IrCombineToSym(mat_ir, const_ir)
            new_assignment = IrAssignment(new_var, output)
            new_assignments.append(new_assignment)
            output_vars.append(new_var)
            continue
        elif json_obj["method"] == "get_sparse_tensor_blocks":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSparseTensorBlocks(input_ir)
        elif json_obj["method"] == "get_abs_elem_sparse_d_key":
            if "json_list_" in json_obj["input"]:
                # print(len(output_vars), int(json_obj["input"].split("_")[-1]))
                # input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
                pass
            else:
                raise Exception("NOT IMPLEMENTED")
            key = json_obj["key"]
            output = IrGetAbsElemSparseDKey(None, key)
        elif json_obj["method"] == "get_poly_exp_sparse_const":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetPolyExpSparseConst(input_ir)
        elif json_obj["method"] == "get_poly_exp_sparse_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetPolyExpSparseMat(input_ir)
        elif json_obj["method"] == "get_sym_exp_sparse_const":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSymExpSparseConst(input_ir)
        elif json_obj["method"] == "get_sym_exp_sparse_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrGetSymExpSparseMat(input_ir)
        elif json_obj["method"] == "expand_symexp_mat":
            if "json_list_" in json_obj["input"]:
                input_ir = output_vars[int(json_obj["input"].split("_")[-1])]
            else:
                raise Exception("NOT IMPLEMENTED")
            output = IrExpandSymExp(input_ir)
        elif json_obj["method"] == "get_kth_layer_bias":
            output = IrGetKthLayerNetworkParam(json_obj["input"], "bias")
        elif json_obj["method"] == "get_kth_layer_weight":
            output = IrGetKthLayerNetworkParam(json_obj["input"], "weight")
        else:
            raise Exception(f"Unknown method {json_obj['method']} in replay at output {json_obj.get('output')}")
        
        new_assignment = IrAssignment(new_var, output)
        new_assignments.append(new_assignment)
        output_vars.append(new_var)

        if(len(output_vars) != json_obj["output"] + 1):
            print(f"Output list length: {len(output_vars)}, expected: {json_obj['output'] + 1}")
        assert(len(output_vars) == json_obj["output"] + 1)

    return output_vars[-1], new_assignments



# For the del statements
def collect_ir_var_names(expr, names=None):
    if names is None:
        names = set()
    if isinstance(expr, IrVar):
        names.add(expr.name)
    elif isinstance(expr, int):
        pass
    elif hasattr(expr, "children"):
        for child in expr.children:
            collect_ir_var_names(child, names)
    return names


def make_ttb_del_statement(new_assignments, keep_var_names=None):
    if keep_var_names is None:
        keep_var_names = set()
    var_names = []
    for assignment in new_assignments:
        if not isinstance(assignment, IrAssignment):
            continue
        lhs = assignment.children[0]
        if not isinstance(lhs, IrVar):
            continue
        if lhs.name.startswith("ttb_var_") and lhs.name not in keep_var_names:
            var_names.append(lhs.name)
    if not var_names:
        return None
    return IrDel(var_names)


def tensor_to_block_block(block, layer_index, ir_list = None, while_iteration=None):
    if ir_list is None:
        ir_list = block.children
    length = len(ir_list)
    index = 0
    for i in range(length):
        l = ir_list[index]
        if isinstance(l, IrAssignment):
            new_expr, new_assignments = convert_to_ir_ttb(l.children[1], layer_index, while_iteration)
            new_children = [l.children[0], new_expr]
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
            # Create the IR node for the del statements
            del_stmt = make_ttb_del_statement(new_assignments)
            if del_stmt is not None:
                ir_list.insert(index + 1, del_stmt)
                index += 1
        elif isinstance(l, IrTransRetBasic):
            new_children = []
            new_assignments = []
            for child in l.children:
                new_expr, new_assignments_inner = convert_to_ir_ttb(child, layer_index, while_iteration)
                new_children.append(new_expr)
                new_assignments += new_assignments_inner
            l.update_parent_child(new_children)
            for j in range(len(new_assignments)):
                ir_list.insert(index, new_assignments[j])
                index += 1
            
            # We want to collect the var names that are used in the return expression, so that we can 
            # exclude them from the del statements
            keep_var_names = set()
            for child in new_children:
                keep_var_names |= collect_ir_var_names(child)
            del_stmt = make_ttb_del_statement(new_assignments, keep_var_names=keep_var_names)
            if del_stmt is not None:
                ir_list.insert(index, del_stmt)
                index += 1
        index += 1


def replace_all_occurrences_expr(expr, var_map):
    if isinstance(expr, IrVar) and expr.name in var_map.keys(): 
        return var_map[expr.name]
    if isinstance(expr, (int, float, list)):
        return expr
    if expr is None:
        return expr
    for i in range(len(expr.children)):
        new_child = replace_all_occurrences_expr(expr.children[i], var_map)
        expr.children[i] = new_child
    return expr

#FEAT: light traversal clone
def clone_traversal_body(statements):
    """Clone executable IR and def links without copying analysis consumers.

    Use/parent edges lead back out into the enclosing CFG and are not read by
    traversal replay. Later passes recompute liveness from the emitted stream.
    Keeping def links in the clone still permits affine substitution matching.
    """
    memo, seen, pending = {}, set(), [statements]
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            pending.extend(obj)
        elif isinstance(obj, (IrAst, IrMetadataElement)):
            for key, value in vars(obj).items():
                if key in ('uses', 'parents') and isinstance(value, list):
                    memo[id(value)] = []
                else:
                    pending.append(value)
    return copy.deepcopy(statements, memo)


def remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node):
    root_block = cfg.ir[root_node]
    first_while_block = cfg.ir[first_while_node]
    second_while_block = cfg.ir[second_while_node]
    exit_block = cfg.ir[exit_node]

    tensor_to_block_block(root_block, layer_index)
    tensor_to_block_block(exit_block, layer_index)


    layer_types = layer_parents = None
    if globals.fuse_affine_subst.get_flag() and globals.network_path is not None:
        layer_types, layer_parents = fuse_affine_subst._load_topology(globals.network_path)
    affine_types = (LayerType.Linear, LayerType.Conv2D)
    #FIX FOR: tested the Flag object (always true), so Add counted as affine even with the builtin Add rule.
    if program_add_rule.get_flag():  # a program-defined Add has L == U, like Affine
        affine_types += (LayerType.Add,)

    ir_list = root_block.children
    for i in range(num_iterations):
        body = first_while_block.children + second_while_block.children
        combined_list = clone_traversal_body(body)  # skips use/parent analysis edges
        if layer_parents is not None:
            selected = fuse_affine_subst.selected_layers_at(layer_index, first_while_block.while_number, i)
            is_affine = bool(selected) and all(layer_types.get(x) in affine_types for x in selected)
            fuse_affine_subst.fuse_iteration(combined_list, is_affine)
        tensor_to_block_block(None, layer_index=layer_index, ir_list=combined_list, while_iteration=i)
        ir_list += combined_list

    ir_list += exit_block.children

    root_block.update_parent_child(ir_list)

    predecessors_exit = cfg.predecessors[exit_node]
    for pred in predecessors_exit:
        cfg.successors[pred].remove(exit_node)
        if root_node not in cfg.successors[pred]:
            cfg.successors[pred].append(root_node)
    cfg.successors[root_node] = cfg.successors[exit_node]
    root_block.jump = exit_block.jump
    root_block.inner_jump = exit_block.inner_jump


    del cfg.ir[first_while_node]
    del cfg.ir[second_while_node]
    del cfg.ir[break_node]
    del cfg.ir[exit_node]

    del cfg.successors[first_while_node]
    del cfg.successors[second_while_node]
    del cfg.successors[break_node]
    del cfg.successors[exit_node]    

    del cfg.predecessors[first_while_node]
    del cfg.predecessors[second_while_node]
    del cfg.predecessors[break_node]
    del cfg.predecessors[exit_node]

    cfg.nodes.remove(first_while_node)
    cfg.nodes.remove(second_while_node)
    cfg.nodes.remove(break_node)
    cfg.nodes.remove(exit_node)


def unroll_while(cfg, layer_index):
    i = 0
    while True:
        live_nodes = get_live_nodes(cfg, layer_index)
        ordered_live_nodes = [node for node in cfg.nodes if node in live_nodes]
        if i >= len(ordered_live_nodes):
            break

        node = ordered_live_nodes[i]
        
        block = cfg.ir[node]
        if block.inner_jump is None:
            tensor_to_block_block(block, layer_index)
            i+=1
        elif len(block.inner_jump) == 3:
            tensor_to_block_block(block, layer_index)
            i+=1
        elif not isinstance(block.inner_jump[1], IrWhileBlock):
            tensor_to_block_block(block, layer_index)
            i+=1
        else:
            root_node = node
            first_while_node = cfg.successors[node][0]
            first_while_block = cfg.ir[first_while_node]
            second_while_block = first_while_block.jump[1]
            second_while_node = cfg.get_block_id(second_while_block)
            break_node = cfg.get_block_id(first_while_block.inner_jump[1])
            exit_node = cfg.get_block_id(block.jump[1])
            while_number = first_while_block.while_number
            filename = f"jit_while/while_iterations_layer_{layer_index}_while_{while_number}.json"
            json_obj = load_capture(filename)
            num_iterations = json_obj["num_iterations"]
            remove_while(layer_index, num_iterations, cfg, root_node, first_while_node, second_while_node, exit_node, break_node)


def tensor_to_block_cfg(cfg, layer_index):
    live_nodes = get_live_nodes(cfg, layer_index)
    for node in cfg.nodes:
        if node not in live_nodes:
            continue
        block = cfg.ir[node]
        tensor_to_block_block(block, layer_index)


def collapse_cfg_to_single_block(cfg, layer_index):
    """
    Resolve all control flow in a per-layer reuse CFG into a single
    straight-line block. Starting at the entry block we walk the profiled live
    path -- exactly the order codegen emits -- pruning each profiled branch to
    its taken arm, and concatenate every visited block's instructions into the
    entry block. Afterwards the CFG has one node with no `jump`/`inner_jump`, so
    downstream passes (subexp_inlining, codegen) see a flat instruction stream.

    Reuse requires control flow to be fully resolved by the profile: any block
    whose control flow does not collapse to a single successor -- an unprofiled
    ternary, a plain `if`, or a `while` that was not unrolled -- is a hard error
    rather than a fallback to a runtime branch.
    """
    order = []
    visited = set()

    def visit(block):
        if block is None or id(block) in visited:
            return
        visited.add(id(block))
        order.append(block)
        if block.inner_jump is not None:
            if len(block.inner_jump) == 3:
                taken = get_profiled_branch(layer_index, block.block_id)
                if taken == 'then':
                    visit(block.inner_jump[1])
                elif taken == 'else':
                    visit(block.inner_jump[2])
                else:
                    raise RuntimeError(
                        f'reuse: profiled branch at block {block.block_id} '
                        f'(layer {layer_index}) is unresolved; cannot linearize')
            else:
                raise RuntimeError(
                    f'reuse: unresolvable control flow at layer {layer_index} '
                    f'(a plain `if` or an un-unrolled `while`); reuse requires '
                    f'fully profiled/unrolled control flow')
        if block.jump is not None:
            visit(block.jump[1])

    entry_block = cfg.ir[cfg.entry_node]
    visit(entry_block)

    entry_block.children = [
        instr for block in order for instr in block.children]
    entry_block.inner_jump = None
    entry_block.jump = None

    entry = cfg.entry_node
    cfg.nodes = [entry]
    cfg.ir = {entry: entry_block}
    cfg.successors = {entry: []}
    cfg.predecessors = {entry: []}


def abs_elem_param(key):
    """Name of the threaded parameter that replaces abs_elem.d[key]."""
    return 'd_' + key


def _rewrite_abs_elem_reads(expr, params, seen):
    """Point every abs_elem.d[key] read in `expr` at its threaded parameter."""
    if expr is None or isinstance(expr, (int, float, str, list, type)):
        return expr
    if isinstance(expr, IrGetAbsElemSparseDKey):
        return params[expr.key]
    if id(expr) in seen:
        return expr
    seen.add(id(expr))
    for i in range(len(expr.children)):
        expr.children[i] = _rewrite_abs_elem_reads(expr.children[i], params, seen)
    return expr


def splice_update(cfg, layer_index, shape_fields):
    """Append layer_index's replayed update to a collapsed CFG and thread d_<key> through.

    The op's abstract-shape outputs are bound to temporaries, each field's update tape is
    replayed against its temporary, and the terminator grows the new d_<key> values."""
    capture = load_capture('jit_update/update_' + str(layer_index) + '.json')
    block = cfg.ir[cfg.entry_node]
    index = next(i for i, stmt in enumerate(block.children)
                 if isinstance(stmt, IrTransRetBasic))
    ret = block.children[index]
    if len(ret.children) != len(shape_fields):
        raise RuntimeError(
            f"fused flow: layer {layer_index} returns {len(ret.children)} values but the "
            f"certifier declares {len(shape_fields)} shape fields")

    params = {key: IrVar(abs_elem_param(key), copy_metadata(ret.children[i].irMetadata))
              for i, key in enumerate(shape_fields)}

    prefix, shape_vars, out_vars = [], [], []
    for i, key in enumerate(shape_fields):
        field = capture[key]
        if field['operand'] != i:
            raise RuntimeError(
                f"fused flow: update capture for layer {layer_index} binds {key!r} to "
                f"abs_shape_{field['operand']}, but it is shape field {i}")
        shape_var = IrVar(f'ttb_upd_{layer_index}_{key}', copy_metadata(ret.children[i].irMetadata))
        prefix.append(IrAssignment(shape_var, ret.children[i]))
        shape_vars.append(shape_var)

        replay = IrUpdateReplay(shape_var, key, field['records'],
                                copy_metadata(ret.children[i].irMetadata))
        out_expr, assignments = convert_to_ir_ttb(replay, layer_index, None)
        out_var = IrVar(abs_elem_param(key) + '_out', copy_metadata(ret.children[i].irMetadata))
        prefix.extend(assignments)
        prefix.append(IrAssignment(out_var, out_expr))
        out_vars.append(out_var)

    ret.update_parent_child(shape_vars + out_vars)
    block.children[index:index] = prefix

    seen = set()
    for stmt in block.children:
        _rewrite_abs_elem_reads(stmt, params, seen)
    return [abs_elem_param(key) for key in shape_fields]


def tensor_to_block(ir):
    # TODO: DEBUG THE FOLLOWING LINE
    # uses.populate_uses_defs(ir)
    filename = "jit_layers/layers.json"
    json_obj = load_capture(filename)
    shape_fields = list(ir.shape.keys())
    for transformer in ir.tstore.keys():
        for i in range(len(ir.tstore[transformer])):
            transformerIr = ir.tstore[transformer][i]
            key = transformerIr.op.lower()
            if key not in json_obj:
                raise RuntimeError(
                    f"reuse: op {transformerIr.op!r} is declared in the certifier but "
                    f"flow() recorded no layer list for it; layers.json has {sorted(json_obj)}")
            layer_indices = json_obj[key]


            new_cfgs = {}
            for j, layer_index in enumerate(layer_indices):
                cfg = deepcopy_cfg_with_fresh_identifiers(transformerIr.cfg)
                unroll_while(cfg, layer_index)
                collapse_cfg_to_single_block(cfg, layer_index)
                if globals.fused_flow.get_flag():
                    transformerIr.layerwise_params = (
                        transformerIr.params + splice_update(cfg, layer_index, shape_fields))
                new_cfgs[layer_index] = cfg

            transformerIr.layerwise_cfgs = new_cfgs



# Llist params flow() cannot reconstruct
_FLOW_UNBOUND = ('prev', 'curr', 'prev1', 'prev2')

# per-layer scalars, bound from flow.json when a body reads one
_FLOW_SCALARS = ('poly_size', 'curr_size', 'prev_size', 'input_size')


#FEAT: tree rename collects nodes without following analysis edges
def _collect_ir_tree_nodes(obj, nodes):
    visited, pending = set(), [obj]
    while pending:
        obj = pending.pop()
        #FIX FOR: most of the walk was visiting numbers and strings in metadata lists.
        if type(obj) in tree_walk.ATOMS or id(obj) in visited:
            continue
        visited.add(id(obj))
        if isinstance(obj, IrAst):
            nodes[id(obj)] = obj
        if isinstance(obj, dict):
            pending.extend(obj.keys())
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            pending.extend(obj)
        elif hasattr(obj, '__dict__'):
            pending.extend(v for k, v in vars(obj).items() if k not in tree_walk.TREE_SKIP)


def _rename_vars(obj, rename):
    """Rewrite IrVar names and IrDel entries in place."""
    nodes = {}
    _collect_ir_tree_nodes(obj, nodes)
    for node in nodes.values():
        if isinstance(node, IrVar):
            node.name = rename.get(node.name, node.name)
        elif isinstance(node, IrDel):
            node.var_names = [rename.get(n, n) for n in node.var_names]


def _bound_names(stmts):
    return [s.children[0].name for s in stmts
            if isinstance(s, IrAssignment) and isinstance(s.children[0], IrVar)]


def splice_flow(ir, shape_fields):
    """Concatenate every layer's collapsed block into one flow block.

    Sets ir.flow_block and ir.flow_entry; layer-local names get an _L<n> suffix
    and the d_<key> state is threaded from each layer's d_<key>_out."""
    order = [row['layer'] for row in load_capture('jit_flow/flow.json')]
    cfgs = {}
    for transformer in ir.tstore.keys():
        for opStmtIr in ir.tstore[transformer]:
            if opStmtIr.layerwise_cfgs is None:
                continue
            for layer_index, cfg in opStmtIr.layerwise_cfgs.items():
                cfgs[layer_index] = cfg
    if sorted(cfgs) != sorted(order):
        raise RuntimeError(
            f"sroa: flow.json covers layers {sorted(order)} but the specialized "
            f"cfgs cover {sorted(cfgs)}")

    rows = {row['layer']: row for row in load_capture('jit_flow/flow.json')}
    entry = {key: abs_elem_param(key) + '_L0' for key in shape_fields}
    state = dict(entry)
    stmts, terminator = [], None
    layer_of = {}
    for layer_index in order:
        cfg = cfgs[layer_index]
        block = cfg.ir[cfg.entry_node]
        body = block.children
        suffix = '_L' + str(layer_index)

        used = set()
        for stmt in body:
            collect_ir_var_names(stmt, used)
        clash = sorted(used.intersection(_FLOW_UNBOUND))
        if clash:
            raise RuntimeError(
                f"sroa: layer {layer_index} reads {clash}, which flow() does not bind")

        rename = {name: name + suffix
                  for name in _bound_names(body) if not name.startswith('ttb_var_')}
        scalars = [name for name in _FLOW_SCALARS if name in used]
        for name in scalars:
            rename[name] = name + suffix
        for key in shape_fields:
            rename[abs_elem_param(key)] = state[key]
        _rename_vars(block, rename)

        for name in scalars:
            stmts.append(IrAssignment(
                IrVar(name + suffix, [IrMetadataElement([1], 'Int', [1], True)]),
                IrConst(int(rows[layer_index][name]), 'Int')))

        terminator = next(s for s in body if isinstance(s, IrTransRetBasic))
        added = [s for s in body if s is not terminator]
        stmts.extend(added)
        for stmt in added:                 # keep a ref; a freed id gets reused
            layer_of[id(stmt)] = (stmt, layer_index)
        for key in shape_fields:
            state[key] = rename[abs_elem_param(key) + '_out']

    ir.flow_block = IrBlock(stmts + [IrTransRetBasic(
        list(terminator.children[:len(shape_fields)]))])
    ir.flow_entry = entry
    ir.flow_splice_layers = layer_of
