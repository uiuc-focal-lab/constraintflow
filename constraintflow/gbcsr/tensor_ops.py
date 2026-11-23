import torch
import builtins
import math
import operator
from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.lib.globals import *
from constraintflow.lib import globals as G

input_size = 784

def check_type_equality(x, y):
    if x == y:
        return True
    if x in [float, int, torch.float, torch.int] and y in [float, int, torch.float, torch.int]:
        return True
    return False

types = {bool: torch.bool, int: torch.int, float: torch.float}

def checkTypes(x, y):
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            if not check_type_equality(x.type, y.type):
            # if x.type != y.type:
                print(x.type, y.type)
                raise Exception('TYPE MISMATCH')
        if isinstance(y, float) or isinstance(y, int) or isinstance(y, bool):
            if not check_type_equality(x.type, type(y)):
            # if type(y) != x.type:
                raise Exception('TYPE MISMATCH')
        if isinstance(y, torch.Tensor):
            if not check_type_equality(types[x.type], y.dtype):
            # if types[x.type] != y.dtype:
                print(x.type, y.dtype)
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool):
            if not check_type_equality(y.type, type(x)):
            # if type(x) != y.type:
                raise Exception('TYPE MISMATCH')
    elif isinstance(x, SparseTensor):
        if isinstance(y, torch.Tensor):
            if not check_type_equality(x.type, y.dtype):
            # if x.type != y.dtype:
                raise Exception('TYPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, torch.Tensor):
            if not check_type_equality(x.dtype, y.type):
            # if y.type != x.dtype:
                raise Exception('TYPE MISMATCH')
    elif not check_type_equality(type(x), type(y)):
    # elif type(x) != type(y):
        print(type(x), type(y))
        raise Exception('TYPE MISMATCH')

def checkShapes(x, y):
    if isinstance(x, SparseTensor):
        if isinstance(y, float) or isinstance(y, int):
            return
        elif isinstance(y, torch.Tensor):
            if not (x.total_size == torch.tensor(y.shape)).all():
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif isinstance(y, SparseTensor):
            if not (x.total_size == y.total_size).all():
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
    elif isinstance(y, SparseTensor):
        if isinstance(x, float) or isinstance(x, int):
            return True
        elif isinstance(x, torch.Tensor):
            if not (y.total_size == torch.tensor(x.shape)).all():
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
    
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.shape != y.shape:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')

def sanityCheck(x, y):
    # return
    start_time = time.perf_counter()
    checkTypes(x, y)
    checkShapes(x, y)
    end_time = time.perf_counter()
    sanity_time.update_total_time(end_time - start_time)

def unary(x, op, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if isinstance(x, torch.Tensor):
        json_list = []
        json_list.append({"method": "noop", "input": "lhs", "output": 0})
        op_name = '-' if op == operator.neg else 'not' if op == operator.not_ else 'sigma'
        json_list.append({"method": "simple_unary", "input": "json_list_0", "op": op_name, "output": 1})
        res = op(x)
        if dummy_mode:
            if layer_index is not None and counter is not None:
                capture_path = f"jit_unary/unary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                save_capture(capture_path, json_list)
    elif isinstance(x, SparseTensor):
        json_list = []
        json_list.append({"method": "noop", "input": "lhs", "output": 0})
        res = x.unary(op, json_list=json_list, lhs_index=0)
        if dummy_mode:
            if layer_index is not None and counter is not None:
                capture_path = f"jit_unary/unary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                save_capture(capture_path, json_list)
    else:
        json_list = []
        json_list.append({"method": "noop", "input": "lhs", "output": 0})
        op_name = '-' if op == operator.neg else 'not' if op == operator.not_ else 'sigma'
        json_list.append({"method": "simple_unary", "input": "json_list_0", "op": op_name, "output": 1})
        res = op(x)
        if dummy_mode:
            if layer_index is not None and counter is not None:
                capture_path = f"jit_unary/unary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                save_capture(capture_path, json_list)
    unary_time.update_total_time(time.perf_counter() - start_time)
    return res

def any(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    json_list = [
        {"method": "noop", "input": "lhs", "output": 0},
    ]
    if type(x) == torch.Tensor:
        json_list.append({"method": "any", "input": "json_list_0", "output": len(json_list)})
        res = x.any()
    else:
        res = x.any(json_list=json_list, lhs_index=0)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            save_capture(f"jit_any/any_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
    any_time.update_total_time(time.perf_counter() - start_time)
    return res

def all(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    
    json_list = [{"method": "noop", "input": "lhs", "output": 0}]
    if type(x) == torch.Tensor:
        json_list.append({"method": "all", "input": "json_list_0", "output": len(json_list)})
        res = x.all()
    else:
        #It never takes this branch
        nknkkjk
        res = x.all()

    if dummy_mode:
        if layer_index is not None and counter is not None:
            save_capture(f"jit_all/all_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
    all_time.update_total_time(time.perf_counter() - start_time)
    return res

# def all(x):
#     if type(x)!=torch.Tensor:
#         raise Exception('TYPE MISMATCH')
#     return x.all()

def binary(x, y, op, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None, parent_json_list = None, x_index = -1, y_index = -1):
    total_start_time = time.perf_counter()
    start_time = time.perf_counter()
    sanityCheck(x, y)
    binary_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
    if parent_json_list is None:
        json_list = []
    else:
        json_list = parent_json_list
    if isinstance(x, SparseTensor):
        start_time = time.perf_counter()
        # A caller that already recorded an operand passes its index instead.
        if x_index == -1:
            lhs_idx = len(json_list)
            json_list.append({"method": "noop", "input": "lhs", "output": len(json_list)})
        else:
            lhs_idx = x_index
        if y_index == -1:
            rhs_idx = len(json_list)
            json_list.append({"method": "noop", "input": "rhs", "output": len(json_list)})
        else:
            rhs_idx = y_index

        res = x.binary(y, op, json_list = json_list, lhs_index=lhs_idx, rhs_index=rhs_idx)
        binary_tensor_ops_x_sparsity.update_total_time(time.perf_counter() - start_time)

        
        
    elif isinstance(y, SparseTensor):
        lhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "lhs", "output": len(json_list)}
        json_list.append(json_obj)
        rhs_idx = len(json_list)
        json_obj = {"method": "noop", "input": "rhs", "output": len(json_list)}
        json_list.append(json_obj)

        start_time = time.perf_counter()
        temp = convert_dense_to_sparse(x, y.total_size, json_list=json_list, x_index=lhs_idx)
        lhs_idx = len(json_list)-1
        binary_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = temp.binary(y, op, json_list = json_list, lhs_index=lhs_idx, rhs_index=rhs_idx)
        binary_tensor_ops_y_sparsity.update_total_time(time.perf_counter() - start_time)
        
    else:
        start_time = time.perf_counter()
        json_obj = {
            "method": "simple_binary",
            "lhs": "lhs",
            "rhs": "rhs",
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = op(x, y)
        binary_tensor_ops_no_sparse.update_total_time(time.perf_counter() - start_time)
    total_binary_tensor_ops.update_total_time(time.perf_counter() - total_start_time)
    # if inside_while:
    #     print(while_iteration)
    if dummy_mode and parent_json_list is None:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            # if inside_while:
            #     print(while_iteration)
            #     print(capture_path)
            
            save_capture(capture_path, json_list)
    return res

def cf_max(x, y, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            json_list = []
            json_list.append({"method": "noop", "input": "lhs", "output": 0})
            json_list.append({"method": "noop", "input": "rhs", "output": 1})
            res = sparse_max(x, y, json_list=json_list, x_json_index=0, y_json_index=1)
            if dummy_mode:
                if layer_index is not None and counter is not None:
                    capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    save_capture(capture_path, json_list)
            where_time.update_total_time(time.perf_counter() - start_time)
            return res
    res = torch.max(x, y)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def cf_min(x, y, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    start_time = time.perf_counter()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            json_list = []
            json_list.append({"method": "noop", "input": "lhs", "output": 0})
            json_list.append({"method": "noop", "input": "rhs", "output": 1})
            res = sparse_min(x, y, json_list=json_list, x_json_index=0, y_json_index=1)
            if dummy_mode:
                if layer_index is not None and counter is not None:
                    capture_path = f"jit_binary/binary_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    save_capture(capture_path, json_list)
            where_time.update_total_time(time.perf_counter() - start_time)
            return res
    res = torch.min(x, y)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def lcm(a, b):
    if isinstance(a, float) or isinstance(a, int) or isinstance(a, bool):
        return b
    if isinstance(b, float) or isinstance(b, int) or isinstance(b, bool):
        return a
    assert(a.shape[0] == b.shape[0])
    total_size = []
    for j in range(len(a)):
        total_size.append(math.lcm(int(a[j].item()), int(b[j].item())))
    return torch.tensor(total_size)

def const_to_sparse(c, total_size, json_list=None):
    # Blockless, so the constructor records "blocks": [] literally -- no ref to thread.
    return SparseTensor([], [], total_size.shape[0], total_size, type=type(c), dense_const=c, og_json_list=json_list)

def where(x, y, z, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    start_time = time.perf_counter()
    json_list = []
    json_list.append({"method": "noop", "input": "cond", "output": 0})   # index 0
    json_list.append({"method": "noop", "input": "lhs",  "output": 1})   # index 1
    json_list.append({"method": "noop", "input": "rhs",  "output": 2})   # index 2
    
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor):
        checkShapes(x, y)
        sanityCheck(y, z)
        res = torch.where(x, y, z)
    if isinstance(x, bool) and isinstance(y, float) and isinstance(z, float):
        if x:
            res = y
        else:
            res = z
    
    if isinstance(x, SparseTensor):
        x_size = x.total_size
    elif isinstance(x, torch.Tensor):
        x_size = torch.tensor(x.shape)
    else:
        x_size = 0

    if isinstance(y, SparseTensor):
        y_size = y.total_size
    elif isinstance(y, torch.Tensor):
        y_size = torch.tensor(y.shape)
    else:
        y_size = 0

    if isinstance(z, SparseTensor):
        z_size = z.total_size
    elif isinstance(z, torch.Tensor):
        z_size = torch.tensor(z.shape)
    else:
        z_size = 0

    total_size = lcm(x_size, lcm(y_size, z_size))
    if isinstance(x, torch.Tensor):
        x1 = convert_dense_to_sparse(x, json_list=json_list, x_index=0)
        x_json_index = len(json_list)-1

    elif isinstance(x, bool):
        x1 = const_to_sparse(x, total_size, json_list)
        x_json_index = x1.json_index

    else:
        x1 = x
        x_json_index = 0

    if isinstance(y, torch.Tensor):
        y1 = convert_dense_to_sparse(y, json_list=json_list, x_index=1)
        y_json_index = len(json_list)-1
    elif isinstance(y, float):
        y1 = const_to_sparse(y, total_size, json_list)
        y_json_index = y1.json_index
    else:
        y1 = y
        y_json_index = 1


    if isinstance(z, torch.Tensor):
        z1 = convert_dense_to_sparse(z, json_list=json_list, x_index=2)
        z_json_index = len(json_list)-1
    elif isinstance(z, float):
        z1 = const_to_sparse(z, total_size, json_list)
        z_json_index = z1.json_index
    else:
        z1 = z
        z_json_index = 2
    checkShapes(x1, y1)
    sanityCheck(y1, z1)

    res = sp_where(x1, y1, z1, json_list = json_list, x_json_index = x_json_index, y_json_index = y_json_index, z_json_index = z_json_index)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_where/where_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            save_capture(capture_path, json_list)
    where_time.update_total_time(time.perf_counter() - start_time)
    return res

def inner_prod(x, y, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    total_start_time = time.perf_counter()
    # time.sleep(0.00625)
    checkTypes(x, y)
    matmul_tensor_ops_expenses.just_update_total_time(time.perf_counter() - total_start_time)
    start_time = time.perf_counter()
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            if x.total_size.shape[0] == y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-2]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if (x.total_size[:-2] != y.total_size[:-2]).all():
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.total_size.shape[0]:
                if x.total_size[-1] != y.total_size[-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
                if (x.total_size[:-2] != y.total_size[:-1]).all():
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
            matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
            json_list = []
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "noop",
                "input": "rhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = x.matmul(y, json_list = json_list, lhs_index=0, rhs_index=1)

            if dummy_mode:
                if layer_index is not None and counter is not None:
                    # capture_occurrence = _next_jit_occurrence(_jit_save_occurrence, layer_index, counter)
                    capture_path = f"jit_matmul/matmul_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
                    
                    save_capture(capture_path, json_list)
        else:
            if x.total_size.shape[0] == y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-2]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            elif x.total_size.shape[0] > y.shape.shape[0]:
                if x.total_size[-1] != y.shape[-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
                if x.total_size[:-2] != y.shape[:-1]:
                    print(x.total_size, y.shape)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.shape)
                raise Exception('SHAPE MISMATCH')
            matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
            res = x.matmul(y)
    elif isinstance(y, SparseTensor):
        if x.shape.shape[0] == y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-2]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.total_size.shape[0]:
            if x.shape[-1] != y.total_size[-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.total_size[:-1]:
                print(x.shape, y.total_size)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.total_size)
            raise Exception('SHAPE MISMATCH')
        x = convert_dense_to_sparse(x)
        matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = x.matmul(y)
    else:
        if x.shape.shape[0] == y.shape.shape[0]:
            if x.shape[-1] != y.shape[-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-2]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        elif x.shape.shape[0] > y.shape.shape[0]:
            if x.shape[-1] != y.shape[-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
            if x.shape[:-2] != y.shape[:-1]:
                print(x.shape, y.shape)
                raise Exception('SHAPE MISMATCH')
        else:
            print(x.shape, y.shape)
            raise Exception('SHAPE MISMATCH')
        matmul_tensor_ops_expenses.update_total_time(time.perf_counter() - start_time)
        res = x@y

    matmul_tensor_ops.update_total_time(time.perf_counter() - total_start_time)
    return res



def convert_to_float(x):
    if isinstance(x, torch.Tensor):
        res = x.float()
    elif isinstance(x, SparseTensor):
        res = x.float()
    else:
        return x
    return res


# This does not seem to be used anywhere
# def get_default_stop1(shape):
#     return SparseTensor([], [], len(shape), torch.tensor(shape), type=bool, dense_const=False)

def get_default_stop(shape, abs_elem, batch_size, curr_size, poly_size, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    res = []
    res_start_indices = []
    res_end_indices = []
    json_list = []
    live_layers = abs_elem.live_layers
    json_list.append({"method": "initialise", "value": "[]", "output": 0})
    current_list_index = 0

    for i in range(len(abs_elem.network)):
        if i in live_layers:
            res_start_indices.append(torch.tensor([0, 0, abs_elem.network[i].start]))
            res_end_indices.append(torch.tensor([batch_size, curr_size, abs_elem.network[i].end]))
            const_block = ConstBlock(False, torch.tensor([batch_size, curr_size, abs_elem.network[i].size]), json_list)
            res.append(const_block)

            block_index = const_block.json_index
            json_obj = {
                "method": "append_list",
                "list": f"json_list_{current_list_index}",
                "value": f"json_list_{block_index}",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            current_list_index = len(json_list) - 1
    
    temp = SparseTensor(res_start_indices, res, len(shape), torch.tensor(shape), res_end_indices, type=bool, dense_const=True, og_json_list=json_list, blocks_index=current_list_index)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_defaultstop/stop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            save_capture(capture_path, json_list)
    return temp

# This does not seem to be used anywhere
# def get_default_stop2(shape):
#     global input_size
#     vertices_stop_default = torch.zeros(shape)
#     vertices_stop_default[:, 0:834] = 1
#     vertices_stop_default = vertices_stop_default.bool()
#     return vertices_stop_default

def get_max_priority(sp_tensor, active_vertices: SparseTensor, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    priorities = []
    active_blocks = []
    for i in range(sp_tensor.num_blocks):
        # print(f'sp_tensor.blocks[i] type: {type(sp_tensor.blocks[i])}')
        assert(isinstance(sp_tensor.blocks[i], ConstBlock))
        active_blocks.append(active_vertices.exists_sub_block(sp_tensor.start_indices[i], sp_tensor.end_indices[i]))
        if active_blocks[-1]:
            priorities.append(sp_tensor.blocks[i].block)
        else:
            priorities.append(float('-inf'))
    if len(priorities) == 0:
        max_priority = float('-inf')
    else:
        max_priority = max(priorities)
    res_blocks = []
    res_start_indices = []
    res_end_indices = []
    
    json_list = []
    json_list.append({"method": "initialise", "value": "[]", "output": 0})
    current_list_index = 0
    for i in range(sp_tensor.num_blocks):
        if active_blocks[i] and priorities[i] == max_priority:
            # if active_vertices.get_sparse_custom_range(sp_tensor.start_indices[i], sp_tensor.end_indices[i]).any():
            const_block = ConstBlock(True, sp_tensor.blocks[i].total_shape, json_list)
            res_blocks.append(const_block)
            res_start_indices.append(sp_tensor.start_indices[i])
            res_end_indices.append(sp_tensor.end_indices[i])
            json_list.append({
                "method": "append_list",
                "list": f"json_list_{current_list_index}",
                "value": f"json_list_{const_block.json_index}",
                "output": len(json_list),
            })
            current_list_index = len(json_list) - 1
            # continue
    res = SparseTensor(res_start_indices, res_blocks, sp_tensor.dims, sp_tensor.total_size, end_indices=res_end_indices, type=bool, dense_const=False, og_json_list=json_list, blocks_index=current_list_index)
    if dummy_mode:
        capture_path = f"jit_priority/priority_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        save_capture(capture_path, json_list)
        ranges = [(int(s[-1]), int(e[-1])) for s, e in zip(res.start_indices, res.end_indices)]
        selected = [ident for ident, start, end in G.capture_layers
                    if builtins.any(start < hi and lo < end for lo, hi in ranges)]
        save_capture(f"jit_selection/selection_{layer_index}_{while_number}_{while_iteration}.json",
                     {'layers': selected, 'ranges': ranges,
                      'priority': float(max_priority), 'substitutions': {}})
    return res

def filter_trav_exp_stop(trav_exp, stop, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    json_list = []
    json_obj = {
        "method": "noop",
        "input": "lhs",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    lhs_index = len(json_list)-1

    json_list.append({"method": "noop", "input": "rhs", "output": len(json_list)})
    stop_index = len(json_list)-1
    stop_float = stop.float(json_list=json_list, lhs_index=stop_index)

    polyexp_stop_mat = binary(trav_exp.mat, stop_float, operator.mul, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, parent_json_list=json_list, x_index=lhs_index, y_index=stop_float.json_index)
    json_obj = {
        "method": "create_similar",
        "input": f"json_list_{lhs_index}",
        "mat": f"json_list_{len(json_list)-1}",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    polyexp_stop = trav_exp.create_similar(mat = polyexp_stop_mat)
    if dummy_mode:
        capture_path = f"jit_polyexp_stop/stop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        save_capture(capture_path, json_list)

    return polyexp_stop

def filter_trav_exp_not_stop(trav_exp, stop, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    json_list = []
    json_obj = {
        "method": "noop",
        "input": "lhs",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    lhs_index = len(json_list)-1

    if isinstance(trav_exp.const, SparseTensor):
        polyexp_not_stop_const = SparseTensor([], [], trav_exp.const.dims, trav_exp.const.total_size, type=float, dense_const=0, og_json_list=json_list)
        polyexp_not_stop_const_index = polyexp_not_stop_const.json_index
    else:
        polyexp_not_stop_const = 0
        json_obj = {"method":"scalar_cost", "value": 0, "output": len(json_list)}
        json_list.append(json_obj)
        polyexp_not_stop_const_index = len(json_list)-1
    json_list.append({"method": "noop", "input": "rhs", "output": len(json_list)})
    stop_index = len(json_list)-1
    stop_not = stop.unary(operator.not_, json_list=json_list, lhs_index=stop_index)
    stop_float = stop_not.float(json_list=json_list, lhs_index=stop_not.json_index)
    polyexp_not_stop_mat = binary(trav_exp.mat, stop_float, operator.mul, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, parent_json_list=json_list, x_index=lhs_index, y_index=stop_float.json_index)
    if dummy_mode:
        from constraintflow.gbcsr.traversal_capture import validate_substitution
        validate_substitution(polyexp_not_stop_mat, layer_index, counter, while_number, while_iteration)
    polyexp_not_stop = trav_exp.create_similar(mat = polyexp_not_stop_mat, const = polyexp_not_stop_const)
    json_obj = {
        "method": "create_similar",
        "input": f"json_list_{lhs_index}",
        "mat": f"json_list_{len(json_list)-1}",
        "const": f"json_list_{polyexp_not_stop_const_index}",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    if dummy_mode:
        capture_path = f"jit_polyexp_not_stop/notstop_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        save_capture(capture_path, json_list)
    return polyexp_not_stop



def get_dims(x, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
    if isinstance(x, SparseTensor):
        res = x.dims
    elif isinstance(x, torch.Tensor):
        res = x.dim()
    else:
        assert(False), f'get_dims type of x: {type(x)}'
        res = 1
    json_list = [
        {"method": "noop", "input": "lhs", "output": 0},
        {"method": "get_dims", "input": "json_list_0", "output": 1},
    ]
    if dummy_mode:
        if layer_index is not None and counter is not None:
            save_capture(f"jit_get_dims/get_dims_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
    return res

def get_shape_1(x):
    if isinstance(x, SparseTensor):
        return x.total_size[1]
    if not isinstance(x, torch.Tensor):
        raise Exception('TYPE MISMATCH')
    return x.shape[1]

def get_shape_0(x):
    if (not isinstance(x, torch.Tensor)) or (not isinstance(x, SparseTensor)):
        raise Exception('TYPE MISMATCH')
    if isinstance(x, SparseTensor):
        return x.total_size[0]
    return x.shape[0]

def repeat(mat, repeat_dims, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    json_list = []

    start_time = time.perf_counter()
    if isinstance(mat, float):
        json_obj = {
            "method": "tensor_ones",
            "repeat_dims": repeat_dims.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "multiplication",
            "lhs": "lhs",
            "rhs": "json_list_" + str(len(json_list)-1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat*torch.ones(*(repeat_dims.tolist()))
    elif isinstance(mat, torch.Tensor):
        json_obj = {
            "method": "tensor_repeat",
            "lhs": "lhs",
            "repeat_dims": repeat_dims.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.repeat(*(repeat_dims.tolist()))
    else:
        json_obj = {
            "method": "noop",
            "input": "lhs",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.repeat(repeat_dims, json_list=json_list, lhs_index=0)
    repeat_time.update_total_time(time.perf_counter() - start_time)

    if dummy_mode:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_repeat/repeat_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
            
            save_capture(capture_path, json_list)
    return res

def add_dimension_const(value, repeat_dims, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    json_list = []
    res = SparseTensor([], [], 0, torch.tensor([]), dense_const=value, type=type(value), og_json_list=json_list)
    for i in range(len(repeat_dims)):
        res = res.unsqueeze(i, json_list=json_list, lhs_index=res.json_index)
    res = res.repeat(repeat_dims, json_list=json_list, lhs_index=res.json_index)

    if dummy_mode:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_add_dim_const/add_dim_const_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"

            save_capture(capture_path, json_list)
    return res

def clamp(mat, const, min_true, layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):
    start_time = time.perf_counter()
    json_list = []
    if isinstance(mat, float):
        if min_true:
            if mat>const:
                res = mat
                json_obj = {
                    "method": "noop",
                    "input": "lhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                
            else:
                res = const
                json_obj = {
                    "method": "noop",
                    "input": "rhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
        else:
            if mat<const:
                res = mat 
                json_obj = {
                    "method": "noop",
                    "input": "lhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
            else:
                res = const
                json_obj = {
                    "method": "noop",
                    "input": "rhs",
                    "output": len(json_list),
                }
                json_list.append(json_obj)
        clamp_op_expense.update_total_time(time.perf_counter() - start_time)
    elif isinstance(mat, torch.Tensor):
        if min_true:
            clamp_op_expense.update_total_time(time.perf_counter() - start_time)
            res = mat.clamp(min=const)
            
        else:
            clamp_op_expense.update_total_time(time.perf_counter() - start_time)
            res = mat.clamp(max=const)
        json_obj = {
            "method": "tensor_clamp",
            "lhs": "lhs",
            "const": const,
            "min_true": min_true,
            "output": len(json_list),
        }
        json_list.append(json_obj)
    else:
        json_obj = {
            "method": "noop",
            "input": "lhs",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res = mat.clamp(const, min_true, json_list=json_list, lhs_index=0)
    clamp_total_time.update_total_time(time.perf_counter() - start_time)
    if dummy_mode:
        if layer_index is not None and counter is not None:
            capture_path = f"jit_clamp/clamp_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"

            save_capture(capture_path, json_list)
    # clamp_time.update_total_time(time.perf_counter() - start_time)
    return res


# ---------------------------------------------------------------------------
# Concat -- traced replacements for flow_sparse.py's original inline Concat
# block re-stitch (see constraintflow/compiler/builtin_ops.py, which is where
# these get called from the synthesized `Concat` op's generic CFG, and
# constraintflow/compiler/ir.py's IrConcatStitch/IrConcatStitchMat, whose
# codegen (constraintflow/compiler/codeGen.py) emits calls to these).
#
# Both parents are assumed -- as the original inline code assumed -- to each
# match exactly one existing block per relevant dimension; that is true for
# every certifier/network this repo builds today. get_block_id only compares
# start/end index tensors (always concrete, never meta, even under
# simulacrum), so calling it directly here (untraced) is safe; only the
# *result* -- which block_id was picked -- needs to be captured, since at
# reuse time there is no live SparseTensor to call get_block_id on again.
# ---------------------------------------------------------------------------

def concat_stitch_2d(abs_elem, key, source, prev1, prev2, layer_index=None, counter=None,
                      inside_while=False, while_number=None, while_iteration=None):
    """2-D case: plain Float/Int/Bool (source='direct', reads abs_elem.d[key])
    or a PolyExp/SymExp .const (source='const', reads abs_elem.d[key].const)."""
    assert source in ('direct', 'const')
    src = abs_elem.d[key] if source == 'direct' else abs_elem.d[key].const
    model = abs_elem.network
    batch_size = abs_elem.batch_size
    par1, par2 = prev1.llist[0], prev2.llist[0]
    out_size = model[par1].size + model[par2].size

    json_list = []
    d_idx = len(json_list)
    if source == 'direct':
        json_list.append({"method": "get_abs_elem_sparse_d_key", "input": "json_list_-1", "key": key, "output": d_idx})
    else:
        pre_idx = len(json_list)
        json_list.append({"method": "get_abs_elem_sparse_d_key", "input": "json_list_-1", "key": key, "output": pre_idx})
        d_idx = len(json_list)
        json_list.append({"method": "get_poly_exp_sparse_const", "input": f"json_list_{pre_idx}", "output": d_idx})

    start_indices = []
    blocks = []
    list_idx = len(json_list)
    json_list.append({"method": "initialise", "value": "[]", "output": list_idx})
    new_start_index = 0
    for par in (par1, par2):
        start_index = torch.tensor([0, model[par].start])
        end_index = torch.tensor([batch_size, model[par].end])
        block_ids = src.get_block_id(start_index, end_index)[0]
        #FIX FOR: zid N18 Concat crashed when a parent was the implicit zero (no stored block).
        if not block_ids and src.dense_const == 0:
            # The parent's value is the implicit zero (e.g. zid.cf sets Z = 0 past
            # layer 4), so it adds no block to the concatenation either.
            new_start_index += model[par].size
            continue
        assert len(block_ids) == 1, 'Concat: a parent range must match exactly one existing block'
        block_id = block_ids[0]
        block = src.blocks[block_id]
        block_idx = len(json_list)
        json_list.append({"method": "extract_block", "input": f"json_list_{d_idx}", "block_id": block_id, "output": block_idx})
        list_idx_new = len(json_list)
        json_list.append({"method": "append_list", "list": f"json_list_{list_idx}", "value": f"json_list_{block_idx}", "output": list_idx_new})
        list_idx = list_idx_new

        start_indices.append(torch.tensor([0, new_start_index]))
        blocks.append(block)
        new_start_index += model[par].size

    total_size = torch.tensor([batch_size, out_size])
    res = SparseTensor(start_indices, blocks, 2, total_size, og_json_list=json_list, blocks_index=list_idx)
    if dummy_mode and layer_index is not None and counter is not None:
        capture_path = f"jit_concat/concat_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        save_capture(capture_path, json_list)
    return res


def concat_stitch_mat(abs_elem, key, prev1, prev2, layer_index=None, counter=None,
                       inside_while=False, while_number=None, while_iteration=None):
    """3-D case: a PolyExp's .mat. Unlike the 2-D case, a parent's neuron range
    can span multiple existing blocks along the trailing (poly) dimension, so
    each parent contributes get_block_id()'s whole match list, each placed at
    the parent's shifted neuron offset with its own poly-dimension sub-range."""
    src = abs_elem.d[key].mat
    model = abs_elem.network
    batch_size = abs_elem.batch_size
    par1, par2 = prev1.llist[0], prev2.llist[0]
    out_size = model[par1].size + model[par2].size
    poly_size = src.total_size[-1]

    json_list = []
    pre_idx = len(json_list)
    json_list.append({"method": "get_abs_elem_sparse_d_key", "input": "json_list_-1", "key": key, "output": pre_idx})
    d_idx = len(json_list)
    json_list.append({"method": "get_poly_exp_sparse_mat", "input": f"json_list_{pre_idx}", "output": d_idx})

    start_indices = []
    blocks = []
    list_idx = len(json_list)
    json_list.append({"method": "initialise", "value": "[]", "output": list_idx})
    new_start_index = 0
    for par in (par1, par2):
        start_index = torch.tensor([0, model[par].start, 0])
        end_index = torch.tensor([batch_size, model[par].end, poly_size])
        block_ids, block_start_indices, block_end_indices = src.get_block_id(start_index, end_index)
        for i in range(len(block_ids)):
            block_id = block_ids[i]
            block = src.blocks[block_id]
            block_idx = len(json_list)
            json_list.append({"method": "extract_block", "input": f"json_list_{d_idx}", "block_id": block_id, "output": block_idx})
            list_idx_new = len(json_list)
            json_list.append({"method": "append_list", "list": f"json_list_{list_idx}", "value": f"json_list_{block_idx}", "output": list_idx_new})
            list_idx = list_idx_new

            start_indices.append(torch.tensor([0, new_start_index, block_start_indices[i][2]]))
            blocks.append(block)
        new_start_index += model[par].size

    total_size = torch.tensor([batch_size, out_size, poly_size])
    res = SparseTensor(start_indices, blocks, 3, total_size, og_json_list=json_list, blocks_index=list_idx)
    if dummy_mode and layer_index is not None and counter is not None:
        capture_path = f"jit_concat/concat_mat_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
        save_capture(capture_path, json_list)
    return res
