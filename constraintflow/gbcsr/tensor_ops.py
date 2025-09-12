import torch 
import math
import operator
from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.lib.globals import *

input_size = 784

def check_type_equality(x, y):
    if x == y:
        return True
    if x in [float, int, torch.float, torch.int] and y in [float, int, torch.float, torch.int]:
        return True
    return False

types = {bool: torch.bool, int: torch.int, float: torch.float}
# equivalent_types = {(int, )}

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
    start_time = time.time()
    checkTypes(x, y)
    checkShapes(x, y)
    end_time = time.time()
    sanity_time.update_total_time(end_time - start_time)

def unary(x, op):
    start_time = time.time()
    if isinstance(x, torch.Tensor):
        res = op(x)
    elif isinstance(x, SparseTensor):
        res = x.unary(op)
    else:
        res = op(x)
    unary_time.update_total_time(time.time() - start_time)
    return res

def any(x):
    start_time = time.time()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    
    res = x.any()
    any_time.update_total_time(time.time() - start_time)
    return res

def all(x):
    start_time = time.time()
    if type(x)!=torch.Tensor and type(x)!=SparseTensor:
        raise Exception('TYPE MISMATCH')
    
    res = x.all()
    all_time.update_total_time(time.time() - start_time)
    return res

# def all(x):
#     if type(x)!=torch.Tensor:
#         raise Exception('TYPE MISMATCH')
#     return x.all()

def binary(x, y, op):
    start_time = time.time()
    # time.sleep(0.0025)
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        res = x.binary(y, op)
    elif isinstance(y, SparseTensor):
        res = convert_dense_to_sparse(x, y.total_size).binary(y, op)
    else:
        res = op(x, y)
    binary_time.update_total_time(time.time() - start_time)
    return res

def cf_max(x, y):
    start_time = time.time()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            res = sparse_max(x, y)
            where_time.update_total_time(time.time() - start_time)
            return res
    res = torch.max(x, y)
    where_time.update_total_time(time.time() - start_time)
    return res

def cf_min(x, y):
    start_time = time.time()
    sanityCheck(x, y)
    if isinstance(x, SparseTensor):
        if isinstance(y, SparseTensor):
            res = sparse_min(x, y)
            where_time.update_total_time(time.time() - start_time)
            return res
    res = torch.min(x, y)
    where_time.update_total_time(time.time() - start_time)
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

def const_to_sparse(c, total_size):
    return SparseTensor([], [], total_size.shape[0], total_size, type=type(c), dense_const=c)

def where(x, y, z):
    start_time = time.time()
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
        x1 = convert_dense_to_sparse(x)
    elif isinstance(x, bool):
        x1 = const_to_sparse(x, total_size)
    else:
        x1 = x

    if isinstance(y, torch.Tensor):
        y1 = convert_dense_to_sparse(y)
    elif isinstance(y, float):
        y1 = const_to_sparse(y, total_size)
    else:
        y1 = y

    if isinstance(z, torch.Tensor):
        z1 = convert_dense_to_sparse(z)
    elif isinstance(z, float):
        z1 = const_to_sparse(z, total_size)
    else:
        z1 = z
    checkShapes(x1, y1)
    sanityCheck(y1, z1)

    res = sp_where(x1, y1, z1)
    where_time.update_total_time(time.time() - start_time)
    return res

def inner_prod(x, y):
    t1 = time.time()
    # time.sleep(0.00625)
    checkTypes(x, y)
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
                if x.total_size[:-2] != y.total_size[:-1]:
                    print(x.total_size, y.total_size)
                    raise Exception('SHAPE MISMATCH')
            else:
                print(x.total_size, y.total_size)
                raise Exception('SHAPE MISMATCH')
            res = x.matmul(y)
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
        res = convert_dense_to_sparse(x).matmul(y)
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
        res = x@y

    matmul_time.update_total_time(time.time() - t1)
    return res



def convert_to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float() 
    if isinstance(x, SparseTensor):
        return x.float()

def get_default_stop1(shape):
    return SparseTensor([], [], len(shape), torch.tensor(shape), type=bool, dense_const=False)

def get_default_stop(shape, abs_elem, batch_size, curr_size, poly_size):
    res = []
    res_start_indices = []
    res_end_indices = []
    live_layers = list((abs_elem.d['llist']))
    for i in range(len(abs_elem.network)):
        if live_layers[i]:
            res_start_indices.append(torch.tensor([0, 0, abs_elem.network[i].start]))
            res_end_indices.append(torch.tensor([batch_size, curr_size, abs_elem.network[i].end]))
            res.append(ConstBlock(False, torch.tensor([batch_size, curr_size, abs_elem.network[i].size])))
    return SparseTensor(res_start_indices, res, len(shape), torch.tensor(shape), res_end_indices, type=bool, dense_const=False)

def get_default_stop2(shape):
    global input_size
    vertices_stop_default = torch.zeros(shape)
    vertices_stop_default[:, 0:834] = 1
    vertices_stop_default = vertices_stop_default.bool()
    return vertices_stop_default

def get_max_priority(sp_tensor, active_vertices):
    priorities = []
    for i in range(sp_tensor.num_blocks):
        assert(isinstance(sp_tensor.blocks[i], ConstBlock))
        if active_vertices.exists_sub_block(sp_tensor.start_indices[i], sp_tensor.end_indices[i]):
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
    
    for i in range(sp_tensor.num_blocks):
        if priorities[i] == max_priority:
            # if active_vertices.get_sparse_custom_range(sp_tensor.start_indices[i], sp_tensor.end_indices[i]).any():
            res_blocks.append(ConstBlock(True, sp_tensor.blocks[i].total_shape))
            res_start_indices.append(sp_tensor.start_indices[i])
            res_end_indices.append(sp_tensor.end_indices[i])
            # continue
    return SparseTensor(res_start_indices, res_blocks, sp_tensor.dims, sp_tensor.total_size, end_indices=res_end_indices, type=bool, dense_const=False)

def filter_trav_exp_stop(trav_exp, stop):
    stop_float = convert_to_float(stop)
    polyexp_stop_mat = trav_exp.mat.binary(stop_float, operator.mul)
    polyexp_stop = trav_exp.create_similar(mat = polyexp_stop_mat)
    return polyexp_stop

def filter_trav_exp_not_stop(trav_exp, stop):
    if isinstance(trav_exp.const, SparseTensor):
        polyexp_not_stop_const = SparseTensor([], [], trav_exp.const.dims, trav_exp.const.total_size, type=float, dense_const=0)
    else:
        polyexp_not_stop_const = 0
    stop_float = convert_to_float(stop.unary(operator.not_))
    polyexp_not_stop_mat = trav_exp.mat.binary(stop_float, operator.mul)
    polyexp_not_stop = trav_exp.create_similar(mat = polyexp_not_stop_mat, const = polyexp_not_stop_const)
    return polyexp_not_stop



def get_dims(x):
    if isinstance(x, SparseTensor):
        return x.dims
    if isinstance(x, torch.Tensor):
        return x.dim()
    assert(False)
    return 1

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

def repeat(mat, repeat_dims):
    start_time = time.time()
    if isinstance(mat, float):
        res = mat*torch.ones(*(repeat_dims.tolist()))
    elif isinstance(mat, torch.Tensor):
        res = mat.repeat(*(repeat_dims.tolist()))
    else:
        res = mat.repeat(repeat_dims)
    repeat_time.update_total_time(time.time() - start_time)
    return res

def clamp(mat, min_true, const):
    start_time = time.time()
    if isinstance(mat, float):
        if min_true:
            if mat>const:
                res = mat 
            else:
                res = const
        else:
            if mat<const:
                res = mat 
            else:
                res = const
    elif isinstance(mat, torch.Tensor):
        if min_true:
            res = mat.clamp(min=const)
        else:
            res = mat.clamp(max=const)
    else:
        res = mat.clamp(const, min_true)
    clamp_time.update_total_time(time.time() - start_time)
    return res