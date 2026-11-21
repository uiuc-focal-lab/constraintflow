import builtins
from pprint import pprint
import inspect
from typing import Any
import numpy as np
import torch
from constraintflow.lib.s_shaped import FUNCTIONS
import math
import copy
import time
import functools
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from constraintflow.gbcsr.sparse_block import *
from constraintflow.lib.globals import *

from constraintflow.gbcsr.plot import *
from constraintflow.gbcsr.op_helper import *
import json
import os
import operator
# dummy_mode = dummy_mode.get_flag()


import operator

# Map Python built-in types to torch dtypes; avoids float -> float64 (Python
# float is 64-bit, but we want 32-bit throughout for performance parity).
#FEAT: Python float maps to the run dtype (torch.float is fp64 under --dtype float64).
_TORCH_DTYPE = {float: torch.float, bool: torch.bool, int: torch.int32}
def _tdtype(t):
    return _TORCH_DTYPE.get(t, t)

OP_MAP = {
    "add": operator.add,
    "sub": operator.sub,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "eq": operator.eq,
    "or_": operator.or_,
    "and_": operator.and_,
}

def get_operator_func(name: str):
    try:
        return OP_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported operator: {name}")

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    return tensor

def write_jit_capture_file(directory, prefix, layer_index, counter, inside_while, while_number, while_iteration, json_list):
    rel_path = f"{directory}/{prefix}_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json"
    save_capture(rel_path, json_list)

def tensor_list_to_list(lst):
    return [tensor_to_list(x) for x in lst]



def plot_block(ax, batch_idx, y_start, z_start, x_size, y_size, z_size, color='blue'):
    x_min = batch_idx 
    x_max = batch_idx + x_size  

    vertices = [
        [x_min, y_start, z_start], [x_max, y_start, z_start],
        [x_max, y_start + y_size, z_start], [x_min, y_start + y_size, z_start],
        [x_min, y_start, z_start + z_size], [x_max, y_start, z_start + z_size],
        [x_max, y_start + y_size, z_start + z_size], [x_min, y_start + y_size, z_start + z_size]
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  
        [vertices[4], vertices[5], vertices[6], vertices[7]],  
        [vertices[0], vertices[1], vertices[5], vertices[4]],  
        [vertices[2], vertices[3], vertices[7], vertices[6]],  
        [vertices[0], vertices[3], vertices[7], vertices[4]],  
        [vertices[1], vertices[2], vertices[6], vertices[5]]   
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=color, alpha=.25))


def custom_assert(x):
    return True

def convert_dense_to_sparse(x, total_shape=None, json_list=[], x_index=-1):
    if isinstance(x, torch.Tensor):
        type=float 
        dense_const = 0.0
        if x.dtype == torch.bool:
            type = bool
            dense_const = False

        block = DenseBlock(x, json_list, x_index)
        json_obj = {
            "method": "initialise",
            "name": "block",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "append_list",
            "list": "json_list_" + str(len(json_list)-1),
            "value": "json_list_" + str(len(json_list)-2),
            "output": len(json_list)
        }
        json_list.append(json_obj)
        return SparseTensor([torch.zeros(x.dim(), dtype=torch.int64)], [block], x.dim(), torch.as_tensor(x.shape), type=type, dense_const=dense_const, og_json_list=json_list, blocks_index=len(json_list)-1)
    elif isinstance(x, float):
        dense_const = x
        type = float
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x, og_json_list=json_list)
    elif isinstance(x, int):
        dense_const = x
        type = int
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x, og_json_list=json_list)
    elif isinstance(x, bool):
        dense_const = x
        type = bool
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x, og_json_list=json_list)
    else:
        raise Exception('TYPE MISMATCH')

def compare_index(index_1, index_2):
    d = index_1 - index_2
    for j in range(len(d)):
        if d[j] > 0:
            return 1
        elif d[j] < 0:
            return -1
    return 0

def sort_tuple(temp):
    def lex_cmp(a, b):
        diff = a[0] - b[0]
        for i in range(len(diff)):
            if diff[i]>0:
                return 1
            elif diff[i]<0:
                return -1
        assert(False)
        return 0
    sorted(temp, key=functools.cmp_to_key(lex_cmp))
    return temp

def interval_overlap(interval_1, interval_2):
    if interval_1[1] <= interval_2[0] or interval_2[1] <= interval_1[0]:
        return False 
    return True

def block_overlap(block_1, block_2):
    block_1_start = block_1[0]
    block_1_end = block_1[1]
    block_2_start = block_2[0]
    block_2_end = block_2[1]

    dims = len(block_1_start)
    
    for i in range(dims):
        if not interval_overlap([block_1_start[i], block_1_end[i]], [block_2_start[i], block_2_end[i]]):
            return False 
    return True

def full_overlap(block_1, block_2):
    block_1_start = block_1[0]
    block_1_end = block_1[1]
    block_2_start = block_2[0]
    block_2_end = block_2[1]
    
    if (block_1_start == block_2_start).all() and (block_1_end == block_2_end).all():
        return True
    return False

# block_2 contains block_1
def contained(block_1, block_2):
    block_1_start = block_1[0]
    block_1_end = block_1[1]
    block_2_start = block_2[0]
    block_2_end = block_2[1]
    
    if (block_1_start >= block_2_start).all().item() and (block_1_end <= block_2_end).all().item():
        return True
    return False

def union_block(block_1, block_2):
    block_1_start = block_1[0]
    block_1_end = block_1[1]
    block_2_start = block_2[0]
    block_2_end = block_2[1]
    res_start = torch.min(block_1_start, block_2_start)
    res_end = torch.max(block_1_end, block_2_end)
    
    return res_start, res_end

def intersection_block(block_1, block_2):
    if not block_overlap(block_1, block_2):
        raise Exception('No Overlap')
    block_1_start = block_1[0]
    block_1_end = block_1[1]
    block_2_start = block_2[0]
    block_2_end = block_2[1]
    res_start = torch.max(block_1_start, block_2_start)
    res_end = torch.min(block_1_end, block_2_end)
    # print(res_start, res_end)
    
    return res_start, res_end

def get_slice(start_index, end_index):
    dims = start_index.shape[0]
    s = []
    for j in range(dims):
        s.append(slice(int(start_index[j]), int(end_index[j])))
    return s

def find_connected_blocks(start_indices, end_indices):
    n = len(start_indices)
    visited = [False] * n
    blocks = []
    if n == 0:
        return {}
    #FIX FOR: comparing every pair in Python was a large part of compile time; same test, computed at once.
    starts = np.array([[int(x) for x in s] for s in start_indices]).reshape(n, -1)
    ends = np.array([[int(x) for x in e] for e in end_indices]).reshape(n, -1)
    overlap = ((starts[:, None] < ends[None]) & (starts[None] < ends[:, None])).all(-1)
    def dfs(node, block):
        visited[node] = True
        block.append(node)
        for neighbor in np.flatnonzero(overlap[node]).tolist():
            if not visited[neighbor]:
                dfs(neighbor, block)
    for node in range(n):
        if not visited[node]:
            block = []
            dfs(node, block)
            blocks.append(block)
    overlap_classes = dict()
    for block in blocks:
        overlap_classes[block[0]] = block[1:]
    return overlap_classes

def sp_tensor_from_overlap_classes(overlap_classes, start_indices, end_indices, blocks, total_size, dims, dense_const, type, json_list=[], res_blocks_json_list_index=0):
    res_blocks = []
    json_obj = {
        "method": "initialise",
        "name": "res_blocks",
        "value": "[]",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    res_blocks_json_list_index_local = len(json_list)-1
    res_start_indices = []
    res_end_indices = []
    for i in overlap_classes.keys():
        json_obj = {
            "method": "index_lookup",
            "input": "json_list_" + str(res_blocks_json_list_index),
            "index": i,
            "output": len(json_list)
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "initialise",
            "name": "block_i",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "append_list",
            "list": "json_list_" + str(len(json_list)-1),
            "value": "json_list_" + str(len(json_list)-2),
            "output": len(json_list)
        }
        json_list.append(json_obj)
        # Constructed after the list scaffolding rather than before it, so the append_list
        # slot can be threaded in; none of those records depend on the tensor itself.
        block = SparseTensor([start_indices[i]], [blocks[i]], dims, total_size, [end_indices[i]], og_json_list=json_list, blocks_index=len(json_list)-1)
        index_i = block.json_index
        for j in overlap_classes[i]:
            json_obj = {
                "method": "index_lookup",
                "input": "json_list_" + str(res_blocks_json_list_index),
                "index": j,
                "output": len(json_list)
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "initialise",
                "name": "block_j",
                "value": "[]",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(len(json_list)-1),
                "value": "json_list_" + str(len(json_list)-2),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            # Constructed after the list scaffolding rather than before it, so the append_list
            # slot can be threaded in; none of those records depend on the tensor itself.
            new_block = SparseTensor([start_indices[j]], [blocks[j]], dims, total_size, [end_indices[j]], og_json_list=json_list, blocks_index=len(json_list)-1)
            index_j = new_block.json_index
            
            block = block.binary(new_block, operator.add, json_list=json_list, lhs_index=index_i, rhs_index=index_j)
            index_i = len(json_list) - 1

        res_start_indices += block.start_indices
        res_end_indices += block.end_indices
        res_blocks += block.blocks 

        blocks_index = len(json_list)-1
        for i in range(len(block.blocks)):
            json_obj = {
                "method": "extract_block",
                "input": "json_list_" + str(blocks_index),
                "index": i,
                "output": len(json_list)
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(res_blocks_json_list_index_local),
                "value": "json_list_" + str(len(json_list)-1),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            res_blocks_json_list_index_local = len(json_list)-1

    res = SparseTensor(res_start_indices, res_blocks, dims, total_size, res_end_indices, type, dense_const, og_json_list=json_list, blocks_index=res_blocks_json_list_index_local)
    return res

# first is contained in second
def split_blocks(start_index_1, end_index_1, start_index_2, end_index_2, block_id_1, block_id_2):
    if (contained([start_index_1, end_index_1], [start_index_2, end_index_2])):
        if full_overlap([start_index_1, end_index_1], [start_index_2, end_index_2]):
            return [start_index_1], [end_index_1], [([block_id_1], [block_id_2])]
        total_size_1 = end_index_1 - start_index_1
        total_size_2 = end_index_2 - start_index_2
        if (total_size_1 - total_size_2 != 0).sum() != 1:
            print(total_size_1, total_size_2)
            raise NotImplementedError
        unequal_dim = torch.nonzero(total_size_1 - total_size_2).item()
        if (start_index_1 == start_index_2).all():
            new_start_index_1 = start_index_1
            new_end_index_1 = end_index_1
            new_end_index_2 = end_index_2
            new_total_size_2 = total_size_2
            new_total_size_2[unequal_dim] -= total_size_1[unequal_dim]
            new_start_index_2 = new_end_index_2 - new_total_size_2
            return [new_start_index_1, new_start_index_2], [new_end_index_1, new_end_index_2], [([block_id_1], [block_id_2]), ([], [block_id_2])]
        elif (end_index_1 == end_index_2).all():
            new_start_index_2 = start_index_1
            new_end_index_2 = end_index_1
            new_start_index_1 = start_index_2
            new_total_size_1 = total_size_2
            new_total_size_1[unequal_dim] -= total_size_1[unequal_dim]
            new_end_index_1 = new_start_index_1 + new_total_size_1
            return [new_start_index_1, new_start_index_2], [new_end_index_1, new_end_index_2], [([], [block_id_2]), ([block_id_1], [block_id_2])]
        else:
            new_start_index_2 = start_index_1
            new_end_index_2 = end_index_1
            new_start_index_1 = start_index_2
            new_total_size_1 = end_index_1 - start_index_2
            new_total_size_1[unequal_dim] -= total_size_1[unequal_dim]
            new_end_index_1 = new_start_index_1 + new_total_size_1
            new_end_index_3 = end_index_2
            new_total_size_3 = end_index_2 - start_index_1
            new_total_size_3[unequal_dim] -= total_size_1[unequal_dim]
            new_start_index_3 = new_end_index_3 - new_total_size_3
            return [new_start_index_1, new_start_index_2, new_start_index_3], [new_end_index_1, new_end_index_2, new_end_index_3], [([], [block_id_2]), ([block_id_1], [block_id_2]), ([], [block_id_2])]

    elif (contained([start_index_2, end_index_2], [start_index_1, end_index_1])):
        a, b, c = split_blocks(start_index_2, end_index_2, start_index_1, end_index_1, block_id_2, block_id_1)
        new_c = [(i, j) for (j, i) in c]
        return a, b, new_c

    else:
        temp = start_index_1 - start_index_2
        if (temp != 0).sum() != 1:
            raise NotImplementedError
        unequal_dim = torch.nonzero(temp).item()
        if temp[unequal_dim] < 0:
            new_start_index_2 = start_index_2
            new_end_index_2 = end_index_1
            new_total_size_2 = end_index_1 - start_index_2


            new_start_index_1 = start_index_1
            new_total_size_1 = end_index_1 - start_index_1
            new_total_size_1[unequal_dim] -= new_total_size_2[unequal_dim]
            new_end_index_1 = start_index_1 + new_total_size_1

            new_end_index_3 = end_index_2
            new_total_size_3 = end_index_2 - start_index_2
            new_total_size_3[unequal_dim] -= new_total_size_2[unequal_dim]
            new_start_index_3 = new_end_index_3 - new_total_size_3
            return [new_start_index_1, new_start_index_2, new_start_index_3], [new_end_index_1, new_end_index_2, new_end_index_3], [([block_id_1], []), ([block_id_1], [block_id_2]), ([], [block_id_2])]
        else:
            a, b, c = split_blocks(start_index_2, end_index_2, start_index_1, end_index_1, block_id_2, block_id_1)
            new_c = [(i, j) for (j, i) in c]
            return a, b, new_c





class SparseTensor:
    def __init__(self, start_indices, blocks, dims, total_size, end_indices = None, type=float, dense_const = 0.0
                , og_json_list = None, blocks_index = -1,
                layer_index = None, counter = None, inside_while = False, while_number = None, while_iteration = None):


        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list
        if not inductor_mode.get_flag():
            t1 = time.perf_counter()

        if blocks_index != -1:
            blocks_json_list_entry = "json_list_" + str(blocks_index)
        elif len(blocks) == 0:
            blocks_json_list_entry = []
        else:
            json_obj = {
                "method": "noop",
                "input": "rhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            blocks_json_list_entry = "json_list_" + str(len(json_list)-1)

        self.start_indices = [torch.as_tensor(x, dtype=torch.int64) for x in start_indices]
        self.blocks = blocks
        self.total_size = torch.as_tensor(total_size, dtype=torch.int64)
        self.dims = dims
        # assert(dims == total_size.shape[0])
        self.num_blocks = len(start_indices)
        self.type = type
        self.dense_const = dense_const
        # assert(self.num_blocks == len(blocks))
        # if self.num_blocks!=0:
        #     assert(self.dims == len(start_indices[0]))
        self.end_indices = (None if end_indices is None else
                            [torch.as_tensor(x, dtype=torch.int64) for x in end_indices])
        if self.end_indices is None:
            self.end_indices = []
            for i in range(self.num_blocks):
                self.end_indices.append(self.start_indices[i] + blocks[i].total_shape)
                if not inductor_mode.get_flag():
                    if not  (self.end_indices[i] <= self.total_size).all():
                        print(self.end_indices)
                        print(self.total_size)
                    assert((self.end_indices[i] <= self.total_size).all())
        # else:
        #     for i in range(self.num_blocks):
        #         assert((end_indices[i] - start_indices[i] >= 0).all())
        # for i in range(self.num_blocks):
        #     for j in range(i+1, self.num_blocks):
        #         if block_overlap([start_indices[i], self.end_indices[i]], [start_indices[j], self.end_indices[j]]):
        #             print(start_indices[i], self.end_indices[i])
        #             print(start_indices[j], self.end_indices[j])
        #         assert(not(block_overlap([start_indices[i], self.end_indices[i]], [start_indices[j], self.end_indices[j]])))
        
        if not inductor_mode.get_flag():
            sparse_tensor_init_time.update_op_time(time.perf_counter()-t1)
        
        if self.num_blocks > 0:
            if self.check_dense():
            # if (self.end_indices[0]-self.start_indices[0] == self.total_size).all():
                if self.type==float:
                    self.dense_const = 0.0
                elif self.type==bool:
                    self.dense_const = False
        
        delete_indices = []
        for i in range(self.num_blocks):
            if isinstance(self.blocks[i], ConstBlock) or (dummy_mode and self.blocks[i].block_type == 'C'):
                block_val = self.blocks[i].block
                if not isinstance(block_val, torch.Tensor) or (block_val.numel() == 1 and not block_val.is_meta):
                    if block_val == self.dense_const and (self.dense_const == 0.0 or self.dense_const == False):
                        delete_indices.append(i)

        delete_indices.reverse()
        
        json_obj = {
            "method": "SparseTensorConstructor",
            "start_indices": [tensor_to_list(x) for x in self.start_indices],
            "blocks": blocks_json_list_entry,
            "dims": self.dims,
            "total_size": self.total_size.tolist(),
            "end_indices": [tensor_to_list(x) for x in self.end_indices],
            "type": self.type.__name__,
            "dense_const": self.dense_const,
            "delete_indices": delete_indices,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if not inductor_mode.get_flag():
            sparse_tensor_init_time.update_total_time(time.perf_counter()-t1)

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                write_jit_capture_file('jit_SpTen', 'SpTen', layer_index, counter, inside_while, while_number, while_iteration, json_list)
                self.json_list = None

        # if self.num_blocks > len(delete_indices):
        #     for i in delete_indices:
        #         del self.start_indices[i]
        #         del self.end_indices[i]
        #         del self.blocks[i]
        #     self.num_blocks -= len(delete_indices)

        # for i in range(self.num_blocks):
        #     assert self.start_indices[i].dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
        #     assert self.end_indices[i].dtype in {torch.int8, torch.int16, torch.int32, torch.int64}

    def plot_3d(self, fig=None, ax=None):
        assert(self.dims == 3)
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for idx, b in zip(self.start_indices, self.blocks):
            color = 'blue'
            if isinstance(b, DiagonalBlock):
                color = 'red'
            elif isinstance(b, KernelBlock):
                color = 'green'
            elif isinstance(b, ConstBlock):
                color = 'yellow'
            elif isinstance(b, PatchesBlock):
                color = 'cyan'
            elif isinstance(b, RepeatBlock):
                color = 'yellow'
            else:
                assert(isinstance(b, DenseBlock))
            batch_idx, y_start, z_start = idx.tolist()
            x_size, y_size, z_size = b.total_shape  

            plot_block(ax, batch_idx, y_start, z_start, x_size, y_size, z_size, color)

        ax.set_xlabel('Batch Index (x)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([0, self.total_size[0]]) 
        ax.set_ylim([0, self.total_size[1]-1])  
        ax.set_zlim([0, self.total_size[2]-1])  


        plt.show()

    def __str__(self):
        res = f"\
Num Blocks: {self.num_blocks} \n\
Start Indices: {self.start_indices} \n\
End Indices: {self.end_indices} \n\
Total Size: {self.total_size} \n\
Type: {self.type} \n\
Dense Const: {self.dense_const} \n\
Blocks Types: "
        for i in range(self.num_blocks):
            res += f"{i}-{self.blocks[i].block_type}"
            if self.blocks[i].block_type == 'C':
                res += '-'
                res += str(self.blocks[i].block)
            res += ', '
        res = res[:-2] + '\n'
        res += "Blocks Shapes: "
        for i in range(self.num_blocks):
            res += f"{i}-{self.blocks[i].total_shape}, "
        res = res[:-2] + '\n'
        res += "Blocks Values:\n"
        for i in range(self.num_blocks):
            res += f"  {i}: {self.blocks[i].block}\n"
        return res

    def expand_symexp_mat(self, symexp_count):
        assert(self.dense_const==0)
        self.total_size[-1] = symexp_count
        return self

    def get_dense(self, dummy: bool=False, json_list=None, template_index=-1, simulacrum=False):
        trace = json_list is not None
        if trace:
            # res = ones(total_size) * dense_const
            json_list.append({
                "method": "tensor_ones",
                "repeat_dims": self.total_size.tolist(),
                "output": len(json_list),
            })
            ones_index = len(json_list) - 1
            json_list.append({
                "method": "scalar_const",
                "value": self.dense_const,
                "output": len(json_list),
            })
            const_index = len(json_list) - 1
            json_list.append({
                "method": "torch_mul",
                "lhs": "json_list_" + str(ones_index),
                "rhs": "json_list_" + str(const_index),
                "output": len(json_list),
            })
            res_index = len(json_list) - 1

        if dummy_mode or dummy:
            res = torch.empty(tuple(self.total_size.tolist()), device="meta")
        else:
            res = torch.ones(list(self.total_size), dtype=_tdtype(self.type))*self.dense_const

        for i in range(self.num_blocks):
            if trace:
                json_list.append({
                    "method": "extract_block",
                    "input": "json_list_" + str(template_index),
                    "index": i,
                    "output": len(json_list),
                })
                extracted_index = len(json_list) - 1
                block_dense, block_dense_index = self.blocks[i].get_dense(
                    json_list=json_list, template_index=extracted_index, simulacrum=True
                )
                # Build the destination view index: use 0 (full slice ':')
                # for dimensions that span the whole tensor (e.g. batch),
                # otherwise an explicit [start, end] range.
                index = []
                for j in range(self.start_indices[i].shape[0]):
                    s_j = int(self.start_indices[i][j])
                    e_j = int(self.end_indices[i][j])
                    if s_j == 0 and e_j == int(self.total_size[j]):
                        index.append(0)
                    else:
                        index.append([s_j, e_j])
                json_list.append({
                    "method": "assign_to_view",
                    "input": "json_list_" + str(res_index),
                    "base": "json_list_" + str(res_index),
                    "index": index,
                    "value": "json_list_" + str(block_dense_index),
                    "output": len(json_list),
                })
                res_index = len(json_list) - 1
                if not (dummy_mode or dummy):
                    s = get_slice(self.start_indices[i], self.end_indices[i])
                    res[tuple(s)] = block_dense
            else:
                s = get_slice(self.start_indices[i], self.end_indices[i])
                res[tuple(s)] = self.blocks[i].get_dense()

        if trace and simulacrum:
            return res, res_index
        return res
    
    def get_dense_custom_range(self, start_index, end_index):
        if self.dense_const == 0.0 or self.dense_const == False:
            res = torch.zeros((end_index - start_index).to(int).tolist(), dtype=_tdtype(self.type))
        else:
            res = self.dense_const * torch.ones((end_index - start_index).to(int).tolist(), dtype=_tdtype(self.type))

        res_block = [start_index, end_index]
        for i in range(self.num_blocks):
            if block_overlap([self.start_indices[i], self.end_indices[i]], res_block):
                intersection_start_indices, intersection_end_indices = intersection_block([self.start_indices[i], self.end_indices[i]], res_block)
                src_start_indices = intersection_start_indices - self.start_indices[i]
                src_end_indices = intersection_end_indices - self.start_indices[i]
                dst_start_indices = intersection_start_indices - start_index
                dst_end_indices = intersection_end_indices - start_index
                src_block = self.blocks[i].get_dense()[tuple(get_slice(src_start_indices, src_end_indices))]
                res[tuple(get_slice(dst_start_indices, dst_end_indices))] = src_block
        return res

    # NOTE: Use at your own risk. It does not create a copy.
    def get_sub_block_custom_range(self, start_index, end_index, block_id, tensor=True, return_extraction_mode=False, json_list=None, lhs_index=-1):
        assert json_list != None
        start_time = time.perf_counter()
        trace = json_list is not None
        if not trace:
            json_list = []
        block_start_index = self.start_indices[block_id]
        sparse_block = self.blocks[block_id]
        extracted_block_index = None
        if trace:
            json_obj = {
                "method": "extract_block",
                "input": "json_list_" + str(lhs_index),
                "index": block_id,
                "output": len(json_list)
            }
            json_list.append(json_obj)
            extracted_block_index = len(json_list)-1

        if (block_start_index == start_index).all() and (self.end_indices[block_id] == end_index).all():
            extraction_mode = "full_block"
            if tensor:
                if trace:
                    json_obj = {
                        "method": "object_lookup",
                        "input": "json_list_" + str(extracted_block_index),
                        "object": "block",
                        "output": len(json_list)
                    }
                    json_list.append(json_obj)
                    result_json_index = len(json_list) - 1
                ret = sparse_block.block
            else:
                if trace:
                    json_obj = {
                        "method": "noop",
                        "input": "json_list_" + str(extracted_block_index),
                        "output": len(json_list)
                    }
                    json_list.append(json_obj)
                    result_json_index = len(json_list) - 1
                ret = sparse_block

            if return_extraction_mode:
                if trace:
                    return ret, extraction_mode, result_json_index
                return ret, extraction_mode
            
            if trace:
                return ret, result_json_index
            return ret
        if (isinstance(sparse_block, KernelBlock) or isinstance(sparse_block, PatchesBlock)) or (dummy_mode and sparse_block.block_type in {'K', 'P'}):
            if len(sparse_block.total_shape)==4:
                if (block_start_index[:-1] == start_index[:-1]).all() and (self.end_indices[block_id][:-1] == end_index[:-1]).all():
                    if trace:
                        json_obj = {
                            "method": "object_lookup",
                            "input": "json_list_" + str(extracted_block_index),
                            "object": "block",
                            "output": len(json_list)
                        }
                        json_list.append(json_obj)
                        block_json_index = len(json_list) - 1


                    res, res_index = sparse_block.create_similar(sparse_block.block, json_list = json_list, template_index = block_json_index, simulacrum = True)

                    if trace: 
                        json_obj = {
                            "method": "block_set_total_shape_last_dim",
                            "input": "json_list_" + str(res_index),
                            "value": int((end_index[-1] - start_index[-1]).item()),
                            "output": len(json_list)
                        }
                        json_list.append(json_obj)
                        result_json_index = len(json_list) - 1
                    res.total_shape[-1] = end_index[-1] - start_index[-1]
                    extraction_mode = "kernel_or_patches"
                    if return_extraction_mode:
                        if trace:
                            return res, extraction_mode, result_json_index
                        return res, extraction_mode

                    if trace:
                        return res, result_json_index
                    return res
            raise NotImplementedError
        extraction_mode = "normal"
        if trace:
            res, result_json_index = self.blocks[block_id].get_sub_block_custom_range(
                start_index, end_index, block_start_index,
                json_list=json_list, template_index=extracted_block_index, simulacrum=True)
        else:
            res = self.blocks[block_id].get_sub_block_custom_range(start_index, end_index, block_start_index)
        if tensor:
            ret = res.block
            if trace:
                json_obj = {
                    "method": "object_lookup",
                    "input": "json_list_" + str(result_json_index),
                    "object": "block",
                    "output": len(json_list)
                }
                json_list.append(json_obj)
                result_json_index = len(json_list) - 1
            if return_extraction_mode:
                if trace:
                    return ret, extraction_mode, result_json_index
                return ret, extraction_mode
            if trace:
                return ret, result_json_index
            return ret
        end_time = time.perf_counter()
        sub_block_custom_range_time.update_total_time(end_time-start_time)
        if return_extraction_mode:
            if trace:
                return res, extraction_mode, result_json_index
            return res, extraction_mode
        
        if trace:
            return res, result_json_index
        return res


    def exists_block(self, start_index, end_index):
        for i in range(self.num_blocks):
            if full_overlap([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                if isinstance(self.blocks[i], ConstBlock) or (dummy_mode and self.blocks[i].block_type == 'C'):
                    if self.blocks[i].block == self.dense_const and (self.dense_const == 0.0 or self.dense_const == False):
                        return False
                return True
        return False
    
    def exists_sub_block(self, start_index, end_index):
        for i in range(self.num_blocks):
            if contained([start_index, end_index], [self.start_indices[i], self.end_indices[i]]):
                if isinstance(self.blocks[i], ConstBlock) or (dummy_mode and self.blocks[i].block_type == 'C'):
                    if self.blocks[i].block == self.dense_const and (self.dense_const == 0.0 or self.dense_const == False):
                        return False
                return True
        return False
    
    def get_block_id(self, start_index, end_index):
        res = []
        res_start_indices = []
        res_end_indices = []
        for i in range(self.num_blocks):
            if contained([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                res.append(i)
                res_start_indices.append(self.start_indices[i])
                res_end_indices.append(self.end_indices[i])
        # if len(res)==0:
        #     raise Exception('No Block Found')
        return res, res_start_indices, res_end_indices
    
    def get_sparse_custom_range(self, start_index, end_index, json_list=None, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None, lhs_index=-1):
        owns_capture = json_list is None and dummy_mode
        if json_list is None:
            json_list = []
        if dummy_mode and not owns_capture:
            assert lhs_index != -1

        start_time = time.perf_counter()
        new_index = [start_index, end_index]
        blocks = []
        res_start_indices = []
        res_end_indices = []
        block_list_in_json_idx: int = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'initialise',
            'name': 'ret_blocks',
            'value': '[]',
            'output': block_list_in_json_idx
        }
        json_list.append(json_obj)
        for i in range(self.num_blocks):
            if contained([self.start_indices[i], self.end_indices[i]], new_index):
                intersection_start_indices, intersection_end_indices = intersection_block([self.start_indices[i], self.end_indices[i]], new_index)
                res_start_indices.append(intersection_start_indices)
                res_end_indices.append(intersection_end_indices)
                blocks.append(self.blocks[i])
                block_i_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'extract_block',
                    'input': 'json_list_' + str(lhs_index),
                    'index': i,
                    'output': block_i_idx
                }
                json_list.append(json_obj)
                json_obj: dict[str, Any] = {
                    'method': 'append_list',
                    'list': 'json_list_' + str(block_list_in_json_idx),
                    'value': 'json_list_' + str(block_i_idx),
                    'output': len(json_list)
                }
                block_list_in_json_idx = len(json_list)
                json_list.append(json_obj)
            elif block_overlap([self.start_indices[i], self.end_indices[i]], new_index):
                raise NotImplementedError
                intersection_start_indices, intersection_end_indices = intersection_block([self.start_indices[i], self.end_indices[i]], new_index)
                res_start_indices.append(intersection_start_indices)
                res_end_indices.append(intersection_end_indices)
                src_start_indices = intersection_start_indices - self.start_indices[i]
                src_end_indices = intersection_end_indices - self.start_indices[i]
                src_block = self.blocks[i].get_dense()[tuple(get_slice(src_start_indices, src_end_indices))]
                block_i_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'extract_block',
                    'input': 'json_list_' + str(lhs_index),
                    'index': i,
                    'output': block_i_idx
                }
                json_list.append(json_obj)
                dense_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'block_get_dense',
                    'input': 'json_list_' + str(block_i_idx),
                    'output': dense_idx
                }
                json_list.append(json_obj)
                src_block_idx = len(json_list)
                blocks.append(DenseBlock(src_block))
        end_time = time.perf_counter()
        get_sparse_range_time.update_total_time(end_time-start_time)
        ret_st = SparseTensor(res_start_indices, blocks, self.dims, self.total_size, res_end_indices, type=self.type, dense_const=self.dense_const, og_json_list=json_list, blocks_index=block_list_in_json_idx)
        if owns_capture:
            write_jit_capture_file(
                'jit_SparseTensor_get_sparse_custom_range',
                'SparseTensor_get_sparse_custom_range',
                layer_index,
                counter,
                inside_while,
                while_number,
                while_iteration,
                json_list
            )
        return ret_st if (not dummy_mode) or owns_capture else (ret_st, len(json_list)-1)
    


    def reduce_size(self, start_index, end_index, total_size, json_list=None, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None, lhs_index=-1):
        owns_capture = json_list is None and dummy_mode
        if json_list is None:
            json_list = []
        if dummy_mode:
            assert lhs_index != -1
            assert not owns_capture
        start_time = time.perf_counter()
        start_indices = []
        end_indices = []
        for i in range(self.num_blocks):
            # assert(contained([self.start_indices[i], self.end_indices[i]], [start_index, end_index]))
            start_indices.append((self.start_indices[i]-start_index))
            end_indices.append((self.end_indices[i]-start_index))
        end_time = time.perf_counter()
        reduce_size_time.update_total_time(end_time-start_time)
        json_obj: dict[str, Any] = {
            'method': 'get_sparse_tensor_blocks',
            'input': 'json_list_' + str(lhs_index),
            'output': len(json_list)
        }
        json_list.append(json_obj)
        # Constructed after the block-list record rather than before it, so that record's
        # slot can be threaded in as the blocks ref.
        ret_st = SparseTensor(start_indices, self.blocks, self.dims, total_size, end_indices, type=self.type, dense_const=self.dense_const, og_json_list=json_list, blocks_index=len(json_list) - 1)
        return ret_st if not dummy_mode else (ret_st, ret_st.json_index)

    def increase_size(self, start_index, new_total_size):
        assert((self.total_size + start_index <= new_total_size).all())
        res_start_indices = []   
        res_end_indices = []   
        res_blocks = []   
        for i in range(self.num_blocks):
            res_start_indices.append(start_index + self.start_indices[i])   
            res_end_indices.append(start_index + self.end_indices[i])
            res_blocks.append(self.blocks[i].copy())
        return SparseTensor(res_start_indices, res_blocks, self.dims, new_total_size, res_end_indices, type=self.type, dense_const=self.dense_const)

    
    
    
    def merge_no_overlap(self, sp_tensor):
        assert(self.dims == sp_tensor.dims)
        assert(self.type == sp_tensor.type)
        assert(self.dense_const == sp_tensor.dense_const)
        assert((self.total_size == sp_tensor.total_size).all())
        temp = [(i, 0, j) for j, i in enumerate(self.start_indices)] + [(i, 1, j) for j, i in enumerate(sp_tensor.start_indices)]
        sort_tuple(temp)
        start_indices = []
        end_indices = []
        blocks = []
        for i in range(len(temp)):
            if temp[i][1] == 0:
                start_indices.append(self.start_indices[temp[i][2]])
                end_indices.append(self.end_indices[temp[i][2]])
                blocks.append(self.blocks[temp[i][2]])
            else:
                start_indices.append(sp_tensor.start_indices[temp[i][2]])
                end_indices.append(sp_tensor.end_indices[temp[i][2]])
                blocks.append(sp_tensor.blocks[temp[i][2]])
        return SparseTensor(start_indices, blocks, self.dims, self.total_size, end_indices, type=self.type, dense_const=self.dense_const)

    
    def union_tensors(self, sp_tensor, indices=False):
        start_time = time.perf_counter()
        assert((self.total_size == sp_tensor.total_size).all())
        if sp_tensor.num_blocks == 0:
            res_start_indices = [i for i in self.start_indices]
            res_end_indices = [i for i in self.end_indices]
            if indices:
                res_indices = [([i], []) for i in range(self.num_blocks)]
                return res_start_indices, res_end_indices, res_indices
            return res_start_indices, res_end_indices
        if self.num_blocks == 0:
            res_start_indices = [i for i in sp_tensor.start_indices]
            res_end_indices = [i for i in sp_tensor.end_indices]
            if indices:
                res_indices = [([], [i]) for i in range(sp_tensor.num_blocks)]
                return res_start_indices, res_end_indices, res_indices
            return res_start_indices, res_end_indices
        
        start_indices = self.start_indices + sp_tensor.start_indices
        end_indices = self.end_indices + sp_tensor.end_indices
        overlap_classes = find_connected_blocks(start_indices, end_indices)
        res_start_indices = []
        res_end_indices = []
        res_indices = []
        for key in overlap_classes.keys():
            if len(overlap_classes[key]) == 1:
                if key < len(self.start_indices):
                    block_id_1 = key 
                    block_id_2 = overlap_classes[key][0] - len(self.start_indices)
                else:
                    block_id_1 = overlap_classes[key][0]
                    block_id_2 = key - len(self.start_indices)
                start_indices, end_indices, res_indices_temp = split_blocks(self.start_indices[block_id_1], self.
                end_indices[block_id_1], sp_tensor.start_indices[block_id_2], sp_tensor.end_indices[block_id_2], block_id_1, block_id_2)

                res_start_indices += start_indices
                res_end_indices += end_indices
                res_indices += res_indices_temp
            elif len(overlap_classes[key]) == 0:
                if key < len(self.start_indices):
                    block_id_1 = key 
                    res_start_indices.append(self.start_indices[block_id_1])
                    res_end_indices.append(self.end_indices[block_id_1])
                    res_indices.append(([block_id_1], []))
                else:
                    block_id_2 = key - len(self.start_indices)
                    res_start_indices.append(sp_tensor.start_indices[block_id_2])
                    res_end_indices.append(sp_tensor.end_indices[block_id_2])
                    res_indices.append(([], [block_id_2]))
            else:
                temp_start_indices = []
                temp_end_indices = []
                temp_res_indices = []
                
                overlap_block_ids = [key] + overlap_classes[key]
                for j in range(len(overlap_block_ids)):
                    if overlap_block_ids[j] < len(self.start_indices):
                        temp_start_indices.append(self.start_indices[overlap_block_ids[j]])
                        temp_end_indices.append(self.end_indices[overlap_block_ids[j]])
                        temp_res_indices.append(([overlap_block_ids[j]], []))
                
                
                for k in range(len(overlap_block_ids)):
                    new_temp_start_indices = []
                    new_temp_end_indices = []
                    new_temp_res_indices = []
                    if overlap_block_ids[k] >= len(self.start_indices):
                        for j in range(len(temp_start_indices)):
                            if len(temp_res_indices[j][0]) == 0:
                                new_temp_start_indices.append(temp_start_indices[j])
                                new_temp_end_indices.append(temp_end_indices[j])
                                new_temp_res_indices.append(temp_res_indices[j])
                                continue
                            assert(len(temp_res_indices[j][0]) == 1)
                            block_id_1 = temp_res_indices[j][0][0]
                            block_id_2 = overlap_block_ids[k] - len(self.start_indices)
                            if block_overlap([temp_start_indices[j], temp_end_indices[j]], [sp_tensor.start_indices[block_id_2], sp_tensor.end_indices[block_id_2]]):
                                start_indices, end_indices, res_indices_temp = split_blocks(temp_start_indices[j], temp_end_indices[j], sp_tensor.start_indices[block_id_2], sp_tensor.end_indices[block_id_2], block_id_1, block_id_2)
                                new_temp_start_indices += start_indices
                                new_temp_end_indices += end_indices
                                new_temp_res_indices += res_indices_temp
                            else:
                                new_temp_start_indices.append(temp_start_indices[j])
                                new_temp_end_indices.append(temp_end_indices[j])
                                new_temp_res_indices.append(temp_res_indices[j])
                        temp_start_indices = new_temp_start_indices
                        temp_end_indices = new_temp_end_indices
                        temp_res_indices = new_temp_res_indices
                
                to_be_removed = []
                for i in range(len(temp_start_indices)):
                    for j in range(i+1, len(temp_start_indices)):
                        if contained([temp_start_indices[i], temp_end_indices[i]], [temp_start_indices[j], temp_end_indices[j]]):
                            to_be_removed.append(j)
                            temp_res_indices[i] = (list(set(temp_res_indices[i][0] + temp_res_indices[j][0])), list(set(temp_res_indices[i][1] + temp_res_indices[j][1])))
                        elif contained([temp_start_indices[j], temp_end_indices[j]], [temp_start_indices[i], temp_end_indices[i]]):
                            to_be_removed.append(i)
                            temp_res_indices[j] = (list(set(temp_res_indices[j][0] + temp_res_indices[i][0])), list(set(temp_res_indices[j][1] + temp_res_indices[i][1])))
                to_be_removed = sorted(list(set(to_be_removed)), reverse=True)
                for i in to_be_removed:
                    del temp_start_indices[i]
                    del temp_end_indices[i]
                    del temp_res_indices[i]

                res_start_indices += temp_start_indices
                res_end_indices += temp_end_indices
                res_indices += temp_res_indices

        end_time = time.perf_counter()
        union_tensors_time.update_total_time(end_time - start_time)
        if indices:
            return res_start_indices, res_end_indices, res_indices
        return res_start_indices, res_end_indices


    def copy(self):
        res_start_indices = [i.clone() for i in self.start_indices]
        res_end_indices = [i.clone() for i in self.end_indices]
        res_blocks = [i.copy() for i in self.blocks]
        return SparseTensor(res_start_indices, res_blocks, self.dims, self.total_size, res_end_indices, type=self.type, dense_const=self.dense_const)
    
    def check_dense(self):
        """Full coverage of the disjoint block partition, without tensor overflow.

        Shape products are host metadata. Even int64 products can overflow for
        symbolic tensors, so use arbitrary-precision Python integers here.
        Over-counting is not proof of coverage (e.g. invalid overlapping blocks).
        """
        volume = lambda shape: math.prod(int(d) for d in shape)
        return sum(volume(b.total_shape) for b in self.blocks) == volume(self.total_size)

    def unary(self, op, json_list=[], lhs_index=-1):
        if op == operator.not_:
            assert(self.type == bool)
            dense_const = not(self.dense_const)
            op_name = 'not'
        elif op == operator.neg:
            assert(self.type == float or self.type == int)
            dense_const = -self.dense_const
            op_name = '-'
        elif op == 'sigma':
            assert(self.type == float or self.type == int)
            dense_const = 1 / (1 + math.exp(-self.dense_const))
            op_name = 'sigma'
        #FEAT: table-backed tangent points of S-shaped activations.
        elif op in ('sigma_tangent', 'tanh_tangent'):
            assert(self.type == float or self.type == int)
            dense_const = float(FUNCTIONS[op](torch.tensor(float(self.dense_const))))
            op_name = op
        else:
            assert(False)

        json_obj = {
            "method": "initialise",
            "name": "res_blocks",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        current_list_index = len(json_list) - 1

        blocks = []
        for i in range(self.num_blocks):
            json_obj = {
                "method": "extract_block",
                "input": "json_list_" + str(lhs_index),
                "index": i,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_index = len(json_list) - 1

            cr, result_block_json_index = self.blocks[i].unary(op, json_list=json_list, template_index=block_json_index, simulacrum=True)

            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(current_list_index),
                "value": "json_list_" + str(result_block_json_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            current_list_index = len(json_list) - 1

            blocks.append(cr)

        return SparseTensor(self.start_indices, blocks, self.dims, self.total_size, end_indices=self.end_indices, type=self.type, dense_const=dense_const, og_json_list=json_list, blocks_index=current_list_index)
    

    def binary(self, sp_tensor, op, json_list = [], lhs_index=-1, rhs_index=-1, dummy: bool=False):
        
        binary_sparse_tensor_count.update_num_used()
        total_start_time = time.perf_counter()
        if op not in all_ops:
            raise Exception(f"NOT IMPLEMENTED {op}")
        if (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()!=1):
            assert(False)
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        if isinstance(sp_tensor, float) or isinstance(sp_tensor, bool) or isinstance(sp_tensor, int):
            original_type = type(sp_tensor)
            sp_tensor = SparseTensor([], [], self.dims, self.total_size, [], type(sp_tensor), sp_tensor, og_json_list=json_list)
            rhs_index = sp_tensor.json_index
        elif (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()==1):
            original_type = type(sp_tensor)
            sp_tensor = SparseTensor([], [], self.dims, self.total_size, [], type(sp_tensor.item()), sp_tensor.item(), og_json_list=json_list)
            rhs_index = sp_tensor.json_index
        # else:
        #     json_obj = {
        #         "method": "noop",
        #         "input": "json_list_" + str(rhs_index),
        #         "output": len(json_list),
        #     }
        #     json_list.append(json_obj)

        assert((self.total_size == sp_tensor.total_size).all())
        assert(self.dims == sp_tensor.dims)
        assert(self.type == sp_tensor.type) or (self.type in [int, float] and sp_tensor.type in [int, float])

        new_type = self.type
        if op in comparison_ops:
            new_type = bool
        if op == operator.truediv and sp_tensor.check_dense():
            dense_const = 0.0
        else:
            dense_const = op(self.dense_const, sp_tensor.dense_const)
            if op == operator.mul and (self.dense_const == 0.0 or sp_tensor.dense_const == 0.0):
                dense_const = 0.0

        

        
        
        # indices[i][0] indices from self, indices[i][1] indices from sp_tensor
        start_indices, end_indices, indices = self.union_tensors(sp_tensor, indices=True)
        res_start_indices = []
        res_end_indices = []
        res_blocks = []

        
        json_obj = {
            "method": "initialise",
            "name": "res_blocks",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_blocks_json_list_index = len(json_list) - 1

        block_operations = []
        block_mappings = []

        binary_sparse_tensor_expenses.update_total_time(time.perf_counter()-total_start_time)

        

        for i in range(len(start_indices)):
            start_index = start_indices[i]
            end_index = end_indices[i]
            block_operation = None
            block_mapping = None
            block_json_index = None
            
            # Only from sp_tensor
            if len(indices[i][0]) == 0:
                start_time = time.perf_counter()
                # essentially get block from sp_tensor that corresponds to the range start_index, end_index
                block_id_2 = indices[i][1][0]
                block, rhs_extraction_mode, rhs_json_index = sp_tensor.get_sub_block_custom_range(
                    start_index, 
                    end_index, 
                    block_id_2, 
                    False, 
                    return_extraction_mode=True, 
                    json_list=json_list, 
                    lhs_index=rhs_index
                    )


                # json_obj = {
                #     "method": "get_sub_block_custom_range",
                #     "lhs": "json_list_" + str(rhs_index),
                #     "start_index": start_index.tolist(),
                #     "end_index": end_index.tolist(),
                #     "block_id": block_id_2,
                #     "tensor": False,
                #     "output": len(json_list)
                # }

                # json_list.append(json_obj)


                binary_sparse_tensor_dom2_expenses.update_total_time(time.perf_counter()-start_time)
                start = time.perf_counter()
                block_json_index = rhs_json_index

                # checking if it can be turned into a unary operation 
                if identity_element(op) == self.dense_const:
                    # Only sub is not the identity here; truediv is unhandled, as before.
                    if op == operator.sub:
                        block, block_json_index = block.unary(
                            operator.neg,
                            json_list=json_list,
                            template_index=rhs_json_index,
                            simulacrum=True,
                        )
                    else:
                        assert op in (operator.add, operator.mul, operator.and_, operator.or_)
                    binary_fixed_costs.update_total_time(time.perf_counter()-start)
                elif annihilator_element(op) == self.dense_const and dense_const == self.dense_const:
                    binary_fixed_costs.update_total_time(time.perf_counter()-start)
                    continue
                else:
                    temp_block = ConstBlock(self.dense_const, block.total_shape, json_list)
                    const_json_index = temp_block.json_index

                    binary_fixed_costs.update_total_time(time.perf_counter()-start)
                    # Index Dependant
                    before_binary_json_len = len(json_list)

                    block, block_json_index = temp_block.binary(block, op, json_list = json_list,
                    lhs_index = len(json_list) - 1,
                    rhs_index = rhs_json_index)
                    # json_obj = {
                    #     "method": "binary",
                    #     "op": op.__name__,
                    #     "lhs": "json_list_" + str(const_json_index),
                    #     "rhs": "json_list_" + str(rhs_json_index),
                    #     "output": len(json_list)
                    # }
                    # json_list.append(json_obj)

                binary_sparse_tensor_dom2.update_total_time(time.perf_counter()-start_time)
            elif len(indices[i][1]) == 0:
                start_time = time.perf_counter()
                block_id_1 = indices[i][0][0]
                block, lhs_extraction_mode, lhs_json_index = self.get_sub_block_custom_range(
                    start_index, 
                    end_index, 
                    block_id_1, 
                    False, 
                    return_extraction_mode=True, 
                    json_list=json_list, 
                    lhs_index=lhs_index
                )
                binary_sparse_tensor_dom1_expenses.update_total_time(time.perf_counter()-start_time)
                start = time.perf_counter()
                block_json_index = lhs_json_index


                # json_obj = {
                #     "method": "get_sub_block_custom_range",
                #     "lhs": "json_list_" + str(lhs_index),
                #     "start_index": start_index.tolist(),
                #     "end_index": end_index.tolist(),
                #     "block_id": block_id_1,
                #     "tensor": False,
                #     "output": len(json_list)
                # }

                # json_list.append(json_obj)


                if identity_element(op) == sp_tensor.dense_const:
                    binary_fixed_costs.update_total_time(time.perf_counter()-start)
                    # raise Exception("NOT IMPLEMENTED")
                    pass 
                elif annihilator_element(op) == sp_tensor.dense_const and dense_const == sp_tensor.dense_const:
                    binary_fixed_costs.update_total_time(time.perf_counter()-start)
                    continue
                else:
                    temp_block = ConstBlock(sp_tensor.dense_const, block.total_shape, json_list)
                    const_json_index = temp_block.json_index

                    binary_fixed_costs.update_total_time(time.perf_counter()-start_time)
                    # Index Dependant
                    before_binary_json_len = len(json_list)
                    block, block_json_index = block.binary(temp_block, op, json_list = json_list,
                    lhs_index = block_json_index,
                    rhs_index = const_json_index)

                    # json_obj = {
                    #     "method": "binary",
                    #     "op": op.__name__,
                    #     "lhs": "json_list_" + str(lhs_json_index),
                    #     "rhs": "json_list_" + str(const_json_index),
                    #     "output": len(json_list)
                    # }
                    # json_list.append(json_obj)

                binary_sparse_tensor_dom1.update_total_time(time.perf_counter()-start_time)
            else:
                start_time = time.perf_counter()
                block_id_1 = indices[i][0][0]
                block_id_2 = indices[i][1][0]
                block_1, lhs_extraction_mode, lhs_json_index = self.get_sub_block_custom_range(
                    start_index, 
                    end_index, 
                    block_id_1, 
                    False, 
                    return_extraction_mode=True, 
                    json_list=json_list, 
                    lhs_index=lhs_index
                )
                block_2, rhs_extraction_mode, rhs_json_index = sp_tensor.get_sub_block_custom_range(
                    start_index, 
                    end_index, 
                    block_id_2, 
                    False, 
                    return_extraction_mode=True, 
                    json_list=json_list, 
                    lhs_index=rhs_index
                )
                
                binary_sparse_tensor_overlap_expenses.update_total_time(time.perf_counter()-start_time)
                # Index Dependant
                before_binary_json_len = len(json_list)
                block, block_json_index = block_1.binary(block_2, op, json_list = json_list,
                lhs_index = lhs_json_index,
                rhs_index = rhs_json_index)
                binary_sparse_tensor_overlap.update_total_time(time.perf_counter()-start_time)


                # json_obj_1 = {
                #     "method": "get_sub_block_custom_range",
                #     "lhs": "json_list_" + str(lhs_index),
                #     "start_index": start_index.tolist(),
                #     "end_index": end_index.tolist(),
                #     "block_id": block_id_1,
                #     "tensor": False,
                #     "output": len(json_list)
                # }

                # json_list.append(json_obj_1)

                # json_obj_2 = {
                # #     "method": "get_sub_block_custom_range",
                # #     "lhs": "json_list_" + str(rhs_index),
                # #     "start_index": start_index.tolist(),
                # #     "end_index": end_index.tolist(),
                # #     "block_id": block_id_2,
                # #     "tensor": False,
                # #     "output": len(json_list)
                # }

                # json_list.append(json_obj_2)

                # json_obj = {
                #     "method": "binary",
                #     "op": op.__name__,
                #     "lhs": "json_list_" + str(lhs_json_index),
                #     "rhs": "json_list_" + str(rhs_json_index),
                #     "output": len(json_list)
                # }
                # json_list.append(json_obj)
            
            start = time.perf_counter()
            #FIX FOR: result ConstBlocks equal to dense_const were stored as explicit blocks (original constraintflow skips them), so zero blocks piled up in stored polyexps and were later densified.
            if (isinstance(block, ConstBlock) or block.block_type == 'C') and block.block == dense_const:
                continue
            res_blocks.append(block)

            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(res_blocks_json_list_index),
                "value": "json_list_" + str(block_json_index),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            res_blocks_json_list_index = len(json_list)-1

            res_start_indices.append(start_indices[i])
            res_end_indices.append(end_indices[i])
            block_operations.append(block_operation)
            block_mappings.append(block_mapping)
            binary_fixed_costs.update_total_time(time.perf_counter()-start)
        
        
        
        

       
        start = time.perf_counter()
        res = SparseTensor(res_start_indices, res_blocks, self.dims, self.total_size, res_end_indices, new_type, dense_const, og_json_list=json_list, blocks_index=res_blocks_json_list_index)
        binary_fixed_costs.update_total_time(time.perf_counter()-start)
        total_binary_sparse_tensor.update_total_time(time.perf_counter()-total_start_time)

        

        
        return res
            
        

    def any(self, json_list=None, lhs_index=0):
        trace = json_list is not None

        res = False
        # return self.num_blocks > 0
        if not self.check_dense():
            if self.dense_const == True:
                if trace:
                    json_list.append(
                        {
                            "method": "bool_value",
                            "value": True,
                            "output": len(json_list)
                        }
                    )
                res = True
            elif self.type != bool and self.dense_const != 0.0:
                if trace:
                    json_list.append(
                        {
                            "method": "bool_value",
                            "value": True,
                            "output": len(json_list)
                        }
                    )
                res = True
        if not res:
            if trace:
                json_obj = {
                    "method": "initialise",
                    "name": "bool_value",
                    "value": False,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                acc_json_list_index = len(json_list) - 1

            for i in range(self.num_blocks):
                if trace:
                    json_obj = {
                        "method": "extract_block",
                        "input": "json_list_" + str(lhs_index),
                        "index": i,
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                    block_json_index = len(json_list) - 1

                    block_any, any_json_index = self.blocks[i].any(json_list=json_list, template_index=block_json_index, simulacrum=True)

                    json_obj = {
                        "method": "simple_binary",
                        "op": "or_",
                        "lhs": "json_list_" + str(acc_json_list_index),
                        "rhs": "json_list_" + str(any_json_index),
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                    acc_json_list_index = len(json_list) - 1

                    if block_any:
                        res = True
                        break
                else:
                    if self.blocks[i].any():
                        res = True
                        break
        return res
    
    def float(self, json_list=[], lhs_index=-1):
        json_list.append({
            "method": "initialise",
            "name": "res_blocks",
            "value": "[]",
            "output": len(json_list),
        })
        current_list_index = len(json_list) - 1

        blocks = []
        for i in range(self.num_blocks):
            json_list.append({
                "method": "extract_block",
                "input": "json_list_" + str(lhs_index),
                "index": i,
                "output": len(json_list),
            })
            block_json_index = len(json_list) - 1

            cr, result_block_json_index = self.blocks[i].float(json_list=json_list, template_index=block_json_index, simulacrum=True)

            json_list.append({
                "method": "append_list",
                "list": "json_list_" + str(current_list_index),
                "value": "json_list_" + str(result_block_json_index),
                "output": len(json_list),
            })
            current_list_index = len(json_list) - 1

            blocks.append(cr)

        return SparseTensor(self.start_indices, blocks, self.dims, self.total_size, end_indices=self.end_indices, type=float, dense_const=float(self.dense_const), og_json_list=json_list, blocks_index=current_list_index)
    
    def matmul(self, sp_tensor, json_list=[], lhs_index=-1, rhs_index=-1):
        total_start_time = time.perf_counter()
        # json_list = []
        
        if isinstance(sp_tensor, torch.Tensor):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        

        assert(self.type == float or self.type == int)
        assert(sp_tensor.type == float or sp_tensor.type == int)
        assert(self.dense_const == 0.0)
        
        if sp_tensor.dense_const != 0.0 and sp_tensor.check_dense()==False:
            dense_sp_tensor, dense_sp_tensor_index = sp_tensor.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            # Re-wrap the densified rhs as a single DenseBlock SparseTensor in
            # the trace, and point rhs_index at it for downstream extractions.
            rhs_db = DenseBlock(dense_sp_tensor, json_list, dense_sp_tensor_index)
            rhs_blocks_index = len(json_list)
            json_list.append({
                "method": "initialise",
                "name": "rhs_blocks",
                "value": "[]",
                "output": rhs_blocks_index,
            })
            rhs_appended_index = len(json_list)
            json_list.append({
                "method": "append_list",
                "list": "json_list_" + str(rhs_blocks_index),
                "value": "json_list_" + str(rhs_db.json_index),
                "output": rhs_appended_index,
            })
            sp_tensor = SparseTensor([torch.tensor([0]*len(dense_sp_tensor.shape))], [rhs_db], sp_tensor.dims, sp_tensor.total_size, type=float, dense_const=0.0, og_json_list=json_list, blocks_index=rhs_appended_index)
            rhs_index = sp_tensor.json_index

            dense_self, dense_self_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
            # Re-wrap the densified lhs (self) likewise, updating lhs_index.
            lhs_db = DenseBlock(dense_self, json_list, dense_self_index)
            lhs_blocks_index = len(json_list)
            json_list.append({
                "method": "initialise",
                "name": "lhs_blocks",
                "value": "[]",
                "output": lhs_blocks_index,
            })
            lhs_appended_index = len(json_list)
            json_list.append({
                "method": "append_list",
                "list": "json_list_" + str(lhs_blocks_index),
                "value": "json_list_" + str(lhs_db.json_index),
                "output": lhs_appended_index,
            })
            # self is re-pointed field by field below rather than replaced, so this tensor
            # exists only to emit the record and hand back the slot lhs_index needs.
            traced_lhs = SparseTensor([torch.tensor([0]*len(dense_self.shape))], [lhs_db], len(dense_self.shape), torch.tensor(dense_self.shape), type=float, dense_const=0.0, og_json_list=json_list, blocks_index=lhs_appended_index)
            lhs_index = traced_lhs.json_index
            self.start_indices = [torch.tensor([0]*len(dense_self.shape))]
            self.blocks = [lhs_db]
            self.dims = len(dense_self.shape)
            self.total_size = torch.tensor(dense_self.shape)
            self.end_indices = [torch.tensor(dense_self.shape)]
            self.dense_const = 0.0
            self.num_blocks = 1
        assert(sp_tensor.dense_const == 0.0 or sp_tensor.check_dense()==True)
        
        


        # Mirror the dispatch below: a lower-rank rhs is a vector contracted along its last dim.
        if sp_tensor.dims <= 2 or self.dims > sp_tensor.dims:
            assert(sp_tensor.total_size[-1].item() == self.total_size[-1].item())
        else:
            assert(sp_tensor.total_size[-2].item() == self.total_size[-1].item())
            rhs_batch = sp_tensor.total_size[:-2]
            lhs_batch = self.total_size[-sp_tensor.dims:-2]
            # Leading dims broadcast; the block extents below are unioned with min/max.
            assert(((rhs_batch == lhs_batch) | (rhs_batch == 1) | (lhs_batch == 1)).all())
        
        start_indices = []
        end_indices = []
        blocks = []
        matmul_operations = []
        matmul_mappings = []
        res_total_size = None

        multiplicands = []

        json_obj = {
            "method": "initialise",
            "name": "res_blocks",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_blocks_json_list_index = len(json_list) - 1

        
        if self.dims == sp_tensor.dims:
            res_total_size = torch.concat([self.total_size[:-1], sp_tensor.total_size[-1:]])
            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-total_start_time)
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    start = time.perf_counter()
                    if sp_tensor.dims == 2:
                        # Save this
                        start_index = torch.concat([self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]])
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                    else:
                        if not block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-2], sp_tensor.end_indices[j][:-2]]):
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            continue
                        if not block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            continue
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)

                        start = time.perf_counter()
                        if full_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            # Store i, j
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(lhs_index),
                                "index": i,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            lhs_block_json_index = len(json_list) - 1

                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(rhs_index),
                                "index": j,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            rhs_block_json_index = len(json_list) - 1

                            block, block_json_index = self.blocks[i].matmul_equal_dims(
                                sp_tensor.blocks[j],
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )


                        elif contained([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):    
                            start_index = torch.concat((sp_tensor.start_indices[j][:-2], self.start_indices[i][-1:], sp_tensor.start_indices[j][-1:]))
                            end_index = torch.concat((sp_tensor.end_indices[j][:-2], self.end_indices[i][-1:], sp_tensor.end_indices[j][-1:]))
                            block_1, rhs_extraction_mode, rhs_block_json_index = sp_tensor.get_sub_block_custom_range(
                                start_index,
                                end_index,
                                j,
                                tensor=False,
                                return_extraction_mode=True,
                                json_list=json_list,
                                lhs_index=rhs_index,
                            )
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            # Store i, and block_1 parameters
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(lhs_index),
                                "index": i,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            lhs_block_json_index = len(json_list) - 1

                            block, block_json_index = self.blocks[i].matmul_equal_dims(
                                block_1,
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )
                            
                        elif contained([sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]], [self.start_indices[i][-1:], self.end_indices[i][-1:]]):    
                            start_index = torch.concat((self.start_indices[i][:-1], sp_tensor.start_indices[j][-2:-1]))
                            end_index = torch.concat((self.end_indices[i][:-1], sp_tensor.end_indices[j][-2:-1]))
                            block_1, lhs_extraction_mode, lhs_block_json_index = self.get_sub_block_custom_range(
                                start_index,
                                end_index,
                                i,
                                tensor=False,
                                return_extraction_mode=True,
                                json_list=json_list,
                                lhs_index=lhs_index,
                            )
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            # Store block_1 parameters and j
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(rhs_index),
                                "index": j,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            rhs_block_json_index = len(json_list) - 1

                            block, block_json_index = block_1.matmul_equal_dims(
                                sp_tensor.blocks[j],
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )
                        start = time.perf_counter()
                        blocks.append(block)

                        json_obj = {
                            "method": "append_list",
                            "list": "json_list_" + str(res_blocks_json_list_index),
                            "value": "json_list_" + str(block_json_index),
                            "output": len(json_list)
                        }
                        json_list.append(json_obj)
                        res_blocks_json_list_index = len(json_list)-1


                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-2], self.start_indices[i][:-2]), self.start_indices[i][-2:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-2], self.end_indices[i][:-2]), self.end_indices[i][-2:-1], sp_tensor.end_indices[j][-1:]])
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                    start = time.perf_counter()
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    multiplicands.append((i, j))
                    matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
        elif self.dims > sp_tensor.dims:
            res_total_size = self.total_size[:-1]
            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-total_start_time)
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    start = time.perf_counter()
                    if sp_tensor.dims == 1:
                        # In this case, I just need to start the start and end indices
                        start_index = self.start_indices[i][:-1]
                        end_index = self.end_indices[i][:-1]
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                    else:
                        if not block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-1], sp_tensor.end_indices[j][:-1]]):
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            continue
                        if not block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            continue
                        
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                        start = time.perf_counter()
                        
                        if full_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):       
                            block_1 = self.blocks[i]
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)    
                            # Store i, j
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(lhs_index),
                                "index": i,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            lhs_block_json_index = len(json_list) - 1

                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(rhs_index),
                                "index": j,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            rhs_block_json_index = len(json_list) - 1

                            block, block_json_index = block_1.matmul_unequal_dims(
                                sp_tensor.blocks[j],
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )
                        elif contained([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            start_index = torch.concat((sp_tensor.start_indices[j][:-1], self.start_indices[i][-1:]))
                            end_index = torch.concat((sp_tensor.end_indices[j][:-1], self.end_indices[i][-1:]))
                            block_1, rhs_extraction_mode, rhs_block_json_index = sp_tensor.get_sub_block_custom_range(
                                start_index,
                                end_index,
                                j,
                                tensor=False,
                                return_extraction_mode=True,
                                json_list=json_list,
                                lhs_index=rhs_index,
                            )
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            # Store i, and sp_tensor parameters
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(lhs_index),
                                "index": i,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            lhs_block_json_index = len(json_list) - 1

                            block, block_json_index = self.blocks[i].matmul_unequal_dims(
                                block_1,
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )
                        elif contained([sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]], [self.start_indices[i][-1:], self.end_indices[i][-1:]]):
                            start_index = torch.concat((self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]))
                            end_index = torch.concat((self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]))
                            block_1, lhs_extraction_mode, lhs_block_json_index = self.get_sub_block_custom_range(
                                start_index,
                                end_index,
                                i,
                                tensor=False,
                                return_extraction_mode=True,
                                json_list=json_list,
                                lhs_index=lhs_index,
                            )
                            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
                            # Store i, and block_1 parameters
                            
                            json_obj = {
                                "method": "extract_block",
                                "input": "json_list_" + str(rhs_index),
                                "index": j,
                                "output": len(json_list)
                            }
                            json_list.append(json_obj)
                            rhs_block_json_index = len(json_list) - 1

                            block, block_json_index = block_1.matmul_unequal_dims(
                                sp_tensor.blocks[j],
                                json_list=json_list,
                                lhs_index=lhs_block_json_index,
                                rhs_index=rhs_block_json_index,
                            )
                        
                        start = time.perf_counter()
                        blocks.append(block)

                        json_obj = {
                            "method": "append_list",
                            "list": "json_list_" + str(res_blocks_json_list_index),
                            "value": "json_list_" + str(block_json_index),
                            "output": len(json_list)
                        }
                        json_list.append(json_obj)
                        res_blocks_json_list_index = len(json_list)-1

                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-1], self.start_indices[i][:-2]), self.start_indices[i][-2:-1]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-1], self.end_indices[i][:-2]), self.end_indices[i][-2:-1]])
                        matmul_operations.append({"operation":"matmul_unequal_dims"})
                        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)

                    start = time.perf_counter()
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
        else:
            matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-total_start_time)
            assert(False)

        start = time.perf_counter()
        overlap_classes = find_connected_blocks(start_indices, end_indices)
        res = sp_tensor_from_overlap_classes(overlap_classes, start_indices, end_indices, blocks, res_total_size, len(res_total_size), self.dense_const, self.type, json_list, res_blocks_json_list_index)
        matmul_sparse_tensor_expenses.update_total_time(time.perf_counter()-start)
        
        return res
    
    def add_block_no_overlap(self, start_index, end_index, block):
        index = self.num_blocks
        flag = False
        for i in range(self.num_blocks):
            d = self.start_indices[i]-start_index
            for j in range(len(d)):
                if d[j]>0:
                    flag = True 
                    index = i
                    break
                elif d[j] < 0:
                    break
            if flag:
                break
        self.start_indices.insert(index, start_index)
        self.end_indices.insert(index, end_index)
        self.blocks.insert(index, block)
        self.num_blocks += 1

    # AVAL: this function is incorrect. what if there are no blocks in self?
    def overwrite_block(self, start_index, end_index, block):
        for i in range(self.num_blocks):
            if block_overlap([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                if contained([start_index, end_index], [self.start_indices[i], self.end_indices[i]]):
                    s = get_slice(start_index-self.start_indices[i], end_index-self.start_indices[i])
                    new_blocks, new_start_indices = self.blocks[i].overwrite_dense_block(block, start_index-self.start_indices[i], tuple(s))
                    new_start_indices = [x+self.start_indices[i] for x in new_start_indices]
                    if len(new_blocks) == 1:
                        self.blocks[i] = new_blocks[0]
                        self.start_indices[i] = new_start_indices[0]
                        self.end_indices[i] = new_start_indices[0]+self.blocks[i].total_shape
                    elif len(new_blocks) == 2:
                        self.blocks[i] = new_blocks[0]
                        self.blocks.insert(i+1, new_blocks[1])
                        self.start_indices[i] = new_start_indices[0]
                        self.start_indices.insert(i+1, new_start_indices[1])
                        self.end_indices[i] = new_start_indices[0]+self.blocks[i].total_shape
                        self.end_indices.insert(i+1, new_start_indices[1]+self.blocks[i+1].total_shape)
                        self.num_blocks += 1
                    else:
                        raise Exception('NOT IMPLEMENTED')
                    return 
                else:
                    if contained([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                        if self.num_blocks == 1:
                            self.blocks[0].block = block.block
                            self.start_indices[0] = start_index
                            self.end_indices[0] = end_index
                            return
                    raise Exception('NOT IMPLEMENTED')
        self.add_block_no_overlap(start_index, end_index, block)

    
    def overwrite_from_index(self, sp_tensor, index):
        for i in range(sp_tensor.num_blocks):
            new_start_index = sp_tensor.start_indices[i]+index
            new_end_index = sp_tensor.end_indices[i]+index
            self.overwrite_block(new_start_index, new_end_index, sp_tensor.blocks[i])
        return self
    
    # TODO: CHECK THE OVERWRITE CODE. WE MAY NEED TO DO - res = self.copy()
    def overwrite(self, sp_tensor):
        # res = self.copy()
        for i in range(sp_tensor.num_blocks):
            self.overwrite_block(sp_tensor.start_indices[i], sp_tensor.end_indices[i], sp_tensor.blocks[i])
        return self
        
    def squeeze(self, index, json_list=None, lhs_index=-1, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
        start_time = time.perf_counter()
        # time.sleep(0.005)
        assert(self.total_size[index].item() == 1)
        owns_capture = json_list is None and dummy_mode 
        if owns_capture:
            json_list = [{"method": "noop", "input": "lhs", "output": 0}]
            lhs_index = 0
        trace = json_list is not None
        if not trace:
            json_list = []
        dims = self.dims-1
        total_size = torch.concat([self.total_size[:index], self.total_size[index+1:]])
        start_indices = []
        end_indices = []
        blocks = []
        if trace:
            json_obj = {
                "method": "initialise",
                "name": "blocks",
                "value": "[]",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            blocks_json_list_index = len(json_list) - 1
        for i in range(self.num_blocks):

            if trace:
                json_obj = {
                    "method": "extract_block",
                    "input": "json_list_" + str(lhs_index),
                    "index": i,
                    "output": len(json_list)
                }
                json_list.append(json_obj)

            block, block_index = self.blocks[i].squeeze(index, json_list=json_list, template_index=len(json_list)-1, simulacrum=True)
            blocks.append(block)
            if trace:
                json_obj = {
                    "method": "append_list",
                    "list": "json_list_" + str(blocks_json_list_index),
                    "value": "json_list_" + str(block_index),
                    "output": len(json_list)
                }
                json_list.append(json_obj)
                blocks_json_list_index = len(json_list) - 1
            
            start_indices.append(torch.concat([self.start_indices[i][:index], self.start_indices[i][index+1:]]))
            end_indices.append(torch.concat([self.end_indices[i][:index], self.end_indices[i][index+1:]]))
            
        # blocks_json_list_index only exists under trace, so the threaded call is guarded the
        # same way the hand-written record was.
        if trace:
            x = SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const, og_json_list=json_list, blocks_index=blocks_json_list_index)
        else:
            x = SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
        if owns_capture:
            write_jit_capture_file("jit_squeeze", "squeeze", layer_index, counter, inside_while, while_number, while_iteration, json_list)
        end_time = time.perf_counter()
        squeeze_time.update_total_time(end_time - start_time)
        return x
    
    def unsqueeze(self, index, json_list=None, lhs_index=-1, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
        if not inductor_mode.get_flag():
            start_time = time.perf_counter()
        owns_capture = json_list is None and dummy_mode
        if owns_capture:
            json_list = [{"method": "noop", "input": "lhs", "output": 0}]
            lhs_index = 0
        trace = json_list is not None
        if not trace:
            json_list = []
        if trace and not owns_capture:
            assert lhs_index != -1
        dims = self.dims+1
        total_size = torch.concat([self.total_size[:index], torch.tensor([1]), self.total_size[index:]])
        total_size = total_size.type(torch.int64)
        start_indices = []
        end_indices = []
        blocks = []
        if trace:
            json_obj = {
                "method": "initialise",
                "name": "blocks",
                "value": "[]",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            blocks_json_list_index = len(json_list) - 1
        for i in range(self.num_blocks):
            if trace:
                json_obj = {
                    "method": "extract_block",
                    "input": "json_list_" + str(lhs_index),
                    "index": i,
                    "output": len(json_list)
                }
                json_list.append(json_obj)

            block, block_index = self.blocks[i].unsqueeze(index, json_list=json_list, template_index=len(json_list)-1, simulacrum=True)
            blocks.append(block)
            start_indices.append(torch.concat([self.start_indices[i][:index], torch.tensor([0]), self.start_indices[i][index:]]))
            end_indices.append(torch.concat([self.end_indices[i][:index], torch.tensor([1]), self.end_indices[i][index:]]))
            if trace:
                json_obj = {
                    "method": "append_list",
                    "list": "json_list_" + str(blocks_json_list_index),
                    "value": "json_list_" + str(block_index),
                    "output": len(json_list)
                }
                json_list.append(json_obj)
                blocks_json_list_index = len(json_list) - 1

        if not inductor_mode.get_flag():
            end_time = time.perf_counter()
            unsqueeze_time.update_total_time(end_time-start_time)
        # blocks_json_list_index only exists under trace, so the threaded call is guarded the
        # same way the hand-written record was.
        if trace:
            result = SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const, og_json_list=json_list, blocks_index=blocks_json_list_index)
        else:
            result = SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
        if owns_capture:
            write_jit_capture_file("jit_unsqueeze", "unsqueeze", layer_index, counter, inside_while, while_number, while_iteration, json_list)
        return result
        
    
    def repeat(self, repeat_dims, json_list=None, lhs_index=-1):
        # if json_list is not None:
        # print('repeat:')
        # caller = inspect.stack()[1]
        # fun = caller.function
        # file = caller.filename
        # line = caller.lineno
        # print(f'repeat called as non-owner by {file}:{fun}:{line}')
        # print('----------------------------------')
        # pprint(inspect.stack())
        if json_list is None:
            json_list = []
        total_size = self.total_size * repeat_dims
        start_indices = []
        end_indices = []
        blocks = []
        json_obj = {
            "method": "initialise",
            "name": "blocks",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        blocks_json_list_index = len(json_list) - 1
        for i in range(self.num_blocks):
            start_indices.append(self.start_indices[i]*repeat_dims)
            end_indices.append(self.end_indices[i]*repeat_dims)

            json_obj = {
                "method": "extract_block",
                "input": "json_list_" + str(lhs_index),
                "index": i,
                "output": len(json_list)
            }
            json_list.append(json_obj)

            block, block_index = self.blocks[i].repeat(repeat_dims, json_list=json_list, template_index=len(json_list)-1, simulacrum=True)
            blocks.append(block)
            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(blocks_json_list_index),
                "value": "json_list_" + str(block_index),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            blocks_json_list_index = len(json_list) - 1
        return SparseTensor(start_indices, blocks, self.dims, total_size, end_indices, self.type, self.dense_const, og_json_list=json_list, blocks_index=blocks_json_list_index)

    def clamp(self, const, min_true, json_list=[], lhs_index=-1):
        start_time = time.perf_counter()
        blocks = []
        json_obj = {
            "method": "initialise",
            "name": "blocks",
            "value": "[]",
            "output": len(json_list),
        }
        json_list.append(json_obj)
        blocks_json_list_index = len(json_list) - 1
        clamp_sparse_tensor_expense.update_total_time(time.perf_counter()-start_time)
        for i in range(self.num_blocks):
            
            json_obj = {
                "method": "extract_block",
                "input": "json_list_" + str(lhs_index),
                "index": i,
                "output": len(json_list)
            }
            json_list.append(json_obj)
            block_json_index = len(json_list) - 1

            cr, cr_index = self.blocks[i].clamp(const, min_true, json_list=json_list, template_index=block_json_index, simulacrum=True)

            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(blocks_json_list_index),
                "value": "json_list_" + str(cr_index),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            blocks_json_list_index = len(json_list) - 1
            append_start_time = time.perf_counter()
            blocks.append(cr)
            clamp_sparse_tensor_expense.update_total_time(time.perf_counter()-append_start_time)
        
        return SparseTensor(self.start_indices, blocks, self.dims, self.total_size, self.end_indices, self.type, self.dense_const, og_json_list=json_list, blocks_index=blocks_json_list_index)


    def sum(self, dim, json_list=None, lhs_index=-1, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
        start_time = time.perf_counter()
        owns_capture = json_list is None and dummy_mode
        if owns_capture:
            json_list = [{"method": "noop", "input": "lhs", "output": 0}]
            lhs_index = 0
        trace = json_list is not None
        if not trace:
            json_list = []
        total_size = torch.concat([self.total_size[:dim], self.total_size[dim+1:]])
        start_indices = []
        end_indices = []
        blocks = []
        if trace:
            json_list.append({
                "method": "initialise",
                "name": "blocks",
                "value": "[]",
                "output": len(json_list),
            })
            blocks_json_list_index = len(json_list) - 1
        else:
            blocks_json_list_index = 0
        for i in range(self.num_blocks):
            start_indices.append(torch.concat([self.start_indices[i][:dim], self.start_indices[i][dim+1:]]))
            end_indices.append(torch.concat([self.end_indices[i][:dim], self.end_indices[i][dim+1:]]))
            if trace:
                json_list.append({
                    "method": "extract_block",
                    "input": "json_list_" + str(lhs_index),
                    "index": i,
                    "output": len(json_list),
                })
            block, block_index = self.blocks[i].sum(dim, json_list=json_list, template_index=len(json_list)-1, simulacrum=True)
            blocks.append(block)
            if trace:
                json_list.append({
                    "method": "append_list",
                    "list": "json_list_" + str(blocks_json_list_index),
                    "value": "json_list_" + str(block_index),
                    "output": len(json_list),
                })
                blocks_json_list_index = len(json_list) - 1
        overlap_classes = find_connected_blocks(start_indices, end_indices)
        res = sp_tensor_from_overlap_classes(overlap_classes, start_indices, end_indices, blocks, total_size, self.dims-1, self.dense_const, self.type,
                                             json_list=json_list, res_blocks_json_list_index=blocks_json_list_index)
        if owns_capture:
            write_jit_capture_file("jit_sum", "sum", layer_index, counter, inside_while, while_number, while_iteration, json_list)
        sum_time.update_total_time(time.perf_counter() - start_time)
        return res
    
def sparse_max(x:SparseTensor, y:SparseTensor, json_list=[], x_json_index=-1, y_json_index=-1):
    z = x.binary(y, operator.gt, json_list=json_list, lhs_index=x_json_index, rhs_index=y_json_index)
    cond_json_index = len(json_list) - 1
    return sp_where(z, x, y, json_list=json_list, x_json_index=cond_json_index, y_json_index=x_json_index, z_json_index=y_json_index)

def sparse_min(x:SparseTensor, y:SparseTensor, json_list=[], x_json_index=-1, y_json_index=-1):
    z = x.binary(y, operator.lt, json_list=json_list, lhs_index=x_json_index, rhs_index=y_json_index)
    cond_json_index = len(json_list) - 1
    return sp_where(z, x, y, json_list=json_list, x_json_index=cond_json_index, y_json_index=x_json_index, z_json_index=y_json_index)


def sp_where(x: SparseTensor, y: SparseTensor, z: SparseTensor, json_list=[], x_json_index=-1, y_json_index=-1, z_json_index=-1):
    assert((x.total_size == y.total_size).all() and (y.total_size == z.total_size).all())
    assert(x.dims == y.dims and x.dims == z.dims)
    assert(y.type==z.type or (y.type in [int, float] and z.type in [int, float]))
    assert(x.type == bool)

    new_type = y.type
    if x.dense_const:
        dense_const = y.dense_const
    else:
        dense_const = z.dense_const
    
    start_indices, end_indices, yz_indices = y.union_tensors(z, indices=True)
    
    dummy_blocks = []
    for i in range(len(start_indices)):
        dummy_blocks.append(ConstBlock(0, end_indices[i]-start_indices[i]))
    dummy_sp_tensor = SparseTensor(start_indices, dummy_blocks, y.dims, y.total_size, end_indices, y.type, y.dense_const)
    
    start_indices, end_indices, indices = x.union_tensors(dummy_sp_tensor, indices=True)

    res_start_indices = []
    res_end_indices = []
    res_blocks = []

    json_obj = {
        "method": "initialise",
        "name": "res_blocks",
        "value": "[]",
        "output": len(json_list),
    }
    json_list.append(json_obj)
    res_blocks_json_list_index = len(json_list) - 1


    for i in range(len(start_indices)):
        start_index = start_indices[i]
        end_index = end_indices[i]
        if len(indices[i][0]) == 0:
            yz_index = indices[i][1][0]
            y_indices = yz_indices[yz_index][0]
            z_indices = yz_indices[yz_index][1]
            if x.dense_const and len(y_indices) == 1:
                block, block_json_index = y.get_sub_block_custom_range(
                    start_index,
                    end_index,
                    y_indices[0],
                    False,
                    json_list=json_list,
                    lhs_index=y_json_index,
                )
            elif not(x.dense_const) and len(z_indices) == 1:
                block, block_json_index = z.get_sub_block_custom_range(
                    start_index,
                    end_index,
                    z_indices[0],
                    False,
                    json_list=json_list,
                    lhs_index=z_json_index,
                )
            else:
                continue
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block)

            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(res_blocks_json_list_index),
                "value": "json_list_" + str(block_json_index),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            res_blocks_json_list_index = len(json_list) - 1


        elif len(indices[i][1]) > 0:
            x_index = indices[i][0][0]
            x_block, x_block_index = x.get_sub_block_custom_range(
                start_index,
                end_index,
                x_index,
                False,
                json_list=json_list,
                lhs_index=x_json_index,
            )

            yz_index = indices[i][1][0]
            y_indices = yz_indices[yz_index][0]
            z_indices = yz_indices[yz_index][1]

            if len(y_indices) == 0:
                y_block = ConstBlock(y.dense_const, end_index-start_index, json_list)
                y_block_index = y_block.json_index
            else:
                y_block, y_block_index = y.get_sub_block_custom_range(
                    start_index,
                    end_index,
                    y_indices[0],
                    False,
                    json_list=json_list,
                    lhs_index=y_json_index,
                )

            if len(z_indices) == 0:
                z_block = ConstBlock(z.dense_const, end_index-start_index, json_list)
                z_block_index = z_block.json_index
            else:
                z_block, z_block_index = z.get_sub_block_custom_range(
                    start_index,
                    end_index,
                    z_indices[0],
                    False,
                    json_list=json_list,
                    lhs_index=z_json_index,
                )

            block, block_json_index = sp_where_block(
                x_block,
                y_block,
                z_block,
                json_list=json_list,
                x_index=x_block_index,
                y_index=y_block_index,
                z_index=z_block_index,
                simulacrum=True,
            )

            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block)

            json_obj = {
                "method": "append_list",
                "list": "json_list_" + str(res_blocks_json_list_index),
                "value": "json_list_" + str(block_json_index),
                "output": len(json_list)
            }
            json_list.append(json_obj)
            res_blocks_json_list_index = len(json_list) - 1

    return SparseTensor(res_start_indices, res_blocks, x.dims, x.total_size, res_end_indices, new_type, dense_const, og_json_list=json_list, blocks_index=res_blocks_json_list_index)
