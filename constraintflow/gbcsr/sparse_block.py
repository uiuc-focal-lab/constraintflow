from numpy import block
import copy
import os
import sys
import copyreg
import math
#FEAT: weak references for the selection-weight cache.
import weakref
from typing import Optional
from torch.utils.weak import WeakIdKeyDictionary
from constraintflow.lib import s_shaped
from constraintflow.lib.s_shaped import sigma_tangent, tanh_tangent
import torch
import torch.nn.functional as F
import time
import operator
from constraintflow.gbcsr.op_helper import *
from constraintflow.lib.globals import *

def meta_tensor(block_shape, dtype=None):
    return torch.empty(block_shape, device='meta', dtype=dtype)

def _sync():
    device_mode.sync()

def get_slice(start_index, end_index):
    dims = start_index.shape[0]
    s = []
    for j in range(dims):
        s.append(slice(int(start_index[j]), int(end_index[j])))
    return s

def get_diagonal(block, diag_index, json_list = None, template_index = None):
    if json_list is not None:
        json_obj = {
            "method": "torch_diagonal",
            "input": "json_list_" + str(template_index),
            "dim1": diag_index-1,
            "dim2": diag_index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
    res = block.diagonal(dim1=diag_index-1, dim2=diag_index)
    permutation = list(range(len(res.shape)))
    last_index = permutation[-1]
    permutation = permutation[:-1]
    permutation.insert(diag_index-1, last_index)
    if json_list is not None:
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(len(json_list) - 1),
            "permutation": permutation,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        return res.permute(permutation), len(json_list) - 1
    return res.permute(permutation)

def operation(x, y, op):
    start_time = time.perf_counter()
    # if baseline_gpu_mode:
    #    torch.cuda.synchronize()
    # print(f'{type(x)} {type(y)} {x.device} {y.device}')
    z = op(x, y)
    # if baseline_gpu_mode:
        # torch.cuda.synchronize()
    binary_profilier.update_total_time(time.perf_counter() - start_time)
    return z

def unary_operation(x, op, json_list = None, block_index = -1, unary_index = -1, concrete_unary = False):
    start_time = time.perf_counter()
    #_sync()
    start_op_time = time.perf_counter()
    if op == 'sigma':
        if json_list is not None:
            json_list.append({
                "method": "torch_sigmoid",
                "block": "json_list_" + str(block_index),
                "output" : len(json_list)
            })
        z = torch.sigmoid(x)
    #FEAT: table-backed tangent points of S-shaped activations.
    elif op in s_shaped.FUNCTIONS:
        if json_list is not None:
            json_list.append({
                "method": "torch_sigmoid",
                "op": op,
                "block": "json_list_" + str(block_index),
                "output" : len(json_list)
            })
        z = s_shaped.FUNCTIONS[op](x)
    else:
        if json_list is not None:
            
            if unary_index != -1:
                json_list.append({
                    "method": "simple_unary",
                    "input": "json_list_" + str(block_index),
                    "operation": "json_list_" + str(unary_index),
                    "output" : len(json_list)
                })
            elif concrete_unary:
                if op == operator.not_:
                    op_name = 'not'
                elif op == operator.neg:
                    op_name = '-'
                json_list.append({
                    "method": "simple_unary",
                    "input": "json_list_" + str(block_index),
                    "op": op_name,
                    "output" : len(json_list)
                })
            else:
                AssertionError("Not Implemented")

        z = op(x)
    #_sync()
    unary_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
    unary_profilier.update_total_time(time.perf_counter() - start_time)
    if json_list is not None:
        return z, len(json_list) - 1
    return z

def where_block(x, y, z, json_list=None, cond_index=-1, lhs_index=-1, rhs_index=-1):
    start_time = time.time()
    if json_list is not None:
        json_list.append({
            "method": "torch_where",
            "cond": "json_list_" + str(cond_index),
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "output": len(json_list),
        })
    w = torch.where(x, y, z)
    where_time.update_op_time(time.time() - start_time)
    if json_list is not None:
        return w, len(json_list) - 1
    return w

class SparseBlock:
    repeat_dims = []
    def __init__(self, block, total_shape, block_type='D'):
        self.total_shape = torch.as_tensor(total_shape, dtype=torch.int64)
        self.block_type = block_type
        if block_type != 'C' and dummy_mode:
            block_shape = block.shape if isinstance(block, torch.Tensor) else self.total_shape
            if isinstance(block, bool) or (isinstance(block, torch.Tensor) and block.dtype == torch.bool):
                block_dtype = torch.bool
            else:
                block_dtype = torch.float
            self.block = meta_tensor(block_shape, dtype=block_dtype)
            return

        if isinstance(block, torch.Tensor) and not block.is_meta:
            if not inductor_mode.get_flag():
                start_transfer = time.perf_counter()

            block = block.to(device_mode.get_device())
            if not inductor_mode.get_flag():
                binary_profilier.update_data_transfer_time(time.perf_counter() - start_transfer)
        
        if isinstance(block, bool) or (isinstance(block, torch.Tensor) and block.dtype == torch.bool):
            self.block = block 
        else:
            if isinstance(block, torch.Tensor):
                self.block = block.type(torch.float)
            else:
                self.block = float(block)

        

    def copy(self):
        new_block = self.block 
        if isinstance(new_block, torch.Tensor):
            new_block = new_block.clone()
        return self.create_similar(new_block)

    def __copy__(self):
        return self.copy()

    def __reduce__(self):
        return (copyreg._reconstructor, (type(self), object, None), self.__dict__)

    def __deepcopy__(self, memo):
        copied = self.copy()
        if memo is not None:
            memo[id(self)] = copied
        return copied

    def get_dense(self, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def repeat(self, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def unsqueeze(self, index, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def squeeze(self, index, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def matmul_equal_dims(self, sp_block, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def matmul_unequal_dims(self, sp_block, *args, **kwargs):
        raise Exception(f'Not implemented for {type(self)}')
    
    def binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        # assert((self.total_shape == sp_block.total_shape).all())
        start_time = time.perf_counter()
        if isinstance(sp_block, ConstBlock):
            if sp_block.block == identity_element(op):
                json_obj = {
                    "method": "noop",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return self, len(json_list) - 1
            elif sp_block.block == annihilator_element(op):
                json_obj = {
                    "method": "noop",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return sp_block, len(json_list) - 1
            if op(0, sp_block.block) in [0, False]:
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1

                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                new_block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=new_block, json_list=json_list, template_index=binary_index, simulacrum=True)
                return res, res_index 
        if op in disjunction_ops:
            res, res_index = self.disjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index 
        elif op in conjunction_ops:
            res, res_index = self.conjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index 
        else:
            # assert(False)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            pass
    
    def disjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=block, json_list=json_list, template_index=binary_index, simulacrum=True)
                return res, res_index 
                
            block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            traced_res = DenseBlock(block, json_list, len(json_list) - 1)
            return traced_res, traced_res.json_index
        elif not isinstance(self, type(sp_block)):
            block_1 = self
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1, lhs_index = self.convert_to_patches(json_list=json_list, index=lhs_index,  simulacrum=True)
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            # Identity-skip + conv-path coefficients (a residual Add's back-substitution
            # meeting at the shared input): keep them compact as patches instead of
            # densifying both. Falls through to the dense path if the geometry doesn't fit.
            if isinstance(self, DiagonalBlock) and isinstance(sp_block, PatchesBlock):
                converted = self.as_unit_patches(sp_block, json_list, lhs_index)
                if converted is not None:
                    return converted[0].disjunctive_binary(sp_block, op, json_list, converted[1], rhs_index)
            if isinstance(self, PatchesBlock) and isinstance(sp_block, DiagonalBlock):
                converted = sp_block.as_unit_patches(self, json_list, rhs_index)
                if converted is not None:
                    return self.disjunctive_binary(converted[0], op, json_list, lhs_index, converted[1])
            if isinstance(self, ConstBlock):
                self_block, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
                block_1 = DenseBlock(self_block, json_list, lhs_index)
                lhs_index = len(json_list) - 1
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            if isinstance(sp_block, KernelBlock):
                block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, template_index=rhs_index)
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
            if isinstance(sp_block, ConstBlock):
                block_temp, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
                block_2 = DenseBlock(block_temp, json_list, rhs_index)
                rhs_index = len(json_list) - 1
                return block_1.disjunctive_binary(block_2, op, json_list, lhs_index, rhs_index)
        block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
        json_obj = {
            "method": "torch_binary",
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block = operation(block_1, block_2, op)
        traced_res = DenseBlock(block, json_list, len(json_list) - 1)
        return traced_res, traced_res.json_index
            
    def conjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        # The two RepeatBlock cases below are new -- og_cfg has no RepeatBlock
        # handling here either and always falls back to full materialization.
        if isinstance(self, type(sp_block)):
            if not isinstance(self, (KernelBlock, PatchesBlock)) or self.parameters() == sp_block.parameters():
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=block, json_list=json_list, template_index=binary_index,  simulacrum=True)
                return res, res_index 
            block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            traced_res = DenseBlock(block, json_list, len(json_list) - 1)
            return traced_res, traced_res.json_index
        elif not isinstance(self, type(sp_block)):
            block_1 = self 
            block_2 = sp_block
            if isinstance(self, KernelBlock):
                block_1, lhs_index = self.convert_to_patches(json_list=json_list, template_index=lhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            if isinstance(self, ConstBlock):
                self_block, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
                block_1 = DenseBlock(self_block, json_list, lhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=block_1.json_index, rhs_index=rhs_index)
            if isinstance(sp_block, KernelBlock):
                block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, index=rhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            if isinstance(sp_block, ConstBlock):
                sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
                dense_rhs = DenseBlock(sp_block, json_list, rhs_index)
                return block_1.conjunctive_binary(block_2, op, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_rhs.json_index)
            if isinstance(sp_block, RepeatBlock) and isinstance(self, DenseBlock):
                # sp_block's compact block broadcasts fine against self's full
                # block via ordinary torch rules, so no need to materialize it
                # first. Restricted to self=DenseBlock since Diagonal/Patches use
                # different compact-block layouts where this isn't safe to assume.
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block = operation(self.block, sp_block.block, op)
                res, res_index = self.create_similar(block=block, json_list=json_list, template_index=len(json_list) - 1, simulacrum=True)
                return res, res_index
            if isinstance(self, RepeatBlock) and isinstance(sp_block, DenseBlock):
                # Symmetric case: self is the compact side here instead of sp_block.
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block = operation(self.block, sp_block.block, op)
                res, res_index = sp_block.create_similar(block=block, json_list=json_list, template_index=len(json_list) - 1, simulacrum=True)
                return res, res_index
        if isinstance(self, DiagonalBlock) and isinstance(sp_block, DenseBlock):
            block_1 = self.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1

            block_2 = sp_block.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1

            block_2, rhs_index = get_diagonal(block_2, self.diag_index, json_list=json_list, template_index=rhs_index)
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = operation(block_1, block_2, op)
            res, res_index = self.create_similar(block=block, json_list=json_list, template_index=len(json_list)-1,  simulacrum=True)
            return res, res_index 
        elif isinstance(sp_block, DiagonalBlock) and isinstance(self, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_1, lhs_index = get_diagonal(self.block, sp_block.diag_index, json_list=json_list, template_index=len(json_list)-1)
            block_2 = sp_block.block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)

            block = operation(block_1, block_2, op)
            res, res_index =  sp_block.create_similar(block=block, json_list=json_list, template_index=len(json_list)-1,  simulacrum=True)
            return res, res_index 
        block_1, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index,  simulacrum=True)
        block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index,  simulacrum=True)
        block = operation(block_1, block_2, op)
        json_obj = {
            "method": "torch_binary",
            "lhs": "json_list_" + str(lhs_index),
            "rhs": "json_list_" + str(rhs_index),
            "op": op.__name__,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        traced_res = DenseBlock(block, json_list, len(json_list) - 1)
        return traced_res, traced_res.json_index



    def unary(self, op, json_list=[], template_index=-1, simulacrum=False):
        if isinstance(self.block, torch.Tensor) and self.block.dtype == bool or isinstance(self.block, bool):
            # assert(op == operator.not_)
            pass
        else:
            # assert(op == operator.neg)
            pass
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        block_index = len(json_list) - 1
        new_block, unary_index = unary_operation(self.block, op, json_list=json_list, block_index=block_index, concrete_unary = True)
        res, res_index = self.create_similar(block=new_block, json_list=json_list, template_index=unary_index, simulacrum=True)
        if simulacrum:
            return res, res_index
        return res
    
    def float(self, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        block_index = len(json_list) - 1
        json_list.append({
            "method": "torch_float",
            "input": "json_list_" + str(block_index),
            "output": len(json_list),
        })
        float_index = len(json_list) - 1
        res, res_index = self.create_similar(block=(self.block).float(), json_list=json_list, template_index=float_index, simulacrum=True)
        if simulacrum:
            return res, res_index
        return res

    def sum(self, dim):
        raise Exception(f'Not implemented for {type(self)}')
    
    def size(self):
        raise Exception(f'Not implemented for {type(self)}')
    
    def clamp(self, const, min_true, json_list=[], template_index=-1, simulacrum=False):
        start_time_total = time.perf_counter()
        # _sync()
        if min_true:
            clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time_total)
            start_op_time = time.perf_counter()
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            extract_index = len(json_list) - 1
            json_list.append({
                "method": "torch_clamp",
                "input": "json_list_" + str(extract_index),
                "const": const,
                "min_true": True,
                "output": len(json_list),
            })
            clamp_index = len(json_list) - 1
            new_block = self.block.clamp(min=const)
            clamp_sparse_block_op_time.update_total_time(time.perf_counter() - start_op_time)
        else:
            clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time_total)
            start_op_time = time.perf_counter()
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            extract_index = len(json_list) - 1
            json_list.append({
                "method": "torch_clamp",
                "input": "json_list_" + str(extract_index),
                "const": const,
                "min_true": False,
                "output": len(json_list),
            })
            clamp_index = len(json_list) - 1
            new_block = self.block.clamp(max=const)
            clamp_sparse_block_op_time.update_total_time(time.perf_counter() - start_op_time)
        start_time = time.perf_counter()
        # _sync()
        res, res_index = self.create_similar(block=new_block, json_list=json_list, template_index=clamp_index, simulacrum=True)
        clamp_sparse_block_expense.update_total_time(time.perf_counter() - start_time)
        if simulacrum:
            return res, res_index
        return res

    def type(self):
        if isinstance(self, DiagonalBlock):
            return 'Diagonal'
        if isinstance(self, KernelBlock):
            return 'Kernel'
        if isinstance(self, DenseBlock):
            return 'Dense'
        
    def overwrite_dense_block(self, sp_block, start_index, s):
        raise Exception(f'Not implemented')
    
    def any(self, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.time()
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        block_index = len(json_list) - 1
        json_list.append({
            "method": "torch_any",
            "input": "json_list_" + str(block_index),
            "output": len(json_list),
        })
        res_index = len(json_list) - 1
        if isinstance(self.block, torch.Tensor) and self.block.is_meta and dummy_mode:
            res = True
        else:
            res = self.block.any()
        if simulacrum:
            return res, res_index
        return res



# Upper bound on the intermediates of one patch composition (KernelBlock @
# PatchesBlock, DenseBlock @ PatchesBlock); larger ones are split by origin channel.
PATCH_COMPOSE_CHUNK_BYTES = 1024 * 1024 ** 2


def channel_step(channels, channel_bytes):
    """Largest divisor of `channels` whose slice keeps the intermediates in budget."""
    fits = [d for d in range(1, channels + 1)
            if channels % d == 0 and d * channel_bytes <= PATCH_COMPOSE_CHUNK_BYTES]
    return max(fits or [1])


def cat_columns(pieces, piece_indices, json_list):
    """Concatenate per-channel results along the last (column) dim."""
    if len(pieces) == 1:
        return pieces[0], piece_indices[0]
    json_list.append({
        "method": "torch_cat",
        "inputs": ["json_list_" + str(i) for i in piece_indices],
        "dim": -1,
        "output": len(json_list),
    })
    return torch.cat(pieces, dim=-1), len(json_list) - 1


# cuDNN's transposed-conv kernel choice is erratic across shapes: some fall ~15x slower
# once the batch reaches 65536 (chunking fixes them), chunking slows others down, and a
# stride-1 transposed conv often runs several times faster as the equivalent forward conv.
# So the lowering is picked per call shape by timing the candidates on the first real
# (warmup) call. Under torch.compile the op is opaque, so the choice is made at run time.
CONV_TRANSPOSE_MAX_BATCH = 32768
# An alternative must beat the chunked lowering by this factor to be chosen.
CONV_TRANSPOSE_MIN_GAIN = 0.95
# The dense-matrix (Toeplitz) lowering is tried only while that matrix stays small and
# does at most this many times the conv's multiply-adds.
TOEPLITZ_MAX_ELEMENTS = 1 << 24
TOEPLITZ_MAX_WASTE = 16
#FEAT: larger Toeplitz limits on the CPU.
TOEPLITZ_MAX_ELEMENTS_CPU = 1 << 26
TOEPLITZ_MAX_WASTE_CPU = 64
# Stride-1 rows are also tried tiled grid x grid into images up to this side length.
PACK_GRIDS = (2, 4, 8)
PACK_MAX_SIDE = 128
_conv_transpose_choice = {}


def _pair(v):
    return [int(e) for e in v] if isinstance(v, (list, tuple)) else [int(v), int(v)]


def _split_batch(fn, x):
    if x.shape[0] <= CONV_TRANSPOSE_MAX_BATCH:
        return fn(x)
    return torch.cat([fn(c) for c in x.split(CONV_TRANSPOSE_MAX_BATCH)])


def _packed_lowering(weight, x_spatial, padding, grid):
    """A stride-1 transposed conv over many small rows, run on grid x grid rows tiled into
    one image. Each row sits in a cell padded with k-1 zeros, exactly the reach of its
    output, so cells stay independent; cuDNN is far more efficient on the larger images."""
    cin, cout, kx, ky = weight.shape
    h, w = x_spatial
    cell = (h + kx - 1, w + ky - 1)

    def packed(x):
        n = x.shape[0]
        images = -(-n // (grid * grid))
        if images * grid * grid != n:
            x = torch.cat([x, x.new_zeros(images * grid * grid - n, cin, h, w)])
        img = x.new_zeros(images, cin, grid, cell[0], grid, cell[1])
        img[:, :, :, :h, :, :w] = x.view(images, grid, grid, cin, h, w).permute(0, 3, 1, 4, 2, 5)
        out = F.conv_transpose2d(img.view(images, cin, grid * cell[0], grid * cell[1]), weight)
        out = out.narrow(2, 0, grid * cell[0]).narrow(3, 0, grid * cell[1])
        out = out.unflatten(2, (grid, cell[0])).unflatten(4, (grid, cell[1])).permute(0, 2, 4, 1, 3, 5)
        out = out[..., padding[0]:cell[0] - padding[0], padding[1]:cell[1] - padding[1]]
        return out.reshape(images * grid * grid, cout, *out.shape[-2:])[:n]

    return packed


def _subpixel_lowering(weight, x_spatial, s, padding, output_padding):
    """A stride-s transposed conv as one stride-1 forward conv computing all s*s output
    phases (as extra output channels, each with a ceil(k/s)-sized sub-kernel), followed
    by interleaving the phases into the output: same multiply-adds, forward kernels."""
    cin, cout, kx, ky = weight.shape
    m = [-(-kx // s), -(-ky // s)]
    out_spatial = [(x_spatial[d] - 1) * s - 2 * padding[d] + weight.shape[2 + d] + output_padding[d]
                   for d in range(2)]
    # Output row Y = s*q + r - p of phase r; per phase, the first row, its q, and how many
    # rows the phase conv reaches (rows past it, from output_padding, get no taps: zero).
    phases = []
    partial = False
    for r in range(s):
        rows = []
        for d in range(2):
            first = (r - padding[d]) % s
            q = (first + padding[d] - r) // s
            count = min(len(range(first, out_spatial[d], s)), x_spatial[d] + m[d] - 1 - q)
            partial = partial or count < len(range(first, out_spatial[d], s))
            rows.append((first, q, count))
        phases.append(rows)
    kernel = []

    def subpixel(x):
        if not kernel:
            # Tap t = s*j' + r of phase r, flipped (j = m-1-j') for the forward conv.
            w = F.pad(weight, (0, m[1] * s - ky, 0, m[0] * s - kx))
            w = w.reshape(cin, cout, m[0], s, m[1], s).flip(2, 4)
            kernel.append(w.permute(3, 5, 1, 0, 2, 4).reshape(s * s * cout, cin, m[0], m[1]).contiguous())
        z = F.conv2d(x, kernel[0], padding=(m[0] - 1, m[1] - 1))
        z = z.view(x.shape[0], s, s, cout, z.shape[2], z.shape[3])
        out = (x.new_zeros if partial else x.new_empty)(x.shape[0], cout, *out_spatial)
        for rx in range(s):
            (fx, qx, nx) = phases[rx][0]
            for ry in range(s):
                (fy, qy, ny) = phases[ry][1]
                out[:, :, fx:fx + s * nx:s, fy:fy + s * ny:s] = z[:, rx, ry, :, qx:qx + nx, qy:qy + ny]
        return out

    return subpixel


def _phase_lowering(weight, x_spatial, s, padding, output_padding):
    """Like the sub-pixel lowering, but one forward conv per output phase with only that
    phase's taps (no zero-padded kernel), when k is not a multiple of s."""
    cin, cout, kx, ky = weight.shape
    out_spatial = [(x_spatial[d] - 1) * s - 2 * padding[d] + weight.shape[2 + d] + output_padding[d]
                   for d in range(2)]
    plan = []
    for d, k in enumerate((kx, ky)):
        per_phase = []
        for r in range(s):
            taps = len(range(r, k, s))
            first = (r - padding[d]) % s
            q = (first + padding[d] - r) // s
            count = len(range(first, out_spatial[d], s))
            per_phase.append((r, taps, first, q, min(count, x_spatial[d] + taps - 1 - q), count))
        plan.append(per_phase)
    #FIX FOR: a phase with no taps (kernel smaller than the stride, e.g. a ResNet 1x1 stride-2
    # shortcut) is skipped below, so its outputs were never written into the new_empty buffer and
    # held recycled memory (the previous image's data): unsound, run-order-dependent bounds.
    partial = any(c < full or t == 0 for per_phase in plan for (_, t, _, _, c, full) in per_phase)

    def phases(x):
        out = (x.new_zeros if partial else x.new_empty)(x.shape[0], cout, *out_spatial)
        for (rx, tx, fx, qx, nx, _) in plan[0]:
            for (ry, ty, fy, qy, ny, _) in plan[1]:
                if tx == 0 or ty == 0 or nx <= 0 or ny <= 0:
                    continue
                sub = weight[:, :, rx::s, ry::s].flip(2, 3).transpose(0, 1).contiguous()
                z = F.conv2d(x, sub, padding=(tx - 1, ty - 1))
                out[:, :, fx:fx + s * nx:s, fy:fy + s * ny:s] = z[:, :, qx:qx + nx, qy:qy + ny]
        return out

    return phases


#FEAT: channel-selection weights (at most one nonzero input channel per tap and output channel, e.g. the
# MaxPool expansion's convs) as strided gathers; keyed by storage and version so in-place updates re-check,
# with a weak reference so a freed weight's address is not mistaken for a new weight.
_selection_cache = {}


def _channel_selection(weight, taps=None):
    """(source channel, scale, taps): [kx, ky, out] sources and scales, and the taps with any nonzero; None for
    any other weight. taps (flat [a0, b0, a1, b1, ...]) names a selection weight known at compile time."""
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype, weight.device)
    hit = _selection_cache.get(key)
    if hit is None or hit[0]() is None:
        nonzero = weight != 0
        sel = None
        #FIX FOR: the selection test and the taps list read the weight back to the host (3 D2H per weight per run).
        if taps is not None or (weight.dim() == 4 and int(nonzero.sum(0).max()) <= 1):
            src = nonzero.to(torch.uint8).argmax(0)
            used = list(zip(taps[::2], taps[1::2])) if taps is not None else nonzero.any(0).any(0).nonzero().tolist()
            sel = (src.permute(1, 2, 0).contiguous(), weight.gather(0, src[None])[0].permute(1, 2, 0).contiguous(),
                   [tuple(t) for t in used])
        _selection_cache[key] = hit = (weakref.ref(weight), sel)
    return hit[1]


_selection_maps = {}


def _selection_map(sel, x_spatial, stride, padding, output_padding, out_spatial, device):
    """Disjoint taps (k <= stride): per output element, its flat source index in one row of x and its scale
    (0 where no tap reaches). Cached per selection and geometry."""
    key = (id(sel), tuple(x_spatial), tuple(stride), tuple(padding), tuple(output_padding), device)
    if key not in _selection_maps:
        src, scale, _ = sel
        kx, ky = src.shape[0], src.shape[1]
        h, w = x_spatial
        axes = []
        for d, (k, n_in) in enumerate(((kx, h), (ky, w))):
            full = torch.arange(out_spatial[d], device=device) + padding[d]
            tap, pos = full % stride[d], torch.div(full, stride[d], rounding_mode='floor')
            axes.append((tap.clamp(max=k - 1), pos.clamp(max=n_in - 1), (tap < k) & (pos < n_in)))
        (ta, pa, va), (tb, pb, vb) = axes
        src_map = src[ta[:, None], tb[None, :]].permute(2, 0, 1)           # [out, H, W]
        index = src_map * (h * w) + (pa[:, None] * w + pb[None, :])[None]
        scale_map = scale[ta[:, None], tb[None, :]].permute(2, 0, 1) * (va[:, None] & vb[None, :])[None]
        _selection_maps[key] = (sel, index.reshape(-1).contiguous(), scale_map.contiguous())
    return _selection_maps[key][1:]


#FEAT: index map of a channel-selection transposed conv with disjoint taps, for compiled code that gathers
# (and fuses the gather into its consumer) instead of convolving.
def selection_gather_map(weight, x_spatial, stride, padding, output_padding):
    """(flat source index, scale [out, H, W]) of conv_transpose2d(x, weight) for x of spatial size x_spatial."""
    sel = _channel_selection(weight)
    if sel is None or weight.shape[2] > stride[0] or weight.shape[3] > stride[1]:
        raise ValueError('program compiled for a channel-selection weight with disjoint taps; '
                         f'this {list(weight.shape)} weight is not one: recompile')
    out_spatial = [(x_spatial[d] - 1) * stride[d] - 2 * padding[d] + weight.shape[2 + d] + output_padding[d]
                   for d in range(2)]
    return _selection_map(sel, x_spatial, stride, padding, output_padding, out_spatial, weight.device)


def _selection_lowering(sel, weight, x_spatial, stride, padding, output_padding):
    """A transposed conv with a channel-selection weight. Disjoint taps (k <= stride, e.g. a 2x2/stride-2 or
    1x1 conv): every output element is one scaled input element, so one gather through an index map.
    Overlapping taps: per tap, gather the source channels, scale, and add into its strided output positions."""
    src, scale, taps = sel
    kx, ky = weight.shape[2], weight.shape[3]
    full = [(x_spatial[d] - 1) * stride[d] + weight.shape[2 + d] + output_padding[d] for d in range(2)]
    out_spatial = [full[d] - 2 * padding[d] for d in range(2)]
    cout = weight.shape[1]

    if kx <= stride[0] and ky <= stride[1]:
        def select(x):
            index, scale_map = _selection_map(sel, x_spatial, stride, padding, output_padding, out_spatial, x.device)
            out = x.reshape(x.shape[0], -1).index_select(1, index).view(x.shape[0], cout, *out_spatial)
            return out.mul_(scale_map)

        return select

    def select(x):
        h, w = x_spatial
        out = x.new_zeros(x.shape[0], cout, *full)
        for a, b in taps:
            view = out[:, :, a:a + stride[0] * (h - 1) + 1:stride[0], b:b + stride[1] * (w - 1) + 1:stride[1]]
            view.add_(x.index_select(1, src[a, b]) * scale[a, b].view(1, -1, 1, 1))
        return out[:, :, padding[0]:padding[0] + out_spatial[0], padding[1]:padding[1] + out_spatial[1]]

    return select


#FIX FOR: the "forward" lowering flipped its weight on the GPU at every call; one flip per weight per run is enough.
_flipped_weights = WeakIdKeyDictionary()


def _forward_conv(x, weight, padding):
    """Stride-1 transposed conv as the forward conv with the kernel flipped and in/out channels swapped."""
    hit = _flipped_weights.get(weight)
    if hit is None or hit[0] != weight._version:
        hit = _flipped_weights[weight] = (weight._version, weight.transpose(0, 1).flip(2, 3).contiguous())
    return F.conv2d(x, hit[1], padding=(weight.shape[2] - 1 - padding[0], weight.shape[3] - 1 - padding[1]))


def _conv_transpose_lowerings(weight, x_spatial, stride, padding, output_padding, sel=None):
    def plain(x):
        return F.conv_transpose2d(x, weight, stride=stride, padding=padding,
                                  output_padding=output_padding)

    lowerings = {"chunked": lambda x: _split_batch(plain, x), "plain": plain}
    #FEAT: selection-weight lowering, chosen by the per-shape timing like the others.
    if sel is not None:
        lowerings["select"] = _selection_lowering(sel, weight, x_spatial, stride, padding, output_padding)
    kx, ky = weight.shape[2], weight.shape[3]
    if (stride == [1, 1] and output_padding == [0, 0]
            and padding[0] < kx and padding[1] < ky):
        # Stride 1: the forward conv with the kernel flipped and in/out channels swapped.
        def forward(x):
            return _forward_conv(x, weight, padding)

        lowerings["forward"] = forward
        lowerings["forward_chunked"] = lambda x: _split_batch(forward, x)
        #FEAT: forward_native lowering on the CPU (oneDNN off).
        if not weight.is_cuda:
            def forward_native(x):
                enabled = torch._C._get_mkldnn_enabled()
                torch._C._set_mkldnn_enabled(False)
                try:
                    return forward(x)
                finally:
                    torch._C._set_mkldnn_enabled(enabled)

            lowerings["forward_native"] = forward_native
        for grid in PACK_GRIDS:
            if grid * (x_spatial[0] + kx - 1) <= PACK_MAX_SIDE and grid * (x_spatial[1] + ky - 1) <= PACK_MAX_SIDE:
                lowerings[f"packed{grid}"] = _packed_lowering(weight, x_spatial, padding, grid)
    if stride[0] == stride[1] > 1:
        subpixel = _subpixel_lowering(weight, x_spatial, stride[0], padding, output_padding)
        lowerings["subpixel"] = subpixel
        lowerings["subpixel_chunked"] = lambda x: _split_batch(subpixel, x)
        #FEAT: subpixel_cl lowering on the CPU (channels-last input).
        if not weight.is_cuda:
            lowerings["subpixel_cl"] = lambda x: subpixel(x.contiguous(memory_format=torch.channels_last))
        if kx % stride[0] or ky % stride[1]:
            lowerings["phases"] = _phase_lowering(weight, x_spatial, stride[0], padding, output_padding)
    in_size = weight.shape[0] * x_spatial[0] * x_spatial[1]
    out_shape = [weight.shape[1]] + [(x_spatial[d] - 1) * stride[d] - 2 * padding[d]
                                     + weight.shape[2 + d] + output_padding[d] for d in range(2)]
    out_size = out_shape[0] * out_shape[1] * out_shape[2]
    conv_macs = in_size * weight.shape[1] * kx * ky
    #FEAT: larger Toeplitz limits on the CPU.
    max_elements, max_waste = ((TOEPLITZ_MAX_ELEMENTS, TOEPLITZ_MAX_WASTE) if weight.is_cuda
                               else (TOEPLITZ_MAX_ELEMENTS_CPU, TOEPLITZ_MAX_WASTE_CPU))
    if in_size * out_size <= max_elements and in_size * out_size <= max_waste * conv_macs:
        # Small rows (patches): one GEMM against the conv's dense matrix, built by pushing
        # the identity through the conv. More FLOPs, but cuDNN runs these shapes far below
        # GEMM throughput.
        def toeplitz(x):
            eye = torch.eye(in_size, device=x.device, dtype=x.dtype).reshape(in_size, -1, *x_spatial)
            matrix = plain(eye).reshape(in_size, out_size)
            return (x.reshape(x.shape[0], in_size) @ matrix).reshape(x.shape[0], *out_shape)

        lowerings["toeplitz"] = toeplitz
    return lowerings


def _time_lowering(fn, x, iters=3):
    fn(x)
    #FEAT: time lowerings on the CPU.
    if not x.is_cuda:
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - start)
        return sorted(times)[len(times) // 2]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def _choose_lowering(key, x, lowerings):
    choice = _conv_transpose_choice.get(key)
    if choice is not None:
        return choice
    choice = "chunked"
    #FEAT: conv-transpose tuning on the CPU.
    if len(lowerings) > 1:
        times = {}
        for name, fn in lowerings.items():
            try:
                times[name] = _time_lowering(fn, x)
            except torch.cuda.OutOfMemoryError:
                pass
            except RuntimeError as e:
                if x.is_cuda or "allocate" not in str(e):
                    raise
        if "chunked" in times:
            best = min(times, key=times.get)
            if times[best] < CONV_TRANSPOSE_MIN_GAIN * times["chunked"]:
                choice = best
    _conv_transpose_choice[key] = choice
    return choice


@torch.library.custom_op("cf::conv_transpose2d", mutates_args=())
def _conv_transpose2d_op(x: torch.Tensor, weight: torch.Tensor, stride: list[int],
                         padding: list[int], output_padding: list[int], may_select: bool,
                         selection_taps: Optional[list[int]] = None) -> torch.Tensor:
    #FEAT: selection weights get their own tuning key (a dense weight of the same shape must not reuse it).
    #FIX FOR: the selection check read the weight back to the host (one D2H per conv weight per batch); the
    # compiler passes may_select=False for weights it already knows are dense.
    sel = _channel_selection(weight, selection_taps) if may_select else None
    key = (tuple(x.shape), tuple(x.stride()), tuple(weight.shape), x.dtype, x.device,
           tuple(stride), tuple(padding), tuple(output_padding), sel is not None)
    #FIX FOR: every call built all candidate lowerings in Python though the tuner needs them only once (N33: ~330 calls per image).
    choice = _conv_transpose_choice.get(key)
    if choice == "plain":
        return F.conv_transpose2d(x, weight, stride=stride, padding=padding, output_padding=output_padding).contiguous()
    if choice == "chunked":
        return _split_batch(lambda part: F.conv_transpose2d(part, weight, stride=stride, padding=padding,
                                                            output_padding=output_padding), x).contiguous()
    if choice == "forward":
        return _forward_conv(x, weight, padding).contiguous()
    if choice == "forward_chunked":
        return _split_batch(lambda part: _forward_conv(part, weight, padding), x).contiguous()
    lowerings = _conv_transpose_lowerings(weight, list(x.shape[2:]), stride, padding, output_padding, sel)
    # The fake kernel promises a contiguous result.
    return lowerings[_choose_lowering(key, x, lowerings)](x).contiguous()


@_conv_transpose2d_op.register_fake
def _(x, weight, stride, padding, output_padding, may_select, selection_taps=None):
    size = [x.shape[0], weight.shape[1]]
    for d in range(2):
        size.append((x.shape[2 + d] - 1) * stride[d] - 2 * padding[d]
                    + weight.shape[2 + d] + output_padding[d])
    return x.new_empty(size)


#FIX FOR: cf::conv_transpose2d had no autograd formula, so conv-net bounds could not be differentiated.
def _conv_transpose2d_setup_context(ctx, inputs, output):
    x, weight, stride, padding, output_padding, _may_select, _selection_taps = inputs
    ctx.save_for_backward(x, weight)
    ctx.conv_args = (stride, padding, output_padding)


def _conv_transpose2d_backward(ctx, grad):
    x, weight = ctx.saved_tensors
    stride, padding, output_padding = ctx.conv_args
    #FIX FOR: cuDNN's transposed-conv weight gradient is several times slower than a patch GEMM on
    # back-substitution shapes (huge batch, few channels); the backward is tuned per shape like the forward.
    grad_x, grad_weight = _conv_transpose2d_backward_op(
        grad.contiguous(), x, weight, stride, padding, output_padding,
        [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
    return (grad_x if ctx.needs_input_grad[0] else None), (grad_weight if ctx.needs_input_grad[1] else None), None, None, None, None, None


_conv_transpose_bwd_choice = {}


def _conv_transpose_bwd_lowerings(x, weight, stride, padding, output_padding, mask):
    def cudnn(grad):
        gx, gw, _ = torch.ops.aten.convolution_backward(
            grad, x, weight, None, stride, padding, [1, 1], True, output_padding, 1, mask + [False])
        return gx, gw

    def patch_gemm(grad):
        # Rows = (sample, x position), columns = the (out channel, kernel tap) patch of grad it touches.
        n, ci, hx, wx = x.shape
        kx, ky = weight.shape[2:]
        padded = F.pad(grad, (padding[1], padding[1], padding[0], padding[0])) if any(padding) else grad
        patches = padded.unfold(2, kx, stride[0]).unfold(3, ky, stride[1])[:, :, :hx, :wx]
        rows = patches.permute(0, 2, 3, 1, 4, 5).reshape(n * hx * wx, -1)
        gx = gw = None
        if mask[0]:
            gx = (rows @ weight.reshape(ci, -1).T).reshape(n, hx * wx, ci).transpose(1, 2).reshape(x.shape)
        if mask[1]:
            gw = (x.reshape(n, ci, -1).transpose(1, 2).reshape(-1, ci).T @ rows).reshape(weight.shape)
        return gx, gw

    return {"cudnn": cudnn, "patch_gemm": patch_gemm}


@torch.library.custom_op("cf::conv_transpose2d_backward", mutates_args=())
def _conv_transpose2d_backward_op(grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, stride: list[int],
                                  padding: list[int], output_padding: list[int], mask: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
    lowerings = _conv_transpose_bwd_lowerings(x, weight, stride, padding, output_padding, mask)
    key = (tuple(grad.shape), tuple(x.shape), tuple(weight.shape), x.dtype, x.device,
           tuple(stride), tuple(padding), tuple(output_padding), tuple(mask))
    choice = _conv_transpose_bwd_choice.get(key)
    if choice is None:
        choice = "cudnn"
        if x.is_cuda:
            times = {}
            for name, fn in lowerings.items():
                try:
                    times[name] = _time_lowering(fn, grad)
                except torch.cuda.OutOfMemoryError:
                    pass
            best = min(times, key=times.get, default="cudnn")
            if best != "cudnn" and times[best] < CONV_TRANSPOSE_MIN_GAIN * times.get("cudnn", float("inf")):
                choice = best
        _conv_transpose_bwd_choice[key] = choice
    gx, gw = lowerings[choice](grad)
    # custom ops cannot return None: unused gradients come back as empty placeholders.
    return (gx.contiguous() if gx is not None else x.new_empty(0)), (gw.contiguous() if gw is not None else weight.new_empty(0))


@_conv_transpose2d_backward_op.register_fake
def _(grad, x, weight, stride, padding, output_padding, mask):
    return (x.new_empty(x.shape) if mask[0] else x.new_empty(0)), (weight.new_empty(weight.shape) if mask[1] else weight.new_empty(0))


_conv_transpose2d_op.register_autograd(_conv_transpose2d_backward,
                                       setup_context=_conv_transpose2d_setup_context)


def conv_transpose2d(x, weight, stride=1, padding=0, output_padding=0, may_select=True, selection_taps=None):
    return _conv_transpose2d_op(x, weight, _pair(stride), _pair(padding), _pair(output_padding), may_select,
                                selection_taps)


#FIX FOR: Inductor merged F.pad with the computation feeding it and returned wrong coefficients (ResNet50 row chunks).
@torch.library.custom_op("cf::realized_pad", mutates_args=())
def realized_pad(x: torch.Tensor, pad: list[int]) -> torch.Tensor:
    return F.pad(x, pad).contiguous()


@realized_pad.register_fake
def _(x, pad):
    size = list(x.shape)
    for d in range(len(pad) // 2):
        size[-1 - d] += pad[2 * d] + pad[2 * d + 1]
    return x.new_empty(size)


def patches_to_dense(pieces, batch, oc, ox, oy, ic, kx, ky, ix, iy, px, py, sx, sy):
    """Gather only valid image coefficients, without a padded scatter buffer.

    Every dense matrix row is one patch, so there are no overlapping writes or
    reductions. Coordinates depend only on geometry and fuse into the gather
    under Inductor. Independent x/y strides support rectangular feature maps.
    """
    rows = ox * oy
    x = torch.arange(ix, device=pieces.device)[None, :] + px - (
        torch.arange(ox, device=pieces.device)[:, None] * sx)
    y = torch.arange(iy, device=pieces.device)[None, :] + py - (
        torch.arange(oy, device=pieces.device)[:, None] * sy)
    valid = ((x >= 0) & (x < kx))[:, None, :, None] & (
        (y >= 0) & (y < ky))[None, :, None, :]
    index = (x.clamp(0, kx - 1)[:, None, :, None] * ky
             + y.clamp(0, ky - 1)[None, :, None, :])
    source = pieces.reshape(-1, oc, rows, ic, kx * ky).expand(batch, oc, rows, ic, kx * ky)
    index = index.reshape(1, 1, rows, 1, ix * iy).expand(batch, oc, rows, ic, ix * iy)
    result = source.gather(-1, index)
    return torch.where(valid.reshape(1, 1, rows, 1, ix * iy), result, 0).reshape(
        batch, oc * rows, ic * ix * iy)


def col2im_columns(out_x, out_y, ker_x, ker_y, padded_rows, padded_cols, stride,
                   device=None):
    """Destination column of each (output position, kernel tap) in the flattened
    padded input -- pure geometry, so the compiled graph folds it to a constant."""
    flat = torch.arange(out_x * out_y, device=device)
    row = torch.div(flat, out_y, rounding_mode="trunc") * padded_rows * stride
    col = torch.fmod(flat, out_y) * stride
    return ((row + col).view(-1, 1, 1)
            + (torch.arange(ker_x, device=device) * padded_cols).view(1, -1, 1)
            + torch.arange(ker_y, device=device).view(1, 1, -1)
            ).reshape(out_x * out_y, ker_x * ker_y)


def _expanded_matvec(a, b):
    # A batch-expanded matrix times per-sample vectors is one GEMM, not B GEMVs over the same matrix.
    if (a.dim() == 3 and b.dim() == 3 and b.shape[-1] == 1 and a.shape[0] > 1
            and a.stride(0) == 0 and b.shape[0] == a.shape[0]):
        return torch.matmul(b.squeeze(-1), a[0].transpose(0, 1)).unsqueeze(-1)
    return torch.matmul(a, b)


def bcast_matmul(a, b):
    #FEAT: a single-sample matrix-vector product under torch.compile is a multiply-sum, so an elementwise-derived
    # matrix (|W|, clamp(W)) fuses into the sum instead of being written out for cuBLAS (N33: 3 x 62.5 MiB).
    if (torch.compiler.is_compiling() and a.dim() >= 2 and b.dim() >= 2
            and math.prod(a.shape[:-2]) == 1 and math.prod(b.shape[:-2]) == 1):
        if b.shape[-1] == 1:
            return (a * b.transpose(-1, -2)).sum(-1, keepdim=True)
        if a.shape[-2] == 1:
            return (a.transpose(-1, -2) * b).sum(-2, keepdim=True)
    if a.dim() == 3 and b.dim() == 3 and a.shape[0] > 1 and b.shape[0] == a.shape[0] and a.stride(0) != 0:
        #FEAT: per-sample matrices times one batch-expanded matrix or vector: one tall GEMM over the stacked rows.
        #FEAT: a tiny inner dimension (Inductor unrolls sums up to 8 terms) is a multiply-sum that fuses with its
        # neighbours; a memory-bound tall GEMM writes and rereads every product (lsnc_relu: K <= 8).
        if b.stride(0) == 0 and a.shape[-1] <= 8:
            return (a.unsqueeze(-1) * b[:1]).sum(-2)
        if b.stride(0) == 0:
            return torch.matmul(a.reshape(-1, a.shape[-1]), b[0]).view(a.shape[0], a.shape[1], b.shape[-1])
        #FEAT: per-sample matrices times per-sample vectors: a fusable multiply-sum, not B tiny GEMVs.
        if b.shape[-1] == 1:
            return (a * b.transpose(1, 2)).sum(-1, keepdim=True)
    return _expanded_matvec(a, b)


#FIX FOR: in training, per-sample matrices times a batch-expanded one ran as a bmm whose backward built a
# per-sample [B, K, N] gradient of the shared matrix; one GEMM over B*M rows gives a single [K, N] gradient GEMM.
def training_bcast_matmul(a, b):
    #FEAT: training builds run without gradients too (SpecBounds verification); then no backward to tune for.
    if not torch.is_grad_enabled():
        return bcast_matmul(a, b)
    # Matrices only: for a vector b the per-sample gradient is small, and reshaping a non-contiguous a would copy it.
    if a.dim() == 3 and b.dim() == 3 and b.shape[0] > 1 and b.shape[-1] > 1 and b.stride(0) == 0 and a.stride(0) != 0:
        return torch.matmul(a.reshape(-1, a.shape[-1]), b[0]).reshape(a.shape[0], a.shape[1], b.shape[-1])
    # Not bcast_matmul: its verification rewrites (vector GEMMs, multiply-sums) are not tuned for backward.
    return _expanded_matvec(a, b)


#FIX FOR: in training, composing a kernel with patches ('pckabo,mkab->pcmabo') ran as one bmm per kernel tap whose
# weight gradient reduces ~B*patch*positions rows into a tiny matrix (one cuBLAS call took ~90 ms on zid N26);
# expanding the kernel over the batch splits that reduction per sample (split-K), and is more accurate.
def training_einsum(equation, a, b):
    if equation == 'pckabo,mkab->pcmabo' and a.shape[0] > 1:
        return _KernelPatchCompose.apply(a, b)
    return torch.einsum(equation, a, b)


#FIX FOR: the per-sample split's kernel gradient was B*taps tiny GEMMs ([16,1728]@[1728,32] each, ~20 ms on zono N26);
# one GEMM per kernel tap over all (sample, patch, position) rows lets cuBLAS split K itself (~4.5 ms, same fp64 error).
class _KernelPatchCompose(torch.autograd.Function):
    @staticmethod
    def forward(a, b):
        return torch.einsum('pckabo,pmkab->pcmabo', a, b.expand(a.shape[0], *b.shape))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        p, c, k, kx, ky, o = a.shape
        m = b.shape[0]
        # Rows (sample, patch, position) per kernel tap, shared by both gradients.
        rows_g = grad.permute(3, 4, 0, 1, 5, 2).reshape(kx * ky, -1, m)
        grad_a = grad_b = None
        if ctx.needs_input_grad[1]:
            rows_a = a.permute(3, 4, 0, 1, 5, 2).reshape(kx * ky, -1, k)
            grad_b = torch.stack([rows_g[t].T @ rows_a[t] for t in range(kx * ky)]).view(kx, ky, m, k).permute(2, 3, 0, 1)
        if ctx.needs_input_grad[0]:
            kernel = b.permute(2, 3, 0, 1).reshape(kx * ky, m, k)
            grad_a = torch.bmm(rows_g, kernel).view(kx, ky, p, c, o, k).permute(2, 3, 5, 0, 1, 4)
        return grad_a, grad_b


#FIX FOR: torch.where evaluates both arms, so a division in the unselected arm (ReLU slope u/(u-l) with
# u-l == 0 or tiny) gave NaN gradients (0/0, 0 * inf). Gradients are exactly 0 where the incoming one is.
class _SafeDiv(torch.autograd.Function):
    @staticmethod
    def forward(num, den):
        return num / torch.where(den == 0, torch.ones_like(den), den)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        num, den = ctx.saved_tensors
        den = torch.where(den == 0, torch.ones_like(den), den)
        live = grad != 0
        grad_over_den = torch.where(live, grad / den, torch.zeros_like(grad))
        grad_num = grad_over_den if ctx.needs_input_grad[0] else None
        grad_den = None
        if ctx.needs_input_grad[1]:
            # -grad * num / den**2, as two divisions so den**2 cannot underflow on its own.
            grad_den = torch.where(live, -grad_over_den * (num / den), torch.zeros_like(grad_over_den))
        if grad_num is not None and grad_num.shape != num.shape:
            grad_num = grad_num.sum_to_size(num.shape)
        if grad_den is not None and grad_den.shape != den.shape:
            grad_den = grad_den.sum_to_size(den.shape)
        return grad_num, grad_den


def safe_div(num, den):
    if not isinstance(den, torch.Tensor):
        return operator.truediv(num, den)
    if not isinstance(num, torch.Tensor):
        num = torch.as_tensor(num, dtype=den.dtype, device=den.device)
    return _SafeDiv.apply(num, den)


# Called by name from generated flow code; codeGen imports and flow_shapes evaluates them.
RUNTIME_HELPERS = {f.__name__: f for f in (s_shaped.sigma_tangent, s_shaped.tanh_tangent, col2im_columns, patches_to_dense, conv_transpose2d, bcast_matmul, training_bcast_matmul, training_einsum, safe_div)}
# Imported only by --training builds, so verification code is unchanged.
TRAINING_HELPERS = {'training_bcast_matmul', 'training_einsum'}


def identifySparseBlockType(block):
    if isinstance(block, torch.Tensor) and block.dtype == torch.bool:
        return "BoolTensorSparse"
    elif isinstance(block, bool):
        return "BoolScalarSparse"
    elif isinstance(block, torch.Tensor):
        return "FloatTensorSparse"
    else:
        return "ScalarSparse"
class DenseBlock(SparseBlock):
    def __init__(self, block, og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        if block_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_list_index = len(json_list) - 1
        else:
            block_json_list_index = block_index

        total_shape = torch.tensor(block.shape)
        super().__init__(block, total_shape, 'D')
        self.batch_size = total_shape[0]

        SparseBlockType = identifySparseBlockType(block)

        json_obj = {
            "method": "DenseBlockConstructor",
            "block": "json_list_" + str(block_json_list_index),
            "total_shape": total_shape.tolist(),
            "batch_size": int(self.batch_size),
            "SparseBlockType":  SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if dummy_mode and self.json_list is not None and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_DenseBlock/DenseBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None
            
    @classmethod
    def DenseBlock_JIT(cls, block, og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            raise AssertionError("og_json_list cannot be None when using DenseBlock_JIT")



        obj = cls(block, og_json_list, block_index, layer_index, counter, inside_while, while_number, while_iteration)
        return obj, obj.json_index
        

                

        

    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({"method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list)
        })
        if simulacrum:
            return self.block, len(json_list) - 1
        return self.block
    
    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block_index = len(json_list) - 1

        res = RepeatBlock(self.block, self.total_shape*repeat_dims, json_list, block_index)
        if simulacrum:
            return res, res.json_index
        return res
        start_time = time.time()
        expand_dims = torch.tensor(self.block.shape) * repeat_dims
        new_block = self.block.expand(*expand_dims)
        repeat_time.update_op_time(time.time() - start_time)
        res = DenseBlock(new_block)
        # res = DenseBlock(self.block.repeat(*repeat_dims))
        return res
    
    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        if not inductor_mode.get_flag():
            start_time = time.time()
        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1

        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        res = self.block.unsqueeze(index)

        res_block = DenseBlock(res, json_list, res_index)

        if simulacrum:
            return res_block, res_block.json_index
        if not inductor_mode.get_flag():
            end_time = time.time()
            unsqueeze_time.update_op_time(end_time-start_time)
        return res_block
    
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.time()


        json_obj = {
            "method": "extract_sparse_block",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_squeeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        new_block = self.block.squeeze(index)

        res = DenseBlock(new_block, json_list, res_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        end_time = time.time()
        squeeze_time.update_op_time(end_time - start_time)
        return res
    
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block
            
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = sp_block.block

            
            json_obj = {
                    "method": "torch_matmul",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a @ b


            res = DenseBlock(c, json_list, len(json_list) - 1)
            res_index = res.json_index
        elif isinstance(sp_block, DiagonalBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            lhs = self.block

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            rhs = sp_block.block
            
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(rhs_index),
                "index": sp_block.diag_index-1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            rhs = rhs.unsqueeze(sp_block.diag_index-1)
            
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = lhs * rhs

            res = DenseBlock(res, json_list, len(json_list) - 1)
            res_index = res.json_index

        elif isinstance(sp_block, KernelBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            kernel = sp_block.block

            kx = sp_block.kx
            ky = sp_block.ky
            sx = sp_block.sx
            sy = sp_block.sy
            px = sp_block.px
            py = sp_block.py
            ox = sp_block.ox
            oy = sp_block.oy
            ix = sp_block.ix
            iy = sp_block.iy
            num_kernels = sp_block.num_kernels
            batch_size = int(self.batch_size)
            curr_size = self.block.shape[-2]
            new_px = (ix + 2*px - kx) % sx
            new_py = (iy + 2*py - ky) % sy

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(lhs_index),
                "shape": (batch_size*curr_size, num_kernels, ox, oy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            input_tensor = self.block.reshape(batch_size*curr_size, num_kernels, ox, oy)

            json_obj = {
                "method": "F.conv_transpose2d",
                "input": "json_list_" + str(lhs_index),
                "weight": "json_list_" + str(rhs_index),
                "stride": (sx, sy),
                "padding": (px, py),
                "output_padding": (new_px, new_py),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            output_tensor = conv_transpose2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py))

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size, curr_size, -1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = DenseBlock(output_tensor.reshape(batch_size, curr_size, -1), json_list, res_index)
            res_index = res.json_index
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]

                res = ConstBlock(0, new_total_shape, json_list)
                res_index = res.json_index
                
            else:
                raise NotImplementedError
        elif isinstance(sp_block, PatchesBlock):
            # DenseBlock @ PatchesBlock, ported from baseline: contract the dense
            # rows with each position's patch, then fold back into input space.
            # Avoids densifying sp_block (the old fallback's get_dense() could
            # allocate a single buffer in the hundreds of MB to a few GB).
            P = sp_block
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c_index = len(json_list) - 1
            C = self.block                       # (B, M, K1*Pox*Poy)
            B = C.shape[0]
            M = C.shape[1]
            K1 = P.num_kernels
            Pox, Poy = P.ox, P.oy
            c0 = P.num_channels
            Pkx, Pky = P.kx, P.ky
            S = Pox * Poy
            pw = c0 * Pkx * Pky

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(c_index),
                "shape": (B, M, K1, S),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            cr_index = len(json_list) - 1
            Cr = C.reshape(B, M, K1, S)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(cr_index),
                "permutation": (0, 3, 1, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            cr_index = len(json_list) - 1
            Cr = Cr.permute(0, 3, 1, 2)

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(cr_index),
                "shape": (B * S, M, K1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            cr_index = len(json_list) - 1
            Cr = Cr.reshape(B * S, M, K1)

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": P.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pb_index = len(json_list) - 1
            Pb = P.block
            if Pb.shape[0] != B:
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(pb_index),
                    "shape": (B,) + tuple(Pb.shape[1:]),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                pb_index = len(json_list) - 1
                Pb = Pb.expand(B, *Pb.shape[1:])

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(pb_index),
                "shape": (B, K1, S, pw),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pr_index = len(json_list) - 1
            Pr = Pb.reshape(B, K1, S, pw)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(pr_index),
                "permutation": (0, 2, 1, 3),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pr_index = len(json_list) - 1
            Pr = Pr.permute(0, 2, 1, 3)

            # The per-position products hold M rows for every patch entry before
            # the fold sums them; compose origin-channel slices to bound them.
            step = channel_step(c0, 4 * int(B) * S * max(M, K1) * Pkx * Pky)
            pieces, piece_indices = [], []
            for ca in range(0, c0, step):
                cch = min(step, c0 - ca)
                pch = cch * Pkx * Pky
                prc, prc_index = Pr, pr_index
                if cch != c0:
                    json_obj = {
                        "method": "torch_slice",
                        "input": "json_list_" + str(pr_index),
                        "index": [0, 0, 0, [ca * Pkx * Pky, (ca + cch) * Pkx * Pky]],
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                    prc_index = len(json_list) - 1
                    prc = Pr[:, :, :, ca * Pkx * Pky:(ca + cch) * Pkx * Pky]

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(prc_index),
                    "shape": (B * S, K1, pch),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                prc_index = len(json_list) - 1
                prc = prc.reshape(B * S, K1, pch)

                json_obj = {
                    "method": "torch_matmul",
                    "lhs": "json_list_" + str(cr_index),
                    "rhs": "json_list_" + str(prc_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                t_index = len(json_list) - 1
                T = torch.matmul(Cr, prc)            # (B*S, M, pch), == torch.bmm(Cr, prc)

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(t_index),
                    "shape": (B, S, M, pch),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                t_index = len(json_list) - 1
                T = T.reshape(B, S, M, pch)

                json_obj = {
                    "method": "torch_permute",
                    "input": "json_list_" + str(t_index),
                    "permutation": (0, 2, 3, 1),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                t_index = len(json_list) - 1
                T = T.permute(0, 2, 3, 1)

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(t_index),
                    "shape": (B * M, pch, S),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                t_index = len(json_list) - 1
                T = T.reshape(B * M, pch, S)

                json_obj = {
                    "method": "F.fold",
                    "input": "json_list_" + str(t_index),
                    "output_size": (P.ix, P.iy),
                    "kernel_size": (Pkx, Pky),
                    "stride": (P.sx, P.sy),
                    "padding": (P.px, P.py),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                fold_index = len(json_list) - 1
                out = F.fold(T, output_size=(P.ix, P.iy), kernel_size=(Pkx, Pky),
                             stride=(P.sx, P.sy), padding=(P.px, P.py))

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(fold_index),
                    "shape": (B, M, cch * P.ix * P.iy),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                piece_indices.append(len(json_list) - 1)
                pieces.append(out.reshape(B, M, cch * P.ix * P.iy))

            out, out_index = cat_columns(pieces, piece_indices, json_list)
            res = DenseBlock(out, json_list, out_index)
            res_index = res.json_index
        elif isinstance(sp_block, RepeatBlock) and sp_block.only_one_repeat and sp_block.repeat_dims[-1] > 1:
            # New case -- og_cfg has a comment here noting this exact idea but
            # never implemented it, and falls back to full materialization too.
            # sp_block only broadcasts along the trailing (N) dim, and matmul is
            # linear there, so matmul-ing the single compact column and
            # re-attaching the N-wide broadcast to the output equals matmul-ing
            # the expanded version. Also avoids a footgun: torch.matmul needs a
            # contiguous rhs, so feeding it an expanded (non-contiguous) view
            # makes PyTorch silently clone() it to full size internally.
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = sp_block.block

            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a @ b
            c_index = len(json_list) - 1

            new_total_shape = torch.cat([self.total_shape[:-1], sp_block.total_shape[-1:]])
            res = RepeatBlock(c, new_total_shape, json_list, c_index)
            res_index = res.json_index
        else:
            start_op_time = time.perf_counter()
            block_2, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)


            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            lhs = self.block


            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = lhs @ block_2

            res = DenseBlock(c, json_list, len(json_list) - 1)
            res_index = res.json_index
        return res, res_index
        
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block 

    


            rhs_block, rhs_index = sp_block.unsqueeze(-1, json_list=json_list, template_index=rhs_index, simulacrum=True)

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = rhs_block.block
            

            json_obj = {
                "method": "torch_matmul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a @ b

            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": -1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = (c).squeeze(-1)


            res = DenseBlock(res, json_list, res_index)
            res_index = res.json_index
        elif isinstance(sp_block, DiagonalBlock):
            raise NotImplementedError
        elif isinstance(sp_block, KernelBlock):
            raise NotImplementedError
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]
                res = ConstBlock(0, new_total_shape, json_list)
                res_index = res.json_index
            else:
                raise NotImplementedError
        elif isinstance(sp_block, RepeatBlock):
            # warnings.warn(f'Matmul with unequal dims inefficient for {type(self)} and {type(sp_block)}')

            sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)

            sp_block = DenseBlock(sp_block, json_list, rhs_index)
            rhs_index = sp_block.json_index

            res, res_index = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            return res, res_index
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        return res, res_index
        
    
    def convert_to_patches(self, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        raise Exception(f'Not an efficient implementation')
        flag = self.block.dtype == torch.bool 
        if flag:
            block = self.block.float() # batch_size, curr_size, prev_size
        else:
            block = self.block # batch_size, curr_size, prev_size
        batch_size, curr_size, prev_size = block.shape
        block = block.reshape(block.size(0) * block.size(1), num_channels, ix, iy) # batch_size*curr_size, num_channels, ix, iy
        block = F.unfold(block, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy)) # batch_size*curr_size, num_channels * kx * ky, ox * oy
        block = block.reshape(batch_size, curr_size, *block.shape[-2:]) # batch_size, curr_size, num_channels * kx * ky, ox * oy
        block = block.repeat(1, 1, 1, num_kernels) # batch_size, curr_size, num_channels * kx * ky, num_kernels * ox * oy
        block = block.permute(0, 1, 3, 2) # batch_size, curr_size, curr_size, num_channels * kx * ky
        block = get_diagonal(block, 2) # batch_size, curr_size, num_channels * kx * ky   
        if flag:
            block = block.bool()
        end_time = time.time()
        mat_to_patches_time.update_total_time(end_time - start_time)
        return PatchesBlock(block, self.total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)
        
    
        
    def sum(self, dim, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_sum",
            "input": "json_list_" + str(template_index),
            "dim": dim,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        sum_index = len(json_list) - 1
        res = DenseBlock(self.block.sum(dim), json_list, sum_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    
    def size(self):
        return self.block.size()
        
    def overwrite_dense_block(self, sp_block, start_index, s):
        new_block = self.block 
        new_block[s] = sp_block.block
        return [DenseBlock(new_block)], [torch.tensor([0]*len(self.total_shape))]
        
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index, json_list=[], template_index=-1, simulacrum=False):
        local_start = start_index - block_start_index
        local_end = end_index - block_start_index
        s = get_slice(local_start, local_end)
        if simulacrum:
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            tensor_index = len(json_list) - 1
            json_list.append({
                "method": "torch_slice",
                "input": "json_list_" + str(tensor_index),
                "index": [[int(local_start[j]), int(local_end[j])] for j in range(local_start.shape[0])],
                "output": len(json_list),
            })
            slice_index = len(json_list) - 1
            res = DenseBlock(self.block[tuple(s)], json_list, slice_index)
            return res, res.json_index
        return DenseBlock(self.block[tuple(s)])
    
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        res = DenseBlock(block, json_list, template_index)
        if simulacrum:
            return res, res.json_index
        return res

class KernelBlock(SparseBlock):
    def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, KernelBlock.__init__, block, total_shape, ix, iy, ox, oy, sx, sy, px, py)
        return super().__new__(cls)

    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        if block_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_list_index = len(json_list) - 1
        else:
            block_json_list_index = block_index


        super().__init__(block, total_shape, 'K')
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        self.kx = block.shape[-2]
        self.ky = block.shape[-1]
        self.num_channels = block.shape[1]
        self.num_kernels = block.shape[0]

        SparseBlockType = identifySparseBlockType(block)
        json_obj = {
            "method": "KernelBlockConstructor",
            "block": "json_list_" + str(block_json_list_index),
            "total_shape": total_shape.tolist() if isinstance(total_shape, torch.Tensor) else list(total_shape),
            "ix": self.ix,
            "iy": self.iy,
            "ox": self.ox,
            "oy": self.oy,
            "sx": self.sx,
            "sy": self.sy,
            "px": self.px,
            "py": self.py,
            "kx": int(self.kx),
            "ky": int(self.ky),
            "num_channels": int(self.num_channels),
            "num_kernels": int(self.num_kernels),
            "SparseBlockType":  SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_KernelBlock/KernelBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None

    def __str__(self):
        res = f'KernelBlock: \n \
            Total Shape: {self.total_shape} \n \
            Block Shape: {self.block.shape} \n \
            Num Kernels: {self.num_kernels} \n \
            Num Channels: {self.num_channels} \n \
            Kernel Shape: {self.block.shape[-2:]} \n \
            Stride: {self.sx, self.sy} \n \
            Padding: {self.px, self.py} \n \
            Input Size: {self.ix, self.iy} \n \
            Output Size: {self.ox, self.oy} \n'
        return res
    
    def parameters(self):
        return [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]

    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_float",
            "input": "json_list_" + str(len(json_list) - 1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        kernel_index = len(json_list) - 1
        kernel = self.block.float()

        px = self.px
        py = self.py
        sx = self.sx
        sy = self.sy
        ix = self.ix
        iy = self.iy
        kx = self.kx
        ky = self.ky
        ox = self.ox
        oy = self.oy
        num_kernels = self.num_kernels
        new_px = (ix + 2*px - kx) % sx
        new_py = (iy + 2*py - ky) % sy
        curr_size = num_kernels*ox*oy

        eye = torch.eye(num_kernels*ox*oy).unsqueeze(0).reshape(curr_size, num_kernels, ox, oy)
        json_obj = {
            "method": "torch_eye",
            "size": num_kernels*ox*oy,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(len(json_list) - 1),
            "index": 0,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (curr_size, num_kernels, ox, oy),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "F.conv_transpose2d",
            "input": "json_list_" + str(len(json_list) - 1),
            "weight": "json_list_" + str(kernel_index),
            "stride": (sx, sy),
            "padding": (px, py),
            "output_padding": (new_px, new_py),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (1, curr_size, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        res = F.conv_transpose2d(eye, kernel, stride=(sx, sy), padding=(px, py), output_padding=(new_px, new_py)).reshape(1, curr_size, -1)
        if simulacrum:
            return res, res_index
        return res
    
    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        new_total_shape = self.total_shape*repeat_dims
        res = KernelBlock(self.block, new_total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, json_list, len(json_list) - 1)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res

    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):

        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        new_total_shape = torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]])
        res = KernelBlock(self.block, new_total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, json_list, len(json_list) - 1)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        new_total_shape = torch.concat([self.total_shape[:index], self.total_shape[index+1:]])
        res = KernelBlock(self.block, new_total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, json_list, len(json_list) - 1)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    

    
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DiagonalBlock):
            # AUTO-LIRPA
            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            kernel_index = len(json_list) - 1

            kernel = self.block
            px = self.px
            py = self.py
            sx = self.sx
            sy = self.sy
            ix = self.ix
            iy = self.iy
            ox = self.ox
            oy = self.oy
            kx = self.kx
            ky = self.ky
            num_kernels = self.num_kernels
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            
            start_op_time = time.perf_counter()

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            x = sp_block.block.view(batch_size, num_channels, ix, iy)

            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(rhs_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy)) # batch_size, num_channels * kx * ky, ox * oy
            
            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(len(json_list) - 1),
                "permutation": (0, 2, 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(len(json_list) - 1),
                "repeats": (1, num_kernels, 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            x_unf_index = len(json_list) - 1
            x_unf = x_unf.permute(0,2,1).repeat(1, num_kernels, 1)
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            

            k_new, k_new_index = self.convert_to_patches(json_list=json_list, template_index=lhs_index, simulacrum=True)

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(k_new_index),
                "block_type": k_new.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            k_new_index = len(json_list) - 1
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(x_unf_index),
                "rhs": "json_list_" + str(k_new_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            patches_index = len(json_list) - 1
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            k_new = k_new.block
            patches = x_unf * k_new
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, patches_index)
            res_index = res.json_index
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()
                new_total_shape[-1] = sp_block.total_shape[-1]
                res = ConstBlock(0, new_total_shape, json_list)
                res_index = res.json_index
            else:
                raise NotImplementedError
        elif isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_block_index = len(json_list) - 1
            b = sp_block.block # batch_size, prev_size, sym_size

            batch_size = b.shape[0]
            sym_size = b.shape[2]
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            start_op_time = time.perf_counter()
            json_obj = {
                "method": "torch_transpose",
                "input": "json_list_" + str(rhs_block_index),
                "dims": (1, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            b = b.transpose(1,2) # batch_size, sym_size, prev_size

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size*sym_size, self.num_channels, self.ix, self.iy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            b_index = len(json_list) - 1
            b = b.reshape(b.shape[0]*b.shape[1], self.num_channels, self.ix, self.iy)
            
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            kernel_index = len(json_list) - 1
            json_obj = {
                "method": "F.conv2d",
                "input": "json_list_" + str(b_index),
                "weight": "json_list_" + str(kernel_index),
                "stride": (self.sx, self.sy),
                "padding": (self.px, self.py),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = F.conv2d(b, self.block, stride=(self.sx, self.sy), padding=(self.px, self.py)) # batch_size*sym_size, num_kernels, ox, oy
            
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size, sym_size, -1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = res.reshape(batch_size, sym_size, -1)

            json_obj = {
                "method": "torch_transpose",
                "input": "json_list_" + str(len(json_list) - 1),
                "dims": (1, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = res.transpose(1,2) # batch_size, curr_size, sym_size
            
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(res, json_list, len(json_list) - 1)
            res_index = res.json_index
        elif isinstance(sp_block, RepeatBlock):
            # New case -- og_cfg has no RepeatBlock handling here and always
            # densifies first.
            if sp_block.only_one_repeat and sp_block.repeat_dims[-1] > 1:
                # Only the sym (last) dim is broadcast, and conv2d has no
                # cross-image mixing, so convolving the single compact column
                # once and re-attaching the broadcast to the output equals
                # convolving every duplicate column separately, but far cheaper.
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_block_index = len(json_list) - 1
                b = sp_block.block # batch_size, prev_size, 1

                batch_size = b.shape[0]
                sym_size = b.shape[2]
                start_op_time = time.perf_counter()
                json_obj = {
                    "method": "torch_transpose",
                    "input": "json_list_" + str(rhs_block_index),
                    "dims": (1, 2),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                b = b.transpose(1,2) # batch_size, 1, prev_size

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(len(json_list) - 1),
                    "shape": (batch_size*sym_size, self.num_channels, self.ix, self.iy),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                b_index = len(json_list) - 1
                b = b.reshape(b.shape[0]*b.shape[1], self.num_channels, self.ix, self.iy)

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                kernel_index = len(json_list) - 1
                json_obj = {
                    "method": "F.conv2d",
                    "input": "json_list_" + str(b_index),
                    "weight": "json_list_" + str(kernel_index),
                    "stride": (self.sx, self.sy),
                    "padding": (self.px, self.py),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res = F.conv2d(b, self.block, stride=(self.sx, self.sy), padding=(self.px, self.py)) # batch_size*1, num_kernels, ox, oy

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(len(json_list) - 1),
                    "shape": (batch_size, sym_size, -1),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res = res.reshape(batch_size, sym_size, -1)

                json_obj = {
                    "method": "torch_transpose",
                    "input": "json_list_" + str(len(json_list) - 1),
                    "dims": (1, 2),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                res = res.transpose(1,2) # batch_size, curr_size, 1
                res_index = len(json_list) - 1

                equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
                new_total_shape = torch.cat([torch.tensor([batch_size, res.shape[1]]), sp_block.total_shape[-1:]])
                res = RepeatBlock(res, new_total_shape, json_list, res_index)
                res_index = res.json_index
            else:
                # General case (repeat spans more than the sym dim): densify and
                # retry. .expand() is a view, so this costs no more than an
                # equivalent DenseBlock rhs would already require.
                dense_block, dense_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
                dense_sp_block = DenseBlock(dense_block, json_list, dense_index)
                res, res_index = self.matmul_equal_dims(dense_sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_sp_block.json_index)
        elif isinstance(sp_block, PatchesBlock):
            # KernelBlock @ PatchesBlock -> PatchesBlock, ported from baseline. The
            # old fallback densified sp_block (hundreds of MB to a few GB for a
            # mid-size Conv2D+SymExp layer); this composes the two conv-like
            # layers directly. KB (=self) maps P's output space -> KB's output
            # space; P (=sp_block) maps in-space -> P's output (= KB's input).
            # Result maps in-space -> KB's output, receptive field enlarged by
            # (KB.kx-1)*P.sx + P.kx.
            P = sp_block
            KB = self
            # Once the enlarged receptive field covers P's whole input map the
            # patch holds more columns than the dense matrix, and the einsum
            # below is kbx*kby times larger still: convolve the dense form.
            if (compact_patches
                    and ((KB.kx - 1) * P.sx + P.kx) * ((KB.ky - 1) * P.sy + P.ky) >= P.ix * P.iy):
                dense_block, dense_index = P.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
                dense_sp_block = DenseBlock(dense_block, json_list, dense_index)
                return self.matmul_equal_dims(dense_sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=dense_sp_block.json_index)
            B = P.batch_size
            K2 = KB.num_kernels          # result output channels
            K1 = KB.num_channels         # == P.num_kernels (the "mid" channels)
            c0 = P.num_channels          # result input channels
            Pkx, Pky = P.kx, P.ky
            Pox, Poy = P.ox, P.oy
            kbx, kby = KB.kx, KB.ky
            ppatch = c0 * Pkx * Pky
            # Scatter-add each contribution into the enlarged result patch: outer index
            # (da, db) is placed at stride P.sx inside the patch -> a fold operation.
            RKX = (kbx - 1) * P.sx + Pkx
            RKY = (kby - 1) * P.sy + Pky

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": P.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            p_block_index = len(json_list) - 1

            # Treat P's per-position patch weights as a batch dim, P.num_kernels (=K1)
            # as channels, and (P.ox, P.oy) as the spatial image over mid-space.
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(p_block_index),
                "shape": (B, K1, Pox, Poy, ppatch),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            m_index = len(json_list) - 1
            M = P.block.reshape(B, K1, Pox, Poy, ppatch)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(m_index),
                "permutation": (0, 4, 1, 2, 3),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            m_index = len(json_list) - 1
            M = M.permute(0, 4, 1, 2, 3)

            json_obj = {
                "method": "extract_sparse_block",
                "input": "json_list_" + str(lhs_index),
                "block_type": KB.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            kb_block_index = len(json_list) - 1

            # The unfold and einsum below hold kbx*kby copies of every patch entry
            # (tens of GB on ResNet18 zonotope). Origin channels compose
            # independently and are the outermost result column, so compose
            # channel slices and concatenate their columns.
            step = channel_step(c0, 4 * int(B) * Pkx * Pky * kbx * kby * KB.ox * KB.oy * max(K1, K2))
            pieces, piece_indices = [], []
            for ca in range(0, c0, step):
                cch = min(step, c0 - ca)
                pch = cch * Pkx * Pky
                mc, mc_index = M, m_index
                if cch != c0:
                    json_obj = {
                        "method": "torch_slice",
                        "input": "json_list_" + str(m_index),
                        "index": [0, [ca * Pkx * Pky, (ca + cch) * Pkx * Pky]],
                        "output": len(json_list),
                    }
                    json_list.append(json_obj)
                    mc_index = len(json_list) - 1
                    mc = M[:, ca * Pkx * Pky:(ca + cch) * Pkx * Pky]

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(mc_index),
                    "shape": (B * pch, K1, Pox, Poy),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                mc_index = len(json_list) - 1
                mc = mc.reshape(B * pch, K1, Pox, Poy)

                # Gather KB's receptive field over mid-space.
                json_obj = {
                    "method": "F.unfold",
                    "input": "json_list_" + str(mc_index),
                    "kernel_size": (kbx, kby),
                    "padding": (KB.px, KB.py),
                    "stride": (KB.sx, KB.sy),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                u_index = len(json_list) - 1
                U = F.unfold(mc, kernel_size=(kbx, kby), stride=(KB.sx, KB.sy), padding=(KB.px, KB.py))
                # U: (B*pch, K1*kbx*kby, KB.ox*KB.oy)
                out_spatial = U.shape[-1]

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(u_index),
                    "shape": (B, pch, K1, kbx, kby, out_spatial),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                u_index = len(json_list) - 1
                U = U.reshape(B, pch, K1, kbx, kby, out_spatial)

                # Contract KB's kernel over the mid channels K1, keeping (k2, da, db).
                json_obj = {
                    "method": "torch_einsum",
                    "equation": "pckabo,mkab->pcmabo",
                    "lhs": "json_list_" + str(u_index),
                    "rhs": "json_list_" + str(kb_block_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                h_index = len(json_list) - 1
                H = torch.einsum('pckabo,mkab->pcmabo', U, KB.block)
                # H: (B, pch, K2, kbx, kby, out_spatial)

                json_obj = {
                    "method": "torch_permute",
                    "input": "json_list_" + str(h_index),
                    "permutation": (0, 2, 5, 1, 3, 4),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                h_index = len(json_list) - 1
                H = H.permute(0, 2, 5, 1, 3, 4)

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(h_index),
                    "shape": (B * K2 * out_spatial, pch, kbx * kby),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                h_index = len(json_list) - 1
                H = H.reshape(B * K2 * out_spatial, pch, kbx * kby)

                json_obj = {
                    "method": "F.fold",
                    "input": "json_list_" + str(h_index),
                    "output_size": (RKX, RKY),
                    "kernel_size": (Pkx, Pky),
                    "stride": (P.sx, P.sy),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                fold_index = len(json_list) - 1
                fold_out = F.fold(H, output_size=(RKX, RKY), kernel_size=(Pkx, Pky), stride=(P.sx, P.sy))
                # fold_out: (B*K2*out_spatial, cch, RKX, RKY)

                json_obj = {
                    "method": "torch_reshape",
                    "input": "json_list_" + str(fold_index),
                    "shape": (B, K2 * out_spatial, cch * RKX * RKY),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                piece_indices.append(len(json_list) - 1)
                pieces.append(fold_out.reshape(B, K2 * out_spatial, cch * RKX * RKY))

            block, block_index = cat_columns(pieces, piece_indices, json_list)

            new_sx = KB.sx * P.sx
            new_sy = KB.sy * P.sy
            new_px = KB.px * P.sx + P.px
            new_py = KB.py * P.sy + P.py
            new_total_shape = torch.concat([
                torch.max(self.total_shape[:-2], sp_block.total_shape[:-2]),
                self.total_shape[-2:-1],
                sp_block.total_shape[-1:]])
            res = PatchesBlock(block, new_total_shape, P.ix, P.iy, KB.ox, KB.oy,
                               new_sx, new_sy, new_px, new_py, RKX, RKY, c0, K2,
                               json_list, block_index)
            res_index = res.json_index
        else:
            temp, temp_index = self.convert_to_patches(json_list=json_list, template_index=lhs_index, simulacrum=True)
            res, res_index = temp.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=temp_index, rhs_index=rhs_index)
            return res, res_index
        equal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)

        return res, res_index


    def compute_patches_stride_padding(self, patches_padding, patches_stride, op_padding, op_stride):
        """
        Compute stride and padding after a conv layer with patches mode.
        """
        for p in (patches_padding, patches_stride, op_padding, op_stride):
            # assert isinstance(p, int) or (isinstance(p, (list, tuple)) and (len(p) == 2 or len(p) == 4))
            pass
        # If p is int, then same padding on all 4 sides.
        # If p is 2-tuple, then it is padding p[0] on both sides of H, p[1] on both sides of W
        # If p is 4-tuple, then it is padding p[2], p[3] on top and bottom sides of H, p[0] and p[1] on left and right sides of W

        # If any of the inputs are not tuple/list, we convert them to tuple.
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p, p) if isinstance(p, int) else p for p in [patches_padding, op_padding, patches_stride, op_stride]]
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p[1], p[1], p[0], p[0]) if len(p) == 2 else p for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]]
        # Compute the new padding and stride after this layer.
        new_padding = tuple(pp * os + op  for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
        new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))

        return new_padding, new_stride
    
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            kernel_index = len(json_list) - 1
            kernel = self.block

            sx = self.sx
            sy = self.sy
            px = self.px
            py = self.py
            ix = self.ix
            iy = self.iy
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            
            start_op_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(rhs_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            input_tensor = sp_block.block.reshape(batch_size, num_channels, ix, iy)
            

            json_obj = {
                "method": "F.conv2d",
                "input": "json_list_" + str(rhs_index),
                "weight": "json_list_" + str(kernel_index),
                "stride": (sx, sy),
                "padding": (px, py),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = F.conv2d(input_tensor, kernel, stride=(sx, sy), padding=(px, py))
            

            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (batch_size, -1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = block.reshape(batch_size, -1)

            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            
            res = DenseBlock(block, json_list, len(json_list) - 1)
            res_index = res.json_index
        elif isinstance(sp_block, DiagonalBlock):
            raise NotImplementedError
        elif isinstance(sp_block, KernelBlock):
            raise NotImplementedError
        elif isinstance(sp_block, ConstBlock):
            if sp_block.block == 0:
                new_total_shape = self.total_shape.clone()[:-1]

                res = ConstBlock(0, new_total_shape, json_list)
                res_index = res.json_index
            else:
                raise NotImplementedError
        elif isinstance(sp_block, RepeatBlock):
            # Densify and retry, as matmul_equal_dims' general case does; .expand() is a
            # view, so this costs no more than an equivalent DenseBlock rhs already would.
            dense_block, dense_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            dense_sp_block = DenseBlock(dense_block, json_list, dense_index)
            res, res_index = self.matmul_unequal_dims(dense_sp_block, json_list=json_list,
                                                       lhs_index=lhs_index, rhs_index=dense_sp_block.json_index)
            unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
            return res, res_index
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res, res_index
    
    def convert_to_patches(self, json_list=[], template_index=-1, simulacrum=False):
        num_kernels = self.num_kernels
        ox = self.ox
        oy = self.oy
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_view",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (1, num_kernels, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(len(json_list) - 1),
            "index": -1,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_repeat",
            "input": "json_list_" + str(len(json_list) - 1),
            "repeats": (1, 1, 1, ox*oy),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(len(json_list) - 1),
            "permutation": (0, 1, 3, 2),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        patches = self.block.view(1, num_kernels, -1).unsqueeze(-1).repeat(1, 1, 1, ox*oy)
        patches = patches.permute(0,1,3,2).reshape(patches.size(0),num_kernels*ox*oy, -1)
        json_obj = {
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": (patches.size(0),num_kernels*ox*oy, -1),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        
        res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, len(json_list) - 1)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res

        # The following doesnt work for some reason
        patches = self.block.view(1, num_kernels, -1).repeat(1, ox*oy, 1)
        return PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels)


    
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        res = KernelBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res

class DiagonalBlock(SparseBlock):
    def __new__(cls, block, total_shape, diag_index,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, DiagonalBlock.__init__, block, total_shape, diag_index)
        return super().__new__(cls)

    def __init__(self, block, total_shape, diag_index,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        if block_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_list_index = len(json_list) - 1
        else:
            block_json_list_index = block_index

        super().__init__(block, total_shape, 'Diag')
        self.diag_index = diag_index
        self.batch_size = self.total_shape[0]

        SparseBlockType = identifySparseBlockType(block)
        json_obj = {
            "method": "DiagonalBlockConstructor",
            "block": "json_list_" + str(block_json_list_index),
            "total_shape": total_shape.tolist() if isinstance(total_shape, torch.Tensor) else list(total_shape),
            "diag_index": self.diag_index,
            "batch_size": int(self.batch_size),
            "SparseBlockType":  SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_DiagonalBlock/DiagonalBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None

    def as_unit_patches(self, partner, json_list=[], template_index=-1):
        """This [batch, n, n] diagonal as a 1x1 PatchesBlock over `partner`'s grid, or None.

        Valid when the diagonal is a per-neuron identity map of a K x H x W feature map
        onto itself: partner's rows are that map (num_kernels*ox*oy == n), its columns the
        same map (num_channels == num_kernels, ix == ox, iy == oy) and its stride is 1.
        Row k*H*W + p then holds a single tap at channel k, which as patches is
        [batch, K*H*W, K] with d[k, p] on the channel diagonal. Uses ~n*K memory instead
        of the dense path's n*n.
        """
        p = partner
        if not (len(self.total_shape) == 3 and self.block.dim() == 2
                and int(self.total_shape[1]) == int(self.total_shape[2])):
            return None
        n = int(self.total_shape[1])
        K, H, W = p.num_kernels, p.ox, p.oy
        if (n != K * H * W or p.num_channels != K or (p.ix, p.iy) != (H, W)
                or (p.sx, p.sy) != (1, 1) or int(self.block.shape[1]) != n):
            return None
        B = int(self.block.shape[0])
        steps = [("torch_view", "shape", [B, K, H * W]),
                 ("torch_permute", "permutation", [0, 2, 1]),
                 ("torch_diag_embed", None, None),
                 ("torch_permute", "permutation", [0, 2, 1, 3]),
                 ("torch_reshape", "shape", [B, n, K])]
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        for method, key, value in steps:
            obj = {"method": method, "input": "json_list_" + str(len(json_list) - 1),
                   "output": len(json_list)}
            if key is not None:
                obj[key] = value
            json_list.append(obj)
        block = torch.diag_embed(self.block.view(B, K, H * W).permute(0, 2, 1)
                                 ).permute(0, 2, 1, 3).reshape(B, n, K)
        res = PatchesBlock(block, self.total_shape, H, W, H, W, 1, 1, 0, 0, 1, 1, K, K,
                           json_list, len(json_list) - 1)
        return res, res.json_index

    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        if self.diag_index == len(self.total_shape):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_diag_embed",
                "input": "json_list_" + str(template_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res_index = len(json_list) - 1
            res = torch.diag_embed(self.block)
            if simulacrum:
                return res, res_index
            return res
        
        # Apply diag_embed at the correct dimension
        # We'll permute, apply diag_embed, and permute back
        shape = list(self.block.shape)
        diag_index = self.diag_index-1

        # Step 1: move the diagonal dim to the last
        perm = list(range(len(shape)))
        perm.pop(diag_index)
        perm.append(diag_index)

        # DUSH: On block
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(template_index),
            "permutation": perm,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        diag_moved = self.block.permute(perm)
       

        # DUSH : On block
        # Step 2: embed to diagonal matrices (adds 1 dimension at the end)
        json_obj = {
            "method": "torch_diag_embed",
            "input": "json_list_" + str(template_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        diag_expanded = torch.diag_embed(diag_moved)  # last dim becomes (c, c)

        # Step 3: move dims back to restore original ordering + new dimension
        # Insert the extra dim right after diag_index
        new_perm = list(range(len(shape)-1))
        new_perm.insert(diag_index, len(new_perm))
        new_perm.insert(diag_index+1, len(new_perm))

        json_obj = {
            "method": "torch_permute",
            "input": "json_list_" + str(template_index),
            "permutation": new_perm,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        res_index = len(json_list) - 1
        decompressed = diag_expanded.permute(new_perm)

        if simulacrum:
            return decompressed, res_index
        return decompressed
    
    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        # # assert(repeat_dims[self.diag_index]==1 and repeat_dims[self.diag_index-1]==1)
        start_time = time.time()

        new_repeat_dims = torch.concat([repeat_dims[:self.diag_index], repeat_dims[self.diag_index+1:]])
        expand_dims = torch.tensor(self.block.shape) * new_repeat_dims
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_expand",
            "input": "json_list_" + str(template_index),
            "shape": expand_dims.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        new_block = self.block.expand(*expand_dims)
        repeat_time.update_op_time(time.time() - start_time)
        res = DiagonalBlock(new_block, self.total_shape*repeat_dims, self.diag_index, json_list, template_index)
        res_index = res.json_index
        # res = DiagonalBlock(self.block.repeat(*new_repeat_dims), self.total_shape*repeat_dims, self.diag_index)
        if simulacrum:
            return res, res_index
        return res


    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        # assert(index!=self.diag_index)
        start_time = time.time()
        if index<self.diag_index:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(template_index),
                "index": index,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = self.block.unsqueeze(index)
            end_time = time.time()
            unsqueeze_time.update_op_time(end_time-start_time)


            res = DiagonalBlock(res, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.diag_index+1, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(template_index),
            "index": index-1,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        res = self.block.unsqueeze(index-1)
        end_time = time.time()
        unsqueeze_time.update_op_time(end_time-start_time)

        res = DiagonalBlock(res, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.diag_index, json_list, len(json_list) - 1)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        if index==self.diag_index:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            res = DenseBlock(self.block, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        elif index<self.diag_index:
            start_time = time.time()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(template_index),
                "index": index,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1

            new_block = self.block.squeeze(index)
            end_time = time.time()
            squeeze_time.update_op_time(end_time - start_time)

            res = DiagonalBlock(new_block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.diag_index-1, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        else:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(template_index),
                "index": index-1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = self.block.squeeze(index-1)
            res = DiagonalBlock(res, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.diag_index, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        
    def sum(self, dim, json_list=[], template_index=-1, simulacrum=False):
        if dim<self.diag_index-1:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(template_index),
                "dim": dim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            res = self.block.sum(dim)
            res = DiagonalBlock(res, torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]]), self.diag_index, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        elif dim>self.diag_index:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(template_index),
                "dim": dim-1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            res = self.block.sum(dim-1)
            res = DiagonalBlock(res, torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]]), self.diag_index, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res
        else:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            res = DenseBlock(self.block, json_list, len(json_list) - 1)
            res_index = res.json_index
            if simulacrum:
                return res, res_index
            return res

    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        total_start_time = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            start_op_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_index),
                "index": self.diag_index,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block.unsqueeze(self.diag_index)

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = sp_block.block
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            c = a * b
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(c, json_list, len(json_list) - 1)
            res_index = res.json_index
        #FIX FOR: Diagonal @ Diagonal raised NotImplementedError (hit by polyzono's Neuron_add backsubstitution).
        elif (isinstance(sp_block, DiagonalBlock)
              and self.diag_index == sp_block.diag_index == len(self.total_shape) - 1
              and len(self.total_shape) == len(sp_block.total_shape)
              and torch.equal(self.total_shape, sp_block.total_shape)):
            # Both are diagonal on the last two (matrix) axes with identical shapes:
            # diag(d1) @ diag(d2) = diag(d1 * d2), one multiply of the stored diagonals.
            start_op_time = time.perf_counter()
            json_list.append({"method": "sparse_block_extract", "input": "json_list_" + str(lhs_index),
                              "block_type": self.block_type, "output": len(json_list)})
            lhs_index = len(json_list) - 1
            json_list.append({"method": "sparse_block_extract", "input": "json_list_" + str(rhs_index),
                              "block_type": sp_block.block_type, "output": len(json_list)})
            rhs_index = len(json_list) - 1
            json_list.append({"method": "torch_mul", "lhs": "json_list_" + str(lhs_index),
                              "rhs": "json_list_" + str(rhs_index), "output": len(json_list)})
            block = self.block * sp_block.block
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DiagonalBlock(block, self.total_shape.clone(), self.diag_index, json_list, len(json_list) - 1)
            res_index = res.json_index
        elif isinstance(sp_block, DiagonalBlock):
            # Any other layout (diagonal on earlier axes, differing indices or shapes): densify both
            # and use the dense matmul, which makes no layout assumption.
            lhs, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
            lhs = DenseBlock(lhs, json_list, lhs_index)
            rhs, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            rhs = DenseBlock(rhs, json_list, rhs_index)
            return lhs.matmul_equal_dims(rhs, json_list=json_list, lhs_index=lhs.json_index, rhs_index=rhs.json_index)
        #FIX FOR: Diagonal @ Patches raised "Unrecognized sparse block type" (hit by deeppoly_efficient's
        # Neuron_add backsubstitution, whose Relu L/U are the pre-activation conv's own patches).
        elif isinstance(sp_block, PatchesBlock):
            batches = (int(self.total_shape[0]), int(sp_block.total_shape[0]))
            if (len(self.total_shape) == len(sp_block.total_shape) == 3
                    and self.diag_index == 2
                    and int(self.total_shape[1]) == int(sp_block.total_shape[1]) == sp_block.block.shape[1]
                    and (batches[0] == batches[1] or 1 in batches)):
                # diag(d) @ P scales row i of P by d[i]: the KernelBlock case below without the conversion.
                start_op_time = time.perf_counter()
                json_list.append({"method": "sparse_block_extract", "input": "json_list_" + str(rhs_index),
                                  "block_type": sp_block.block_type, "output": len(json_list)})
                rhs_index = len(json_list) - 1
                json_list.append({"method": "sparse_block_extract", "input": "json_list_" + str(lhs_index),
                                  "block_type": self.block_type, "output": len(json_list)})
                lhs_index = len(json_list) - 1
                json_list.append({"method": "torch_unsqueeze", "input": "json_list_" + str(lhs_index),
                                  "index": self.diag_index, "output": len(json_list)})
                lhs_index = len(json_list) - 1
                json_list.append({"method": "torch_mul", "lhs": "json_list_" + str(lhs_index),
                                  "rhs": "json_list_" + str(rhs_index), "output": len(json_list)})
                block = self.block.unsqueeze(self.diag_index) * sp_block.block
                equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
                total_shape = sp_block.total_shape.clone()
                total_shape[0] = max(batches)
                res = PatchesBlock(block, total_shape, sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy,
                                   sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky,
                                   sp_block.num_channels, sp_block.num_kernels, json_list, len(json_list) - 1)
                res_index = res.json_index
            else:
                # Any other layout: densify both and use the dense matmul.
                lhs, lhs_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
                lhs = DenseBlock(lhs, json_list, lhs_index)
                rhs, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
                rhs = DenseBlock(rhs, json_list, rhs_index)
                return lhs.matmul_equal_dims(rhs, json_list=json_list, lhs_index=lhs.json_index, rhs_index=rhs.json_index)
        elif isinstance(sp_block, KernelBlock):

            block_2, rhs_index = sp_block.convert_to_patches(json_list=json_list, template_index=rhs_index, simulacrum=True)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            block_2 = block_2.block # batch_size, curr_size, num_channels * kx * ky
            start_op_time = time.perf_counter()
            
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_index),
                "index": self.diag_index,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            block_1 = self.block.unsqueeze(self.diag_index)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block = block_1 * block_2
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(block, sp_block.total_shape, sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy, sp_block.sx, sp_block.sy, sp_block.px, sp_block.py, sp_block.kx, sp_block.ky, sp_block.num_channels, sp_block.num_kernels, json_list, len(json_list) - 1)
            res_index = res.json_index
        #FIX FOR: Diagonal @ RepeatBlock with real repeats (residual Add after fully connected layers, e.g. Cersyve)
        # raised "Unrecognized sparse block type"; densify the repeat as for repeat_dims == 1.
        elif isinstance(sp_block, RepeatBlock):
            sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            sp_block = DenseBlock(sp_block, json_list, rhs_index)
            rhs_index = sp_block.json_index
            res, res_index = self.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            return res, res_index
        else:
            raise Exception(f'Unrecognized sparse block type: {type(sp_block)}')
        equal_matmul_profilier.update_total_time(time.perf_counter() - total_start_time)

        return res, res_index
        

    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        print(sp_block.block_type)
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            start_op_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(lhs_index),
                "index": self.diag_index,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            a = self.block.unsqueeze(self.diag_index)
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(rhs_index),
                "index": -1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            b = sp_block.block.unsqueeze(-1)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = a * b
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": -1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            res = res.squeeze(-1)
            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(res, json_list, len(json_list) - 1)
            res_index = res.json_index
        #FIX FOR: Diagonal @ RepeatBlock with real repeats (a residual Add after fully connected layers, e.g. Cersyve)
        # hit the unimplemented base case; densify the repeat as for repeat_dims == 1.
        elif isinstance(sp_block, RepeatBlock):
            sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            sp_block = DenseBlock(sp_block, json_list, rhs_index)
            rhs_index = sp_block.json_index
            res, res_index = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            return res, res_index
        elif isinstance(sp_block, ConstBlock) and self.diag_index == len(self.total_shape)-1:
            # Contraction is over the diagonal's own last dim, so the const scales
            # the compressed diagonal directly -- no densifying either side.
            new_total_shape = self.total_shape.clone()[:-1]
            if sp_block.block == 0:
                res = ConstBlock(0, new_total_shape, json_list)
                res_index = res.json_index
            else:
                start_op_time = time.perf_counter()
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_mul",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block = self.block * sp_block.block
                unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
                res = DenseBlock(block, json_list, len(json_list) - 1)
                res_index = res.json_index
        else:
            res, res_index = super().matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res, res_index
    
    
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index, json_list=[], template_index=-1, simulacrum=False):
        r_start = torch.concat([start_index[:self.diag_index], start_index[self.diag_index+1:]])
        r_end = torch.concat([end_index[:self.diag_index], end_index[self.diag_index+1:]])
        r_block_start = torch.concat([block_start_index[:self.diag_index], block_start_index[self.diag_index+1:]])
        local_start = r_start - r_block_start
        local_end = r_end - r_block_start
        s = get_slice(local_start, local_end)
        block = self.block[tuple(s)]
        # Constructed inside the branch now so the tape and the slice's slot can be threaded
        # in; the untraced path builds the same block without touching json_list.
        if simulacrum:
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            tensor_index = len(json_list) - 1
            json_list.append({
                "method": "torch_slice",
                "input": "json_list_" + str(tensor_index),
                "index": [[int(local_start[j]), int(local_end[j])] for j in range(local_start.shape[0])],
                "output": len(json_list),
            })
            slice_index = len(json_list) - 1
            res = DiagonalBlock(block, torch.concat([torch.tensor(block.shape[:self.diag_index]), torch.tensor(block.shape[self.diag_index-1:])]),  self.diag_index, json_list, slice_index)
            return res, res.json_index
        res = DiagonalBlock(block, torch.concat([torch.tensor(block.shape[:self.diag_index]), torch.tensor(block.shape[self.diag_index-1:])]),  self.diag_index)
        return res

    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        res = DiagonalBlock(block, self.total_shape, self.diag_index, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    
    def binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, RepeatBlock):
            if (sp_block.repeat_dims[self.diag_index] > 1) and sp_block.only_one_repeat:
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                block_1 = self.block 

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_squeeze",
                    "input": "json_list_" + str(rhs_index),
                    "index": self.diag_index,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                block_2 = sp_block.block.squeeze(self.diag_index)

                json_obj = {
                    "method": "torch_binary",
                    "op": op.__name__,
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)


                res, res_index = self.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.update_total_time(time.perf_counter() - start_time_total)
                return res, res_index
            elif (sp_block.repeat_dims[self.diag_index-1] > 1) and sp_block.only_one_repeat:
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                block_1 = self.block 

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_squeeze",
                    "input": "json_list_" + str(rhs_index),
                    "index": self.diag_index-1,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                block_2 = sp_block.block.squeeze(self.diag_index-1)
                
                json_obj = {
                    "method": "torch_binary",
                    "op": op.__name__,
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)
                res, res_index = self.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.update_total_time(time.perf_counter() - start_time_total)
                return res, res_index
        return super().binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
        
class PatchesBlock(SparseBlock):
    def __new__(cls, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels, 
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, PatchesBlock.__init__, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)
        return super().__new__(cls)

    def __init__(self, block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels, 
                 og_json_list = None, block_index = -1,
                 layer_index = None, counter = None, inside_while = False,
                 while_number = None, while_iteration = None):
        # block.shape = batch, num_kernels*ox*oy, num_channels*kx*ky 
        # self.total_shape = total_shape

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list
        
        if block_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_list_index = len(json_list) - 1
        else:
            block_json_list_index = block_index


        super().__init__(block, total_shape, 'P')
        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py
        self.kx = kx
        self.ky = ky
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.batch_size = block.shape[0]

        SparseBlockType = identifySparseBlockType(block)
        json_obj = {
            "method": "PatchesBlockConstructor",
            "block": "json_list_" + str(block_json_list_index),
            "total_shape": total_shape.tolist() if isinstance(total_shape, torch.Tensor) else list(total_shape),
            "ix": self.ix,
            "iy": self.iy,
            "ox": self.ox,
            "oy": self.oy,
            "sx": self.sx,
            "sy": self.sy,
            "px": self.px,
            "py": self.py,
            "kx": self.kx,
            "ky": self.ky,
            "num_channels": self.num_channels,
            "num_kernels": self.num_kernels,
            "SparseBlockType":  SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_list = json_list
        self.json_index = len(json_list) - 1

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_PatchesBlock/PatchesBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None

    def parameters(self):
        return [self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels]
        
    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        res =  PatchesBlock(self.block, self.total_shape*repeat_dims, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res


    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        res = PatchesBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        res = PatchesBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res
    

    # AVAL: understand this function
    def compute_patches_stride_padding(self, patches_padding, patches_stride, op_padding, op_stride):
        """
        Compute stride and padding after a conv layer with patches mode.
        """
        for p in (patches_padding, patches_stride, op_padding, op_stride):
            # assert isinstance(p, int) or (isinstance(p, (list, tuple)) and (len(p) == 2 or len(p) == 4))
            pass
        # If p is int, then same padding on all 4 sides.
        # If p is 2-tuple, then it is padding p[0] on both sides of H, p[1] on both sides of W
        # If p is 4-tuple, then it is padding p[2], p[3] on top and bottom sides of H, p[0] and p[1] on left and right sides of W

        # If any of the inputs are not tuple/list, we convert them to tuple.
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p, p) if isinstance(p, int) else p for p in [patches_padding, op_padding, patches_stride, op_stride]]
        full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
                (p[1], p[1], p[0], p[0]) if len(p) == 2 else p for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]]
        # Compute the new padding and stride after this layer.
        new_padding = tuple(pp * os + op  for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
        new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))

        return new_padding, new_stride




    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        # Avoid convolving an oversized receptive field full of padding. The
        # dense representation has exactly the feature-map extent and cuDNN
        # can apply the network's padding directly to it.
        if compact_patches and isinstance(sp_block, KernelBlock):
            next_kx = (self.kx - 1) * sp_block.sx + sp_block.kx
            next_ky = (self.ky - 1) * sp_block.sy + sp_block.ky
            if next_kx * next_ky >= sp_block.ix * sp_block.iy:
                dense, index = self.get_dense(json_list, lhs_index, simulacrum=True)
                dense = DenseBlock(dense, json_list, index)
                return dense.matmul_equal_dims(sp_block, json_list, dense.json_index, rhs_index)
        # AVAL: understand the if case
        if isinstance(sp_block, KernelBlock):
            start_op = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(lhs_index),
                "shape": (self.batch_size*self.num_kernels*self.ox*self.oy, self.num_channels, self.kx, self.ky),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "F.conv_transpose2d",
                "input": "json_list_" + str(lhs_index),
                "weight": "json_list_" + str(rhs_index),
                "stride": (sp_block.sx, sp_block.sy),
                "padding": (0, 0),
                "output_padding": (0, 0),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            patches_index = len(json_list) - 1
            flattened_patches = self.block.reshape(self.batch_size*self.num_kernels*self.ox*self.oy, self.num_channels, self.kx, self.ky)
            patches = conv_transpose2d(flattened_patches, sp_block.block, stride=(sp_block.sx, sp_block.sy))
            kx = patches.shape[-2]
            ky = patches.shape[-1]

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(patches_index),
                "shape": (self.batch_size, self.num_kernels*self.ox*self.oy, -1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            patches_index = len(json_list) - 1
            patches = patches.reshape(self.batch_size, self.num_kernels*self.ox*self.oy, -1)
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op)

            # assert(kx == (self.kx-1)*sp_block.sx + sp_block.kx)
            # assert(ky == (self.ky-1)*sp_block.sy + sp_block.ky)
            new_padding, new_stride = self.compute_patches_stride_padding(patches_padding=(self.px, self.py), patches_stride=(self.sx, self.sy), op_padding=(sp_block.px, sp_block.py), op_stride=(sp_block.sx, sp_block.sy))
            new_total_shape = torch.concat([torch.max(self.total_shape[:-2], sp_block.total_shape[:-2]), self.total_shape[-2:-1], sp_block.total_shape[-1:]])
            res = PatchesBlock(patches, new_total_shape, sp_block.ix, sp_block.iy, self.ox, self.oy, new_stride[0], new_stride[1], new_padding[0], new_padding[1], kx, ky, sp_block.num_channels, self.num_kernels, json_list, patches_index)
            res_index = res.json_index

        elif isinstance(sp_block, DiagonalBlock):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            patches_index = len(json_list) - 1
            patches = self.block
            px = self.px
            py = self.py
            sx = self.sx
            sy = self.sy
            ix = self.ix
            iy = self.iy
            kx = self.kx
            ky = self.ky
            num_kernels = self.num_kernels
            num_channels = self.num_channels
            batch_size = int(sp_block.batch_size)
            start_op_time = time.perf_counter()

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            x = sp_block.block.view(batch_size, num_channels, ix, iy)
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(rhs_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))

            json_obj = {
                "method": "torch_transpose",
                "input": "json_list_" + str(len(json_list) - 1),
                "dims": (1, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(len(json_list) - 1),
                "repeats": (1, num_kernels, 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            unf_index = len(json_list) - 1
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            # x_unf = x_unf.unsqueeze(1).repeat(1, num_kernels, 1, 1)
            if patches.shape[0] != batch_size:
                # patches = patches.repeat(batch_size, 1, 1, 1)
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(patches_index),
                    "shape": (batch_size, patches.size(1), patches.size(2)),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                patches_index = len(json_list) - 1
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))
            # patches = patches.repeat(batch_size, 1, 1, 1)
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(unf_index),
                "rhs": "json_list_" + str(patches_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            mul_index = len(json_list) - 1
            patches = x_unf * patches
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            equal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = PatchesBlock(patches, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, mul_index)
            res_index = res.json_index
        elif isinstance(sp_block, PatchesBlock):
            # PatchesBlock @ PatchesBlock -> PatchesBlock, ported from baseline.
            # self (=Po, outer) maps mid-space -> out-space; sp_block (=Pi, inner)
            # maps in-space -> mid-space. Same idea as KernelBlock @ PatchesBlock
            # above, but the "kernel" weights are position-dependent (from
            # Po.block, not a fixed kernel).
            Po = self
            Pi = sp_block
            B = Pi.batch_size
            K2 = Po.num_kernels          # result output channels
            Km = Po.num_channels         # == Pi.num_kernels (the "mid" channels)
            c0 = Pi.num_channels         # result input channels
            Pikx, Piky = Pi.kx, Pi.ky
            Piox, Pioy = Pi.ox, Pi.oy
            okx, oky = Po.kx, Po.ky
            ppatch = c0 * Pikx * Piky

            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": Pi.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pi_block_index = len(json_list) - 1

            # mid-space feature map from inner patches; gather Po's receptive field.
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(pi_block_index),
                "shape": (B, Km, Piox, Pioy, ppatch),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            m_index = len(json_list) - 1
            M = Pi.block.reshape(B, Km, Piox, Pioy, ppatch)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(m_index),
                "permutation": (0, 4, 1, 2, 3),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            m_index = len(json_list) - 1
            M = M.permute(0, 4, 1, 2, 3)

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(m_index),
                "shape": (B * ppatch, Km, Piox, Pioy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            m_index = len(json_list) - 1
            M = M.reshape(B * ppatch, Km, Piox, Pioy)

            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(m_index),
                "kernel_size": (okx, oky),
                "padding": (Po.px, Po.py),
                "stride": (Po.sx, Po.sy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            u_index = len(json_list) - 1
            U = F.unfold(M, kernel_size=(okx, oky), stride=(Po.sx, Po.sy), padding=(Po.px, Po.py))
            out_spatial = U.shape[-1]    # == Po.ox * Po.oy

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(u_index),
                "shape": (B, ppatch, Km, okx, oky, out_spatial),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            u_index = len(json_list) - 1
            U = U.reshape(B, ppatch, Km, okx, oky, out_spatial)

            # per-position outer patch weights, aligned to the unfolded windows.
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": Po.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            po_block_index = len(json_list) - 1

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(po_block_index),
                "shape": (B, K2, out_spatial, Km, okx, oky),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            w_index = len(json_list) - 1
            W = Po.block.reshape(B, K2, out_spatial, Km, okx, oky)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(w_index),
                "permutation": (0, 1, 3, 4, 5, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            w_index = len(json_list) - 1
            W = W.permute(0, 1, 3, 4, 5, 2)   # (B, K2, Km, okx, oky, out_spatial)

            # contract over the mid channels Km per output position, keep (k2, da, db).
            json_obj = {
                "method": "torch_einsum",
                "equation": "ncmabo,nkmabo->nckabo",
                "lhs": "json_list_" + str(u_index),
                "rhs": "json_list_" + str(w_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            h_index = len(json_list) - 1
            H = torch.einsum('ncmabo,nkmabo->nckabo', U, W)

            json_obj = {
                "method": "torch_permute",
                "input": "json_list_" + str(h_index),
                "permutation": (0, 2, 5, 1, 3, 4),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            h_index = len(json_list) - 1
            H = H.permute(0, 2, 5, 1, 3, 4)

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(h_index),
                "shape": (B * K2 * out_spatial, ppatch, okx * oky),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            h_index = len(json_list) - 1
            H = H.reshape(B * K2 * out_spatial, ppatch, okx * oky)

            RKX = (okx - 1) * Pi.sx + Pikx
            RKY = (oky - 1) * Pi.sy + Piky

            json_obj = {
                "method": "F.fold",
                "input": "json_list_" + str(h_index),
                "output_size": (RKX, RKY),
                "kernel_size": (Pikx, Piky),
                "stride": (Pi.sx, Pi.sy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            fold_index = len(json_list) - 1
            fold_out = F.fold(H, output_size=(RKX, RKY), kernel_size=(Pikx, Piky), stride=(Pi.sx, Pi.sy))

            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(fold_index),
                "shape": (B, K2 * out_spatial, c0 * RKX * RKY),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_index = len(json_list) - 1
            block = fold_out.reshape(B, K2 * out_spatial, c0 * RKX * RKY)

            new_sx = Po.sx * Pi.sx
            new_sy = Po.sy * Pi.sy
            new_px = Po.px * Pi.sx + Pi.px
            new_py = Po.py * Pi.sy + Pi.py
            new_total_shape = torch.concat([
                torch.max(self.total_shape[:-2], sp_block.total_shape[:-2]),
                self.total_shape[-2:-1],
                sp_block.total_shape[-1:]])
            res = PatchesBlock(block, new_total_shape, Pi.ix, Pi.iy, Po.ox, Po.oy,
                               new_sx, new_sy, new_px, new_py, RKX, RKY, c0, K2,
                               json_list, block_index)
            res_index = res.json_index
        else:
            raise NotImplementedError

        equal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res, res_index

    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time_total = time.perf_counter()
        if isinstance(sp_block, DenseBlock):
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            patches_index = len(json_list) - 1
            patches = self.block
            sx = self.sx
            sy = self.sy
            px = self.px
            py = self.py
            ix = self.ix
            iy = self.iy
            kx = self.kx
            ky = self.ky
            num_channels = self.num_channels
            num_kernels = self.num_kernels
            batch_size = int(sp_block.batch_size)

            start_op_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(rhs_index),
                "shape": (batch_size, num_channels, ix, iy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            x = sp_block.block.view(batch_size, num_channels, ix, iy)

            json_obj = {
                "method": "F.unfold",
                "input": "json_list_" + str(rhs_index),
                "kernel_size": (kx, ky),
                "padding": (px, py),
                "stride": (sx, sy),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            unf_index = len(json_list) - 1
            x_unf = F.unfold(x, kernel_size=(kx, ky), padding=(px, py), stride=(sx, sy))

            json_obj = {
                "method": "torch_transpose",
                "input": "json_list_" + str(unf_index),
                "dims": (1, 2),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "torch_repeat",
                "input": "json_list_" + str(len(json_list) - 1),
                "repeats": (1, num_kernels, 1),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            unf_index = len(json_list) - 1
            x_unf = x_unf.transpose(1,2).repeat(1, num_kernels, 1)
            if patches.shape[0] != batch_size:
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(patches_index),
                    "shape": (batch_size, patches.size(1), patches.size(2)),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                patches_index = len(json_list) - 1
                patches = patches.expand(batch_size, patches.size(1), patches.size(2))
            
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(unf_index),
                "rhs": "json_list_" + str(patches_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            mul_index = len(json_list) - 1
            patches = x_unf * patches
            # if baseline_gpu_mode:
            #     torch.cuda.synchronize()
            
            
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(mul_index),
                "dim": -1,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            sum_index = len(json_list) - 1
            ret = patches.sum(dim=-1)

            unequal_matmul_profilier.update_actual_op_time(time.perf_counter() - start_op_time)
            res = DenseBlock(ret, json_list, sum_index)
            res_index = res.json_index
        elif isinstance(sp_block, ConstBlock) and sp_block.block == 0:
            res = ConstBlock(0, self.total_shape[:-1], json_list)
            res_index = res.json_index
        elif isinstance(sp_block, RepeatBlock):
            # warnings.warn(f'Matmul with unequal dims inefficient for {type(self)} and {type(sp_block)}')
            sp_block, rhs_index = sp_block.get_dense(json_list=json_list, template_index=rhs_index, simulacrum=True)
            sp_block = DenseBlock(sp_block, json_list, rhs_index)
            rhs_index = sp_block.json_index
            res, res_index = self.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            return res, res_index
        else:
            raise NotImplementedError

        unequal_matmul_profilier.update_total_time(time.perf_counter() - start_time_total)
        return res, res_index
        
    # AVAL: Understand this function
    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        """Converting a Patches piece into a full dense matrix."""
        start_time = time.time()
        batch_size = self.total_shape[0]
        output_channel, output_x, output_y = self.num_kernels, self.ox, self.oy
        input_channel, kernel_x, kernel_y = self.num_channels, self.kx, self.ky
        input_x, input_y = self.ix, self.iy
        padding = (self.py, self.py, self.px, self.px)
        stride = self.sx

        if compact_patches:
            json_list.append({"method": "sparse_block_extract",
                              "input": "json_list_" + str(template_index),
                              "block_type": self.block_type, "output": len(json_list)})
            geometry = [int(v) for v in (batch_size, output_channel, output_x, output_y,
                                        input_channel, kernel_x, kernel_y, input_x, input_y,
                                        self.px, self.py, self.sx, self.sy)]
            json_list.append({"method": "patches_to_dense",
                              "input": "json_list_" + str(len(json_list) - 1),
                              "geometry": geometry, "output": len(json_list)})
            A_matrix = patches_to_dense(self.block, *geometry)
        else:
            # DUSH: On block
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pieces_index = len(json_list) - 1
            pieces = self.block
            # pieces = self.block.permute(1,0,2,3)
            # pieces = pieces.view(batch_size, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y).transpose(0, 1)
            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(pieces_index),
                "shape": (-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pieces_index = len(json_list) - 1
            pieces = pieces.view(-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y)
            if pieces.shape[0] < batch_size:
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(pieces_index),
                    "shape": (int(batch_size), *pieces.shape[1:]),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                pieces_index = len(json_list) - 1
                # If the batch size of pieces is not equal to the batch size of the total shape, we need to expand it.
                pieces = pieces.expand(batch_size, *pieces.shape[1:])
            pieces_current_index = len(json_list) - 1
            # col2im as one out-of-place scatter along the flattened spatial axis:
            # row p of the matrix holds its patch at padded position
            # (p//oy*stride + kxi, p%oy*stride + kyi). The destination columns are
            # pure geometry, so the index is a constant the compiled graph can fold.
            padded_cols = input_y + padding[0] + padding[1]
            flat_spatial = (input_x + padding[2] + padding[3]) * padded_cols
            rows = output_x * output_y
            scatter_shape = (int(batch_size), output_channel, rows, input_channel, flat_spatial)
            json_obj = {
                "method": "torch_zeros",
                "size": scatter_shape,
                "device": "device_mode.get_device()",
                "dtype": str(pieces.dtype),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            A_matrix_index = len(json_list) - 1
            A_matrix = torch.zeros(*scatter_shape, device=pieces.device, dtype=pieces.dtype)

            # pieces have shape (batch, out_c, out_h, out_w, c, h, w).
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(pieces_index),
                "shape": [int(batch_size), output_channel, rows, input_channel, kernel_x * kernel_y],
                "output": len(json_list),
            }
            json_list.append(json_obj)
            pieces_flat_index = len(json_list) - 1
            pieces_flat = pieces.reshape(batch_size, output_channel, rows, input_channel,
                                         kernel_x * kernel_y)

            cols = col2im_columns(output_x, output_y, kernel_x, kernel_y,
                                   input_x + padding[2] + padding[3], padded_cols, stride,
                                   device=pieces.device)
            cols_expr = ('col2im_columns(' + ', '.join(str(v) for v in (
                output_x, output_y, kernel_x, kernel_y,
                input_x + padding[2] + padding[3], padded_cols, stride))
                + ', device=device_mode.get_device()).view(1, 1, ' + str(rows) + ', 1, '
                + str(kernel_x * kernel_y) + ').expand' + str(scatter_shape[:4] + (kernel_x * kernel_y,)))
            json_obj = {
                "method": "tensor_scatter",
                "input": "json_list_" + str(A_matrix_index),
                "value": "json_list_" + str(pieces_flat_index),
                "dim": -1,
                "index": cols_expr,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            A_matrix = A_matrix.scatter(
                -1, cols.view(1, 1, rows, 1, kernel_x * kernel_y).expand(
                    *scatter_shape[:4], kernel_x * kernel_y), pieces_flat)

            json_obj = {
                "method": "torch_view",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": (int(batch_size), output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1]),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
            json_obj = {
                "method": "torch_slice",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": [0, 0, 0, [padding[2], input_x + padding[2]], [padding[0], input_y + padding[0]]],
                "output": len(json_list),
            }
            json_list.append(json_obj)
            A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]
        
            json_obj = {
                "method": "torch_reshape",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": [int(A_matrix.shape[0]), int(A_matrix.shape[1]), -1],
                "output": len(json_list),
            }
            json_list.append(json_obj)
            A_matrix = A_matrix.reshape(A_matrix.shape[0], A_matrix.shape[1], -1)
            # A_matrix = A_matrix.view(batch_size, output_channel*output_x*output_y, input_channel*input_x*input_y)
        end_time = time.time()
        patches_to_mat_time.update_total_time(end_time - start_time)
        if len(A_matrix.shape)!=len(self.total_shape):
            # AVAL: This is a hack to make sure the shape is correct.
            # We need to reshape it to the total shape.
            if (torch.tensor(A_matrix.shape) == self.total_shape[:-1]).all():
                diffdim = -1
                A_matrix = A_matrix.unsqueeze(diffdim).expand(*self.total_shape)
            else:
                diffdim = -1
                for i in range(len(A_matrix.shape)):
                    if(diffdim == -1 and self.total_shape[i] != A_matrix.shape[i]):
                            diffdim = i
                    if diffdim != -1 and self.total_shape[i+1] != A_matrix.shape[i]:
                        raise NotImplementedError(f'PatchesBlock get_dense: {A_matrix.shape} != {self.total_shape[:-1]}')
                A_matrix = A_matrix.unsqueeze(diffdim).expand(*self.total_shape)
            json_obj = {
                "method": "torch_unsqueeze",
                "input": "json_list_" + str(len(json_list) - 1),
                "index": diffdim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            json_obj = {
                "method": "torch_expand",
                "input": "json_list_" + str(len(json_list) - 1),
                "shape": self.total_shape.tolist(),
                "output": len(json_list),
            }
            json_list.append(json_obj)
        if simulacrum:
            return A_matrix, len(json_list) - 1
        return A_matrix
    
    
    
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        res = PatchesBlock(block, self.total_shape, self.ix, self.iy, self.ox, self.oy, self.sx, self.sy, self.px, self.py, self.kx, self.ky, self.num_channels, self.num_kernels, json_list, template_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res

    
    def sum(self, dim, json_list = [], template_index=-1, simulacrum=False):
        if dim == len(self.total_shape)-1:
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(template_index),
                "dim": dim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            sum_index = len(json_list) - 1
            block = self.block.sum(dim)
            res = DenseBlock(block, json_list, sum_index)
            res_index = res.json_index
            return res, res_index
        else:
            denseblock, denseblock_index = self.get_dense(json_list=json_list, template_index=template_index, simulacrum=True)
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(denseblock_index),
                "dim": dim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            sum_index = len(json_list) - 1
            res = DenseBlock(denseblock.sum(dim), json_list, sum_index)
            res_index = res.json_index
            return res, res_index
        
    def alignment_pad(self, sp_block):
        """(dx, dy) padding that grows self's kernel to sp_block's, or None if the
        two patch geometries are not concentric over the same input and grid."""
        if not isinstance(sp_block, PatchesBlock):
            return None
        if (self.ix, self.iy, self.ox, self.oy, self.sx, self.sy,
                self.num_channels, self.num_kernels) != (
                sp_block.ix, sp_block.iy, sp_block.ox, sp_block.oy,
                sp_block.sx, sp_block.sy,
                sp_block.num_channels, sp_block.num_kernels):
            return None
        dkx, dky = sp_block.kx - self.kx, sp_block.ky - self.ky
        if dkx < 0 or dky < 0 or dkx % 2 or dky % 2:
            return None
        dx, dy = dkx // 2, dky // 2
        # concentric windows: the extra taps must be matched by extra padding
        if sp_block.px - self.px != dx or sp_block.py - self.py != dy:
            return None
        if dx == 0 and dy == 0:
            return None
        return dx, dy

    def pad_kernel(self, dx, dy, json_list=[], template_index=-1):
        """Zero-pad the kernel by (dx, dy) each side; the dense value is unchanged."""
        rows = self.block.shape[1]
        wide = [int(self.block.shape[0]), int(rows), self.num_channels,
                self.kx + 2 * dx, self.ky + 2 * dy]
        flat = [int(self.block.shape[0]), int(rows),
                self.num_channels * (self.kx + 2 * dx) * (self.ky + 2 * dy)]
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        json_list.append({
            "method": "torch_view",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": [int(self.block.shape[0]), int(rows), self.num_channels,
                      self.kx, self.ky],
            "output": len(json_list),
        })
        json_list.append({
            "method": "torch_pad",
            "input": "json_list_" + str(len(json_list) - 1),
            "pad": [dy, dy, dx, dx],
            "output": len(json_list),
        })
        json_list.append({
            "method": "torch_reshape",
            "input": "json_list_" + str(len(json_list) - 1),
            "shape": flat,
            "output": len(json_list),
        })
        block = torch.nn.functional.pad(
            self.block.view(*[int(self.block.shape[0]), int(rows),
                              self.num_channels, self.kx, self.ky]),
            (dy, dy, dx, dx)).reshape(*flat)
        res = PatchesBlock(block, self.total_shape, self.ix, self.iy, self.ox,
                           self.oy, self.sx, self.sy, self.px + dx, self.py + dy,
                           self.kx + 2 * dx, self.ky + 2 * dy,
                           self.num_channels, self.num_kernels,
                           json_list, len(json_list) - 1)
        return res, res.json_index

    def align_pair(self, sp_block, json_list, lhs_index, rhs_index):
        """Grow whichever side has the smaller kernel so both share one geometry."""
        if (compact_patches and isinstance(sp_block, PatchesBlock)
                and max(self.kx, sp_block.kx) * max(self.ky, sp_block.ky) >= self.ix * self.iy):
            return None  # the existing dense fallback is smaller
        pad = self.alignment_pad(sp_block)
        if pad is not None:
            lhs, lhs_index = self.pad_kernel(pad[0], pad[1], json_list, lhs_index)
            return lhs, lhs_index, sp_block, rhs_index
        pad = sp_block.alignment_pad(self) if isinstance(sp_block, PatchesBlock) else None
        if pad is not None:
            rhs, rhs_index = sp_block.pad_kernel(pad[0], pad[1], json_list, rhs_index)
            return self, lhs_index, rhs, rhs_index
        return None

    def disjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        if self.parameters() != getattr(sp_block, "parameters", lambda: None)():
            aligned = self.align_pair(sp_block, json_list, lhs_index, rhs_index)
            if aligned is not None:
                lhs, lhs_index, rhs, rhs_index = aligned
                return SparseBlock.disjunctive_binary(lhs, rhs, op, json_list, lhs_index, rhs_index)
        return SparseBlock.disjunctive_binary(self, sp_block, op, json_list, lhs_index, rhs_index)

    def conjunctive_binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        if self.parameters() != getattr(sp_block, "parameters", lambda: None)():
            aligned = self.align_pair(sp_block, json_list, lhs_index, rhs_index)
            if aligned is not None:
                lhs, lhs_index, rhs, rhs_index = aligned
                return SparseBlock.conjunctive_binary(lhs, rhs, op, json_list, lhs_index, rhs_index)
        return SparseBlock.conjunctive_binary(self, sp_block, op, json_list, lhs_index, rhs_index)

    def binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        start_time = time.perf_counter()
        if isinstance(sp_block, RepeatBlock):
            if sp_block.only_one_repeat and sp_block.repeat_dims[-1] > 1:
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                block_1 = self.block 

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(rhs_index),
                    "shape": list(self.block.shape),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                rhs_index = len(json_list) - 1
                block_2 = sp_block.block.expand(self.block.shape)
                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(lhs_index),
                    "rhs": "json_list_" + str(rhs_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)
                res, res_index = self.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res, res_index
        return super().binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
    
    def total_expand(self, json_list=[], template_index=-1, simulacrum=False):
        original_index = template_index
        new_dims = list(self.total_shape[:-1]) + [(self.block.shape[-1])]
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        extract_index = len(json_list) - 1
        json_obj = {
            "method": "torch_expand",
            "input": "json_list_" + str(extract_index),
            "shape": [int(d) for d in self.total_shape[:-1]] + [int(self.block.shape[-1])],
            "output": len(json_list),
        }
        json_list.append(json_obj)
        expand_index = len(json_list) - 1
        json_obj = {
            "method": "assign_to_block",
            "assign_to": "json_list_" + str(original_index),
            "value": "json_list_" + str(expand_index),
        }
        json_list.append(json_obj)
        self.block = self.block.expand(*new_dims)
        if simulacrum:
            return self, original_index
        return self


class ConstBlock(SparseBlock):
    def __new__(cls, block, total_shape,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, ConstBlock.__init__, block, total_shape)
        return super().__new__(cls)

    def __init__(self, block, total_shape,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        super().__init__(block, total_shape, 'C')
        # assert total_shape.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}

        if block_index == -1:
            block_json_list_entry = self.block.tolist() if isinstance(self.block, torch.Tensor) else self.block
        else:
            block_json_list_entry = "json_list_" + str(block_index)

        SparseBlockType = identifySparseBlockType(self.block)
        json_obj = {
            "method": "ConstBlockConstructor",
            "block": block_json_list_entry,
            "total_shape": total_shape.tolist() if isinstance(total_shape, torch.Tensor) else list(total_shape),
            "SparseBlockType":  SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_ConstBlock/ConstBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None

    # Done
    def get_dense(self, json_list=[], template_index=-1, simulacrum = False):
        json_list.append({
            "method": "extract_sparse_block", 
            "input": "json_list_" + str(template_index), 
            "block_type": self.block_type,
            "output": len(json_list)
        })
        json_list.append({
            "method": "torch_ones",
            "size": self.total_shape.tolist(),
            "output": len(json_list)
        })
        json_list.append({
            "method": "torch_mul",
            "lhs": "json_list_" + str(len(json_list)-1),
            "rhs": "json_list_" + str(len(json_list)-2),
            "output": len(json_list)
        })
        ret = torch.ones(*self.total_shape.tolist()) * self.block
        if simulacrum:
            return ret, len(json_list) - 1
        return ret
    
    # Done
    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block_index = len(json_list) - 1
        res = ConstBlock(self.block, self.total_shape*repeat_dims, json_list, block_index)
        const_index = res.json_index
        if simulacrum:
            return res, const_index
        return res

    # Done
    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        res_index = len(json_list) - 1

        res = ConstBlock(self.block, torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]]), json_list, res_index)
        res_index = res.json_index
        
        if simulacrum:
            return res, res_index
        return res
    
    # Done
    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):

        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        extract_index = len(json_list) - 1

        res = ConstBlock(self.block, torch.concat([self.total_shape[:index], self.total_shape[index+1:]]), json_list, extract_index)
        res_index = res.json_index
        
        if simulacrum:
            return res, res_index
        return res

    # Done
    def overwrite_dense_block(self, sp_block, start_index, s):
        block_shape = sp_block.total_shape
        if (block_shape == self.total_shape).all():
            return [sp_block], [torch.tensor([0]*len(self.total_shape))]
        elif (block_shape != self.total_shape).float().sum() == 1:
            unequal_dim = torch.nonzero(block_shape - self.total_shape).item()
            if (start_index == 0).all():
                block_1 = sp_block
                start_index_1 = torch.tensor([0]*len(self.total_shape))
                end_index_2 = self.total_shape
                total_shape_2 = self.total_shape.clone()
                total_shape_2[unequal_dim] -= block_shape[unequal_dim]
                start_index_2 = end_index_2 - total_shape_2
                block_2 = ConstBlock(self.block, total_shape_2)
                return [block_1, block_2], [start_index_1, start_index_2]
            elif (start_index + block_shape == self.total_shape).all():
                start_index_1 = torch.tensor([0]*len(self.total_shape))
                end_index_1 = self.total_shape.clone()
                end_index_1[unequal_dim] -= block_shape[unequal_dim]
                block_1 = ConstBlock(self.block, end_index_1)
                end_index_2 = self.total_shape
                start_index_2 = end_index_2 - block_shape
                block_2 = sp_block
                return [block_1, block_2], [start_index_1, start_index_2]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError

    # Done
    def binary(self, sp_block, op, json_list=[], lhs_index = -1, rhs_index=-1):
        start_time = time.perf_counter()
        if self.block == identity_element(op):
            start = time.perf_counter()
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
                "meta": "We are fucked"
            })
            extract_index = len(json_list) - 1
            unary_op, lambda_index = binary_to_identity_unary(op, json_list)

            block, block_index = unary_operation(sp_block.block, unary_op, json_list = json_list, block_index = extract_index, unary_index = lambda_index)
            binary_profilier.update_total_time(time.perf_counter() - start)
            res, res_index = sp_block.create_similar(block, json_list=json_list, template_index=block_index, simulacrum=True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif self.block == annihilator_element(op):
            json_list.append({
                "method": "noop",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            })
            return self, len(json_list) - 1

        elif self.block == 0 and op == operator.truediv:
            json_list.append({
                "method": "noop",
                "input": "json_list_" + str(lhs_index),
                "output": len(json_list),
            })
            return self, len(json_list) - 1
        elif op!=operator.truediv and op(self.block, 0) in [0, False]:
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            })
            rhs_index = len(json_list) - 1
            json_list.append({
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            lhs_index = len(json_list) - 1

            json_list.append({
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            })
            binary_index = len(json_list) - 1
            block = operation(self.block, sp_block.block, op)
            # res = sp_block.create_similar(block)
            res, res_index = sp_block.create_similar(
                            block=block,
                            json_list=json_list,
                            template_index=binary_index, simulacrum = True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif op in disjunction_ops:
            res, res_index = self.disjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        elif op in conjunction_ops:
            res, res_index = self.conjunctive_binary(sp_block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=rhs_index)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        else:
            raise NotImplementedError

    # Done
    def any(self, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        block_index = len(json_list) - 1
        json_list.append({
            "method": "const_any",
            "input": "json_list_" + str(block_index),
            "output": len(json_list),
        })
        res_index = len(json_list) - 1
        res = self.block == True
        if simulacrum:
            return res, res_index
        return res


    # Done
    def float(self, json_list=[], template_index=-1, simulacrum=False):
        if self.block == False:
            res, res_index = self.create_similar(0.0, json_list=json_list, template_index=-1, simulacrum=True)
        elif self.block == True:
            res, res_index = self.create_similar(1.0, json_list=json_list, template_index=-1, simulacrum=True)
        else:
            # assert False
            return None
        if simulacrum:
            return res, res_index
        return res

    # Done
    def clamp(self, const, min_true, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.perf_counter()
        if min_true:
            unchanged = self.block >= const
        else:
            unchanged = self.block <= const
        if simulacrum:
            if unchanged:
                json_list.append({
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(template_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                })
                block_index = len(json_list) - 1
                # res stays self, as before; the record describes rebuilding it from the
                # extracted block, so the constructed block is only there to emit it.
                res = self
                traced = ConstBlock(self.block, self.total_shape, json_list, block_index)
            else:
                res = ConstBlock(const, self.total_shape, json_list)
                traced = res
            clamp_const_block_expense.update_total_time(time.perf_counter() - start_time)
            return res, traced.json_index
        if unchanged:
            res = self
        else:
            res = ConstBlock(const, self.total_shape)
        clamp_const_block_expense.update_total_time(time.perf_counter() - start_time)
        return res


    # Done
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        # start_time = time.perf_counter()
        if self.block == 0:
            new_total_shape = self.total_shape.clone()
            new_total_shape[-1] = sp_block.total_shape[-1]

            res = ConstBlock(0, new_total_shape, json_list)
            res_index = res.json_index
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time)
        return res, res_index

    # Done
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        # start_time = time.perf_counter()
        if self.block == 0:
            new_total_shape = self.total_shape.clone()[:-1]

            res = ConstBlock(0, new_total_shape, json_list)
            res_index = res.json_index
        else:
            raise NotImplementedError
        # matmul_time.update_op_time(time.time() - start_time) 
        return res, res_index


    # Done        
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index, json_list=[], template_index=-1, simulacrum=False):
        new_total_shape = end_index - start_index
        # Constructed inside the branch now so the tape and the looked-up const's slot can be
        # threaded in; the untraced path builds the same block without touching json_list.
        if simulacrum:
            json_list.append({
                "method": "object_lookup",
                "input": "json_list_" + str(template_index),
                "object": "block",
                "output": len(json_list),
            })
            const_value_index = len(json_list) - 1
            res = ConstBlock(self.block, new_total_shape, json_list, const_value_index)
            return res, res.json_index
        return ConstBlock(self.block, new_total_shape)
    
    # Done
    def get_patches(self, batch_size, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels):
        block = torch.ones(batch_size, num_kernels*ox*oy, num_channels*kx*ky)*self.block
        return PatchesBlock(block, total_shape, ix, iy, ox, oy, sx, sy, px, py, kx, ky, num_channels, num_kernels)

    # Done     
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum = False):
        res = ConstBlock(block, self.total_shape, json_list, template_index)
        if simulacrum:
            return res, res.json_index
        return res
    
    # Done
    def sum(self, dim, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        block_index = len(json_list) - 1
        # Summing a const over dim is a scale by that dim's size; it goes on the tape as
        # its own scalar_const slot -- a bare tensor here is not JSON serializable.
        count = float(self.total_shape[dim])
        json_obj = {
            "method": "scalar_const",
            "value": count,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        count_index = len(json_list) - 1
        json_obj = {
            "method": "torch_mul",
            "lhs": "json_list_" + str(block_index),
            "rhs": "json_list_" + str(count_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        mul_index = len(json_list) - 1
        res = ConstBlock(self.block * count, torch.concat([self.total_shape[:dim], self.total_shape[dim+1:]]), json_list, mul_index)
        const_index = res.json_index
        if simulacrum:
            return res, const_index
        return res
    


class RepeatBlock(SparseBlock):
    def __new__(cls, block, total_shape,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):
        if _should_coerce_to_dense(cls):
            return _coerce_to_dense_block(cls, RepeatBlock.__init__, block, total_shape)
        return super().__new__(cls)

    def __init__(self, block, total_shape,
                og_json_list = None, block_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        if block_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            block_json_list_index = len(json_list) - 1
        else:
            block_json_list_index = block_index

        super().__init__(block, total_shape, block_type='R')
        self.repeat_dims = self.total_shape / torch.tensor(self.block.shape)
        self.only_one_repeat = (self.repeat_dims != 1).sum() == 1
        SparseBlockType = identifySparseBlockType(block)
        json_obj = {
            "method": "RepeatBlockConstructor",
            "block": "json_list_" + str(block_json_list_index),
            "total_shape": total_shape.tolist() if isinstance(total_shape, torch.Tensor) else list(total_shape),
            "repeat_dims": self.repeat_dims.tolist() if isinstance(self.repeat_dims, torch.Tensor) else list(self.repeat_dims),
            "only_one_repeat": bool(self.only_one_repeat),
            "SparseBlockType": SparseBlockType,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_list = json_list
        self.json_index = len(json_list) - 1

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                save_capture(f"jit_RepeatBlock/RepeatBlock_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
                self.json_list = None
        # assert total_shape.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
        

    def __str__(self):
        ret = f'RepeatBlock: {self.block.shape} with total shape {self.total_shape} and repeat dims {self.repeat_dims}'
        return ret

    def sum(self, dim, json_list=[], template_index=-1, simulacrum=False):
        # New method -- og_cfg's RepeatBlock has no sum() override.
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        if self.block.shape[dim] == 1 and self.total_shape[dim] > 1:
            # dim is fully broadcast here: sum of N copies of x is N*x, so scale
            # the compact value instead of materializing N copies.
            count = float(self.total_shape[dim])
            json_obj = {
                "method": "torch_squeeze",
                "input": "json_list_" + str(template_index),
                "index": dim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            squeezed_index = len(json_list) - 1
            squeezed = self.block.squeeze(dim)
            json_obj = {
                "method": "scalar_const",
                "value": count,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            count_index = len(json_list) - 1
            json_obj = {
                "method": "torch_mul",
                "lhs": "json_list_" + str(squeezed_index),
                "rhs": "json_list_" + str(count_index),
                "output": len(json_list),
            }
            json_list.append(json_obj)
            mul_index = len(json_list) - 1
            res = DenseBlock(squeezed * count, json_list, mul_index)
        else:
            # dim is real (not broadcast): the compact block already holds every
            # value that needs summing.
            json_obj = {
                "method": "torch_sum",
                "input": "json_list_" + str(template_index),
                "dim": dim,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            sum_index = len(json_list) - 1
            res = DenseBlock(self.block.sum(dim), json_list, sum_index)
        res_index = res.json_index
        if simulacrum:
            return res, res_index
        return res

    def get_dense(self, json_list=[], template_index=-1, simulacrum=False):
        # DUSH: On block
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_expand",
            "input": "json_list_" + str(template_index),
            "shape": self.total_shape.tolist(),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        expand_index = len(json_list) - 1
        dense_block = self.block.expand(*self.total_shape)
        if simulacrum:
            return dense_block, expand_index
        return dense_block

    def repeat(self, repeat_dims, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1

        repeat_block = RepeatBlock(self.block, self.total_shape*repeat_dims, json_list, template_index)
        repeat_index = repeat_block.json_index
        if simulacrum:
            return repeat_block, repeat_index
        return repeat_block
    
    def unsqueeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_unsqueeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        unsqueeze_index = len(json_list) - 1

        new_total_shape = torch.concat([self.total_shape[:index], torch.ones(1, dtype=int), self.total_shape[index:]])
        repeat_block = RepeatBlock(self.block.unsqueeze(index), new_total_shape, json_list, unsqueeze_index)
        repeat_index = repeat_block.json_index
        if simulacrum:
            return repeat_block, repeat_index
        return repeat_block

    def squeeze(self, index, json_list=[], template_index=-1, simulacrum=False):
        start_time = time.time()
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        json_obj = {
            "method": "torch_squeeze",
            "input": "json_list_" + str(template_index),
            "index": index,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        squeeze_index = len(json_list) - 1
        new_total_shape = torch.concat([self.total_shape[:index], self.total_shape[index+1:]])
        repeat_block = RepeatBlock(self.block.squeeze(index), new_total_shape, json_list, squeeze_index)
        repeat_index = repeat_block.json_index
        if simulacrum:
            return repeat_block, repeat_index
        return repeat_block
    
    def create_similar(self, block, json_list=[], template_index=-1, simulacrum=False):
        repeat_block = RepeatBlock(block, self.total_shape, json_list, template_index)
        repeat_index = repeat_block.json_index
        if simulacrum:
            return repeat_block, repeat_index
        return repeat_block
    
    def get_sub_block_custom_range(self, start_index, end_index, block_start_index, json_list=[], template_index=-1, simulacrum=False):
        slice_start = start_index - block_start_index
        slice_end = end_index - block_start_index
        new_total_shape = slice_end - slice_start
        slice_start = torch.where(self.repeat_dims > 1, 0, slice_start)
        slice_end = torch.where(self.repeat_dims > 1, 1, slice_end)
        s = get_slice(slice_start, slice_end)
        b = self.block[tuple(s)]
        # Constructed inside the branch now so the tape and the slice's slot can be threaded
        # in; the untraced path builds the same block without touching json_list.
        if simulacrum:
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            })
            tensor_index = len(json_list) - 1
            json_list.append({
                "method": "torch_slice",
                "input": "json_list_" + str(tensor_index),
                "index": [[int(slice_start[j]), int(slice_end[j])] for j in range(slice_start.shape[0])],
                "output": len(json_list),
            })
            slice_index = len(json_list) - 1
            res = RepeatBlock(b, new_total_shape, json_list, slice_index)
            return res, res.json_index
        return RepeatBlock(b, new_total_shape)
    
    def matmul_equal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):

        block_1, block_1_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
        d_block = DenseBlock(block_1, json_list, block_1_index)
        dense_block_index = d_block.json_index
        res, res_index = d_block.matmul_equal_dims(sp_block, json_list=json_list, lhs_index=dense_block_index, rhs_index=rhs_index)
        return res, res_index
    
    def matmul_unequal_dims(self, sp_block, json_list=[], lhs_index=-1, rhs_index=-1):
        block_1, block_1_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
        d_block = DenseBlock(block_1, json_list, block_1_index)
        dense_block_index = d_block.json_index
        res, res_index = d_block.matmul_unequal_dims(sp_block, json_list=json_list, lhs_index=dense_block_index, rhs_index=rhs_index)
        return res, res_index
    
    def binary(self, sp_block, op, json_list=[], lhs_index=-1, rhs_index=-1):
        # Diagonal/Patches/Kernel cases below are ported from og_cfg; the
        # ConstBlock and RepeatBlock cases further down are new -- og_cfg is
        # missing them too and falls back to the same full materialization.
        start_time = time.perf_counter()
        if isinstance(sp_block, DiagonalBlock):
            if (self.repeat_dims[sp_block.diag_index] > 1) and self.only_one_repeat:
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_squeeze",
                    "input": "json_list_" + str(lhs_index),
                    "index": sp_block.diag_index,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block_1_index = len(json_list) - 1
                block_1 = self.block.squeeze(sp_block.diag_index)

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block_2_index = len(json_list) - 1
                block_2 = sp_block.block

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(block_1_index),
                    "rhs": "json_list_" + str(block_2_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)


                res, res_index = sp_block.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res, res_index
            elif (self.repeat_dims[sp_block.diag_index-1] > 1) and self.only_one_repeat:
                
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_squeeze",
                    "input": "json_list_" + str(lhs_index),
                    "index": sp_block.diag_index - 1,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block_1_index = len(json_list) - 1
                block_1 = self.block.squeeze(sp_block.diag_index - 1)

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(rhs_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block_2_index = len(json_list) - 1
                block_2 = sp_block.block

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(block_1_index),
                    "rhs": "json_list_" + str(block_2_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)
                res, res_index = sp_block.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res, res_index
        if isinstance(sp_block, PatchesBlock):
            if self.only_one_repeat and self.repeat_dims[-1] > 1 and len(sp_block.total_shape) == 3: #last condition means patches was not unsqueezed
                sp_block, sp_block_index = sp_block.total_expand(json_list=json_list, template_index=rhs_index, simulacrum=True)
                
                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(lhs_index),
                    "block_type": self.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                lhs_index = len(json_list) - 1
                json_obj = {
                    "method": "torch_expand",
                    "input": "json_list_" + str(lhs_index),
                    "shape": list(sp_block.block.shape),
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                block_1_index = len(json_list) - 1
                block_1 = self.block.expand(sp_block.block.shape)

                json_obj = {
                    "method": "sparse_block_extract",
                    "input": "json_list_" + str(sp_block_index),
                    "block_type": sp_block.block_type,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                sp_block_index = len(json_list) - 1
                block_2 = sp_block.block 

                json_obj = {
                    "method": "torch_binary",
                    "lhs": "json_list_" + str(block_1_index),
                    "rhs": "json_list_" + str(sp_block_index),
                    "op": op.__name__,
                    "output": len(json_list),
                }
                json_list.append(json_obj)
                binary_index = len(json_list) - 1
                block = operation(block_1, block_2, op)
                res, res_index = sp_block.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return res, res_index
        if isinstance(sp_block, KernelBlock):
            if self.only_one_repeat and self.repeat_dims[-1] > 1:
                block, block_index = sp_block.convert_to_patches(json_list=json_list, template_index=rhs_index, simulacrum=True)
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                res, res_index = self.binary(block, op, json_list=json_list, lhs_index=lhs_index, rhs_index=block_index)
                return res, res_index
        if isinstance(sp_block, ConstBlock):
            # Identity/annihilator shortcuts (self AND False == False, self * 0 == 0),
            # same as the base SparseBlock.binary(). Needed so a structurally-uniform
            # result collapses back to ConstBlock instead of a materialized RepeatBlock
            # -- callers like exists_sub_block rely on ConstBlock to detect a range is
            # done, and without this shortcut a traversal that checks that can hang.
            if sp_block.block == identity_element(op):
                json_list.append({
                    "method": "noop",
                    "input": "json_list_" + str(lhs_index),
                    "output": len(json_list),
                })
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return self, len(json_list) - 1
            elif sp_block.block == annihilator_element(op):
                json_list.append({
                    "method": "noop",
                    "input": "json_list_" + str(rhs_index),
                    "output": len(json_list),
                })
                binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
                return sp_block, len(json_list) - 1
            # sp_block.block is a plain scalar, so it can combine directly with
            # self's compact block -- no need to materialize self first.
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            binary_index = len(json_list) - 1
            block = operation(self.block, sp_block.block, op)
            res, res_index = self.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index
        if isinstance(sp_block, RepeatBlock):
            # Both sides are already compact, so combine them directly and let
            # ordinary torch broadcasting handle it -- no materializing needed.
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(lhs_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            lhs_index = len(json_list) - 1
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(rhs_index),
                "block_type": sp_block.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            rhs_index = len(json_list) - 1
            json_obj = {
                "method": "torch_binary",
                "lhs": "json_list_" + str(lhs_index),
                "rhs": "json_list_" + str(rhs_index),
                "op": op.__name__,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            binary_index = len(json_list) - 1
            block = operation(self.block, sp_block.block, op)
            res, res_index = self.create_similar(block, json_list=json_list, template_index=binary_index, simulacrum=True)
            binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
            return res, res_index

        block, block_index = self.get_dense(json_list=json_list, template_index=lhs_index, simulacrum=True)
        block_1 = DenseBlock(block, json_list, block_index)
        dense_block_index = block_1.json_index
        binary_block_expenses.just_update_total_time(time.perf_counter() - start_time)
        res, res_index = block_1.binary(sp_block, op, json_list=json_list, lhs_index=dense_block_index, rhs_index=rhs_index)
        return res, res_index
    
    def unary(self, op, json_list=[], template_index=-1, simulacrum=False):
        json_obj = {
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        }
        json_list.append(json_obj)
        template_index = len(json_list) - 1
        block, block_index = unary_operation(self.block, op, json_list=json_list, block_index=template_index, concrete_unary = True)
        res, res_index = self.create_similar(block, json_list=json_list, template_index=block_index, simulacrum=True)

        if simulacrum:
            return res, res_index
        return res
    
    def any(self, json_list=[], template_index=-1, simulacrum=False):
        json_list.append({
            "method": "sparse_block_extract",
            "input": "json_list_" + str(template_index),
            "block_type": self.block_type,
            "output": len(json_list),
        })
        block_index = len(json_list) - 1
        json_list.append({
            "method": "torch_any",
            "input": "json_list_" + str(block_index),
            "output": len(json_list),
        })
        res_index = len(json_list) - 1
        if isinstance(self.block, torch.Tensor) and self.block.is_meta:
            res = True
        else:
            res = self.block.any()
        if simulacrum:
            return res, res_index
        return res

    def clamp(self, const, min_true, json_list=[], template_index=-1, simulacrum=False):
        start_total_time = time.perf_counter()
        # _sync()
        clamp_repeat_block_expense.update_total_time(time.perf_counter() - start_total_time)
        if min_true:
            start_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_clamp",
                "input": "json_list_" + str(template_index),
                "const": const,
                "min_true": True,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            clamp_index = len(json_list) - 1
            new_block = self.block.clamp(min=const)
            clamp_repeat_block_op_time.update_op_time(time.perf_counter() - start_time)
        else:
            start_time = time.perf_counter()
            json_obj = {
                "method": "sparse_block_extract",
                "input": "json_list_" + str(template_index),
                "block_type": self.block_type,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            template_index = len(json_list) - 1
            json_obj = {
                "method": "torch_clamp",
                "input": "json_list_" + str(template_index),
                "const": const,
                "min_true": False,
                "output": len(json_list),
            }
            json_list.append(json_obj)
            clamp_index = len(json_list) - 1
            new_block = self.block.clamp(max=const)
            clamp_repeat_block_op_time.update_op_time(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        # _sync()
        clamp_repeat_block_expense.update_total_time(time.perf_counter() - start_time)
        repeat_block = RepeatBlock(new_block, self.total_shape, json_list, clamp_index)
        repeat_index = repeat_block.json_index
        if simulacrum:
            return repeat_block, repeat_index
        return repeat_block


DENSE_DEFAULT_COERCED_BLOCK_TYPES = (
)

def _should_coerce_to_dense(cls):
    return dense_default_mode and cls in DENSE_DEFAULT_COERCED_BLOCK_TYPES

def _coerce_to_dense_block(cls, init_fn, *args, **kwargs):
    tmp = object.__new__(cls)
    init_fn(tmp, *args, **kwargs)
    return DenseBlock(tmp.get_dense())



def sp_where_block(x: SparseBlock, y: SparseBlock, z: SparseBlock, dummy: bool=False, json_list=[], x_index=-1, y_index=-1, z_index=-1, simulacrum=False):
    # New vs og_cfg: the "isinstance(_, (DiagonalBlock, PatchesBlock))"
    # zero-shortcut checks below now also match RepeatBlock, and there's a new
    # same-type RepeatBlock-vs-RepeatBlock case -- og_cfg has neither and
    # falls back to full materialization.
    # if dummy_mode or dummy:
    #     res = copy.copy(y)
    #     if isinstance(y.total_shape, torch.Tensor):
    #         res.total_shape = y.total_shape.clone()
    #     else:
    #         res.total_shape = copy.copy(y.total_shape)
    #     if not isinstance(res, ConstBlock):
    #         if isinstance(res.block, torch.Tensor):
    #             res.block = meta_tensor(res.block.shape)
    #         else:
    #             res.block = meta_tensor(res.total_shape)
    #     return res

    if isinstance(x, ConstBlock):
        if x.block:
            return (y, y_index) if simulacrum else y
        else:
            return (z, z_index) if simulacrum else z
    #FIX FOR: where(cond, c, c) on two equal constants broadcast c to cond's shape (a stored all-zero RepeatBlock in ReLU upper bounds) instead of returning the constant.
    elif isinstance(y, ConstBlock) and isinstance(z, ConstBlock) and y.block == z.block:
        return (y, y_index) if simulacrum else y
    elif isinstance(x, DenseBlock):
        if isinstance(y, type(z)):
            if not isinstance(y, (KernelBlock, PatchesBlock)):
                if isinstance(y, DenseBlock):
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = y.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                elif isinstance(y, DiagonalBlock):
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    x_diag, x_diag_index = get_diagonal(x.block, y.diag_index, json_list=json_list, template_index=x_block_index)
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x_diag, y_block, z_block, json_list=json_list, cond_index=x_diag_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = y.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                elif isinstance(y, ConstBlock):
                    y_block, y_block_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                    z_block, z_block_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = x.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                    raise Exception('Check this case')
                else:
                    raise NotImplementedError
                return (res, res_index) if simulacrum else res
            if y.parameters() == z.parameters():
                if isinstance(y, KernelBlock):
                    y, y_index = y.convert_to_patches(json_list=json_list, template_index=y_index, simulacrum=True)
                    z, z_index = z.convert_to_patches(json_list=json_list, template_index=z_index, simulacrum=True)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(x_index),
                    "block_type": x.block_type,
                    "output": len(json_list),
                })
                x_block_index = len(json_list) - 1
                y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                w, w_index = where_block(x.block, y_dense, z_dense, json_list=json_list, cond_index=x_block_index, lhs_index=y_dense_index, rhs_index=z_dense_index)
                res = DenseBlock(w, json_list, w_index)
                res_index = res.json_index
                # res = y.create_similar(block=where_block(x.convert_to_patches(*y.parameters()).block, y.block, z.block))
                return (res, res_index) if simulacrum else res
            block_1, block_1_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
            block_2, block_2_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(x_index),
                "block_type": x.block_type,
                "output": len(json_list),
            })
            x_block_index = len(json_list) - 1
            w, w_index = where_block(x.block, block_1, block_2, json_list=json_list, cond_index=x_block_index, lhs_index=block_1_index, rhs_index=block_2_index)
            res = DenseBlock(w, json_list, w_index)
            res_index = res.json_index
            return (res, res_index) if simulacrum else res
        elif not isinstance(y, type(z)):
            block_1 = y
            block_1_index = y_index
            block_2 = z
            block_2_index = z_index
            flag = False
            if isinstance(y, KernelBlock):
                block_1, block_1_index = y.convert_to_patches(json_list=json_list, template_index=y_index, simulacrum=True)
                flag = True
            if isinstance(y, ConstBlock):
                if y.block == 0 and isinstance(z, (DiagonalBlock, PatchesBlock, RepeatBlock)):
                    json_list.append({
                        "method": "torch_zeros",
                        "size": list(z.block.shape),
                        "device": "device_mode.get_device()",
                        "dtype": str(z.block.dtype),
                        "output": len(json_list),
                    })
                    zeros_index = len(json_list) - 1
                    block_1, block_1_index = z.create_similar(torch.zeros(z.block.shape, device=z.block.device, dtype=z.block.dtype), json_list=json_list, template_index=zeros_index, simulacrum=True)
                else:
                    self_block, self_block_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                    block_1 = DenseBlock(self_block, json_list, self_block_index)
                    block_1_index = block_1.json_index
                flag = True
            if isinstance(z, KernelBlock):
                block_2, block_2_index = z.convert_to_patches(json_list=json_list, template_index=z_index, simulacrum=True)
                flag = True
            if isinstance(z, ConstBlock):
                if z.block == 0 and isinstance(y, (DiagonalBlock, PatchesBlock, RepeatBlock)):
                    json_list.append({
                        "method": "torch_zeros",
                        "size": list(y.block.shape),
                        "device": "device_mode.get_device()",
                        "dtype": str(y.block.dtype),
                        "output": len(json_list),
                    })
                    zeros_index = len(json_list) - 1
                    block_2, block_2_index = y.create_similar(torch.zeros(y.block.shape, device=y.block.device, dtype=y.block.dtype), json_list=json_list, template_index=zeros_index, simulacrum=True)
                else:
                    z_block, z_block_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                    block_2 = DenseBlock(z_block, json_list, z_block_index)
                    block_2_index = block_2.json_index
                flag = True
            if flag:
                res, res_index = sp_where_block(x, block_1, block_2, json_list=json_list, x_index=x_index, y_index=block_1_index, z_index=block_2_index, simulacrum=True)
                return (res, res_index) if simulacrum else res
        block_1, block_1_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
        block_2, block_2_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
        json_list.append({
            "method": "extract_sparse_block",
            "input": "json_list_" + str(x_index),
            "block_type": x.block_type,
            "output": len(json_list),
        })
        x_block_index = len(json_list) - 1
        w, w_index = where_block(x.block, block_1, block_2, json_list=json_list, cond_index=x_block_index, lhs_index=block_1_index, rhs_index=block_2_index)
        res = DenseBlock(w, json_list, w_index)
        res_index = res.json_index
        return (res, res_index) if simulacrum else res

    elif isinstance(x, RepeatBlock):
        if isinstance(y, type(z)):
            if not isinstance(y, (KernelBlock, PatchesBlock)):
                if isinstance(y, DenseBlock):
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = y.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                elif isinstance(y, DiagonalBlock):
                    if x.repeat_dims[y.diag_index] > 1 and x.only_one_repeat:
                        x_squeezed, x_squeezed_index = x.squeeze(y.diag_index, json_list=json_list, template_index=x_index, simulacrum=True)
                    elif x.repeat_dims[y.diag_index-1] > 1 and x.only_one_repeat:
                        x_squeezed, x_squeezed_index = x.squeeze(y.diag_index-1, json_list=json_list, template_index=x_index, simulacrum=True)
                    else:
                        #FIX FOR: a Repeat condition not repeated along the diagonal (ReLU after a fully connected
                        # residual Add, e.g. Cersyve) left x_squeezed unset; use the dense fallback below.
                        block_1, block_1_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                        block_2, block_2_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                        x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
                        w, w_index = where_block(x_dense, block_1, block_2, json_list=json_list, cond_index=x_dense_index, lhs_index=block_1_index, rhs_index=block_2_index)
                        res = DenseBlock(w, json_list, w_index)
                        return (res, res.json_index) if simulacrum else res
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_squeezed_index),
                        "block_type": x_squeezed.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    x_block = x_squeezed.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x_block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = y.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                elif isinstance(y, ConstBlock):
                    json_list.append({
                        "method": "torch_ones",
                        "size": list(x.block.shape),
                        "output": len(json_list),
                    })
                    y_ones_index = len(json_list) - 1
                    json_list.append({
                        "method": "scalar_const",
                        "value": y.block,
                        "output": len(json_list),
                    })
                    y_const_index = len(json_list) - 1
                    json_list.append({
                        "method": "torch_mul",
                        "lhs": "json_list_" + str(y_ones_index),
                        "rhs": "json_list_" + str(y_const_index),
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = torch.ones(x.block.shape, device=x.block.device)*y.block
                    json_list.append({
                        "method": "torch_ones",
                        "size": list(x.block.shape),
                        "output": len(json_list),
                    })
                    z_ones_index = len(json_list) - 1
                    json_list.append({
                        "method": "scalar_const",
                        "value": z.block,
                        "output": len(json_list),
                    })
                    z_const_index = len(json_list) - 1
                    json_list.append({
                        "method": "torch_mul",
                        "lhs": "json_list_" + str(z_ones_index),
                        "rhs": "json_list_" + str(z_const_index),
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = torch.ones(x.block.shape, device=x.block.device)*z.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = x.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                    # raise Exception('Check this case')
                elif isinstance(y, RepeatBlock):
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = x.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                else:
                    raise NotImplementedError
                return (res, res_index) if simulacrum else res
            if y.parameters() == z.parameters():
                if isinstance(y, KernelBlock):
                    y, y_index = y.convert_to_patches(json_list=json_list, template_index=y_index, simulacrum=True)
                    z, z_index = z.convert_to_patches(json_list=json_list, template_index=z_index, simulacrum=True)
                y, y_index = y.total_expand(json_list=json_list, template_index=y_index, simulacrum=True)
                z, z_index = z.total_expand(json_list=json_list, template_index=z_index, simulacrum=True)
                if x.only_one_repeat and x.repeat_dims[-1] > 1:
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    json_list.append({
                        "method": "torch_expand",
                        "input": "json_list_" + str(x_block_index),
                        "shape": list(y.block.shape),
                        "output": len(json_list),
                    })
                    x_expand_index = len(json_list) - 1
                    x_block = x.block.expand(y.block.shape)
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(y_index),
                        "block_type": y.block_type,
                        "output": len(json_list),
                    })
                    y_block_index = len(json_list) - 1
                    y_block = y.block
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(z_index),
                        "block_type": z.block_type,
                        "output": len(json_list),
                    })
                    z_block_index = len(json_list) - 1
                    z_block = z.block
                    w, w_index = where_block(x_block, y_block, z_block, json_list=json_list, cond_index=x_expand_index, lhs_index=y_block_index, rhs_index=z_block_index)
                    res, res_index = y.create_similar(block=w, json_list=json_list, template_index=w_index, simulacrum=True)
                else:
                    json_list.append({
                        "method": "extract_sparse_block",
                        "input": "json_list_" + str(x_index),
                        "block_type": x.block_type,
                        "output": len(json_list),
                    })
                    x_block_index = len(json_list) - 1
                    y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                    z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                    w, w_index = where_block(x.block, y_dense, z_dense, json_list=json_list, cond_index=x_block_index, lhs_index=y_dense_index, rhs_index=z_dense_index)
                    res = DenseBlock(w, json_list, w_index)
                    res_index = res.json_index
                return (res, res_index) if simulacrum else res
            block_1, block_1_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
            block_2, block_2_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(x_index),
                "block_type": x.block_type,
                "output": len(json_list),
            })
            x_block_index = len(json_list) - 1
            w, w_index = where_block(x.block, block_1, block_2, json_list=json_list, cond_index=x_block_index, lhs_index=block_1_index, rhs_index=block_2_index)
            res = DenseBlock(w, json_list, w_index)
            res_index = res.json_index
            return (res, res_index) if simulacrum else res
        elif not isinstance(y, type(z)):
            block_1 = y
            block_1_index = y_index
            block_2 = z
            block_2_index = z_index
            flag = False
            if isinstance(y, KernelBlock):
                block_1, block_1_index = y.convert_to_patches(json_list=json_list, template_index=y_index, simulacrum=True)
                flag = True
            if isinstance(y, ConstBlock):
                if y.block == 0 and isinstance(z, (DiagonalBlock, PatchesBlock, RepeatBlock)):
                    json_list.append({
                        "method": "torch_zeros",
                        "size": list(z.block.shape),
                        "device": "device_mode.get_device()",
                        "dtype": str(z.block.dtype),
                        "output": len(json_list),
                    })
                    zeros_index = len(json_list) - 1
                    block_1, block_1_index = z.create_similar(torch.zeros(z.block.shape, device=z.block.device, dtype=z.block.dtype), json_list=json_list, template_index=zeros_index, simulacrum=True)
                else:
                    self_block, self_block_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                    block_1 = DenseBlock(self_block, json_list, self_block_index)
                    block_1_index = block_1.json_index
                flag = True
            if isinstance(z, KernelBlock):
                block_2, block_2_index = z.convert_to_patches(json_list=json_list, template_index=z_index, simulacrum=True)
                flag = True
            if isinstance(z, ConstBlock):
                if z.block == 0 and isinstance(y, (DiagonalBlock, PatchesBlock, RepeatBlock)):
                    json_list.append({
                        "method": "torch_zeros",
                        "size": list(y.block.shape),
                        "device": "device_mode.get_device()",
                        "dtype": str(y.block.dtype),
                        "output": len(json_list),
                    })
                    zeros_index = len(json_list) - 1
                    block_2, block_2_index = y.create_similar(torch.zeros(y.block.shape, device=y.block.device, dtype=y.block.dtype), json_list=json_list, template_index=zeros_index, simulacrum=True)
                else:
                    z_block, z_block_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                    block_2 = DenseBlock(z_block, json_list, z_block_index)
                    block_2_index = block_2.json_index
                flag = True
            if flag:
                res, res_index = sp_where_block(x, block_1, block_2, json_list=json_list, x_index=x_index, y_index=block_1_index, z_index=block_2_index, simulacrum=True)
                return (res, res_index) if simulacrum else res
        block_1, block_1_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
        block_2, block_2_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
        x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
        w, w_index = where_block(x_dense, block_1, block_2, json_list=json_list, cond_index=x_dense_index, lhs_index=block_1_index, rhs_index=block_2_index)
        res = DenseBlock(w, json_list, w_index)
        res_index = res.json_index
        return (res, res_index) if simulacrum else res



    elif isinstance(x, DiagonalBlock):
        if isinstance(y, ConstBlock):
            if isinstance(z, ConstBlock):
                y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                y_diag, y_diag_index = get_diagonal(y_dense, x.diag_index, json_list=json_list, template_index=y_dense_index)
                z_diag, z_diag_index = get_diagonal(z_dense, x.diag_index, json_list=json_list, template_index=z_dense_index)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(x_index),
                    "block_type": x.block_type,
                    "output": len(json_list),
                })
                x_block_index = len(json_list) - 1
                w, w_index = where_block(x.block, y_diag, z_diag, json_list=json_list, cond_index=x_block_index, lhs_index=y_diag_index, rhs_index=z_diag_index)
                res = DiagonalBlock(w, x.total_shape, x.diag_index, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            elif isinstance(z, DiagonalBlock):
                y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                y_diag, y_diag_index = get_diagonal(y_dense, x.diag_index, json_list=json_list, template_index=y_dense_index)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(x_index),
                    "block_type": x.block_type,
                    "output": len(json_list),
                })
                x_block_index = len(json_list) - 1
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(z_index),
                    "block_type": z.block_type,
                    "output": len(json_list),
                })
                z_block_index = len(json_list) - 1
                z_block = z.block
                w, w_index = where_block(x.block, y_diag, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_diag_index, rhs_index=z_block_index)
                res = DiagonalBlock(w, x.total_shape, x.diag_index, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            elif isinstance(z, DenseBlock):
                x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
                y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(z_index),
                    "block_type": z.block_type,
                    "output": len(json_list),
                })
                z_block_index = len(json_list) - 1
                z_block = z.block
                w, w_index = where_block(x_dense, y_dense, z_block, json_list=json_list, cond_index=x_dense_index, lhs_index=y_dense_index, rhs_index=z_block_index)
                res = DenseBlock(w, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            else:
                raise NotImplementedError
        elif isinstance(y, DiagonalBlock):
            if isinstance(z, ConstBlock):
                z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
                z_diag, z_diag_index = get_diagonal(z_dense, y.diag_index, json_list=json_list, template_index=z_dense_index)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(x_index),
                    "block_type": x.block_type,
                    "output": len(json_list),
                })
                x_block_index = len(json_list) - 1
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(y_index),
                    "block_type": y.block_type,
                    "output": len(json_list),
                })
                y_block_index = len(json_list) - 1
                y_block = y.block
                w, w_index = where_block(x.block, y_block, z_diag, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_diag_index)
                res = DiagonalBlock(w, y.total_shape, y.diag_index, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            elif isinstance(z, DiagonalBlock):
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(x_index),
                    "block_type": x.block_type,
                    "output": len(json_list),
                })
                x_block_index = len(json_list) - 1
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(y_index),
                    "block_type": y.block_type,
                    "output": len(json_list),
                })
                y_block_index = len(json_list) - 1
                y_block = y.block
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(z_index),
                    "block_type": z.block_type,
                    "output": len(json_list),
                })
                z_block_index = len(json_list) - 1
                z_block = z.block
                w, w_index = where_block(x.block, y_block, z_block, json_list=json_list, cond_index=x_block_index, lhs_index=y_block_index, rhs_index=z_block_index)
                res = DiagonalBlock(w, y.total_shape, y.diag_index, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            elif isinstance(z, DenseBlock):
                x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
                y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
                json_list.append({
                    "method": "extract_sparse_block",
                    "input": "json_list_" + str(z_index),
                    "block_type": z.block_type,
                    "output": len(json_list),
                })
                z_block_index = len(json_list) - 1
                z_block = z.block
                w, w_index = where_block(x_dense, y_dense, z_block, json_list=json_list, cond_index=x_dense_index, lhs_index=y_dense_index, rhs_index=z_block_index)
                res = DenseBlock(w, json_list, w_index)
                res_index = res.json_index
                return (res, res_index) if simulacrum else res
            else:
                raise NotImplementedError
        elif isinstance(y, DenseBlock):
            z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
            x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
            json_list.append({
                "method": "extract_sparse_block",
                "input": "json_list_" + str(y_index),
                "block_type": y.block_type,
                "output": len(json_list),
            })
            y_block_index = len(json_list) - 1
            y_block = y.block
            w, w_index = where_block(x_dense, y_block, z_dense, json_list=json_list, cond_index=x_dense_index, lhs_index=y_block_index, rhs_index=z_dense_index)
            res = DenseBlock(w, json_list, w_index)
            res_index = res.json_index
            return (res, res_index) if simulacrum else res
        else:
            raise NotImplementedError
    x_dense, x_dense_index = x.get_dense(json_list=json_list, template_index=x_index, simulacrum=True)
    y_dense, y_dense_index = y.get_dense(json_list=json_list, template_index=y_index, simulacrum=True)
    z_dense, z_dense_index = z.get_dense(json_list=json_list, template_index=z_index, simulacrum=True)
    w, w_index = where_block(x_dense, y_dense, z_dense, json_list=json_list, cond_index=x_dense_index, lhs_index=y_dense_index, rhs_index=z_dense_index)
    res = DenseBlock(w, json_list, w_index)
    res_index = res.json_index
    return (res, res_index) if simulacrum else res
