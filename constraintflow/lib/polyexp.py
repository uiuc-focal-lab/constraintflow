import torch 
from constraintflow.gbcsr.sparse_tensor import *
from typing import Any
 

class PolyExpSparse:
    def __init__(self, network, mat, const,
                og_json_list = None, mat_index = -1, const_index = -1,
                layer_index = None, counter = None, inside_while = False,
                while_number = None, while_iteration = None):

        if og_json_list is None:
            json_list = []
        else:
            json_list = og_json_list

        if mat_index == -1:
            json_obj = {
                "method": "noop",
                "input": "lhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            mat_json_list_index = len(json_list) - 1
        else:
            mat_json_list_index = mat_index

        self.network = network
        self.mat = mat
        self.const = const

        if not isinstance(self.const, SparseTensor) and isinstance(self.const, torch.Tensor):
            self.const = SparseTensor([torch.tensor([0]*self.const.dim())], [SparseBlock(self.const)], self.const.dim(), torch.tensor(self.const.shape), og_json_list = json_list)
            const_json_list_index = self.const.json_index
        elif const_index == -1:
            json_obj = {
                "method": "noop",
                "input": "rhs",
                "output": len(json_list),
            }
            json_list.append(json_obj)
            const_json_list_index = len(json_list) - 1
        else:
            const_json_list_index = const_index

        json_obj = {
            "method": "PolyExpSparseConstructor",
            "mat": "json_list_" + str(mat_json_list_index),
            "const": "json_list_" + str(const_json_list_index),
            "output": len(json_list),
        }
        json_list.append(json_obj)
        self.json_index = len(json_list) - 1
        self.json_list = json_list

        if dummy_mode and og_json_list is None:
            if layer_index is not None and counter is not None:
                write_jit_capture_file('jit_PolyExpSparse', 'PolyExpSparse', layer_index, counter, inside_while, while_number, while_iteration, json_list)
                self.json_list = None

    def copy(self):
        if isinstance(self.mat, SparseTensor):
            new_mat = self.mat.copy()
        elif isinstance(self.mat, torch.Tensor):
            new_mat = self.mat.clone()
        else:
            new_mat = self.mat

        if isinstance(self.const, SparseTensor):
            new_const = self.const.copy()
        elif isinstance(self.const, torch.Tensor):
            new_const = self.const.clone()
        else:
            new_const = self.const

        return PolyExpSparse(self.network, new_mat, new_const)

    def get_mat(self, abs_elem, dense=False, json_list=None, layer_index=None,
                counter=None, inside_while=False, while_number=None,
                while_iteration=None, lhs_index=-1):
        owns_capture = (json_list is None) and dummy_mode
        if owns_capture:
            json_list = [{"method": "noop", "input": "lhs", "output": 0}]
            lhs_index = 0
        trace = json_list is not None
        if not trace:
            json_list = []
        if trace and not owns_capture:
            assert lhs_index != -1
        if isinstance(self.mat, float):
            if trace:
                mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': mat_idx,
                }
                json_list.append(json_obj)
            if owns_capture:
                write_jit_capture_file(
                    'jit_poly_exp_sparse_get_mat',
                    'poly_exp_sparse_get_mat',
                    layer_index,
                    counter,
                    inside_while,
                    while_number,
                    while_iteration,
                    json_list
                )
            return self.mat
        if dense:
            sp_mat_idx = -1
            mat_idx = -1
            if trace:
                mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': mat_idx,
                }
                json_list.append(json_obj)
            block_ret = self.mat.get_dense(
                json_list=json_list if trace else None,
                template_index=mat_idx,
                simulacrum=True,
            )
            if trace:
                block, block_idx = block_ret
            else:
                block = block_ret
            # block_idx only exists under trace, so the threaded call is guarded the same way
            # the hand-written record was.
            if trace:
                dense_block = DenseBlock(block, json_list, block_idx)
                db_idx = dense_block.json_index
            else:
                dense_block = DenseBlock(block)
            if trace:
                block_list_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'initialise',
                    'name': 'get_dense_sb',
                    'value': '[]',
                    'output': block_list_idx,
                }
                json_list.append(json_obj)
                appended_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'append_list',
                    'list': 'json_list_' + str(block_list_idx),
                    'value': 'json_list_' + str(db_idx),
                    'output': appended_idx,
                }
                json_list.append(json_obj)
            # appended_idx only exists under trace, so the threaded call is guarded the same
            # way the hand-written record was.
            if trace:
                sp_mat = SparseTensor([torch.tensor([0]*block.dim())], [dense_block], block.dim(), torch.tensor(block.shape), og_json_list=json_list, blocks_index=appended_idx)
                sp_mat_idx = sp_mat.json_index
            else:
                sp_mat = SparseTensor([torch.tensor([0]*block.dim())], [dense_block], block.dim(), torch.tensor(block.shape))
        else:
            sp_mat_idx = -1
            if trace:
                sp_mat_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'get_poly_exp_sparse_mat',
                    'input': 'json_list_' + str(lhs_index),
                    'output': sp_mat_idx,
                }
                json_list.append(json_obj)
            sp_mat = self.mat
        start, end = abs_elem.live_layers[0], abs_elem.live_layers[-1]
        start, end = self.network[start].start, self.network[end].end
        start_index = torch.zeros(sp_mat.dims, dtype=torch.int64)
        end_index = sp_mat.total_size
        start_index[-1] = start
        end_index[-1] = end
        sp_mat_gscr = sp_mat.get_sparse_custom_range(start_index, end_index, json_list=json_list if trace else None, layer_index=layer_index, counter=counter, inside_while=inside_while, while_number=while_number, while_iteration=while_iteration, lhs_index=sp_mat_idx)
        if dummy_mode:
            sp_mat_gscr, _ = sp_mat_gscr
        if owns_capture:
            write_jit_capture_file(
                'jit_poly_exp_sparse_get_mat',
                'poly_exp_sparse_get_mat',
                layer_index,
                counter,
                inside_while,
                while_number,
                while_iteration,
                json_list
            )
        return sp_mat_gscr
        
    def get_const(self):
        return self.const
    
    def get_dense_layers(self):
        layer = 0
        dense_layers = set()
        for j, i in enumerate(self.mat.start_indices):
            while(True):
                if self.network[layer].start<=i[-1]:
                    break
                layer+=1
            
            while(True):
                dense_layers.add(layer)
                if self.network[layer].start<=self.mat.end_indices[j][-1]:
                    break
                layer+=1
        return list(dense_layers)
    
    def create_similar(self, network=None, mat=None, const=None):
        if network == None:
            network = self.network
        if mat == None:
            mat = self.mat
        if const == None:
            const = self.const
        return PolyExpSparse(network, mat, const)

    def __repr__(self):
        return f"PolyExpSparse(mat={self.mat}, const={self.const})"
