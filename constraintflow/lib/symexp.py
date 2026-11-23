from typing import Any
from constraintflow.gbcsr.sparse_tensor import *
from constraintflow.gbcsr.sparse_block import *
from constraintflow.lib.globals import dummy_mode, inductor_mode

def get_num_eps(mat):
    if mat==None:
        return 0
    num = 0
    for i in range(mat.num_blocks):
        num += mat.blocks[i].total_shape[-1]
    return num

def get_new_eps(network, initial_shape, json_list=None, layer_index=None,
                counter=None, inside_while=False, while_number=None,
                while_iteration=None):
    owns_capture = (json_list is None) and dummy_mode
    if owns_capture:
        json_list = []
    trace = json_list is not None
    if not trace:
        json_list = []

    num = initial_shape[-1].item()
    # json_list is a throwaway list when not tracing, so the record simply goes nowhere.
    const = SparseTensor([], [], len(initial_shape), initial_shape, og_json_list=json_list)
    const_idx = const.json_index
    start_index = torch.concat([torch.zeros(len(initial_shape), dtype=int), torch.tensor([SymExpSparse.count])])

    mat_tensor = torch.ones(num, dtype=int)
    if trace:
        mat_tensor_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'torch_ones',
            'size': [num],
            #FIX FOR: size is a neuron count, not a batch; with num == 1 (a 1-neuron layer, e.g. Cersyve) codegen
            # took the leading 1 for the traced batch size and emitted ones(batch_size).
            'batch_dim': False,
            'output': mat_tensor_idx,
        }
        json_list.append(json_obj)
    for i in range(len(initial_shape)-1):
        mat_tensor = mat_tensor.unsqueeze(0)
        if trace:
            unsqueeze_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'torch_unsqueeze',
                'input': 'json_list_' + str(mat_tensor_idx),
                'index': 0,
                'output': unsqueeze_idx,
            }
            json_list.append(json_obj)
            mat_tensor_idx = unsqueeze_idx
    mat_tensor = mat_tensor.repeat(*(list(initial_shape[:-1]) + [1]))
    if trace:
        repeat_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'torch_repeat',
            'input': 'json_list_' + str(mat_tensor_idx),
            'repeats': [int(d) for d in initial_shape[:-1]] + [1],
            'output': repeat_idx,
        }
        json_list.append(json_obj)
        mat_tensor_idx = repeat_idx

    mat_total_shape = torch.tensor(list(initial_shape) + [num])
    # mat_tensor_idx only exists under trace, so the threaded call is guarded the same way
    # the hand-written record was.
    if trace:
        mat = DiagonalBlock(mat_tensor, mat_total_shape, len(initial_shape), json_list, mat_tensor_idx)
        diag_idx = mat.json_index
    else:
        mat = DiagonalBlock(mat_tensor, mat_total_shape, diag_index=len(initial_shape))
    if trace:
        block_list_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'initialise',
            'name': 'blocks',
            'value': '[]',
            'output': block_list_idx,
        }
        json_list.append(json_obj)
        json_obj: dict[str, Any] = {
            'method': 'append_list',
            'list': 'json_list_' + str(block_list_idx),
            'value': 'json_list_' + str(diag_idx),
            'output': len(json_list),
        }
        json_list.append(json_obj)
        block_list_idx = len(json_list) - 1

    mat_total_size = torch.tensor(list(initial_shape) + [num+SymExpSparse.count])
    # block_list_idx only exists under trace, so the threaded call is guarded the same way
    # the hand-written record was.
    if trace:
        mat = SparseTensor([start_index], [mat], len(initial_shape)+1, mat_total_size, og_json_list=json_list, blocks_index=block_list_idx)
        mat_idx = mat.json_index
    else:
        mat = SparseTensor([start_index], [mat], len(initial_shape)+1, mat_total_size)

    if network.no_sparsity:
        if trace:
            dense_mat, dense_mat_idx = mat.blocks[0].get_dense(json_list=json_list, template_index=diag_idx, simulacrum=True)
        else:
            dense_mat = mat.blocks[0].get_dense()
        # dense_mat_idx only exists under trace, so the threaded call is guarded the same way
        # the hand-written record was.
        if trace:
            mat.blocks[0] = DenseBlock(dense_mat, json_list, dense_mat_idx)
            dense_block_idx = mat.blocks[0].json_index
        else:
            mat.blocks[0] = DenseBlock(dense_mat)
        mat.end_indices[0] = start_index + torch.tensor(dense_mat.shape)
        if trace:
            block_list_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'initialise',
                'name': 'blocks',
                'value': '[]',
                'output': block_list_idx,
            }
            json_list.append(json_obj)
            json_obj: dict[str, Any] = {
                'method': 'append_list',
                'list': 'json_list_' + str(block_list_idx),
                'value': 'json_list_' + str(dense_block_idx),
                'output': len(json_list),
            }
            json_list.append(json_obj)
            block_list_idx = len(json_list) - 1
            traced_mat = SparseTensor([start_index], mat.blocks, len(initial_shape)+1, mat_total_size, type=mat.type, dense_const=mat.dense_const, og_json_list=json_list, blocks_index=block_list_idx)
            mat_idx = traced_mat.json_index

    if trace:
        new_eps_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'new_eps',
            'mat': 'json_list_' + str(mat_idx),
            'const': 'json_list_' + str(const_idx),
            'output': new_eps_idx,
        }
        json_list.append(json_obj)

    SymExpSparse.count += num
    if owns_capture:
        write_jit_capture_file(
            'jit_new_eps',
            'new_eps',
            layer_index,
            counter,
            inside_while,
            while_number,
            while_iteration,
            json_list
        )
    return SymExpSparse(network, mat, const)

class SymExpSparse:
    count = 0
    def __init__(self, network, mat = None, const = 0.0):
        if not inductor_mode.get_flag():
            if SymExpSparse.count < get_num_eps(mat) :
                SymExpSparse.count = get_num_eps(mat)
        self.mat = mat
        self.const = const
        self.network = network
        if mat==None:
            if isinstance(const, SparseTensor):
                self.mat = SparseTensor([], [], const.dims+1, torch.tensor(list(const.total_size) + [SymExpSparse.count]))
            

    def expand_mat(self):
        assert(self.mat.dense_const==0)
        self.mat.total_size[-1] = SymExpSparse.count

    def get_mat(self, sym_size, json_list=None, layer_index=None,
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

        if self.mat == None:
            empty_mat = SparseTensor([], [], 0, torch.tensor([]), og_json_list=json_list)
            st_idx = empty_mat.json_index
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
            return empty_mat

        self.expand_mat()
        if trace:
            mat_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'get_sym_exp_sparse_mat',
                'input': 'json_list_' + str(lhs_index),
                'output': mat_idx,
            }
            json_list.append(json_obj)
            expand_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'expand_symexp_mat',
                'input': 'json_list_' + str(mat_idx),
                'output': expand_idx,
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

    def get_const(self):
        return self.const