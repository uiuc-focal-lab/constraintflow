import torch 
import math
import inspect
from typing import Any
from constraintflow.lib.polyexp import *
from constraintflow.lib.network import LayerType
from constraintflow.gbcsr.sparse_block import DenseBlock, KernelBlock, ConstBlock, DiagonalBlock
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.lib.globals import dummy_mode, save_capture


class Llist:
    """List of layers."""
    def __init__(self, network, initial_shape, start=None, end=None, llist=None):
        # network type: lib.network.Network
        self.network = network
        # Is this understanding correct:
        # initial_shape is the prefixing dimensions for structural/shape
        # convenience, like batching dimensions?
        self.initial_shape: list[int] = initial_shape
        self.start = start
        self.end = end
        # llist type: list[int]
        # list of layers by their layer No. (index in `Network`, which is a
        # list of `Layer`s)
        self.llist = llist
        self.llist_flag = True
        if llist==None:
            self.llist_flag = False

    def get_metadata(self, elem, batch_size, json_list=None, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
        """
        Metadata is neural network-specific information.
        Not certifier-specific information.
        """
        # ---- Is this true, and why? ----
        # Currently, get_metadata only supports consecutive intervals of
        # layers. Must coalesce successfully first.
        self.coalesce()
        owns_capture = (json_list is None) and dummy_mode
        if json_list is None:
            json_list = []
        if not self.llist_flag:
            # type of ret: list[gbscr.sparse_block.SparseBlock]
            json_obj_representing_ret_list: dict[str, Any] = {
                'method': 'initialise',
                'name': 'ret',
                'value': '[]',
                'output': 0,
                "debug": elem
            }
            json_obj_representing_ret_list_idx = 0
            json_list.append(json_obj_representing_ret_list)
            ret = []
            start_indices = []
            temp = 0
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        kth_weight: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_kth_layer_weight',
                            'input': k,
                            'output': kth_weight
                        }
                        json_list.append(json_obj)
                        block = DenseBlock(self.network[k].weight, json_list, kth_weight)
                        dense_idx: int = block.json_index
                        if not self.network[k].last_layer:
                            for i in range(len(self.initial_shape)):
                                block = block.unsqueeze(0, json_list, block.json_index)
                            repeat_dims = [batch_size]
                            for i in range(len(block.total_shape)-1):
                                repeat_dims.append(1)
                            repeat_dims = torch.tensor(repeat_dims)
                            block = block.repeat(repeat_dims, json_list, block.json_index)
                            dense_idx = block.json_index
                        ret.append(block)
                        json_obj: dict[str, Any] = {
                            'method': 'append_list',
                            'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                            'value': 'json_list_' + str(dense_idx),
                            'output': len(json_list)
                        }
                        json_obj_representing_ret_list_idx = len(json_list)
                        json_list.append(json_obj)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].weight.shape[0]
                    elif self.network[k].type == LayerType.Conv2D:
                        ix, iy = self.network[self.network[k].parents[0]].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        kth_weight: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_kth_layer_weight',
                            'input': k,
                            'output': kth_weight
                        }
                        json_list.append(json_obj)
                        # Constructed after the weight lookup rather than before it, so that
                        # record's slot can be threaded in as the payload ref.
                        block = KernelBlock(self.network[k].weight, torch.tensor([self.network[k].size, self.network[self.network[k].parents[0]].size]), ix, iy, ox, oy, sx, sy, px, py, json_list, kth_weight)
                        kernel_idx: int = block.json_index
                        if self.network.no_sparsity:
                            dense_idx: int = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'block_get_dense',
                                'block': 'json_list_' + str(kernel_idx),
                                'output': dense_idx
                            }
                            json_list.append(json_obj)
                            squeeze_idx: int = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'block_squeeze',
                                'input': 'json_list_' + str(dense_idx),
                                'index': 0,
                                'output': squeeze_idx
                            }
                            json_list.append(json_obj)
                            block = DenseBlock(block.get_dense().squeeze(0), json_list, squeeze_idx)
                            dense_idx: int = block.json_index
                            kernel_idx = dense_idx
                        for i in range(len(self.initial_shape)):
                            block = block.unsqueeze(0, json_list, block.json_index)
                        repeat_dims = [batch_size]
                        for i in range(len(block.total_shape)-1):
                            repeat_dims.append(1)
                        repeat_dims = torch.tensor(repeat_dims)
                        block = block.repeat(repeat_dims, json_list, block.json_index)
                        repeat_idx: int = block.json_index
                        ret.append(block)
                        json_obj: dict[str, Any] = {
                            'method': 'append_list',
                            'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                            'value': 'json_list_' + str(repeat_idx),
                            'output': len(json_list)
                        }
                        json_obj_representing_ret_list_idx = len(json_list)
                        json_list.append(json_obj)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].size
                    else:
                        raise NotImplementedError
                elif elem == 'bias' or elem == 'b':
                    bias_idx: int = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_kth_layer_bias',
                        'input': k,
                        'output': bias_idx
                    }
                    json_list.append(json_obj)
                    # Constructed after the bias lookup rather than before it, so that
                    # record's slot can be threaded in as the payload ref.
                    block = DenseBlock(self.network[k].bias, json_list, bias_idx)
                    dense_idx: int = block.json_index
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0, json_list, block.json_index)
                        dense_idx = block.json_index
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(dense_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'layer':
                    # When elem == 'layer', give the layer number of the layer
                    #   which the neuron belongs to.
                    block = ConstBlock(k, torch.tensor([self.network[k].size]), json_list)
                    const_idx: int = block.json_index
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0, json_list, block.json_index)
                        const_idx = block.json_index
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(const_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'last_layer':
                    # When elem == 'last_layer', give whether the neuron
                    #   belongs to the last layer or not.
                    mat = (k == len(self.network)-1)
                    block = ConstBlock(int(mat), torch.tensor([self.network[k].size]), json_list)
                    const_idx: int = block.json_index
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0, json_list, block.json_index)
                        const_idx = block.json_index
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(const_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                else:
                    raise NotImplementedError
            total_shape = start_indices[-1] + ret[-1].total_shape
            dim = len(total_shape)
            ret_st = SparseTensor(start_indices, ret, dim, total_shape, og_json_list=json_list, blocks_index=json_obj_representing_ret_list_idx)
            st_idx = ret_st.json_index
            if owns_capture:
                write_jit_capture_file(
                    'jit_llist_get_metadata',
                    'llist_get_metadata',
                    layer_index,
                    counter,
                    inside_while,
                    while_number,
                    while_iteration,
                    json_list
                )
            return ret_st
        else:
            raise NotImplementedError
    
    def get_metadata_dummy(self, elem, batch_size, json_list=None, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None):
        """
        Compile-time counterpart of `get_metadata`.
        All sparse blocks replaced with dummy blocks.
        Shape information
        """
        self.coalesce()
        # print(self.start, self.end)
        owns_capture = (json_list is None) and dummy_mode
        if not self.llist_flag:
            if owns_capture:
                json_list = []
            # type of ret: list[gbscr.sparse_block.SparseBlock]
            json_obj_representing_ret_list: dict[str, Any] = {
                'method': 'initialise',
                'name': 'ret',
                'value': '[]',
                'output': 0,
                "debug": elem
            }
            json_obj_representing_ret_list_idx = 0
            json_list.append(json_obj_representing_ret_list)
            ret = []
            start_indices = []
            temp = 0
            for k in range(self.start, self.end):
                if elem == 'weight' or elem == 'w':
                    if self.network[k].type == LayerType.Linear:
                        kth_weight: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_kth_layer_weight',
                            'input': k,
                            'output': kth_weight
                        }
                        json_list.append(json_obj)
                        block = DummyBlock(None, torch.tensor(self.network[k].weight.shape))
                        dense_idx: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'DenseBlock',
                            'input': kth_weight,
                            'output': dense_idx
                        }
                        json_list.append(json_obj)
                        if not self.network[k].last_layer:
                            for i in range(len(self.initial_shape)):
                                block = block.unsqueeze(0, json_list, block.json_index)
                            repeat_dims = [batch_size]
                            for i in range(len(block.total_shape)-1):
                                repeat_dims.append(1)
                            repeat_dims = torch.tensor(repeat_dims)
                            block = block.repeat(repeat_dims, json_list, block.json_index)
                            dense_idx = block.json_index
                        ret.append(block)
                        json_obj: dict[str, Any] = {
                            'method': 'append_list',
                            'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                            'value': 'json_list_' + str(dense_idx),
                            'output': len(json_list)
                        }
                        json_obj_representing_ret_list_idx = len(json_list)
                        json_list.append(json_obj)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].weight.shape[0]
                    elif self.network[k].type == LayerType.Conv2D:
                        ix, iy = self.network[self.network[k].parents[0]].shape[-2:]
                        ox, oy = self.network[k].shape[-2:]
                        sx, sy = self.network[k].stride
                        px, py = self.network[k].padding
                        block = DummyBlock(None, torch.tensor([self.network[k].size, self.network[self.network[k].parents[0]].size]))
                        kth_weight: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_kth_layer_weight',
                            'input': k,
                            'output': kth_weight
                        }
                        json_list.append(json_obj)
                        kernel_idx: int = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'KernelBlock',
                            'block': kth_weight,
                            'total_shape': [self.network[k].size, self.network[self.network[k].parents[0]].size],
                            'ix': ix, 'iy': iy, 'ox': ox, 'oy': oy,
                            'sx': sx, 'sy': sy, 'px': px, 'py': py,
                            'output': kernel_idx
                        }
                        json_list.append(json_obj)
                        if self.network.no_sparsity:
                            dense_idx: int = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'block_get_dense',
                                'block': 'json_list_' + str(kernel_idx),
                                'output': dense_idx
                            }
                            json_list.append(json_obj)
                            squeeze_idx: int = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'block_squeeze',
                                'input': 'json_list_' + str(dense_idx),
                                'index': 0,
                                'output': squeeze_idx
                            }
                            json_list.append(json_obj)
                            block = DummyBlock(None, torch.tensor(block.get_dense().squeeze(0).shape))
                            dense_idx: int = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'DenseBlock',
                                'input': 'json_list_' + str(squeeze_idx),
                                'output': dense_idx
                            }
                            json_list.append(json_obj)
                            kernel_idx = dense_idx
                        for i in range(len(self.initial_shape)):
                            block = block.unsqueeze(0, json_list, block.json_index)
                        repeat_dims = [batch_size]
                        for i in range(len(block.total_shape)-1):
                            repeat_dims.append(1)
                        repeat_dims = torch.tensor(repeat_dims)
                        block = block.repeat(repeat_dims, json_list, block.json_index)
                        repeat_idx: int = block.json_index
                        ret.append(block)
                        json_obj: dict[str, Any] = {
                            'method': 'append_list',
                            'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                            'value': 'json_list_' + str(repeat_idx),
                            'output': len(json_list)
                        }
                        json_obj_representing_ret_list_idx = len(json_list)
                        json_list.append(json_obj)
                        start_index = torch.tensor([0]*len(self.initial_shape) + [temp, 0])
                        start_indices.append(start_index)
                        temp += self.network[k].size
                    else:
                        raise NotImplementedError
                elif elem == 'bias' or elem == 'b':
                    block = DummyBlock(None, torch.tensor(self.network[k].bias.shape))
                    bias_idx: int = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_kth_layer_bias',
                        'input': k,
                        'output': bias_idx
                    }
                    json_list.append(json_obj)
                    dense_idx: int = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'DenseBlock',
                        'input': bias_idx,
                        'output': dense_idx
                    }
                    json_list.append(json_obj)
                    unsqueeze_idcs: list[int] = []
                    for i in range(len(self.initial_shape)):
                        unsqueeze_idcs.append(len(json_list))
                        block = block.unsqueeze(0)
                        json_obj: dict[str, Any] = {
                            'method': 'block_unsqueeze',
                            'input': 'json_list_' + str(unsqueeze_idcs[-1] - 1),
                            'index': 0,
                            'output': unsqueeze_idcs[-1]
                        }
                        dense_idx = unsqueeze_idcs[-1]
                        json_list.append(json_obj)
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(dense_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'layer':
                    # When elem == 'layer', give the layer number of the layer
                    #   which the neuron belongs to.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    # block = DummyBlock(None, torch.tensor([self.network[k].size]))
                    block = ConstBlock(k, torch.tensor([self.network[k].size]))
                    # print(f'get_metadata_dummy `layer` block type: {type(block)}')
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    # print(f'get_metadata_dummy `layer` block type after unsequeeze: {type(block)}')
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(const_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                elif elem == 'last_layer':
                    # When elem == 'last_layer', give whether the neuron
                    #   belongs to the last layer or not.
                    # block = DenseBlock(torch.ones(self.network[k].size, dtype=int)*k)
                    mat = (k == len(self.network)-1)
                    # block = DummyBlock(None, torch.tensor([self.network[k].size]))
                    block = ConstBlock(int(mat), torch.tensor([self.network[k].size]))
                    # print(f'get_metadata_dummy `last_layer` block type: {type(block)}')
                    for i in range(len(self.initial_shape)):
                        block = block.unsqueeze(0)
                    # print(f'get_metadata_dummy `last_layer` block type after unsequeeze: {type(block)}')
                    ret.append(block)
                    json_obj: dict[str, Any] = {
                        'method': 'append_list',
                        'list': 'json_list_' + str(json_obj_representing_ret_list_idx),
                        'value': 'json_list_' + str(const_idx),
                        'output': len(json_list)
                    }
                    json_obj_representing_ret_list_idx = len(json_list)
                    json_list.append(json_obj)
                    start_index = torch.tensor([0]*len(self.initial_shape) + [temp])
                    start_indices.append(start_index)
                    temp += self.network[k].size
                else:
                    raise NotImplementedError
            total_shape = start_indices[-1] + ret[-1].total_shape
            dim = len(total_shape)
            ret_st = SparseTensor(start_indices, ret, dim, total_shape, og_json_list=json_list, blocks_index=json_obj_representing_ret_list_idx)
            st_idx = ret_st.json_index
            # print(f'ret_st: {ret_st}')
            if owns_capture:
                write_jit_capture_file(
                    'jit_llist_get_metadata',
                    'llist_get_metadata',
                    layer_index,
                    counter,
                    inside_while,
                    while_number,
                    while_iteration,
                    json_list
                )
            return ret_st
        else:
            raise NotImplementedError
        
    def coalesce(self):
        if not self.llist_flag:
            return True
        # Can coalesce only if currently operated layers (llist) are
        # consecutive.
        for i in range(len(self.llist)-1):
            if self.llist[i]!=self.llist[i+1]-1:
                return False
        self.start = self.llist[0]
        self.end = self.llist[-1]+1
        self.llist_flag = False
        return True
    
    def decoalesce(self):
        if self.llist_flag:
            return True
        self.llist = []
        for i in range(self.start, self.end):
            self.llist.append(i)
        self.llist_flag = True
        return True
                
    def dot(self, mats, total_size,
            json_list=None, layer_index=None, counter=None,
            inside_while=False, while_number=None, while_iteration=None,
            mats_input='rhs'):
        if not isinstance(mats, list):
            mats: list[SparseTensor] = [mats]
        else:
            assert(False)
        owns_capture = (json_list is None) and dummy_mode
        if json_list is None:
            json_list = []
        rhs_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'noop',
            'input': mats_input,
            'output': rhs_idx
        }
        json_list.append(json_obj)
        mats_0_idx = rhs_idx
        initial_shape = self.initial_shape
        # If `SparseTensor` constructor has no side effects (I'm assuming so
        #   for a constructor) this should have no influence at all, commented.
        # polyexp_const = SparseTensor([], [], 1, torch.tensor([1]))
        polyexp_const = 0.0
        if self.llist_flag:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in self.llist]
        else:
            start_indices = [torch.tensor([0]*len(self.initial_shape) + [self.network[i].start]) for i in range(self.start, self.end)]
        cols = 0
        for j, mat in enumerate(self.llist):
            if self.llist_flag:
                cols += self.network[self.llist[j]].size
            else:
                cols += self.network[self.start+j].size
        assert(mats[0].total_size[-1] == cols)
        
        initial_shape = []
        for j in range(len(self.initial_shape)):
            initial_shape.append(math.lcm(self.initial_shape[j], mats[0].total_size[j].item()))
        
        new_total_size = torch.tensor(initial_shape+[total_size])
        res_blocks_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'initialise',
            'name': 'res_blocks',
            'value': '[]',
            'output': res_blocks_idx
        }
        json_list.append(json_obj)
        for i in range(len(mats[0].blocks)):
            block_i_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'extract_block',
                'input': 'json_list_' + str(mats_0_idx),
                'index': i,
                'output': block_i_idx
            }
            json_list.append(json_obj)
            copy_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'block_copy',
                'input': 'json_list_' + str(block_i_idx),
                'output': copy_idx
            }
            json_list.append(json_obj)
            new_res_blocks_idx = len(json_list)
            json_obj: dict[str, Any] = {
                'method': 'append_list',
                'list': 'json_list_' + str(res_blocks_idx),
                'value': 'json_list_' + str(copy_idx),
                'output': new_res_blocks_idx
            }
            json_list.append(json_obj)
            res_blocks_idx = new_res_blocks_idx
        res_blocks = [i.copy() for i in mats[0].blocks]
        # Both were built in the return statement below, after the capture was written; they
        # are built here instead so their own records land in that capture.
        res_st = SparseTensor(start_indices, res_blocks, len(self.initial_shape)+1, new_total_size, og_json_list=json_list, blocks_index=res_blocks_idx)
        st_idx = res_st.json_index
        const_idx = len(json_list)
        json_obj: dict[str, Any] = {
            'method': 'scalar_const',
            'value': 0.0,
            'output': const_idx
        }
        json_list.append(json_obj)
        res_pes = PolyExpSparse(self.network, res_st, polyexp_const, json_list, mat_index=st_idx, const_index=const_idx)
        pes_idx = res_pes.json_index
        if owns_capture:
            write_jit_capture_file(
                'jit_Llist_dot',
                'Llist_dot',
                layer_index,
                counter,
                inside_while,
                while_number,
                while_iteration,
                json_list
            )
        return res_pes

    def convert_to_poly(self, abs_elem, layer_index=None, counter=None, inside_while=False,
                        while_number=None, while_iteration=None):
        json_list: list[dict[str, Any]] = []
        json_list.append({'method': 'initialise', 'name': 'mats', 'value': '[]', 'output': 0})
        mats_idx = 0

        mats = []
        start_indices = []
        index = 0
        if self.llist:
            for i in self.llist:
                json_list.append({
                    'method': 'tensor_ones',
                    'repeat_dims': [self.network[i].size],
                    #FIX FOR: a layer size, not a batch; a 1-neuron layer (e.g. Cersyve) became ones(batch_size) in codegen.
                    'batch_dim': False,
                    'output': len(json_list),
                })
                json_list.append({
                    'method': 'torch_reshape',
                    'input': 'json_list_' + str(len(json_list) - 1),
                    'shape': [*self.initial_shape, self.network[i].size],
                    'output': len(json_list),
                })
                reshape_idx = len(json_list) - 1
                mat = torch.ones(self.network[i].size).reshape(*self.initial_shape, self.network[i].size)
                block = DiagonalBlock(mat, total_shape=torch.tensor([*self.initial_shape, self.network[i].size, self.network[i].size]), diag_index=len(self.initial_shape) + 1, og_json_list=json_list, block_index=reshape_idx)
                mats.append(block)
                json_list.append({
                    'method': 'append_list',
                    'list': 'json_list_' + str(mats_idx),
                    'value': 'json_list_' + str(block.json_index),
                    'output': len(json_list),
                })
                mats_idx = len(json_list) - 1
                start_indices.append(torch.tensor([0]*len(self.initial_shape) + [index, self.network[i].start]))
                index += self.network[i].size
        else:
            raise NotImplementedError

        polyexp_const = SparseTensor([], [], len(self.initial_shape)+1, torch.tensor(self.initial_shape+[index]), og_json_list=json_list)
        mat_st = SparseTensor(start_indices, mats, len(self.initial_shape)+2, torch.tensor(self.initial_shape+[index, abs_elem.get_poly_size()]), og_json_list=json_list, blocks_index=mats_idx)
        res = PolyExpSparse(self.network, mat_st, polyexp_const, og_json_list=json_list, mat_index=mat_st.json_index, const_index=polyexp_const.json_index)

        if dummy_mode and layer_index is not None and counter is not None:
            save_capture(f"jit_convert_to_poly/convert_to_poly_{layer_index}_{counter}_{inside_while}_{while_number}_{while_iteration}.json", json_list)
        return res
