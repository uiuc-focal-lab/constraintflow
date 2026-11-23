from typing import Any
import inspect

import torch
import copy
from constraintflow.lib.polyexp import *
from constraintflow.lib.symexp import *
from constraintflow.lib.llist import Llist
from constraintflow.lib.globals import dummy_mode


UPDATE_CAPTURE = "jit_update/update_{}.json"


def _shape_parts(types, key, value):
    """(name, SparseTensor) pairs of one shape field: ('',) for Float, const/mat otherwise."""
    if types[key] in ('PolyExp', 'SymExp'):
        return (('const', value.const), ('mat', value.mat))
    return (('', value),)


def _update_snapshot(abs_elem, abs_shape):
    """Pre-update state each shape field's new value is appended onto."""
    snap = {}
    for i, key in enumerate(list(abs_elem.d.keys())[1:]):
        value = abs_shape[i]
        if isinstance(value, (float, int, bool)):
            raise NotImplementedError(
                f"fused flow: update of {key!r} from a scalar abstract shape is not traced")
        new_parts = dict(_shape_parts(abs_elem.types, key, value))
        snap[key] = {
            part: (list(cur.blocks),
                   new_parts[part].dense_const != cur.dense_const and not new_parts[part].check_dense(),
                   new_parts[part].num_blocks)
            for part, cur in _shape_parts(abs_elem.types, key, abs_elem.d[key])
        }
    return snap


def _fresh_block_record(block, json_list, layer_index, key):
    """Record for a block update built itself (the dense_const-mismatch ConstBlock)."""
    if block.block_type != 'C':
        raise NotImplementedError(
            f"fused flow: update at layer {layer_index} put a {type(block).__name__} into "
            f"{key!r} that came from neither the previous state nor the op's output")
    json_list.append({
        'method': 'ConstBlockConstructor',
        'block': block.block.tolist() if isinstance(block.block, torch.Tensor) else block.block,
        'total_shape': block.total_shape.tolist() if isinstance(block.total_shape, torch.Tensor)
                       else list(block.total_shape),
        'SparseBlockType': identifySparseBlockType(block.block),
        'output': len(json_list),
    })
    return len(json_list) - 1


def _tail_source_index(shape, shift, start_index):
    """Which shape block a state block came from: overwrite inserts blocks sorted by start index."""
    target = [int(x) for x in start_index]
    for j, s in enumerate(shape.start_indices):
        if [int(a) + int(b) for a, b in zip(s, shift)] == target:
            return j
    return None


def _write_update_capture(abs_elem, abs_shape, layer_index, snap):
    """Record how each shape field's post-update value is built from d[key] and abs_shape."""
    capture = {}
    for i, key in enumerate(list(abs_elem.d.keys())[1:]):
        # One record list per field: each replays under that field's own irMetadata.
        json_list = [{'method': 'noop', 'input': 'abs_shape_' + str(i), 'output': 0}]
        kind = abs_elem.types[key]
        part_index = {}
        new_parts = dict(_shape_parts(abs_elem.types, key, abs_shape[i]))
        for part, cur in _shape_parts(abs_elem.types, key, abs_elem.d[key]):
            prev_blocks, mismatch, new_count = snap[key][part]
            kept, tail = cur.blocks[:len(prev_blocks)], cur.blocks[len(prev_blocks):]
            if any(a is not b for a, b in zip(kept, prev_blocks)) or len(kept) != len(prev_blocks):
                raise NotImplementedError(
                    f"fused flow: update at layer {layer_index} rewrote existing blocks of {key!r}")
            if len(tail) != (1 if mismatch else 0) + new_count:
                raise NotImplementedError(
                    f"fused flow: update at layer {layer_index} appended {len(tail)} blocks to "
                    f"{key!r}, expected {(1 if mismatch else 0) + new_count}")

            source = 0
            if part:
                json_list.append({
                    'method': 'get_' + ('poly' if kind == 'PolyExp' else 'sym') + '_exp_sparse_' + part,
                    'input': 'json_list_0', 'output': len(json_list)})
                source = len(json_list) - 1
            json_list.append({'method': 'get_abs_elem_sparse_d_key', 'input': 'json_list_-1',
                              'key': key, 'output': len(json_list)})
            prev = len(json_list) - 1
            if part:
                json_list.append({
                    'method': 'get_' + ('poly' if kind == 'PolyExp' else 'sym') + '_exp_sparse_' + part,
                    'input': 'json_list_' + str(prev), 'output': len(json_list)})
                prev = len(json_list) - 1
            json_list.append({'method': 'get_sparse_tensor_blocks',
                              'input': 'json_list_' + str(prev), 'output': len(json_list)})
            blocks = len(json_list) - 1

            # Row offset update() adds to the shape's start indices before overwriting.
            shift = [0, abs_elem.network[layer_index].start] + [0] * (cur.dims - 2)
            for offset, block in enumerate(tail):
                if mismatch and offset == 0:
                    value = _fresh_block_record(block, json_list, layer_index, key)
                else:
                    src = _tail_source_index(new_parts[part], shift,
                                             cur.start_indices[len(prev_blocks) + offset])
                    if src is None:
                        raise NotImplementedError(
                            f"fused flow: update at layer {layer_index} appended a block to {key!r} "
                            f"that matches no block of the abstract shape")
                    json_list.append({'method': 'extract_block', 'input': 'json_list_' + str(source),
                                      'index': src, 'output': len(json_list)})
                    json_list.append({'method': 'block_copy', 'input': 'json_list_' + str(len(json_list) - 1),
                                      'output': len(json_list)})
                    value = len(json_list) - 1
                json_list.append({'method': 'append_list', 'list': 'json_list_' + str(blocks),
                                  'value': 'json_list_' + str(value), 'output': len(json_list)})
                blocks = len(json_list) - 1

            # Constructed only so its __init__ writes the SparseTensorConstructor record.
            part_index[part] = SparseTensor(
                cur.start_indices, cur.blocks, cur.dims, cur.total_size, cur.end_indices,
                cur.type, cur.dense_const, og_json_list=json_list, blocks_index=blocks).json_index

        if '' in part_index:
            output = part_index['']
        elif kind == 'PolyExp':
            output = PolyExpSparse(
                abs_elem.network, abs_elem.d[key].mat, abs_elem.d[key].const, og_json_list=json_list,
                mat_index=part_index['mat'], const_index=part_index['const']).json_index
        else:
            json_list.append({'method': 'SymExpSparse', 'mat': 'json_list_' + str(part_index['mat']),
                              'const': 'json_list_' + str(part_index['const']), 'output': len(json_list)})
            output = len(json_list) - 1
        capture[key] = {'operand': i, 'records': json_list, 'output': output}

    save_capture(UPDATE_CAPTURE.format(layer_index), capture)


class Abs_elem_sparse:
    def __init__(self, d, types, network, batch_size=1, no_sparsity=False):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        # d['llist']: torch.Tensor of torch.bool, currently live layers.
        #   One element larger than network.num_layers, the input layer.
        # d['l'], d['u'], ...: SparseTensor, abstract information about neurons
        # self.d is about the entire NN.
        # `get_elem` uses its parameter `llist` to get only relevant parts of
        #   self.d.
        self.d = d
        # self.types: dict str -> str
        # Keys are the same as d.keys()
        # Meaning: how to interpret each element in self.d. Example:
        # {'l': 'Float', 'u': 'Float', 'L': 'PolyExp', 'U': 'PolyExp', 'llist': 'bool'}
        self.types = types 
        self.network = network
        self.batch_size = batch_size
        self.no_sparsity = no_sparsity
        # Plain list: torch.compile cannot trace nonzero on a data-dependent shape.
        self.live_layers = torch.nonzero(self.d['llist']).flatten().tolist()

    def filter_non_live(self, llist):
        start_time = time.time()
        live_layers = self.live_layers
        # res = copy.deepcopy(llist)
        if llist.llist_flag:
            res_llist = list(set(llist.llist).intersection(set(live_layers)))
            res = Llist(llist.network, llist.initial_shape, llist=res_llist)
        else:
            res_llist = []
            for i in range(llist.start, llist.end):
                if i in live_layers:
                    res_llist.append(i)
            res = Llist(llist.network, llist.initial_shape, llist=res_llist)
            # res.llist = res_llist
            # res.llist_flag = True
            res.coalesce()
        end_time = time.time()
        filter_non_live_time.update_total_time(end_time-start_time)
        return res
    
    def get_poly_size(self):
        l = self.live_layers[-1]
        return self.network[l].end
        

    def get_elem(self, key, llist, json_list=None, layer_index=None, counter=None, inside_while=False, while_number=None, while_iteration=None, lhs_index=-1):
        """
        Gives certifier-specific information about the live neurons.
        
        `lhs_index`: The index of `self` in `json_list`. Currently,
            `Abs_elem_sparse` objects exist in transformers, so they can be
            first-class objects in the JIT json.
        """
        start_time = time.time()
        llist = self.filter_non_live(llist)
        llist_compressed = self.live_layers
        owns_capture = json_list is None and dummy_mode
        if json_list is None:
            json_list = []
        if llist.llist_flag:
            if self.types[key] in ['int', 'float', 'Int', 'Float', 'bool', 'Bool']:
                start_indices = []
                end_indices = []
                blocks = []
                val_const_idx: int = -1
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].start_indices
                    end_indices = self.d[key].end_indices
                    blocks = self.d[key].blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        blocks_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sparse_tensor_blocks',
                            'input': 'json_list_' + str(d_key_idx),
                            'output': blocks_idx
                        }
                        json_list.append(json_obj)
                        val_const = SparseTensor(start_indices, blocks, self.d[key].dims, total_size, end_indices, self.d[key].type, self.d[key].dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                        val_const_idx = val_const.json_index
                    assert val_const_idx != -1
                else:
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        res = self.d[key].get_sparse_custom_range(
                            start_index, end_index,
                            json_list=json_list, layer_index=layer_index,
                            counter=counter, inside_while=inside_while, while_number=while_number,
                            while_iteration=while_iteration,
                            lhs_index=d_key_idx
                        )
                        gscr_st_idx = -1
                        if dummy_mode:
                            res, gscr_st_idx = res
                        else:
                            pass
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                        res_blocks_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sparse_tensor_blocks',
                            'input': 'json_list_' + str(gscr_st_idx),
                            'output': res_blocks_idx
                        }
                        json_list.append(json_obj)
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const, og_json_list=json_list, blocks_index=len(json_list) - 1)
                    val_const_idx = val_const.json_index

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(
                        start_index, end_index, total_size,
                        json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_const_idx
                    )
                    val_const_idx = -1
                    if dummy_mode:
                        val_const, val_const_idx = val_const
                extra_dims = len(llist.initial_shape)-1
                
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(
                        1, json_list=json_list, layer_index=layer_index,
                        counter=counter, inside_while=inside_while,
                        while_number=while_number, while_iteration=while_iteration,
                        lhs_index=val_const_idx
                    )
                    val_const_idx = len(json_list) - 1

                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    assert val_const_idx != -1
                    val_const = val_const.repeat(
                        repeat_shape, json_list=json_list, lhs_index=val_const_idx)
                    if dummy_mode:
                        val_const, val_const_idx = val_const
                if owns_capture:
                    write_jit_capture_file(
                        'jit_Abs_elem_sparse_get_elem',
                        'Abs_elem_sparse_get_elem',
                        layer_index,
                        counter,
                        inside_while,
                        while_number,
                        while_iteration,
                        json_list
                    )
                return val_const
            elif self.types[key] == 'PolyExp':
                start_indices = []
                end_indices = []
                blocks = []
                val_const_idx: int = -1
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].const.start_indices
                    end_indices = self.d[key].const.end_indices
                    blocks = self.d[key].const.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        const_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_poly_exp_sparse_const',
                            'input': 'json_list_' + str(d_key_idx),
                            'output': const_idx
                        }
                        json_list.append(json_obj)
                        blocks_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sparse_tensor_blocks',
                            'input': 'json_list_' + str(const_idx),
                            'output': blocks_idx
                        }
                        json_list.append(json_obj)
                        val_const = SparseTensor(start_indices, blocks, self.d[key].const.dims, total_size, end_indices, self.d[key].const.type, self.d[key].const.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                        val_const_idx = val_const.json_index
                    assert val_const_idx != -1
                else:
                    blocks_idx = len(json_list)
                    blocks_idx_original = blocks_idx
                    json_obj: dict[str, Any] = {
                        'method': 'initialise',
                        'name': 'res_blocks',
                        'value': '[]',
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        const_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_poly_exp_sparse_const',
                            'input': 'json_list_' + str(d_key_idx),
                            'output': const_idx
                        }
                        json_list.append(json_obj)
                        res = self.d[key].const.get_sparse_custom_range(
                            start_index, end_index, json_list=json_list,
                            layer_index=layer_index, counter=counter,
                            inside_while=inside_while, while_number=while_number,
                            while_iteration=while_iteration, lhs_index=const_idx
                        )
                        res_idx = -1
                        if dummy_mode:
                            res, res_idx = res
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                        for j in range(len(res.blocks)):
                            blk_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'extract_block',
                                'input': 'json_list_' + str(res_idx),
                                'block_id': j,
                                'output': blk_idx
                            }
                            json_list.append(json_obj)
                            new_blocks_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'append_list',
                                'list': 'json_list_' + str(blocks_idx),
                                'value': 'json_list_' + str(blk_idx),
                                'output': new_blocks_idx
                            }
                            blocks_idx = new_blocks_idx
                            json_list.append(json_obj)
                    assert blocks_idx != blocks_idx_original
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                    val_const_idx = val_const.json_index

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(
                        start_index, end_index, total_size,
                        json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_const_idx
                    )
                    if dummy_mode:
                        val_const, val_const_idx = val_const
                extra_dims = len(llist.initial_shape)-1
                
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(
                        1, json_list=json_list, layer_index=layer_index,
                        counter=counter, inside_while=inside_while,
                        while_number=while_number, while_iteration=while_iteration,
                        lhs_index=val_const_idx,
                    )
                    val_const_idx = len(json_list) - 1


                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    assert val_const_idx != -1
                    val_const = val_const.repeat(
                        repeat_shape, json_list=json_list, lhs_index=val_const_idx
                    )
                    val_const_idx = len(json_list) - 1


                

                if llist_compressed == llist.llist:
                    start_indices = self.d[key].mat.start_indices
                    end_indices = self.d[key].mat.end_indices
                    blocks = self.d[key].mat.blocks
                    d_key_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_abs_elem_sparse_d_key',
                        'input': 'json_list_' + str(lhs_index),
                        'key': key,
                        'output': d_key_idx
                    }
                    json_list.append(json_obj)
                    mat_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_poly_exp_sparse_mat',
                        'input': 'json_list_' + str(d_key_idx),
                        'output': mat_idx
                    }
                    json_list.append(json_obj)
                    blocks_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_sparse_tensor_blocks',
                        'input': 'json_list_' + str(mat_idx),
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, self.d[key].mat.total_size[-1]])
                        val_mat = SparseTensor(start_indices, blocks, self.d[key].mat.dims, total_size, end_indices, self.d[key].mat.type, self.d[key].mat.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                        val_mat_idx = val_mat.json_index
                else:
                    start_indices = []
                    end_indices = []
                    blocks = []
                    blocks_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'initialise',
                        'name': 'mat_blocks',
                        'value': '[]',
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start, self.network[min(llist_compressed)].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end, self.network[max(llist_compressed)].end])
                        blocks_ids, block_start_indices, block_end_indices = self.d[key].mat.get_block_id(start_index, end_index)
                        for i in range(len(blocks_ids)):
                            block = self.d[key].mat.blocks[blocks_ids[i]]
                            d_key_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'get_abs_elem_sparse_d_key',
                                'input': 'json_list_' + str(lhs_index),
                                'key': key,
                                'output': d_key_idx
                            }
                            json_list.append(json_obj)
                            mat_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'get_poly_exp_sparse_mat',
                                'input': 'json_list_' + str(d_key_idx),
                                'output': mat_idx
                            }
                            json_list.append(json_obj)
                            blk_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'extract_block',
                                'input': 'json_list_' + str(mat_idx),
                                'block_id': blocks_ids[i],
                                'output': blk_idx
                            }
                            json_list.append(json_obj)
                            start_indices.append(torch.tensor([0, llist.network[l].start, block_start_indices[i][2]]))
                            end_indices.append(torch.tensor([self.batch_size, llist.network[l].end, block_end_indices[i][2]]))
                            blocks.append(block)
                            new_blocks_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'append_list',
                                'list': 'json_list_' + str(blocks_idx),
                                'value': 'json_list_' + str(blk_idx),
                                'output': new_blocks_idx
                            }
                            blocks_idx = new_blocks_idx
                            json_list.append(json_obj)
                    
                    val_mat = SparseTensor(start_indices, blocks, len(start_indices[0]), torch.tensor([self.batch_size, self.network[llist.llist[0]].size, self.d[key].mat.total_size[-1]]), end_indices, self.d[key].mat.type, self.d[key].mat.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                    val_mat_idx = val_mat.json_index
                    
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, val_mat.total_size[-1]])
                    total_size = end_index - start_index
                    val_mat = val_mat.reduce_size(
                        start_index, end_index, total_size,
                        json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_mat_idx)
                    if dummy_mode:
                        val_mat, val_mat_idx = val_mat
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    val_mat = val_mat.unsqueeze(
                        1, json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_mat_idx)
                    val_mat_idx = len(json_list) - 1
                
                repeat_shape = torch.tensor(llist.initial_shape + [1,1])
                if not (repeat_shape==1).all():
                    assert val_mat_idx != -1
                    val_mat = val_mat.repeat(
                        repeat_shape, json_list=json_list, lhs_index=val_mat_idx)
                    val_mat_idx = len(json_list) - 1
                
                end_time = time.time()
                
                get_elem_time.update_total_time(end_time-start_time)
                ret_pes = PolyExpSparse(self.network, val_mat, val_const, json_list, mat_index=val_mat_idx, const_index=val_const_idx)
                pes_idx = ret_pes.json_index
                if owns_capture:
                    write_jit_capture_file(
                        'jit_Abs_elem_sparse_get_elem',
                        'Abs_elem_sparse_get_elem',
                        layer_index,
                        counter,
                        inside_while,
                        while_number,
                        while_iteration,
                        json_list
                    )
                return ret_pes
            
            elif self.types[key] == 'SymExp':
                start_indices = []
                end_indices = []
                blocks = []
                val_const_idx: int = -1
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].const.start_indices
                    end_indices = self.d[key].const.end_indices
                    blocks = self.d[key].const.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        const_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sym_exp_sparse_const',
                            'input': 'json_list_' + str(d_key_idx),
                            'output': const_idx
                        }
                        json_list.append(json_obj)
                        blocks_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sparse_tensor_blocks',
                            'input': 'json_list_' + str(const_idx),
                            'output': blocks_idx
                        }
                        json_list.append(json_obj)
                        val_const = SparseTensor(start_indices, blocks, self.d[key].const.dims, total_size, end_indices, self.d[key].const.type, self.d[key].const.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                        val_const_idx = val_const.json_index
                    assert val_const_idx != -1
                else:
                    blocks_idx = len(json_list)
                    blocks_idx_original = blocks_idx
                    json_obj: dict[str, Any] = {
                        'method': 'initialise',
                        'name': 'res_blocks',
                        'value': '[]',
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        d_key_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_abs_elem_sparse_d_key',
                            'input': 'json_list_' + str(lhs_index),
                            'key': key,
                            'output': d_key_idx
                        }
                        json_list.append(json_obj)
                        const_idx = len(json_list)
                        json_obj: dict[str, Any] = {
                            'method': 'get_sym_exp_sparse_const',
                            'input': 'json_list_' + str(d_key_idx),
                            'output': const_idx
                        }
                        json_list.append(json_obj)
                        res = self.d[key].const.get_sparse_custom_range(
                            start_index, end_index, json_list=json_list,
                            layer_index=layer_index, counter=counter,
                            inside_while=inside_while, while_number=while_number,
                            while_iteration=while_iteration, lhs_index=const_idx
                        )
                        res_idx = -1
                        if dummy_mode:
                            res, res_idx = res
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                        for j in range(len(res.blocks)):
                            blk_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'extract_block',
                                'input': 'json_list_' + str(res_idx),
                                'block_id': j,
                                'output': blk_idx
                            }
                            json_list.append(json_obj)
                            new_blocks_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'append_list',
                                'list': 'json_list_' + str(blocks_idx),
                                'value': 'json_list_' + str(blk_idx),
                                'output': new_blocks_idx
                            }
                            blocks_idx = new_blocks_idx
                            json_list.append(json_obj)
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                    val_const_idx = val_const.json_index

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(
                        start_index, end_index, total_size,
                        json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_const_idx
                    )
                    if dummy_mode:
                        val_const, val_const_idx = val_const
                extra_dims = len(llist.initial_shape)-1

                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(
                        1, json_list=json_list, layer_index=layer_index,
                        counter=counter, inside_while=inside_while,
                        while_number=while_number, while_iteration=while_iteration,
                        lhs_index=val_const_idx,
                    )
                    val_const_idx = len(json_list) - 1

                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    assert val_const_idx != -1
                    val_const = val_const.repeat(
                        repeat_shape, json_list=json_list, lhs_index=val_const_idx
                    )
                    val_const_idx = len(json_list) - 1

                # mat part
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].mat.start_indices
                    end_indices = self.d[key].mat.end_indices
                    blocks = self.d[key].mat.blocks
                    d_key_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_abs_elem_sparse_d_key',
                        'input': 'json_list_' + str(lhs_index),
                        'key': key,
                        'output': d_key_idx
                    }
                    json_list.append(json_obj)
                    mat_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_sym_exp_sparse_mat',
                        'input': 'json_list_' + str(d_key_idx),
                        'output': mat_idx
                    }
                    json_list.append(json_obj)
                    blocks_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'get_sparse_tensor_blocks',
                        'input': 'json_list_' + str(mat_idx),
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, self.d[key].mat.total_size[-1]])
                        val_mat = SparseTensor(start_indices, blocks, self.d[key].mat.dims, total_size, end_indices, self.d[key].mat.type, self.d[key].mat.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                        val_mat_idx = val_mat.json_index
                else:
                    start_indices = []
                    end_indices = []
                    blocks = []
                    blocks_idx = len(json_list)
                    json_obj: dict[str, Any] = {
                        'method': 'initialise',
                        'name': 'mat_blocks',
                        'value': '[]',
                        'output': blocks_idx
                    }
                    json_list.append(json_obj)
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start, 0])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end, SymExpSparse.count])
                        blocks_ids, block_start_indices, block_end_indices = self.d[key].mat.get_block_id(start_index, end_index)
                        for i in range(len(blocks_ids)):
                            block = self.d[key].mat.blocks[blocks_ids[i]]
                            d_key_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'get_abs_elem_sparse_d_key',
                                'input': 'json_list_' + str(lhs_index),
                                'key': key,
                                'output': d_key_idx
                            }
                            json_list.append(json_obj)
                            mat_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'get_sym_exp_sparse_mat',
                                'input': 'json_list_' + str(d_key_idx),
                                'output': mat_idx
                            }
                            json_list.append(json_obj)
                            blk_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'extract_block',
                                'input': 'json_list_' + str(mat_idx),
                                'block_id': blocks_ids[i],
                                'output': blk_idx
                            }
                            json_list.append(json_obj)
                            start_indices.append(torch.tensor([0, block_start_indices[i][1], block_start_indices[i][2]]))
                            end_indices.append(torch.tensor([self.batch_size, block_end_indices[i][1], block_end_indices[i][2]]))
                            blocks.append(block)
                            new_blocks_idx = len(json_list)
                            json_obj: dict[str, Any] = {
                                'method': 'append_list',
                                'list': 'json_list_' + str(blocks_idx),
                                'value': 'json_list_' + str(blk_idx),
                                'output': new_blocks_idx
                            }
                            blocks_idx = new_blocks_idx
                            json_list.append(json_obj)

                    val_mat = SparseTensor(start_indices, blocks, 3, torch.tensor([self.batch_size, self.network[llist.llist[0]].size, self.d[key].mat.total_size[-1]]), end_indices, self.d[key].mat.type, self.d[key].mat.dense_const, og_json_list=json_list, blocks_index=blocks_idx)
                    val_mat_idx = val_mat.json_index

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, val_mat.total_size[-1]])
                    total_size = end_index - start_index
                    val_mat = val_mat.reduce_size(
                        start_index, end_index, total_size,
                        json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_mat_idx)
                    if dummy_mode:
                        val_mat, val_mat_idx = val_mat
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    val_mat = val_mat.unsqueeze(
                        1, json_list=json_list, layer_index=layer_index, counter=counter,
                        inside_while=inside_while, while_number=while_number,
                        while_iteration=while_iteration, lhs_index=val_mat_idx)
                    val_mat_idx = len(json_list) - 1

                repeat_shape = torch.tensor(llist.initial_shape + [1,1])
                if not (repeat_shape==1).all():
                    assert val_mat_idx != -1
                    val_mat = val_mat.repeat(
                        repeat_shape, json_list=json_list, lhs_index=val_mat_idx)
                    val_mat_idx = len(json_list) - 1

                end_time = time.time()
                get_elem_time.update_total_time(end_time-start_time)
                ret_ses = SymExpSparse(self.network, val_mat, val_const)
                ses_idx = len(json_list)
                json_obj: dict[str, Any] = {
                    'method': 'SymExpSparse',
                    'mat': 'json_list_' + str(val_mat_idx),
                    'const': 'json_list_' + str(val_const_idx),
                    'output': ses_idx,
                }
                json_list.append(json_obj)
                if owns_capture:
                    write_jit_capture_file(
                        'jit_Abs_elem_sparse_get_elem',
                        'Abs_elem_sparse_get_elem',
                        layer_index,
                        counter,
                        inside_while,
                        while_number,
                        while_iteration,
                        json_list
                    )
                return ret_ses
            
        else:
            raise NotImplementedError
            if self.types[key] == 'int' or self.types[key] == 'float' or self.types[key] == 'Int' or self.types[key] == 'Float':
                start_index = torch.tensor([0, llist.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, llist.network[llist.end].end])
                sp_tensor = self.d[key].get_sparse_custom_range(start_index, end_index)

                start_index = torch.tensor([0, self.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, self.network[llist.end].end])
                total_size = end_index - start_index

                end_time = time.time()
                get_elem_time.update_total_time(end_time-start_time)
                return sp_tensor.reduce_size(start_index, end_index, total_size)
            elif self.types[key] == 'PolyExp':
                start_index = torch.tensor([0, llist.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, llist.network[llist.end].end])
                val_const = self.d[key].const.get_sparse_custom_range(start_index, end_index)

                start_index = torch.tensor([0, self.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, self.network[llist.end].end])
                total_size = end_index - start_index
                val_const = val_const.reduce_size(start_index, end_index, total_size)

                start_index = torch.tensor([0, llist.network[llist.start].start, self.network[min(llist_compressed)].start])
                end_index = torch.tensor([self.batch_size, llist.network[llist.end].end, self.network[max(llist_compressed)].end])
                val_mat = self.d[key].mat.get_sparse_custom_range(start_index, end_index)

                start_index = torch.tensor([0, self.network[llist.start].start, 0])
                end_index = torch.tensor([self.batch_size, self.network[llist.end].end, val_mat.total_size[-1]])
                total_size = end_index - start_index
                val_mat = val_mat.reduce_size(start_index, end_index, total_size)

                end_time = time.time()
                get_elem_time.update_total_time(end_time-start_time)
                return PolyExpSparse(self.network, val_mat, val_const)
            
            elif self.types[key] == 'SymExp':
                raise Exception('NOT IMPLEMENTED')
            
    def update(self, llist, abs_shape, layer_index=None):
        # if dummy_mode:
        #     return self.update_dummy(llist, abs_shape)
        trace = bool(dummy_mode) and bool(fused_flow) and layer_index is not None
        snap = _update_snapshot(self, abs_shape) if trace else None
        llist.decoalesce()
        assert(len(llist.llist) == 1)
        if llist.llist_flag:
            layer = self.network[llist.llist[0]]
            for key, value in zip(list(self.d)[1:], abs_shape):
                if isinstance(value, (float, int, bool)):
                    continue
                bounds = value if self.types[key] in ('Float', 'Int', 'Bool') else value.const
                expected = [self.batch_size, layer.size]
                if bounds.total_size.tolist() != expected:
                    raise ValueError(
                        f"Layer {llist.llist[0]} ({layer.type.name}) field {key!r}: "
                        f"expected bounds shape {expected}, got {bounds.total_size.tolist()}"
                    )
        if llist.llist_flag:
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int', 'Bool']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = self.d[key].total_size
                    if isinstance(abs_shape[i], float) or isinstance(abs_shape[i], int) or isinstance(abs_shape[i], bool):
                        new_val_block = ConstBlock(abs_shape[i], end_index-start_index)
                        new_val = SparseTensor([torch.zeros(self.d[key].dims)], [new_val_block], self.d[key].dims, end_index-start_index, [end_index-start_index], self.d[key].type, abs_shape[i])
                        self.d[key] = self.d[key].overwrite_from_index(new_val, start_index)
                    else:
                        if abs_shape[i].dense_const != self.d[key].dense_const and (not abs_shape[i].check_dense()):
                            temp = SparseTensor(
                                [torch.tensor([0]*abs_shape[i].dims)],
                                [ConstBlock(abs_shape[i].dense_const, abs_shape[i].total_size)],
                                abs_shape[i].dims,
                                abs_shape[i].total_size,
                                [abs_shape[i].total_size],
                                abs_shape[i].type,
                                abs_shape[i].dense_const)
                            self.d[key] = self.d[key].overwrite_from_index(temp, start_index)
                            self.d[key] = self.d[key].overwrite_from_index(abs_shape[i], start_index)
                        else:
                            new_val = (abs_shape[i]).increase_size(start_index, total_size)
                            self.d[key] = self.d[key].overwrite(new_val)
                elif self.types[key] in ['PolyExp']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].const.total_size
                    const = abs_shape[i].const
                    if const.dense_const != self.d[key].const.dense_const and (not const.check_dense()):
                        temp_const = SparseTensor([torch.tensor([0]*const.dims)], [ConstBlock(const.dense_const, const.total_size)], const.dims, const.total_size, [const.total_size], const.type, const.dense_const)
                        self.d[key].const = self.d[key].const.overwrite_from_index(temp_const, start_index)
                        self.d[key].const = self.d[key].const.overwrite_from_index(const, start_index)
                    else:
                        self.d[key].const = self.d[key].const.overwrite((abs_shape[i].const).increase_size(start_index, total_size))

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    total_size = torch.tensor(list(self.d[key].mat.total_size))
                    mat = abs_shape[i].mat
                    if (mat.dense_const != self.d[key].mat.dense_const and (not mat.check_dense())):
                        temp_mat = SparseTensor([torch.tensor([0]*mat.dims)], [ConstBlock(mat.dense_const, mat.total_size)], mat.dims, mat.total_size, [mat.total_size], mat.type, mat.dense_const)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(temp_mat, start_index)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(mat, start_index)
                    else:
                        self.d[key].mat = self.d[key].mat.overwrite((abs_shape[i].mat).increase_size(start_index, total_size))

                elif self.types[key] in ['SymExp']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].const.total_size
                    const = abs_shape[i].const
                    if const.dense_const != self.d[key].const.dense_const and (not const.check_dense()):
                        temp_const = SparseTensor([torch.tensor([0]*const.dims)], [ConstBlock(const.dense_const, const.total_size)], const.dims, const.total_size, [const.total_size], const.type, const.dense_const)
                        self.d[key].const = self.d[key].const.overwrite_from_index(temp_const, start_index)
                        self.d[key].const = self.d[key].const.overwrite_from_index(const, start_index)
                    else:
                        self.d[key].const = self.d[key].const.overwrite((abs_shape[i].const).increase_size(start_index, total_size))

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    total_size = torch.tensor(list(self.d[key].mat.total_size))
                    total_size[-1] = SymExpSparse.count
                    self.d[key].mat.total_size[-1] = SymExpSparse.count
                    mat = abs_shape[i].mat
                    if mat.dense_const != self.d[key].mat.dense_const and (not mat.check_dense()):
                        temp_mat = SparseTensor([torch.tensor([0]*mat.dims)], [ConstBlock(mat.dense_const, mat.total_size)], mat.dims, mat.total_size, [mat.total_size], mat.type, mat.dense_const)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(temp_mat, start_index)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(mat, start_index)
                    else:
                        self.d[key].mat = self.d[key].mat.overwrite((abs_shape[i].mat).increase_size(start_index, total_size))

                else:
                    raise Exception(f'Unrecognized type {self.types[key]}')
            self.d['llist'][llist.llist] = True
            self.live_layers = torch.nonzero(self.d['llist']).flatten().tolist()
            if trace:
                _write_update_capture(self, abs_shape, layer_index, snap)
        else:
            raise Exception('NOT NEEDED')

    def update_dummy(self, llist: Llist, abs_shape):
        llist.decoalesce()
        assert(len(llist.llist) == 1)
        if llist.llist_flag:
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int', 'Bool']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = self.d[key].total_size
                    if isinstance(abs_shape[i], float) or isinstance(abs_shape[i], int) or isinstance(abs_shape[i], bool):
                        new_val_block = ConstBlock(1, end_index-start_index)
                        new_val = SparseTensor([torch.zeros(self.d[key].dims)], [new_val_block], self.d[key].dims, end_index-start_index, [end_index-start_index], self.d[key].type, abs_shape[i])
                        self.d[key] = self.d[key].overwrite_from_index(new_val, start_index)
                    else:
                        if abs_shape[i].dense_const != self.d[key].dense_const and (not abs_shape[i].check_dense()):
                            temp = SparseTensor(
                                [torch.tensor([0]*abs_shape[i].dims)],
                                [ConstBlock(1, abs_shape[i].total_size)],
                                abs_shape[i].dims,
                                abs_shape[i].total_size,
                                [abs_shape[i].total_size],
                                abs_shape[i].type,
                                abs_shape[i].dense_const)
                            self.d[key] = self.d[key].overwrite_from_index(temp, start_index)
                            self.d[key] = self.d[key].overwrite_from_index(abs_shape[i], start_index)
                        else:
                            new_val = (abs_shape[i]).increase_size(start_index, total_size)
                            self.d[key] = self.d[key].overwrite(new_val)
                elif self.types[key] in ['PolyExp']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].const.total_size
                    const = abs_shape[i].const
                    if const.dense_const != self.d[key].const.dense_const and (not const.check_dense()):
                        temp_const = SparseTensor([torch.tensor([0]*const.dims)], [ConstBlock(1, const.total_size)], const.dims, const.total_size, [const.total_size], const.type, const.dense_const)
                        self.d[key].const = self.d[key].const.overwrite_from_index(temp_const, start_index)
                        self.d[key].const = self.d[key].const.overwrite_from_index(const, start_index)
                    else:
                        self.d[key].const = self.d[key].const.overwrite((abs_shape[i].const).increase_size(start_index, total_size))

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    total_size = torch.tensor(list(self.d[key].mat.total_size))
                    mat = abs_shape[i].mat
                    if (mat.dense_const != self.d[key].mat.dense_const and (not mat.check_dense())):
                        temp_mat = SparseTensor([torch.tensor([0]*mat.dims)], [ConstBlock(1, mat.total_size)], mat.dims, mat.total_size, [mat.total_size], mat.type, mat.dense_const)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(temp_mat, start_index)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(mat, start_index)
                    else:
                        self.d[key].mat = self.d[key].mat.overwrite((abs_shape[i].mat).increase_size(start_index, total_size))

                elif self.types[key] in ['SymExp']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].const.total_size
                    const = abs_shape[i].const
                    if const.dense_const != self.d[key].const.dense_const and (not const.check_dense()):
                        temp_const = SparseTensor([torch.tensor([0]*const.dims)], [ConstBlock(1, const.total_size)], const.dims, const.total_size, [const.total_size], const.type, const.dense_const)
                        self.d[key].const = self.d[key].const.overwrite_from_index(temp_const, start_index)
                        self.d[key].const = self.d[key].const.overwrite_from_index(const, start_index)
                    else:
                        self.d[key].const = self.d[key].const.overwrite((abs_shape[i].const).increase_size(start_index, total_size))

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    total_size = torch.tensor(list(self.d[key].mat.total_size))
                    total_size[-1] = SymExpSparse.count
                    self.d[key].mat.total_size[-1] = SymExpSparse.count
                    mat = abs_shape[i].mat
                    if mat.dense_const != self.d[key].mat.dense_const and (not mat.check_dense()):
                        temp_mat = SparseTensor([torch.tensor([0]*mat.dims)], [ConstBlock(1, mat.total_size)], mat.dims, mat.total_size, [mat.total_size], mat.type, mat.dense_const)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(temp_mat, start_index)
                        self.d[key].mat = self.d[key].mat.overwrite_from_index(mat, start_index)
                    else:
                        self.d[key].mat = self.d[key].mat.overwrite((abs_shape[i].mat).increase_size(start_index, total_size))

                else:
                    raise Exception(f'Unrecognized type {self.types[key]}')
            self.d['llist'][llist.llist] = True
            self.live_layers = torch.nonzero(self.d['llist']).flatten().tolist()
        else:
            raise Exception('NOT NEEDED')
