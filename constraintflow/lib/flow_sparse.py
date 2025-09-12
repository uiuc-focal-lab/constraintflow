from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.polyexp import *
from constraintflow.lib.symexp import *
from constraintflow.lib.llist import *
from constraintflow.lib.network import Network, LayerType
from constraintflow.lib.globals import *
from constraintflow.gbcsr.sparse_tensor import SparseTensor
from constraintflow.gbcsr.sparse_block import DenseBlock, DiagonalBlock

import torch
import time



class Flow:
    def __init__(self, abs_elem: Abs_elem_sparse, transformer, model: Network, print_intermediate_results=False, no_sparsity=False):
        self.abs_elem = abs_elem 
        self.transformer = transformer 
        self.model = model
        self.input_size = model.input_size
        self.batch_size = abs_elem.batch_size
        self.print_intermediate_results = print_intermediate_results
        self.no_sparsity = no_sparsity

    def flow(self):
        begin_time = time.time()
        prev_size = self.model.input_size
        size = self.model.input_size
        

        for tmp, layer in enumerate(self.model):
            t_time = time.time()
            poly_size = self.model[list(torch.nonzero(self.abs_elem.d['llist']))[-1].item()].end
            curr_size = self.model[tmp].end-size

            if layer.type == LayerType.ReLU:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Sigmoid:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Sigmoid(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Linear:
                prev = Llist(self.model, [1, 1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)
                
            elif layer.type == LayerType.Conv2D:
                prev = Llist(self.model, [1, 1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Affine(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size)

            elif layer.type == LayerType.Input:
                continue
            elif layer.type == LayerType.Add:
                prev1 = Llist(self.model, [1], None, None, [layer.parents[0]])
                prev2 = Llist(self.model, [1], None, None, [layer.parents[1]])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = []
                for key in self.abs_elem.d.keys():
                    if key == 'llist':
                        continue
                    elif isinstance(self.abs_elem.d[key], SparseTensor):
                        res = self.abs_elem.get_elem(key, prev1).binary(self.abs_elem.get_elem(key, prev2), operator.add)
                        abs_shape.append(res)
                    elif isinstance(self.abs_elem.d[key], PolyExpSparse) or isinstance(self.abs_elem.d[key], SymExpSparse):
                        exp1 = self.abs_elem.get_elem(key, prev1)
                        exp2 = self.abs_elem.get_elem(key, prev2)
                        const1 = exp1.get_const()
                        const2 = exp2.get_const()
                        const = const1.binary(const2, operator.add)
                        mat1 = exp1.get_mat(self.abs_elem)
                        mat2 = exp2.get_mat(self.abs_elem)
                        mat = mat1.binary(mat2, operator.add)
                        if isinstance(self.abs_elem.d[key], PolyExpSparse):
                            abs_shape.append(PolyExpSparse(self.model, mat, const))
                        elif isinstance(self.abs_elem.d[key], SymExpSparse):
                            abs_shape.append(SymExpSparse(self.model, mat, const))
            elif layer.type == LayerType.Concat:
                debug_flag.set_flag()
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = []
                for key in self.abs_elem.d.keys():
                    if key == 'llist':
                        continue
                    if isinstance(self.abs_elem.d[key], SparseTensor):
                        start_indices = []
                        end_indices = []
                        blocks = []
                        new_start_index = 0
                        for par in layer.parents:
                            start_index = torch.tensor([0, self.model[par].start])
                            end_index = torch.tensor([self.batch_size, self.model[par].end])
                            block_id = self.abs_elem.d[key].get_block_id(start_index, end_index)[0][0]
                            block = self.abs_elem.d[key].blocks[block_id]
                            start_indices.append(torch.tensor([0, new_start_index]))
                            end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size]))
                            new_start_index += self.model[par].size
                            blocks.append(block)
                        abs_shape.append(SparseTensor(start_indices, blocks, 2, torch.tensor([self.batch_size, self.model[tmp].size])))
                    elif isinstance(self.abs_elem.d[key], PolyExpSparse) or isinstance(self.abs_elem.d[key], SymExpSparse):
                        debug_flag.set_flag()
                        start_indices = []
                        end_indices = []
                        blocks = []
                        new_start_index = 0
                        for par in layer.parents:
                            start_index = torch.tensor([0, self.model[par].start])
                            end_index = torch.tensor([self.batch_size, self.model[par].end])
                            block_id = self.abs_elem.d[key].const.get_block_id(start_index, end_index)[0][0]
                            block = self.abs_elem.d[key].const.blocks[block_id]
                            start_indices.append(torch.tensor([0, new_start_index]))
                            end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size]))
                            new_start_index += self.model[par].size
                            blocks.append(block)
                        const = SparseTensor(start_indices, blocks, 2, torch.tensor([self.batch_size, self.model[tmp].size]))

                        if isinstance(self.abs_elem.d[key], PolyExpSparse):
                            start_indices = []
                            end_indices = []
                            blocks = []
                            new_start_index = 0
                            for par in layer.parents:
                                start_index = torch.tensor([0, self.model[par].start, 0])
                                end_index = torch.tensor([self.batch_size, self.model[par].end, self.abs_elem.d[key].mat.total_size[-1]])
                                block_ids, block_start_indices, block_end_indices = self.abs_elem.d[key].mat.get_block_id(start_index, end_index)
                                for i in range(len(block_ids)):
                                    block_id = block_ids[i]
                                    block = self.abs_elem.d[key].mat.blocks[block_id]
                                    start_indices.append(torch.tensor([0, new_start_index, block_start_indices[i][2]]))
                                    end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size, block_end_indices[i][2]]))
                                    blocks.append(block)
                                new_start_index += self.model[par].size
                            total_size = torch.tensor([self.batch_size, self.model[tmp].size, self.abs_elem.d[key].mat.total_size[-1]])
                            mat = SparseTensor(start_indices, blocks, 3, total_size)
                            abs_shape.append(PolyExpSparse(self.model, mat, const))
                        elif isinstance(self.abs_elem.d[key], SymExpSparse):
                            start_indices = []
                            end_indices = []
                            blocks = []
                            new_start_index = 0
                            for p, par in enumerate(layer.parents):
                                start_index = torch.tensor([0, self.model[par].start, 0])
                                end_index = torch.tensor([self.batch_size, self.model[par].end, SymExpSparse.count])
                                block_ids, block_start_indices, block_end_indices = self.abs_elem.d[key].mat.get_block_id(start_index, end_index)
                                for i in range(len(block_ids)):
                                    block_id = block_ids[i]
                                    block = self.abs_elem.d[key].mat.blocks[block_id]
                                    if p==0:
                                        
                                        start_indices.append(torch.tensor([0, new_start_index, block_start_indices[i][2]]))
                                        end_indices.append(torch.tensor([self.batch_size, new_start_index+self.model[par].size, block_end_indices[i][2]]))
                                        blocks.append(block)
                                    if p==1:
                                        end_indices[i] = torch.tensor([0, end_indices[i][1]+new_start_index, block_start_indices[i][2]])
                                        if isinstance(block, DenseBlock):
                                            blocks[i] = DenseBlock(torch.concat([blocks[i].block, block.block], dim=1))
                                        elif isinstance(block, DiagonalBlock):
                                            new_total_shape = torch.tensor([block.total_shape[0].item(), block.total_shape[1].item() + blocks[i].total_shape[1].item(), max(block.total_shape[2].item(), blocks[i].total_shape[2].item())])
                                            blocks[i] = DiagonalBlock(torch.concat([blocks[i].block, block.block], dim=1), new_total_shape, diag_index=block.diag_index)
                                new_start_index += self.model[par].size
                            total_size = torch.tensor([self.batch_size, new_start_index+self.model[tmp].size, SymExpSparse.count])
                            mat = SparseTensor(start_indices, blocks, 3, total_size, end_indices=end_indices)
                            abs_shape.append(SymExpSparse(self.model, mat, const))
                    else:
                        assert(False)
            else:
                print(layer.type)
                assert(False)
            size += curr_size
            prev_size = self.model[tmp].size
            self.abs_elem.update(curr, abs_shape)

            if self.print_intermediate_results:
                print(tmp+1, layer.type, layer.shape)
                print(time.time()-t_time)
                print('---------------------------')
                print(f'l: {lb}')
                print(f'u: {ub}')
        lb = (abs_shape[0].get_dense())
        ub = (abs_shape[1].get_dense())
        return lb, ub

