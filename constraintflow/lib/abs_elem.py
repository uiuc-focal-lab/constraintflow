import torch
import copy
from constraintflow.lib.polyexp import *
from constraintflow.lib.symexp import *
from constraintflow.lib.llist import Llist

class Abs_elem_sparse:
    def __init__(self, d, types, network, batch_size=1, no_sparsity=False):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        self.d = d
        self.types = types 
        self.network = network
        self.batch_size = batch_size
        self.no_sparsity = no_sparsity
    
    def filter_non_live(self, llist):
        start_time = time.time()
        live_layers = torch.nonzero(self.d['llist']).flatten().tolist()
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
        l = list(torch.nonzero(self.d['llist']))[-1].item()
        return self.network[l].end
        

    def get_elem(self, key, llist):
        start_time = time.time()
        llist = self.filter_non_live(llist)
        llist_compressed = torch.nonzero(self.d['llist']).flatten().tolist()
        if llist.llist_flag:
            if self.types[key] in ['int', 'float', 'Int', 'Float', 'bool', 'Bool']:
                start_indices = []
                end_indices = []
                blocks = []
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].start_indices
                    end_indices = self.d[key].end_indices
                    blocks = self.d[key].blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        val_const = SparseTensor(start_indices, blocks, self.d[key].dims, total_size, end_indices, self.d[key].type, self.d[key].dense_const)
                else:
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        res = self.d[key].get_sparse_custom_range(start_index, end_index)
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const)
                    

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(1)

                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    val_const = val_const.repeat(repeat_shape)
                return val_const
            elif self.types[key] == 'PolyExp':
                start_indices = []
                end_indices = []
                blocks = []
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].const.start_indices
                    end_indices = self.d[key].const.end_indices
                    blocks = self.d[key].const.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        val_const = SparseTensor(start_indices, blocks, self.d[key].const.dims, total_size, end_indices, self.d[key].const.type, self.d[key].const.dense_const)
                else:
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        res = self.d[key].const.get_sparse_custom_range(start_index, end_index)
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const)
                    

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(1)

                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    val_const = val_const.repeat(repeat_shape)


                

                if llist_compressed == llist.llist:
                    start_indices = self.d[key].mat.start_indices
                    end_indices = self.d[key].mat.end_indices
                    blocks = self.d[key].mat.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, self.d[key].mat.total_size[-1]])
                        val_mat = SparseTensor(start_indices, blocks, self.d[key].mat.dims, total_size, end_indices, self.d[key].mat.type, self.d[key].mat.dense_const)
                else:
                    start_indices = []
                    end_indices = []
                    blocks = []
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start, self.network[min(llist_compressed)].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end, self.network[max(llist_compressed)].end])
                        blocks_ids, block_start_indices, block_end_indices = self.d[key].mat.get_block_id(start_index, end_index)
                        for i in range(len(blocks_ids)):
                            block = self.d[key].mat.blocks[blocks_ids[i]]
                            start_indices.append(torch.tensor([0, llist.network[l].start, block_start_indices[i][2]]))
                            end_indices.append(torch.tensor([self.batch_size, llist.network[l].end, block_end_indices[i][2]]))
                            blocks.append(block)
                    
                    val_mat = SparseTensor(start_indices, blocks, len(start_indices[0]), torch.tensor([self.batch_size, self.network[llist.llist[0]].size, self.d[key].mat.total_size[-1]]), end_indices, self.d[key].mat.type, self.d[key].mat.dense_const)
                    
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, val_mat.total_size[-1]])
                    total_size = end_index - start_index
                    val_mat = val_mat.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    val_mat = val_mat.unsqueeze(1)
                
                repeat_shape = torch.tensor(llist.initial_shape + [1,1])
                if not (repeat_shape==1).all():
                    val_mat = val_mat.repeat(repeat_shape)
                
                end_time = time.time()
                
                get_elem_time.update_total_time(end_time-start_time)
                return PolyExpSparse(self.network, val_mat, val_const)
            
            elif self.types[key] == 'SymExp':
                start_indices = []
                end_indices = []
                blocks = []
                if llist_compressed == llist.llist:
                    start_indices = self.d[key].const.start_indices
                    end_indices = self.d[key].const.end_indices
                    blocks = self.d[key].const.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                        val_const = SparseTensor(start_indices, blocks, self.d[key].const.dims, total_size, end_indices, self.d[key].const.type, self.d[key].const.dense_const)
                else:
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end])
                        res = self.d[key].const.get_sparse_custom_range(start_index, end_index)
                        start_indices += res.start_indices
                        end_indices += res.end_indices
                        blocks += res.blocks
                    val_const = SparseTensor(start_indices, blocks, res.dims, res.total_size, end_indices, res.type, res.dense_const)
                    

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                    total_size = end_index - start_index
                    val_const = val_const.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(1)

                repeat_shape = torch.tensor(llist.initial_shape + [1])
                if not (repeat_shape==1).all():
                    val_const = val_const.repeat(repeat_shape)


                
                start_time_2 = time.time()

                if llist_compressed == llist.llist:
                    start_indices = self.d[key].mat.start_indices
                    end_indices = self.d[key].mat.end_indices
                    blocks = self.d[key].mat.blocks
                    if 0 in llist.llist:
                        total_size = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, self.d[key].mat.total_size[-1]])
                        val_mat = SparseTensor(start_indices, blocks, self.d[key].mat.dims, total_size, end_indices, self.d[key].mat.type, self.d[key].mat.dense_const)
                else:
                    start_indices = []
                    end_indices = []
                    blocks = []
                    for l in llist.llist:
                        start_index = torch.tensor([0, llist.network[l].start, 0])
                        end_index = torch.tensor([self.batch_size, llist.network[l].end, SymExpSparse.count])
                        blocks_ids, block_start_indices, block_end_indices = self.d[key].mat.get_block_id(start_index, end_index)
                        for i in range(len(blocks_ids)):
                            block = self.d[key].mat.blocks[blocks_ids[i]]
                            start_indices.append(torch.tensor([0, block_start_indices[i][1], block_start_indices[i][2]]))
                            end_indices.append(torch.tensor([self.batch_size, block_end_indices[i][1], block_end_indices[i][2]]))
                            blocks.append(block)
                    val_mat = SparseTensor(start_indices, blocks, 3, torch.tensor([self.batch_size, self.network[llist.llist[0]].size, self.d[key].mat.total_size[-1]]), end_indices, self.d[key].mat.type, self.d[key].mat.dense_const)
                    
                    
                    
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, val_mat.total_size[-1]])
                    total_size = end_index - start_index
                    val_mat = val_mat.reduce_size(start_index, end_index, total_size)
                mid_time = time.time()
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    val_mat = val_mat.unsqueeze(1)
                t2 = time.time()
                
                repeat_shape = torch.tensor(llist.initial_shape + [1,1])
                if not (repeat_shape==1).all():
                    val_mat = val_mat.repeat(repeat_shape)
                
                end_time = time.time()
                
                get_elem_time.update_total_time(end_time-start_time)
                return SymExpSparse(self.network, val_mat, val_const)
            
        else:
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
            
    def update(self, llist, abs_shape):
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
                        new_val_block = ConstBlock(abs_shape[i], end_index-start_index)
                        new_val = SparseTensor([torch.zeros(self.d[key].dims)], [new_val_block], self.d[key].dims, end_index-start_index, [end_index-start_index], self.d[key].type, abs_shape[i])
                        self.d[key] = self.d[key].overwrite_from_index(new_val, start_index)
                    else:
                        if abs_shape[i].dense_const != self.d[key].dense_const and (not abs_shape[i].check_dense()):
                            temp = SparseTensor([torch.tensor([0]*abs_shape[i].dims)], [ConstBlock(abs_shape[i].dense_const, abs_shape[i].total_size)], abs_shape[i].dims, abs_shape[i].total_size, [abs_shape[i].total_size], abs_shape[i].type, abs_shape[i].dense_const)
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
        else:
            raise Exception('NOT NEEDED')