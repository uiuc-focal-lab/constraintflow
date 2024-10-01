import torch
import copy
import functools

def compute_size(shape):
    s = 1
    while len(shape)>0:
        s *= shape[0]
        shape = shape[1:]
    return s


def convert_dense_to_sparse(tensor):
    type=float 
    dense_const = 0.0
    if tensor.dtype == torch.bool:
        type = bool
        dense_const = False
    return SparseTensorBlock([torch.zeros(tensor.dim(), dtype=torch.int64)], [tensor], tensor.dim(), torch.tensor(tensor.shape), type=type, dense_const=dense_const)

def sort_tuple(temp):
    def lex_cmp(a, b):
        diff = a[0] - b[0]
        for i in range(len(diff)):
            if diff[i]>0:
                return 1
            elif diff[i]<0:
                return -1
        print(temp)
        assert(False)
        return 0
    sorted(temp, key=functools.cmp_to_key(lex_cmp))
    return temp

class SparseTensorBlock:
    def __init__(self, start_indices, blocks, dims, total_size, end_indices = None, type=float, dense_const = 0.0):
        self.start_indices = start_indices
        self.blocks = blocks
        self.total_size = total_size
        self.dims = dims
        assert(dims == total_size.shape[0])
        self.num_blocks = len(start_indices)
        self.type = type
        self.dense_const = dense_const
        assert(self.num_blocks == len(blocks))
        if self.num_blocks!=0:
            assert(self.dims == len(start_indices[0]))
        self.end_indices = end_indices 
        if self.end_indices is None:
            self.end_indices = []
            for i in range(self.num_blocks):
                self.end_indices.append(start_indices[i] + torch.tensor(blocks[i].shape))
                assert((self.end_indices[i] <= self.total_size).all())
        for i in range(self.num_blocks):
            for j in range(i+1, self.num_blocks):
                assert(not(self.block_overlap([start_indices[i], self.end_indices[i]], [start_indices[j], self.end_indices[j]])))
        
        
        if self.num_blocks > 0:
            if (self.end_indices[0]-self.start_indices[0] == self.total_size).all():
                if self.type==float:
                    self.dense_const = 0.0
                elif self.type==bool:
                    self.dense_const = False


    def compare_index(index_1, index_2):
        d = index_1 - index_2
        flag = True 
        for j in range(len(d)):
            if d[j] > 0:
                return 1
            elif d[j] < 0:
                return -1
        return 0

    def interval_overlap(self, interval_1, interval_2):
        if interval_1[1] <= interval_2[0] or interval_2[1] <= interval_1[0]:
            return False 
        return True
    
    def block_overlap(self, block_1, block_2):
        block_1_start = block_1[0]
        block_1_end = block_1[1]
        block_2_start = block_2[0]
        block_2_end = block_2[1]

        dims = len(block_1_start)
        
        for i in range(dims):
            if not self.interval_overlap([block_1_start[i], block_1_end[i]], [block_2_start[i], block_2_end[i]]):
                return False 
        return True
    
    def full_overlap(self, block_1, block_2):
        block_1_start = block_1[0]
        block_1_end = block_1[1]
        block_2_start = block_2[0]
        block_2_end = block_2[1]
        
        if block_1_start == block_2_start and block_1_end == block_2_end:
            return True
        return False
    
    def contained(self, block_1, block_2):
        block_1_start = block_1[0]
        block_1_end = block_1[1]
        block_2_start = block_2[0]
        block_2_end = block_2[1]
        
        if (block_1_start >= block_2_start).all().item() and (block_1_end <= block_2_end).all().item():
            return True
        return False

    def union_block(self, block_1, block_2):
        block_1_start = block_1[0]
        block_1_end = block_1[1]
        block_2_start = block_2[0]
        block_2_end = block_2[1]
        res_start = torch.min(block_1_start, block_2_start)
        res_end = torch.max(block_1_end, block_2_end)
        
        return res_start, res_end
    
    def intersection_block(self, block_1, block_2):
        if not self.block_overlap(block_1, block_2):
            raise Exception('No Overlap')
        block_1_start = block_1[0]
        block_1_end = block_1[1]
        block_2_start = block_2[0]
        block_2_end = block_2[1]
        res_start = torch.max(block_1_start, block_2_start)
        res_end = torch.min(block_1_end, block_2_end)

        
        return res_start, res_end
    
    def get_slice(self, start_index, end_index):
        dims = start_index.shape[0]
        s = []
        for j in range(dims):
            s.append(slice(int(start_index[j]), int(end_index[j])))
        return s
    
    def get_dense(self):
        res = torch.ones(list(self.total_size), dtype=self.type)*self.dense_const
        for i in range(self.num_blocks):
            s = self.get_slice(self.start_indices[i], self.end_indices[i])
            res[s] = self.blocks[i]
        return res
    
    def get_dense_custom_range(self, start_index, end_index):
        if self.dense_const == 0.0 or self.dense_const == False:    
            res = torch.zeros((end_index-start_index).to(int).tolist(), dtype=self.type)
        else:
            res = self.dense_const * torch.ones((end_index-start_index).to(int).tolist(), dtype=self.type)
        res_block = [start_index, end_index]
        for i in range(self.num_blocks):
            if self.block_overlap([self.start_indices[i], self.end_indices[i]], res_block):
                assert(start_index.dtype == torch.int64)
                intersection_start_indices, intersection_end_indices = self.intersection_block([self.start_indices[i], self.end_indices[i]], res_block)
                src_start_indices = intersection_start_indices - self.start_indices[i]
                src_end_indices = intersection_end_indices - self.start_indices[i]
                dst_start_indices = intersection_start_indices - start_index
                dst_end_indices = intersection_end_indices - start_index

                src_block = self.blocks[i][self.get_slice(src_start_indices, src_end_indices)]
                res[self.get_slice(dst_start_indices, dst_end_indices)] = src_block
        return res
    
    def get_sparse_custom_range(self, start_index, end_index):
        new_index = [start_index, end_index]
        blocks = []
        res_start_indices = []
        res_end_indices = []
        for i in range(self.num_blocks):
            if self.block_overlap([self.start_indices[i], self.end_indices[i]], new_index):
                intersection_start_indices, intersection_end_indices = self.intersection_block([self.start_indices[i], self.end_indices[i]], new_index)
                res_start_indices.append(intersection_start_indices)
                res_end_indices.append(intersection_end_indices)
                src_start_indices = intersection_start_indices - self.start_indices[i]
                src_end_indices = intersection_end_indices - self.start_indices[i]
                src_block = self.blocks[i][self.get_slice(src_start_indices, src_end_indices)]
                blocks.append(src_block)
        return SparseTensorBlock(res_start_indices, blocks, self.dims, copy.deepcopy(self.total_size), res_end_indices)
    


    def reduce_size(self, start_index, end_index, total_size):
        assert((total_size == end_index - start_index).all())
        start_indices = []
        end_indices = []
        for i in range(self.num_blocks):
            assert(self.contained([self.start_indices[i], self.end_indices[i]], [start_index, end_index]))
            start_indices.append((self.start_indices[i]-start_index).type(torch.int64))
            end_indices.append((self.end_indices[i]-start_index).type(torch.int64))
        return SparseTensorBlock(start_indices, copy.deepcopy(self.blocks), self.dims, total_size, end_indices) 

    def increase_size(self, start_index, new_total_size):
        assert((self.total_size <= new_total_size).all())
        start_indices = []   
        end_indices = []   
        for i in range(self.num_blocks):
            start_indices.append(start_index + self.start_indices[i])   
            end_indices.append(start_index + self.end_indices[i])
        return SparseTensorBlock(start_indices, copy.deepcopy(self.blocks), self.dims, new_total_size, end_indices)

    
    
    
    def merge_no_overlap(self, sp_tensor):
        assert(self.dims == sp_tensor.dims)
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
        return SparseTensorBlock(start_indices, blocks, self.dims, copy.deepcopy(self.total_size), end_indices)

    
    def union_tensors(self, sp_tensor):
        if self.num_blocks == 0:
            return copy.deepcopy(sp_tensor.start_indices), copy.deepcopy(sp_tensor.end_indices)
        if sp_tensor.num_blocks == 0:
            return copy.deepcopy(self.start_indices), copy.deepcopy(self.end_indices)
        assert(self.type == sp_tensor.type)
        # assert(self.dense_const == sp_tensor.dense_const)
        assert((self.total_size == sp_tensor.total_size).all())
        overlap_classes = dict()
        for i in range(self.num_blocks):
            overlap_flag = False
            for j in range(sp_tensor.num_blocks):
                if self.block_overlap([self.start_indices[i], self.end_indices[i]], [sp_tensor.start_indices[j], sp_tensor.end_indices[j]]):
                    overlap_flag = True
                    flag = False
                    for key in overlap_classes.keys():
                        if i==key:
                            flag = True
                            if j not in overlap_classes[key][1]:
                                overlap_classes[key][1].append(j)
                        elif i in overlap_classes[key][0]:
                            flag = True
                            if j not in overlap_classes[key][1]:
                                overlap_classes[key][1].append(j)
                        elif j in overlap_classes[key][1]:
                            flag = True 
                            if i not in overlap_classes[key][0]:
                                overlap_classes[key][0].append(i)
                    if not flag:
                        overlap_classes[i] = ([], [j])
            if not overlap_flag:
                overlap_classes[i] = ([], [])
        
        res_start_indices = []
        res_end_indices = []
        
        for j in range(sp_tensor.num_blocks):
            flag = False
            for key in overlap_classes.keys():
                if j in overlap_classes[key][1]:
                    flag = True
                    break
            if not flag:
                res_start_indices.append(sp_tensor.start_indices[j])
                res_end_indices.append(sp_tensor.end_indices[j])
        
        

        for i, key in enumerate(overlap_classes.keys()):
            start_index = self.start_indices[key]
            end_index = self.end_indices[key]
            for j in overlap_classes[key][0]:
                start_index, end_index = self.union_block([start_index, end_index], [self.start_indices[j], self.end_indices[j]])
            for j in overlap_classes[key][1]:
                start_index, end_index = self.union_block([start_index, end_index], [sp_tensor.start_indices[j], sp_tensor.end_indices[j]])
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
        
        temp = [(res_start_indices[i], res_end_indices[i]) for i in range(len(res_start_indices))]
        sort_tuple(temp)
        res_start_indices = [i[0] for i in temp]
        res_end_indices = [i[1] for i in temp]
        
        return res_start_indices, res_end_indices
    
    def intersection_tensors(self, sp_tensor):
        assert(self.type == sp_tensor.type)
        assert(self.dense_const == sp_tensor.dense_const)
        assert((self.total_size == sp_tensor.total_size).all())

        res_start_indices = []
        res_end_indices = []

        for i in range(self.num_blocks):
            for j in range(sp_tensor.num_blocks):
                if self.block_overlap([self.start_indices[i], self.end_indices[i]], [sp_tensor.start_indices[j], sp_tensor.end_indices[j]]):
                    start_index, end_index = self.intersection_block([self.start_indices[i], self.end_indices[i]], [sp_tensor.start_indices[j], sp_tensor.end_indices[j]])
                    res_start_indices.append(start_index)
                    res_end_indices.append(end_index)

        return res_start_indices, res_end_indices

    def copy(self):
        return SparseTensorBlock(copy.deepcopy(self.start_indices), copy.deepcopy(self.blocks), self.dims, self.total_size, copy.deepcopy(self.end_indices), dense_const=self.dense_const, type = self.type)
    
    def is_const(self):
        return self.num_blocks==0
    
    def check_dense(self):
        def mult_list(l):
            if len(l)==1:
                return l[0]
            return mult_list(l[1:])*l[0]
        t = 0
        for i in range(self.num_blocks):
            t += mult_list(list(self.blocks[i].size())) 
        if t < mult_list(list(self.total_size)):
            return False
        return True

    def boolneg(self):
        assert(self.type == bool)
        res = self.copy()
        res.dense_const = not(self.dense_const)
        for i in range(res.num_blocks):
            res.blocks[i] = ~(self.blocks[i])
        return res

    def binary(self, sp_tensor, op):
        if (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()!=1):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        if isinstance(sp_tensor, SparseTensorBlock):
            assert((self.total_size == sp_tensor.total_size).all())
            assert(self.dims == sp_tensor.dims)
            if op == '*':
                if self.dense_const == 0 and sp_tensor.dense_const == 0:
                    res_start_indices, res_end_indices = self.intersection_tensors(sp_tensor)
                elif self.dense_const == 0:
                    res_start_indices, res_end_indices = self.start_indices, self.end_indices
                elif sp_tensor.dense_const == 0:
                    res_start_indices, res_end_indices = sp_tensor.start_indices, sp_tensor.end_indices
                else:
                    res_start_indices, res_end_indices = self.union_tensors(sp_tensor)
            elif op == '/':
                if sp_tensor.dense_const==0 and not(sp_tensor.check_dense()):
                    raise Exception('DIVISION BY ZERO')
                elif self.dense_const == 0:
                    res_start_indices, res_end_indices = self.start_indices,self.end_indices
                else:
                    res_start_indices, res_end_indices = self.union_tensors(sp_tensor)
            else:
                res_start_indices, res_end_indices = self.union_tensors(sp_tensor)
            blocks = []
            type = self.type
            dense_const = self.dense_const
            for i in range(len(res_start_indices)):
                block_1 = self.get_dense_custom_range(res_start_indices[i], res_end_indices[i])
                block_2 = sp_tensor.get_dense_custom_range(res_start_indices[i], res_end_indices[i])
                if op == '+':
                    block = block_1 + block_2 
                    dense_const = self.dense_const + sp_tensor.dense_const
                elif op == '-':
                    block = block_1 - block_2
                    dense_const = self.dense_const - sp_tensor.dense_const
                elif op == '*':
                    block = block_1 * block_2 
                    dense_const = self.dense_const * sp_tensor.dense_const
                elif op == '/':
                    block = block_1 / block_2
                    if sp_tensor.check_dense():
                        dense_const = 0.0
                    else:
                        dense_const = self.dense_const / sp_tensor.dense_const
                elif op == '<':
                    block = block_1 < block_2
                    type = bool
                    dense_const = self.dense_const < sp_tensor.dense_const
                elif op == '>':
                    type = bool
                    block = block_1 > block_2
                    dense_const = self.dense_const > sp_tensor.dense_const
                elif op == '<=':
                    type = bool
                    block = block_1 <= block_2
                    dense_const = self.dense_const <= sp_tensor.dense_const
                elif op == '>=':
                    type = bool
                    block = block_1 >= block_2
                    dense_const = self.dense_const >= sp_tensor.dense_const
                elif op == '==':
                    type = bool
                    block = block_1 == block_2
                    dense_const = self.dense_const == sp_tensor.dense_const
                elif op == '!=':
                    type = bool
                    block = block_1 != block_2
                    dense_const = self.dense_const != sp_tensor.dense_const
                elif op == '&':
                    type = bool
                    block = block_1 & block_2
                    dense_const = self.dense_const & sp_tensor.dense_const
                elif op == '|':
                    type = bool
                    block = block_1 | block_2
                    dense_const = self.dense_const or sp_tensor.dense_const
                else:
                    raise Exception('CHECK OPERATION', op)
                blocks.append(block)
            return SparseTensorBlock(res_start_indices, blocks, self.dims, self.total_size, res_end_indices, type, dense_const)
        elif isinstance(sp_tensor, float) or isinstance(sp_tensor, int) or (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()==1):
            res = self.copy()
            type = self.type
            dense_const = self.dense_const
            if op=='*' and sp_tensor==0.0:
                res.blocks = []
                res.num_blocks = 0
                res.dense_const = 0
                res.start_indices = []
                res.end_indices = []
                return res
            for i in range(res.num_blocks):
                if op == '+':
                    res.blocks[i] = res.blocks[i] + sp_tensor 
                    dense_const = self.dense_const + sp_tensor
                elif op == '-':
                    res.blocks[i] = res.blocks[i] - sp_tensor
                    dense_const = self.dense_const - sp_tensor
                elif op == '*':
                    res.blocks[i] = res.blocks[i] * sp_tensor 
                    dense_const = self.dense_const * sp_tensor
                elif op == '/':
                    res.blocks[i] = res.blocks[i] / sp_tensor
                    dense_const = self.dense_const / sp_tensor
                elif op == '<':
                    type = bool
                    res.blocks[i] = res.blocks[i] < sp_tensor
                    dense_const = self.dense_const < sp_tensor
                elif op == '>':
                    type = bool
                    res.blocks[i] = res.blocks[i] > sp_tensor
                    dense_const = self.dense_const > sp_tensor
                elif op == '<=':
                    type = bool
                    res.blocks[i] = res.blocks[i] <= sp_tensor
                    dense_const = self.dense_const <= sp_tensor
                elif op == '>=':
                    type = bool
                    res.blocks[i] = res.blocks[i] >= sp_tensor
                    dense_const = self.dense_const >= sp_tensor
                elif op == '==':
                    type = bool
                    res.blocks[i] = res.blocks[i] == sp_tensor
                    dense_const = self.dense_const == sp_tensor
                elif op == '!=':
                    type = bool
                    res.blocks[i] = res.blocks[i] != sp_tensor
                    dense_const = self.dense_const != sp_tensor
                elif op == '&':
                    type = bool
                    res.blocks[i] = res.blocks[i] & sp_tensor
                    dense_const = self.dense_const & sp_tensor
                elif op == '|':
                    type = bool
                    res.blocks[i] = res.blocks[i] | sp_tensor
                    dense_const = self.dense_const | sp_tensor
                else:
                    raise Exception('CHECK OPERATION', op)
            res.type = type
            res.dense_const = dense_const
            return res
        
        
        else:
            raise Exception('CHECK TYPE OF THE TENSOR', type(sp_tensor))
        
    def any(self):
        if self.dense_const == True:
            return True
        if self.type != bool and self.dense_const != 0.0:
            return True
        for i in range(self.num_blocks):
            if self.blocks[i].any():
                return True
        return False
    
    def float(self):
        res = self.copy()
        res.type = float
        res.dense_const = float(res.dense_const)
        for i in range(res.num_blocks):
            res.blocks[i] = res.blocks[i].float()
        return res 
    
    def transpose(self):
        blocks = []
        start_indices = []
        end_indices = []
        for i in range(self.num_blocks):
            blocks.append(self.blocks[i].T)
            start_indices.append(torch.flip(self.start_indices[i], dims=[0]))
            end_indices.append(torch.flip(self.end_indices[i], dims=[0]))
        total_size = torch.flip(self.total_size, dims=[0])
        return SparseTensorBlock(start_indices, blocks, self.dims, total_size, end_indices)
    
    def matmul(self, sp_tensor):
        return self.matmul_new(sp_tensor)

    def matmul_old(self, sp_tensor):
        if isinstance(sp_tensor, torch.Tensor):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        mat_1 = self.get_dense()
        mat_2 = sp_tensor.get_dense()

        assert(self.type == float or self.type == int)
        assert(sp_tensor.type == float or sp_tensor.type == int)
        assert(self.dense_const == 0.0)
        assert(sp_tensor.dense_const == 0.0)
        
        if sp_tensor.dims <= 2:
            assert(sp_tensor.total_size[-1].item() == self.total_size[-1].item())
        else:
            assert(sp_tensor.total_size[-2].item() == self.total_size[-1].item())
            assert(sp_tensor.total_size[:-2] == self.total_size[-sp_tensor.dims:-2])
        
        
        start_indices = []
        end_indices = []
        blocks = []
        
        if self.dims == sp_tensor.dims:
            mat = mat_1 @ mat_2
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 2:
                        start_index = torch.concat([self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]])
                    else:
                        if not self.block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-2], sp_tensor.end_indices[j][:-2]]):
                            continue
                        
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-2], self.start_indices[i][:-2]), self.start_indices[i][-2:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-2], self.end_indices[i][:-2]), self.end_indices[i][-2:-1], sp_tensor.end_indices[j][-1:]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        elif self.dims > sp_tensor.dims:
            mat = (mat_1 @ mat_2.unsqueeze(-1)).squeeze(-1)
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 1:
                        start_index = self.start_indices[i][:-1]
                        end_index = self.end_indices[i][:-1]
                    else:
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-1], self.start_indices[i][:-2]), self.start_indices[i][-2:-1]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-1], self.end_indices[i][:-2]), self.end_indices[i][-2:-1]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        else:
            assert(False)

        # print(start_indices)
        # print(end_indices)
        
        
        overlap_classes = dict()
        for i in range(len(start_indices)):
            if i not in overlap_classes:
                flag = False
                for key in overlap_classes.keys():
                    if i in overlap_classes[key]:
                        flag = True 
                        break
                if not flag:
                    overlap_classes[i] = []
            for j in range(i+1, len(start_indices)):
                if self.block_overlap([start_indices[i], end_indices[i]], [start_indices[j], end_indices[j]]):
                    flag = False
                    for key in overlap_classes.keys():
                        if i==key:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                        elif i in overlap_classes[key]:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                    if not flag:
                        overlap_classes[i] = [j]
        res_start_indices = []
        res_end_indices = []
        blocks = []
        res = SparseTensorBlock([], [], mat.dim(), torch.tensor(mat.shape))
        for i, key in enumerate(overlap_classes.keys()):
            start_index = start_indices[key]
            end_index = end_indices[key]
            for j in overlap_classes[key]:
                start_index, end_index = self.union_block([start_index, end_index], [start_indices[j], end_indices[j]])
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            slice = res.get_slice(res_start_indices[-1], res_end_indices[-1])
            blocks.append(mat[slice])
        
        # print(start_indices)
        # print(end_indices)
        return SparseTensorBlock(res_start_indices, blocks, mat.dim(), torch.tensor(mat.size()), res_end_indices)
    
    def matmul_new(self, sp_tensor):
        if isinstance(sp_tensor, torch.Tensor):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        

        assert(self.type == float or self.type == int)
        assert(sp_tensor.type == float or sp_tensor.type == int)
        assert(self.dense_const == 0.0)
        assert(sp_tensor.dense_const == 0.0)
        
        if sp_tensor.dims <= 2:
            assert(sp_tensor.total_size[-1].item() == self.total_size[-1].item())
        else:
            assert(sp_tensor.total_size[-2].item() == self.total_size[-1].item())
            assert(sp_tensor.total_size[:-2] == self.total_size[-sp_tensor.dims:-2])
        
        
        start_indices = []
        end_indices = []
        blocks = []
        res_total_size = None
        
        if self.dims == sp_tensor.dims:
            res_total_size = torch.concat([self.total_size[:-1], sp_tensor.total_size[-1:]])
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 2:
                        start_index = torch.concat([self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]])
                    else:
                        if not self.block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-2], sp_tensor.end_indices[j][:-2]]):
                            continue
                        if not self.block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):
                            continue
                        block = self.blocks[i].type(torch.float) @ sp_tensor.blocks[j].type(torch.float)
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-2], self.start_indices[i][:-2]), self.start_indices[i][-2:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-2], self.end_indices[i][:-2]), self.end_indices[i][-2:-1], sp_tensor.end_indices[j][-1:]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        elif self.dims > sp_tensor.dims:
            res_total_size = self.total_size[:-1]
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 1:
                        start_index = self.start_indices[i][:-1]
                        end_index = self.end_indices[i][:-1]
                    else:
                        if not self.block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-1], sp_tensor.end_indices[j][:-1]]):
                            continue
                        if not self.block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            continue
                        block = (self.blocks[i].type(torch.float) @ sp_tensor.blocks[j].type(torch.float).unsqueeze(-1)).squeeze(-1)
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-1], self.start_indices[i][:-2]), self.start_indices[i][-2:-1]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-1], self.end_indices[i][:-2]), self.end_indices[i][-2:-1]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        else:
            assert(False)

        overlap_classes = dict()
        for i in range(len(start_indices)):
            if i not in overlap_classes:
                flag = False
                for key in overlap_classes.keys():
                    if i in overlap_classes[key]:
                        flag = True 
                        break
                if not flag:
                    overlap_classes[i] = []
            for j in range(i+1, len(start_indices)):
                if self.block_overlap([start_indices[i], end_indices[i]], [start_indices[j], end_indices[j]]):
                    flag = False
                    for key in overlap_classes.keys():
                        if i==key:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                        elif i in overlap_classes[key]:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                    if not flag:
                        overlap_classes[i] = [j]
        res_start_indices = []
        res_end_indices = []
        res_blocks = []
        for i, key in enumerate(overlap_classes.keys()):
            start_index = start_indices[key]
            end_index = end_indices[key]
            for j in overlap_classes[key]:
                start_index, end_index = self.union_block([start_index, end_index], [start_indices[j], end_indices[j]])
            block = SparseTensorBlock([start_index], [blocks[key]], len(start_index), res_total_size)
            for j in overlap_classes[key]:
                new_block = SparseTensorBlock([start_index], [blocks[j]], len(start_index), res_total_size)
                block = block.binary(new_block, '+')
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block.blocks[0])
        return SparseTensorBlock(res_start_indices, res_blocks, len(res_total_size), res_total_size, res_end_indices) 

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

    def overwrite_block(self, start_index, end_index, block):
        for i in range(self.num_blocks):
            if self.block_overlap([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                if self.contained([start_index, end_index], [self.start_indices[i], self.end_indices[i]]):
                    s = self.get_slice(start_index-self.start_indices[i], end_index-self.start_indices[i])
                    self.blocks[i][s] = block
                    return 
                else:
                    raise Exception('NOT IMPLEMENTED')
        self.add_block_no_overlap(start_index, end_index, block)

    def overwrite(self, sp_tensor):
        res = self.copy()
        for i in range(sp_tensor.num_blocks):
            res.overwrite_block(sp_tensor.start_indices[i], sp_tensor.end_indices[i], sp_tensor.blocks[i])
        return res
        
    def squeeze(self, index):
        assert(self.total_size[index].item() == 1)
        dims = self.dims-1
        total_size = torch.concat([self.total_size[:index], self.total_size[index+1:]])
        start_indices = []
        end_indices = []
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(self.blocks[i].squeeze(index))
            start_indices.append(torch.concat([self.start_indices[i][:index], self.start_indices[i][index+1:]]))
            end_indices.append(torch.concat([self.end_indices[i][:index], self.end_indices[i][index+1:]]))
        return SparseTensorBlock(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
    
    def unsqueeze(self, index):
        dims = self.dims+1
        total_size = torch.concat([self.total_size[:index], torch.tensor([1]), self.total_size[index:]])
        total_size = total_size.type(torch.int64)
        start_indices = []
        end_indices = []
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(self.blocks[i].unsqueeze(index))
            start_indices.append(torch.concat([self.start_indices[i][:index], torch.tensor([0]), self.start_indices[i][index:]]))
            end_indices.append(torch.concat([self.end_indices[i][:index], torch.tensor([1]), self.end_indices[i][index:]]))
        return SparseTensorBlock(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
    
    def repeat(self, repeat_dims):
        total_size = self.total_size * repeat_dims
        start_indices = []
        end_indices = []
        blocks = []
        for i in range(self.num_blocks):
            start_indices.append(self.start_indices[i]*repeat_dims)
            end_indices.append(self.end_indices[i]*repeat_dims)
            blocks.append(self.blocks[i].repeat(*repeat_dims))
        return SparseTensorBlock(start_indices, blocks, self.dims, total_size, end_indices, self.type, self.dense_const)

            
    def sum(self, dim):
        start_indices = []
        end_indices = []
        blocks = []
        total_size = torch.concat([self.total_size[:dim], self.total_size[dim+1:]])
        for i in range(self.num_blocks):
            start_indices.append(torch.concat([self.start_indices[i][:dim], self.start_indices[i][dim+1:]]))
            end_indices.append(torch.concat([self.end_indices[i][:dim], self.end_indices[i][dim+1:]]))
            blocks.append(self.blocks[i].sum(dim=dim))

        overlap_classes = dict()
        for i in range(len(start_indices)):
            if i not in overlap_classes:
                flag = False
                for key in overlap_classes.keys():
                    if i in overlap_classes[key]:
                        flag = True 
                        break
                if not flag:
                    overlap_classes[i] = []
            for j in range(i+1, len(start_indices)):
                if self.block_overlap([start_indices[i], end_indices[i]], [start_indices[j], end_indices[j]]):
                    flag = False
                    for key in overlap_classes.keys():
                        if i==key:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                        elif i in overlap_classes[key]:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                    if not flag:
                        overlap_classes[i] = [j]

        res_blocks = []
        res_start_indices = []
        res_end_indices = []
        for key in overlap_classes:
            start_index, end_index = start_indices[key], end_indices[key]
            for b in overlap_classes[key]:
                start_index, end_index = self.union_block([start_index, end_index], [start_indices[b], end_indices[b]])
            current_block = SparseTensorBlock([start_indices[key]], [blocks[key]], self.dims-1, total_size, [end_indices[key]], self.type, self.dense_const).get_dense_custom_range(start_index, end_index)
            for b in overlap_classes[key]:
                current_block += SparseTensorBlock([start_indices[b]], [blocks[b]], self.dims-1, total_size, [end_indices[b]], self.type, self.dense_const).get_dense_custom_range(start_index, end_index)
            res_blocks.append(current_block)
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)


        return SparseTensorBlock(res_start_indices, res_blocks, self.dims-1, total_size, res_end_indices, self.type, self.dense_const)
    
def sp_where(x:SparseTensorBlock, y:SparseTensorBlock, z:SparseTensorBlock):
    if x.dense_const:
        res = y.copy()
    else:
        res = z.copy()
    for i in range(x.num_blocks):
        start_index = x.start_indices[i]
        end_index = x.end_indices[i]
        block_y = y.get_dense_custom_range(start_index, end_index)
        block_z = z.get_dense_custom_range(start_index, end_index)
        block = torch.where(x.blocks[i], block_y, block_z)
        res.overwrite_block(start_index, end_index, block)
    return res