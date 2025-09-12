import torch
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

# def compute_size(shape):
#     s = 1
#     while len(shape)>0:
#         s *= shape[0]
#         shape = shape[1:]
#     return s


def custom_assert(x):
    return True

def convert_dense_to_sparse(x, total_shape=None):
    if isinstance(x, torch.Tensor):
        type=float 
        dense_const = 0.0
        if x.dtype == torch.bool:
            type = bool
            dense_const = False
        return SparseTensor([torch.zeros(x.dim(), dtype=torch.int64)], [DenseBlock(x)], x.dim(), torch.as_tensor(x.shape), type=type, dense_const=dense_const)
    elif isinstance(x, float):
        dense_const = x
        type = float
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x)
    elif isinstance(x, int):
        dense_const = x
        type = int
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x)
    elif isinstance(x, bool):
        dense_const = x
        type = bool
        return SparseTensor([], [], len(total_shape), total_shape, type=type, dense_const=x)
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
    def dfs(node, block):
        visited[node] = True
        block.append(node)
        for neighbor in range(n):
            if not visited[neighbor] and block_overlap([start_indices[node], end_indices[node]], [start_indices[neighbor], end_indices[neighbor]]):
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

def sp_tensor_from_overlap_classes(overlap_classes, start_indices, blocks, total_size, dims, dense_const, type):
    res_blocks = []
    res_start_indices = []
    res_end_indices = []
    for i in overlap_classes.keys():
        block = SparseTensor([start_indices[i]], [blocks[i]], dims, total_size)
        for j in overlap_classes[i]:
            new_block = SparseTensor([start_indices[j]], [blocks[j]], dims, total_size)
            block = block.binary(new_block, operator.add)
        res_start_indices += block.start_indices
        res_end_indices += block.end_indices
        res_blocks += block.blocks 
    res = SparseTensor(res_start_indices, res_blocks, dims, total_size, res_end_indices, type, dense_const)
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
    def __init__(self, start_indices, blocks, dims, total_size, end_indices = None, type=float, dense_const = 0.0):
        t1 = time.time()
        self.start_indices = start_indices
        self.blocks = blocks
        self.total_size = total_size
        self.dims = dims
        # assert(dims == total_size.shape[0])
        self.num_blocks = len(start_indices)
        self.type = type
        self.dense_const = dense_const
        # assert(self.num_blocks == len(blocks))
        # if self.num_blocks!=0:
        #     assert(self.dims == len(start_indices[0]))
        self.end_indices = end_indices 
        if self.end_indices is None:
            self.end_indices = []
            for i in range(self.num_blocks):
                self.end_indices.append(start_indices[i] + torch.as_tensor(blocks[i].total_shape))
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
        
        sparse_tensor_init_time.update_op_time(time.time()-t1)
        
        if self.num_blocks > 0:
            if (self.end_indices[0]-self.start_indices[0] == self.total_size).all():
                if self.type==float:
                    self.dense_const = 0.0
                elif self.type==bool:
                    self.dense_const = False
        
        delete_indices = []
        for i in range(self.num_blocks):
            if isinstance(self.blocks[i], ConstBlock):
                if self.blocks[i].block == self.dense_const and (self.dense_const == 0.0 or self.dense_const == False):
                    delete_indices.append(i)
        delete_indices.reverse()

        sparse_tensor_init_time.update_total_time(time.time()-t1)


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
        return res[:-2] + '\n'

    def expand_symexp_mat(self, symexp_count):
        assert(self.dense_const==0)
        self.total_size[-1] = symexp_count
        return self

    def get_dense(self):
        res = torch.ones(list(self.total_size), dtype=self.type)*self.dense_const
        for i in range(self.num_blocks):
            s = get_slice(self.start_indices[i], self.end_indices[i])
            res[tuple(s)] = self.blocks[i].get_dense()
        return res
    
    def get_dense_custom_range(self, start_index, end_index):
        if self.dense_const == 0.0 or self.dense_const == False:    
            res = torch.zeros((end_index - start_index).to(int).tolist(), dtype=self.type)
        else:
            res = self.dense_const * torch.ones((end_index - start_index).to(int).tolist(), dtype=self.type)

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
    def get_sub_block_custom_range(self, start_index, end_index, block_id, tensor=True):
        start_time = time.time()
        block_start_index = self.start_indices[block_id]
        sparse_block = self.blocks[block_id]
        if (block_start_index == start_index).all() and (self.end_indices[block_id] == end_index).all():
            if tensor:
                return sparse_block.block
            else:
                return sparse_block
        if isinstance(sparse_block, KernelBlock) or isinstance(sparse_block, PatchesBlock):
            if len(sparse_block.total_shape)==4:
                if (block_start_index[:-1] == start_index[:-1]).all() and (self.end_indices[block_id][:-1] == end_index[:-1]).all():
                    res = sparse_block.create_similar(sparse_block.block)
                    res.total_shape[-1] = end_index[-1] - start_index[-1]
                    return res
            raise NotImplementedError
        res = self.blocks[block_id].get_sub_block_custom_range(start_index, end_index, block_start_index)
        if tensor:
            return res.block
        end_time = time.time()
        sub_block_custom_range_time.update_total_time(end_time-start_time)
        return res
    
    def exists_block(self, start_index, end_index):
        for i in range(self.num_blocks):
            if full_overlap([self.start_indices[i], self.end_indices[i]], [start_index, end_index]):
                if isinstance(self.blocks[i], ConstBlock):
                    if self.blocks[i].block == self.dense_const and (self.dense_const == 0.0 or self.dense_const == False):
                        return False
                return True
        return False
    
    def exists_sub_block(self, start_index, end_index):
        for i in range(self.num_blocks):
            if contained([start_index, end_index], [self.start_indices[i], self.end_indices[i]]):
                if isinstance(self.blocks[i], ConstBlock):
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
    
    def get_sparse_custom_range(self, start_index, end_index):
        start_time = time.time()
        new_index = [start_index, end_index]
        blocks = []
        res_start_indices = []
        res_end_indices = []
        for i in range(self.num_blocks):
            if contained([self.start_indices[i], self.end_indices[i]], new_index):
                intersection_start_indices, intersection_end_indices = intersection_block([self.start_indices[i], self.end_indices[i]], new_index)
                res_start_indices.append(intersection_start_indices)
                res_end_indices.append(intersection_end_indices)
                blocks.append(self.blocks[i])
            elif block_overlap([self.start_indices[i], self.end_indices[i]], new_index):
                intersection_start_indices, intersection_end_indices = intersection_block([self.start_indices[i], self.end_indices[i]], new_index)
                res_start_indices.append(intersection_start_indices)
                res_end_indices.append(intersection_end_indices)
                src_start_indices = intersection_start_indices - self.start_indices[i]
                src_end_indices = intersection_end_indices - self.start_indices[i]
                src_block = self.blocks[i].get_dense()[tuple(get_slice(src_start_indices, src_end_indices))]
                blocks.append(DenseBlock(src_block))
        end_time = time.time()
        get_sparse_range_time.update_total_time(end_time-start_time)
        return SparseTensor(res_start_indices, blocks, self.dims, self.total_size, res_end_indices, type=self.type, dense_const=self.dense_const)
    


    def reduce_size(self, start_index, end_index, total_size):
        start_time = time.time()
        start_indices = []
        end_indices = []
        for i in range(self.num_blocks):
            # assert(contained([self.start_indices[i], self.end_indices[i]], [start_index, end_index]))
            start_indices.append((self.start_indices[i]-start_index))
            end_indices.append((self.end_indices[i]-start_index))
        end_time = time.time()
        reduce_size_time.update_total_time(end_time-start_time)
        return SparseTensor(start_indices, self.blocks, self.dims, total_size, end_indices, type=self.type, dense_const=self.dense_const) 

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
        start_time = time.time()
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

        end_time = time.time()
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
        def mult_list(l):
            if len(l)==1:
                return l[0]
            return mult_list(l[1:])*l[0]
        t = 0
        for i in range(self.num_blocks):
            t += mult_list(list(self.blocks[i].total_shape)) 
        if t < mult_list(list(self.total_size)):
            return False
        return True

    def unary(self, op):
        if op == operator.not_:
            assert(self.type == bool)
            dense_const = not(self.dense_const)
        elif op == operator.neg:
            assert(self.type == float or self.type == int)
            dense_const = -self.dense_const
        elif op == 'sigma':
            assert(self.type == float or self.type == int)
            dense_const = 1 / (1 + math.exp(-self.dense_const))
        else:
            assert(False)
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(self.blocks[i].unary(op))
        return SparseTensor(self.start_indices, blocks, self.dims, self.total_size, end_indices=self.end_indices, type=self.type, dense_const=dense_const)
    

    def binary(self, sp_tensor, op):
        start_time = time.time()
        if op not in all_ops:
            raise Exception(f"NOT IMPLEMENTED {op}")
        if (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()!=1):
            assert(False)
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        elif isinstance(sp_tensor, float) or isinstance(sp_tensor, bool) or isinstance(sp_tensor, int):
            sp_tensor = SparseTensor([], [], self.dims, self.total_size, [], type(sp_tensor), sp_tensor)
        elif (isinstance(sp_tensor, torch.Tensor) and sp_tensor.size()==1):
            sp_tensor = SparseTensor([], [], self.dims, self.total_size, [], type(sp_tensor.item()), sp_tensor.item())

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
        
        start_indices, end_indices, indices = self.union_tensors(sp_tensor, indices=True)
        res_start_indices = []
        res_end_indices = []
        res_blocks = []
        # print(self)
        # print(sp_tensor)
        # for i in range(len(start_indices)):
        #     print(i, start_indices[i], end_indices[i], indices[i])
        # print(start_indices, end_indices, indices)
        for i in range(len(start_indices)):
            start_index = start_indices[i]
            end_index = end_indices[i]
            # print('!!!!!!!!!', indices[i])
            if len(indices[i][0]) == 0:
                block = sp_tensor.get_sub_block_custom_range(start_index, end_index, indices[i][1][0], False)
                if identity_element(op) == self.dense_const:
                    block = binary_to_identity_unary(op)(block)
                elif annihilator_element(op) == self.dense_const and dense_const == self.dense_const:
                    continue
                else:
                    temp_block = ConstBlock(self.dense_const, block.total_shape)
                    block = temp_block.binary(block, op)
            elif len(indices[i][1]) == 0:
                block = self.get_sub_block_custom_range(start_index, end_index, indices[i][0][0], False)
                if identity_element(op) == sp_tensor.dense_const:
                    pass 
                elif annihilator_element(op) == sp_tensor.dense_const and dense_const == sp_tensor.dense_const:
                    continue
                else:
                    temp_block = ConstBlock(sp_tensor.dense_const, block.total_shape)
                    block = block.binary(temp_block, op)
            else:
                block_1 = self.get_sub_block_custom_range(start_index, end_index, indices[i][0][0], False)
                block_2 = sp_tensor.get_sub_block_custom_range(start_index, end_index, indices[i][1][0], False)
                block = block_1.binary(block_2, op)
            
            if isinstance(block, ConstBlock) and block.block == dense_const:
                continue
            res_blocks.append(block)
            res_start_indices.append(start_indices[i])
            res_end_indices.append(end_indices[i])
        return SparseTensor(res_start_indices, res_blocks, self.dims, self.total_size, res_end_indices, new_type, dense_const)
    
            
        

    def any(self):
        # return self.num_blocks > 0
        if not self.check_dense():
            if self.dense_const == True:
                return True
            if self.type != bool and self.dense_const != 0.0:
                return True
        for i in range(self.num_blocks):
            if self.blocks[i].any():
                return True
        return False
    
    def float(self):
        blocks = [b.float() for b in self.blocks]
        res = SparseTensor(self.start_indices, blocks, self.dims, self.total_size, self.end_indices, float, float(self.dense_const))
        return res 
    
    def matmul(self, sp_tensor):
        
        if isinstance(sp_tensor, torch.Tensor):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        

        assert(self.type == float or self.type == int)
        assert(sp_tensor.type == float or sp_tensor.type == int)
        assert(self.dense_const == 0.0)
        
        if sp_tensor.dense_const != 0.0 and sp_tensor.check_dense()==False:
            dense_sp_tensor = sp_tensor.get_dense()
            sp_tensor = SparseTensor([torch.tensor([0]*len(dense_sp_tensor.shape))], [DenseBlock(dense_sp_tensor)], sp_tensor.dims, sp_tensor.total_size, type=float, dense_const=0.0)

            dense_self = self.get_dense()
            self.start_indices = [torch.tensor([0]*len(dense_self.shape))]
            self.blocks = [DenseBlock(dense_self)]
            self.dims = len(dense_self.shape)
            self.total_size = torch.tensor(dense_self.shape)
            self.end_indices = [torch.tensor(dense_self.shape)]
            self.dense_const = 0.0
            self.num_blocks = 1
        assert(sp_tensor.dense_const == 0.0 or sp_tensor.check_dense()==True)
        
        


        if sp_tensor.dims <= 2:
            assert(sp_tensor.total_size[-1].item() == self.total_size[-1].item())
        else:
            assert(sp_tensor.total_size[-2].item() == self.total_size[-1].item())
            assert((sp_tensor.total_size[:-2] == self.total_size[-sp_tensor.dims:-2]).all())
        
        start_indices = []
        end_indices = []
        blocks = []
        res_total_size = None

        multiplicands = []

        

        if self.dims == sp_tensor.dims:
            res_total_size = torch.concat([self.total_size[:-1], sp_tensor.total_size[-1:]])
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 2:
                        start_index = torch.concat([self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]])
                    else:
                        if not block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-2], sp_tensor.end_indices[j][:-2]]):
                            continue
                        if not block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):
                            continue
                        
                        if full_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):    
                            block = self.blocks[i].matmul_equal_dims(sp_tensor.blocks[j])
                        elif contained([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):    
                            start_index = torch.concat((sp_tensor.start_indices[j][:-2], self.start_indices[i][-1:], sp_tensor.start_indices[j][-1:]))
                            end_index = torch.concat((sp_tensor.end_indices[j][:-2], self.end_indices[i][-1:], sp_tensor.end_indices[j][-1:]))
                            block_1 = sp_tensor.get_sub_block_custom_range(start_index, end_index, j, tensor=False)
                            block = self.blocks[i].matmul_equal_dims(block_1)
                        elif contained([sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]], [self.start_indices[i][-1:], self.end_indices[i][-1:]]):    
                            start_index = torch.concat((self.start_indices[i][:-1], sp_tensor.start_indices[j][-2:-1]))
                            end_index = torch.concat((self.end_indices[i][:-1], sp_tensor.end_indices[j][-2:-1]))
                            block_1 = self.get_sub_block_custom_range(start_index, end_index, i, tensor=False)
                            block = block_1.matmul_equal_dims(sp_tensor.blocks[j])
                        
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-2], self.start_indices[i][:-2]), self.start_indices[i][-2:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-2], self.end_indices[i][:-2]), self.end_indices[i][-2:-1], sp_tensor.end_indices[j][-1:]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    multiplicands.append((i, j))
        elif self.dims > sp_tensor.dims:
            res_total_size = self.total_size[:-1]
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 1:
                        start_index = self.start_indices[i][:-1]
                        end_index = self.end_indices[i][:-1]
                    else:
                        if not block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-1], sp_tensor.end_indices[j][:-1]]):
                            continue
                        if not block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            continue
                        if full_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):       
                            block_1 = self.blocks[i]               
                            block = block_1.matmul_unequal_dims(sp_tensor.blocks[j])
                        elif contained([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            start_index = torch.concat((sp_tensor.start_indices[j][:-1], self.start_indices[i][-1:]))
                            end_index = torch.concat((sp_tensor.end_indices[j][:-1], self.end_indices[i][-1:]))
                            block_1 = sp_tensor.get_sub_block_custom_range(start_index, end_index, j, tensor=False)
                            block = self.blocks[i].matmul_unequal_dims(block_1)
                        elif contained([sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]], [self.start_indices[i][-1:], self.end_indices[i][-1:]]):
                            start_index = torch.concat((self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]))
                            end_index = torch.concat((self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]))
                            block_1 = self.get_sub_block_custom_range(start_index, end_index, i, tensor=False)
                            block = block_1.matmul_unequal_dims(sp_tensor.blocks[j])
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-1], self.start_indices[i][:-2]), self.start_indices[i][-2:-1]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-1], self.end_indices[i][:-2]), self.end_indices[i][-2:-1]])

                    start_indices.append(start_index)
                    end_indices.append(end_index)
        else:
            assert(False)

        overlap_classes = find_connected_blocks(start_indices, end_indices)
        res = sp_tensor_from_overlap_classes(overlap_classes, start_indices, blocks, res_total_size, len(res_total_size), self.dense_const, self.type)
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
        
    def squeeze(self, index):
        start_time = time.time()
        # time.sleep(0.005)
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
        x = SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
        end_time = time.time()
        squeeze_time.update_total_time(end_time - start_time)
        return x
    
    def unsqueeze(self, index):
        start_time = time.time()
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
        end_time = time.time()
        unsqueeze_time.update_total_time(end_time-start_time)
        return SparseTensor(start_indices, blocks, dims, total_size, end_indices, self.type, self.dense_const)
    
    def repeat(self, repeat_dims):
        total_size = self.total_size * repeat_dims
        start_indices = []
        end_indices = []
        blocks = []
        for i in range(self.num_blocks):
            start_indices.append(self.start_indices[i]*repeat_dims)
            end_indices.append(self.end_indices[i]*repeat_dims)
            blocks.append(self.blocks[i].repeat(repeat_dims))
        return SparseTensor(start_indices, blocks, self.dims, total_size, end_indices, self.type, self.dense_const)
    
    def clamp(self, const, min_true):
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(self.blocks[i].clamp(const, min_true))
        return SparseTensor(self.start_indices, blocks, self.dims, self.total_size, self.end_indices, self.type, self.dense_const)

            
    def sum(self, dim):
        start_indices = []
        end_indices = []
        blocks = []
        total_size = torch.concat([self.total_size[:dim], self.total_size[dim+1:]])
        for i in range(self.num_blocks):
            start_indices.append(torch.concat([self.start_indices[i][:dim], self.start_indices[i][dim+1:]]))
            end_indices.append(torch.concat([self.end_indices[i][:dim], self.end_indices[i][dim+1:]]))
            blocks.append(self.blocks[i].sum(dim))

        
        overlap_classes = find_connected_blocks(start_indices, end_indices)
        res = sp_tensor_from_overlap_classes(overlap_classes, start_indices, blocks, total_size, self.dims-1, self.dense_const, self.type)
        return res
    
def sparse_max(x:SparseTensor, y:SparseTensor):
    z = x.binary(y, operator.gt)
    return sp_where(z, x, y)

def sparse_min(x:SparseTensor, y:SparseTensor):
    z = x.binary(y, operator.lt)
    return sp_where(z, x, y)


def sp_where(x: SparseTensor, y: SparseTensor, z: SparseTensor):
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
    for i in range(len(start_indices)):
        start_index = start_indices[i]
        end_index = end_indices[i]
        if len(indices[i][0]) == 0:
            yz_index = indices[i][1][0]
            y_indices = yz_indices[yz_index][0]
            z_indices = yz_indices[yz_index][1]
            if x.dense_const and len(y_indices) == 1:
                block = y.get_sub_block_custom_range(start_index, end_index, y_indices[0], False)
            elif not(x.dense_const) and len(z_indices) == 1:
                block = z.get_sub_block_custom_range(start_index, end_index, z_indices[0], False)
            else:
                continue
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block)


        elif len(indices[i][1]) > 0:
            x_index = indices[i][0][0]
            x_block = x.get_sub_block_custom_range(start_index, end_index, x_index, False)

            yz_index = indices[i][1][0]
            y_indices = yz_indices[yz_index][0]
            z_indices = yz_indices[yz_index][1]

            if len(y_indices) == 0:
                y_block = ConstBlock(y.dense_const, end_index-start_index)
            else:
                y_block = y.get_sub_block_custom_range(start_index, end_index, y_indices[0], False)

            if len(z_indices) == 0:
                z_block = ConstBlock(z.dense_const, end_index-start_index)
            else:
                z_block = z.get_sub_block_custom_range(start_index, end_index, z_indices[0], False)

            block = sp_where_block(x_block, y_block, z_block)
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block)
    return SparseTensor(res_start_indices, res_blocks, x.dims, x.total_size, res_end_indices, new_type, dense_const)
