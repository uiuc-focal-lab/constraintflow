import torch 
import torch.nn.functional as F

class SparseBlock:
    def __init__(self, block, type='Normal', total_shape=None, ix=0, iy=0, ox=0, oy=0, sx=1, sy=1, px=0, py=0):
        self.block = block
        self.type = type 
        self.total_shape = total_shape

        self.ix = ix
        self.iy = iy
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.px = px
        self.py = py

        if self.type=='Normal':
            self.total_shape = self.block.shape

    # def convert_to_dense(num_kernels, channels, kernel, ix, iy, ox, oy, kx, ky, sx=1, sy=1, px=0, py=0):
    #     w = torch.zeros(ox*oy*num_kernels, ix*iy*channels)
    #     for n in range(num_kernels):
    #         for c in range(channels):
    #             for i in range(ox):
    #                 for j in range(oy):
    #                     if i*sx-px+kx<ix+2*px and j*sy-py+ky<iy+2*py:
    #                         for ki, si in enumerate(range(i*sx-px, i*sx-px+kx)):
    #                             for kj, sj in enumerate(range(j*sy-py, j*sy-py+ky)):
    #                                 if si<0 or si>=ix or sj<0 or sj>=iy:
    #                                     continue
    #                                 oi = i*ox+j
    #                                 oj = si*ix+sj
    #                                 w[n*ox*oy+oi][c*ix*iy + oj] = kernel[n][c][ki][kj]
    #     return w

    def get_dense(self):
        if self.type=='Normal':
            return self.block
        elif self.type=='Kernel':
            assert(False) 

    def size(self):
        if self.type=='Normal':
            return self.block.size()
        elif self.type=='Kernel':
            assert(False) 

    def repeat(self, repeat_dims):
        if self.type=='Normal':
            return SparseBlock(self.block.repeat(*repeat_dims))
        else:
            self.total_shape *= repeat_dims
            return self
    
    def unsqueeze(self, index):
        if self.type=='Normal':
            return SparseBlock(self.block.unsqueeze(index))
        else:
            self.total_shape = torch.concat([self.total_shape[:index], torch.ones(1), self.total_shape[index:]])
            # self.total_shape.insert(index, 1)
            return self
        
    def squeeze(self, index):
        if self.type=='Normal':
            return SparseBlock(self.block.squeeze(index))
        else:
            self.total_shape = torch.concat([self.total_shape[:index], self.total_shape[index+1:]])
            # self.total_shape.insert(index, 1)
            return self
        
    # def binary_with_scalar(self, scalar, op):
    #     if self.type=='Normal':
    #         if op=='*':
    #             block = self.block*scalar
    #         elif op=='/':
    #             block = self.block/scalar
    #         elif op=='+':
    #             block = self.block+scalar
    #         elif op=='-':
    #             block = self.block-scalar
    #         elif op=='|':
    #             block = self.block|scalar
    #         elif op=='&':
    #             block = self.block&scalar
    #         elif op=='>=':
    #             block = self.block>=scalar
    #         elif op=='>':
    #             block = self.block>scalar
    #         elif op=='<=':
    #             block = self.block<=scalar
    #         elif op=='<':
    #             block = self.block<scalar
    #         elif op=='==':
    #             block = self.block==scalar
    #         elif op=='!=':
    #             block = self.block!=scalar
    #         return SparseBlock(block)
    #     elif self.type=='Kernel':
