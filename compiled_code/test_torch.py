import torch 
import torch.nn.functional as F 

def convert_index(i, j, s):
    return i*s+j

def convert_to_fully_connected(num_kernels, channels, kernel, ix, iy, ox, oy, kx, ky, sx=1, sy=1, px=0, py=0):
    w = torch.zeros(ox*oy*num_kernels, ix*iy*channels)
    for n in range(num_kernels):
        for c in range(channels):
            for i in range(ox):
                for j in range(oy):
                    if i*sx-px+kx<ix+2*px and j*sy-py+ky<iy+2*py:
                        for ki, si in enumerate(range(i*sx-px, i*sx-px+kx)):
                            for kj, sj in enumerate(range(j*sy-py, j*sy-py+ky)):
                                if si<0 or si>=ix or sj<0 or sj>=iy:
                                    continue
                                oi = convert_index(i, j, ox)
                                oj = convert_index(si, sj, ix)
                                w[n*ox*oy+oi][c*ix*iy + oj] = kernel[n][c][ki][kj]
    return w

# kx = 2
# ky = 2
# in_size = 5
# out_size = 4
# channels = 1
# num_kernels = 1

# k = torch.randn(num_kernels,channels,kx,ky)
# x = torch.randn(2,2,channels,in_size,in_size)

# x1 = x[0]
# x2 = x[1]
# x3 = x.reshape(4, channels, in_size, in_size)


# y1 = F.conv2d(x1, k, stride=(2,2), padding=(1,1))
# y2 = F.conv2d(x2, k, stride=(2,2), padding=(1,1))
# y3 = F.conv2d(x3, k, stride=(2,2), padding=(1,1))

# y4 = y3.reshape(2,2, y1.shape[-3], y1.shape[-2], y1.shape[-1])

# y5 = y4[0]
# y6 = y4[1]

# print(y2)
# print(y6)
# kjf


# y = F.conv2d(x, k, stride=(2,2), padding=(1,1))
# print(y.shape)
# lkfh
# k = k

# print(k)
# print(x[0][0])

# ox = y.shape[-2]
# oy = y.shape[-1]

# k2 = convert_to_fully_connected(num_kernels, channels, k, in_size, in_size, ox, oy, kx, ky, 2, 2, 1, 1)
# print(k2)

# z = (k2@x.flatten()).reshape(num_kernels, ox,oy)

# print(y)
# print(z)