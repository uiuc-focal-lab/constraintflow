# from common.polyexp import PolyExpNew

# import torch 


# class Nlist:
#     def __init__(self, size=0, start=None, end=None, nlist=None):
#         self.size = size
#         self.start = start
#         self.end = end
#         self.nlist = nlist
#         self.nlist_flag = True 
#         if nlist==None:
#             self.nlist_flag = False

#     def dot(self, w):
#         n = w.size(0)
#         N = self.size 
#         polyexp_const = torch.zeros(n)
#         polyexp_coeff = torch.zeros(N)
#         if w.dim()==2:
#             polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
#             if self.nlist_flag:
#                 if self.nlist.dim() == 1:
#                     self.nlist = self.nlist.unsqueeze(0).expand(n, -1)
#         if self.nlist_flag:
#             polyexp_coeff[torch.arange(n).unsqueeze(1), self.nlist] = w 
#             # polyexp_coeff[self.nlist] = w
#         else:
#             polyexp_coeff[:, self.start:self.end+1] = w 
#         res = PolyExpNew(N, polyexp_coeff, polyexp_const)
#         return res 
    
#     def sum(self):
#         N = self.size 
#         polyexp_coeff = torch.zeros(N)
#         if self.nlist_flag:
#             if self.nlist.dim()>1:
#                 n = self.nlist.size(0)
#                 polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
#                 polyexp_coeff[torch.arange(n).unsqueeze(1), self.nlist] = 1
#                 polyexp_const = torch.zeros(n)
#             else:
#                 polyexp_coeff[self.nlist] = 1
#                 polyexp_const = 0
#         else:
#             if isinstance(self.start, torch.Tensor):
#                 if self.start.shape[0]>1:
#                     n = self.start.size(0)
#                     polyexp_coeff = polyexp_coeff.unsqueeze(0).expand(n, -1).clone()
#                     polyexp_coeff[:, self.start:self.end+1] = 1
#                     polyexp_const = torch.zeros(n)
#                 else:
#                     polyexp_coeff[self.start:self.end+1] = 1
#                     polyexp_const = 0
#             else:
#                 polyexp_coeff[self.start:self.end+1] = 1
#                 polyexp_const = 0
#         res = PolyExpNew(N, polyexp_coeff, polyexp_const)
#         return res 
    
#     def avg(self):
#         res = self.sum()
#         if self.nlist_flag:
#             if self.nlist.dim()>1:
#                 num_elems = self.nlist.shape[1]
#             else:
#                 num_elems = self.nlist.shape[0]
#         else:
#             num_elems = self.end + 1 - self.start
#         res.coeff = res.coeff / num_elems
#         res.const = res.const / num_elems
#         return res 