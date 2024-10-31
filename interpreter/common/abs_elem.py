import torch
import copy
from lib.polyexp import PolyExpNew

class Abs_elem:
    def __init__(self, d, types, shapes):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        self.d = d
        self.types = types 
        self.shapes = shapes

    # def get_elem(self, key, neuron):
    #     if self.types[key] == 'int' or self.types[key] == 'float':
    #         return self.d[key][neuron[0]][neuron[1]]
    #     elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
    #         x = self.d[key][neuron[0]]
    #         idx = neuron[1]
    #         while len(idx) > 1:
    #             x = x[idx[0]]
    #             idx = idx[1:]
    #         return x[idx[0]].copy()
        
    # def get_elem_new_old(self, key, nlist):
    #     if nlist.nlist_flag:
    #         res = self.d[key][nlist.nlist]
    #     else:
    #         res = self.d[key][nlist.start:nlist.end+1]
    #     if self.types[key] == 'int' or self.types[key] == 'float':
    #         return res 
    #     elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
    #         return res.clone()
        
    def get_live_nlist(self, nlist):
        live_neurons = torch.nonzero(self.d['t']).flatten()
        if nlist.nlist_flag:
            selected_live_neurons = live_neurons[torch.isin(live_neurons, nlist.nlist)]
        else:
            selected_live_neurons = live_neurons[live_neurons >= nlist.start & live_neurons <= nlist.end]
        nlist_new = copy.deepcopy(nlist)
        nlist_new.nlist_flag = True 
        nlist_new.nlist = selected_live_neurons 
        return nlist_new

        
    def get_elem_new(self, key, nlist):
        nlist = self.get_live_nlist(nlist)
        if nlist.nlist_flag:
            if self.types[key] == 'int' or self.types[key] == 'float':
                res = self.d[key][nlist.nlist]
                return res.flatten()
            elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
                val_mat = self.d[key].mat[nlist.nlist]
                val_const = self.d[key].const[nlist.nlist].flatten()
                size = self.d[key].size 
                res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
        else:
            if self.types[key] == 'int' or self.types[key] == 'float':
                res = self.d[key][nlist.start:nlist.end+1]
                return res 
            elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
                val_mat = self.d[key].mat[nlist.start:nlist.end+1].clone()
                val_const = self.d[key].const[nlist.start:nlist.end+1].clone()
                size = self.d[key].size 
                res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
            
    # def get_elem_new(self, key, nlist):
    #     live_neurons = self.get_live_nlist(nlist)
    #     flag = True 
    #     # if nlist.nlist_flag:
    #     if flag:
    #         if self.types[key] == 'int' or self.types[key] == 'float':
    #             res = self.d[key][live_neurons]
    #             # res = res.reshape(-1,1)
    #             return res 
    #         elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
    #             val_mat = self.d[key].mat[nlist.nlist]
    #             val_const = self.d[key].const[nlist.nlist]
    #             size = self.d[key].size 
    #             res = PolyExpNew(size=size, mat=val_mat, const=val_const)
    #             return res
    #     else:
    #         if self.types[key] == 'int' or self.types[key] == 'float':
    #             res = self.d[key][nlist.start:nlist.end+1]
    #             # res = res.reshape(-1,1)
    #             return res 
    #         elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
    #             val_mat = self.d[key].mat[nlist.start:nlist.end+1].clone()
    #             val_const = self.d[key].const[nlist.start:nlist.end+1].clone()
    #             size = self.d[key].size 
    #             res = PolyExpNew(size=size, mat=val_mat, const=val_const)
    #             return res
            
        
    def update_elem(self, neuron, vals):
        for i, key in enumerate(self.d.keys()):
            if self.types[key] == 'int' or self.types[key] == 'float':
                if isinstance(vals[i], float) or isinstance(vals[i], int) or isinstance(vals[i], torch.Tensor):
                    self.d[key][neuron[0]][neuron[1]] = vals[i]
                else:
                    print(type(vals[i]))
                    self.d[key][neuron[0]][neuron[1]] = vals[i].const 
            elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
                x = self.d[key][neuron[0]]
                idx = neuron[1]
                while len(idx) > 1:
                    x = x[idx[0]]
                    idx = idx[1:]
                x[idx[0]] = vals[i].copy()

    def update(self, neuron, val):
        for i, key in enumerate(self.d.keys()):
            if isinstance(val[i], torch.Tensor):
                self.d[key][neuron[0]][neuron[1]] = val[i].clone()
            elif isinstance(val[i], float):
                self.d[key][neuron[0]][neuron[1]] = val[i]
            elif isinstance(val[i], int):
                self.d[key][neuron[0]][neuron[1]] = val[i]
            else:
                self.d[key][neuron[0]][neuron[1]] = val[i].copy()
