import torch
import copy
from common.polyexp import PolyExp, SymExp

class Abs_elem:
    def __init__(self, d, types, shapes):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        self.d = d
        self.types = types 
        self.shapes = shapes

    def get_live_nlist(self, nlist):
        live_neurons = torch.nonzero(self.d['t']).flatten()
        if nlist.nlist_flag:
            selected_live_neurons = live_neurons[torch.isin(live_neurons, nlist.nlist)]
        else:
            selected_live_neurons = live_neurons[((live_neurons >= nlist.start) & (live_neurons <= nlist.end))]
        nlist_new = copy.deepcopy(nlist)
        nlist_new.nlist_flag = True 
        nlist_new.nlist = selected_live_neurons 
        return nlist_new

        
    def get_elem_new(self, key, nlist):
        nlist = self.get_live_nlist(nlist)
        if nlist.nlist_flag:
            if self.types[key] == 'int' or self.types[key] == 'float' or self.types[key] == 'Int' or self.types[key] == 'Float':
                res = self.d[key][nlist.nlist]
                return res.flatten()
            elif self.types[key] == 'PolyExp':
                val_mat = self.d[key].mat[nlist.nlist]
                val_const = self.d[key].const[nlist.nlist].flatten()
                size = self.d[key].cols 
                res = PolyExp(len(nlist.nlist), size, val_mat, val_const)
                # res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
            elif self.types[key] == 'ZonoExp':
                print(key)
                val_mat = self.d[key].mat[nlist.nlist]
                val_const = self.d[key].const[nlist.nlist].flatten()
                size = self.d[key].cols 
                res = SymExp(len(nlist.nlist), size, val_mat, val_const, 0, SymExp.count)
                # res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
        else:
            if self.types[key] == 'int' or self.types[key] == 'float':
                res = self.d[key][nlist.start:nlist.end+1]
                return res 
            elif self.types[key] == 'PolyExp':
                val_mat = self.d[key].mat[nlist.start:nlist.end+1].clone()
                val_const = self.d[key].const[nlist.start:nlist.end+1].clone()
                size = self.d[key].cols 
                res = PolyExp(-nlist.start+nlist.end+1, size, val_mat, val_const)
                # res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
            elif self.types[key] == 'ZonoExp':
                val_mat = self.d[key].mat[nlist.start:nlist.end+1].clone()
                val_const = self.d[key].const[nlist.start:nlist.end+1].clone()
                size = self.d[key].cols 
                res = SymExp(-nlist.start+nlist.end+1, size, val_mat, val_const, 0, SymExp.count)
                # res = PolyExpNew(size=size, mat=val_mat, const=val_const)
                return res
            
    # def update_elem(self, neuron, vals):
    #     for i, key in enumerate(self.d.keys()):
    #         if self.types[key] == 'int' or self.types[key] == 'float':
    #             if isinstance(vals[i], float) or isinstance(vals[i], int) or isinstance(vals[i], torch.Tensor):
    #                 self.d[key][neuron[0]][neuron[1]] = vals[i]
    #             else:
    #                 self.d[key][neuron[0]][neuron[1]] = vals[i].const 
    #         elif self.types[key] == 'PolyExp' or self.types[key] == 'ZonoExp':
    #             x = self.d[key][neuron[0]]
    #             idx = neuron[1]
    #             while len(idx) > 1:
    #                 x = x[idx[0]]
    #                 idx = idx[1:]
    #             x[idx[0]] = vals[i].copy()

    def update(self, nlist, abs_shape):
        live_neurons = torch.nonzero(self.d['t']).flatten()
        if nlist.nlist_flag:
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int']:
                    self.d[key][nlist.nlist] = abs_shape[i]
                elif self.types[key] in ['PolyExp']:
                    self.d[key].mat[nlist.nlist, 0:max(live_neurons)+1] = abs_shape[i].mat
                    self.d[key].const[nlist.nlist] = abs_shape[i].const
                elif self.types[key] in ['ZonoExp']:
                    new_eps = torch.zeros(self.d[key].mat.shape[0], SymExp.count - self.d[key].mat.shape[1])
                    # print(new_eps.shape)
                    self.d[key].mat = torch.concat([self.d[key].mat, new_eps], dim=1)
                    # df
                    print(abs_shape[i].mat.shape)
                    print(self.d[key].mat.shape)
                    self.d[key].mat[nlist.nlist] = abs_shape[i].mat
                    self.d[key].const[nlist.nlist] = abs_shape[i].const
                else:
                    jhxdgf
            self.d['t'][nlist.nlist] = True
        else:
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int']:
                    # print(key)
                    # print(self.d[key])
                    self.d[key][nlist.start:nlist.end+1] = abs_shape[i]
                elif self.types[key] in ['PolyExp']:
                    temp = torch.zeros(nlist.end + 1 - nlist.start, self.d[key].mat.shape[1])
                    self.d[key].mat[nlist.start:nlist.end+1][:, 0:max(live_neurons)+1] = abs_shape[i].mat
                    self.d[key].const[nlist.start:nlist.end+1] = abs_shape[i].const
                elif self.types[key] in ['ZonoExp']:
                    # temp = torch.zeros(nlist.end + 1 - nlist.start, self.d[key].mat.shape[1])
                    # self.d[key].mat = torch.concat([self.d[key].mat, temp], dim=0)
                    # self.d[key].const = torch.concat([self.d[key].const, torch.zeros(nlist.end - nlist.start + 1)], dim=0)
                    self.d[key].mat[nlist.start:nlist.end+1] = abs_shape[i].mat
                    self.d[key].const[nlist.start:nlist.end+1] = abs_shape[i].const
                else:
                    jhxdgf
            self.d['t'][nlist.start:nlist.end+1] = True
