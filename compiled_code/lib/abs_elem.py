import torch
import copy
from compiled_code.lib.polyexp import *
from compiled_code.lib.nlist import Llist
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
                return res
            elif self.types[key] == 'ZonoExp':
                val_mat = self.d[key].mat[nlist.nlist]
                val_const = self.d[key].const[nlist.nlist].flatten()
                size = self.d[key].cols 
                res = SymExp(len(nlist.nlist), size, val_mat, val_const, 0, SymExp.count)
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
                return res
            elif self.types[key] == 'ZonoExp':
                val_mat = self.d[key].mat[nlist.start:nlist.end+1].clone()
                val_const = self.d[key].const[nlist.start:nlist.end+1].clone()
                size = self.d[key].cols 
                res = SymExp(-nlist.start+nlist.end+1, size, val_mat, val_const, 0, SymExp.count)
                return res 

    def update(self, nlist, abs_shape, debug_flag=False):
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
                    self.d[key].mat = torch.concat([self.d[key].mat, new_eps], dim=1)
                    self.d[key].mat[nlist.nlist] = abs_shape[i].mat
                    self.d[key].const[nlist.nlist] = abs_shape[i].const
                else:
                    raise Exception('CHECK THIS')
            self.d['t'][nlist.nlist] = True
        else:
            raise Exception('NOT NEEDED')
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int']:
                    self.d[key][nlist.start:nlist.end+1] = abs_shape[i]
                elif self.types[key] in ['PolyExp']:
                    temp = torch.zeros(nlist.end + 1 - nlist.start, self.d[key].mat.shape[1])
                    self.d[key].mat[nlist.start:nlist.end+1][:, 0:max(live_neurons)+1] = abs_shape[i].mat
                    self.d[key].const[nlist.start:nlist.end+1] = abs_shape[i].const
                    
                elif self.types[key] in ['ZonoExp']:
                    self.d[key].mat[nlist.start:nlist.end+1, :] = abs_shape[i].mat
                    self.d[key].const[nlist.start:nlist.end+1] = abs_shape[i].const
                else:
                    raise Exception('CHECK THIS')
            self.d['t'][nlist.start:nlist.end+1] = True

class Abs_elem_sparse:
    def __init__(self, d, types, network, batch_size=1):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        self.d = d
        self.types = types 
        self.network = network
        self.batch_size = batch_size
    
    def filter_non_live(self, llist):
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
        return res
    
    def get_poly_size(self):
        l = list(torch.nonzero(self.d['llist']))[-1].item()
        return self.network[l].end
        

    def get_elem_new(self, key, llist):
        return self.get_elem(key, llist)
        
    def get_elem(self, key, llist):
        llist = self.filter_non_live(llist)
        llist_compressed = torch.nonzero(self.d['llist']).flatten().tolist()
        if llist.llist_flag:
            if self.types[key] == 'int' or self.types[key] == 'float' or self.types[key] == 'Int' or self.types[key] == 'Float':
                res = []
                for l in llist.llist:
                    start_index = torch.tensor([0, llist.network[l].start])
                    end_index = torch.tensor([self.batch_size, llist.network[l].end])
                    res.append(self.d[key].get_sparse_custom_range(start_index, end_index))
                sp_tensor = res[0]
                for i in range(1, len(res)):
                    sp_tensor = sp_tensor.merge_no_overlap(res[i])
                start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                total_size = end_index - start_index
                sp_tensor = sp_tensor.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    sp_tensor = sp_tensor.unsqueeze(1)
                sp_tensor = sp_tensor.repeat(torch.tensor(llist.initial_shape + [1]))
                return sp_tensor
            elif self.types[key] == 'PolyExp':
                res = []
                for l in llist.llist:
                    start_index = torch.tensor([0, llist.network[l].start])
                    end_index = torch.tensor([self.batch_size, llist.network[l].end])
                    res.append(self.d[key].const.get_sparse_custom_range(start_index, end_index))
                val_const = res[0]
                for i in range(1, len(res)):
                    val_const = val_const.merge_no_overlap(res[i])
                start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end])
                total_size = end_index - start_index
                val_const = val_const.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1
                for i in range(extra_dims):
                    val_const = val_const.unsqueeze(1)
                val_const = val_const.repeat(torch.tensor(llist.initial_shape + [1]))

                res = []
                for l in llist.llist:
                    start_index = torch.tensor([0, llist.network[l].start, self.network[min(llist_compressed)].start])
                    end_index = torch.tensor([self.batch_size, llist.network[l].end, self.network[max(llist_compressed)].end])
                    res.append(self.d[key].mat.get_sparse_custom_range(start_index, end_index))
                val_mat = res[0]
                for i in range(1, len(res)):
                    val_mat = val_mat.merge_no_overlap(res[i])
                start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                end_index = torch.tensor([self.batch_size, self.network[max(llist.llist)].end, val_mat.total_size[-1]])
                total_size = end_index - start_index
                val_mat = val_mat.reduce_size(start_index, end_index, total_size)
                extra_dims = len(llist.initial_shape)-1

                
                for i in range(extra_dims):
                    val_mat = val_mat.unsqueeze(1)
                val_mat = val_mat.repeat(torch.tensor(llist.initial_shape + [1,1]))

                return PolyExpSparse(self.network, val_mat, val_const)
            
            elif self.types[key] == 'ZonoExp':
                raise Exception('NOT IMPLEMENTED')
        else:
            if self.types[key] == 'int' or self.types[key] == 'float' or self.types[key] == 'Int' or self.types[key] == 'Float':
                start_index = torch.tensor([0, llist.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, llist.network[llist.end].end])
                sp_tensor = self.d[key].get_sparse_custom_range(start_index, end_index)

                start_index = torch.tensor([0, self.network[llist.start].start])
                end_index = torch.tensor([self.batch_size, self.network[llist.end].end])
                total_size = end_index - start_index
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

                return PolyExpSparse(self.network, val_mat, val_const)
            
            elif self.types[key] == 'ZonoExp':
                raise Exception('NOT IMPLEMENTED')
            
    def update(self, llist, abs_shape):
        # llist = self.filter_non_live(llist)
        llist.decoalesce()
        assert(len(llist.llist) == 1)
        if llist.llist_flag:
            keys = list(self.d.keys())
            for i in range(len(abs_shape)):
                key = keys[i+1]
                if self.types[key] in ['Float', 'Int']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].total_size
                    new_val = (abs_shape[i]).increase_size(start_index, total_size)
                    self.d[key] = self.d[key].overwrite(new_val)
                elif self.types[key] in ['PolyExp']:
                    start_index = torch.tensor([0, self.network[min(llist.llist)].start])
                    total_size = self.d[key].const.total_size
                    self.d[key].const = self.d[key].const.overwrite((abs_shape[i].const).increase_size(start_index, total_size))

                    start_index = torch.tensor([0, self.network[min(llist.llist)].start, 0])
                    total_size = torch.tensor(list(self.d[key].mat.total_size))
                    self.d[key].mat = self.d[key].mat.overwrite((abs_shape[i].mat).increase_size(start_index, total_size))

                elif self.types[key] in ['ZonoExp']:
                    raise Exception('NOT IMPLEMENTED')
                
            self.d['llist'][llist.llist] = True
            # mhsd
        else:
            raise Exception('NOT NEEDED')