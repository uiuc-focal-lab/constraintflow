import torch

class Abs_elem:
    def __init__(self, d, types, shapes):
        if d.keys() != types.keys():
            raise TypeError("abs elem inconsistent")
        self.d = d
        self.types = types 
        self.shapes = shapes

    def get_elem(self, key, neuron):
        if self.types[key] == 'int' or self.types[key] == 'float':
            return self.d[key][neuron[0]][neuron[1]]
        elif self.types[key] == 'PolyExp' or self.types[key] == 'SymExp':
            x = self.d[key][neuron[0]]
            idx = neuron[1]
            while len(idx) > 1:
                x = x[idx[0]]
                idx = idx[1:]
            return x[idx[0]].copy()
        
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
