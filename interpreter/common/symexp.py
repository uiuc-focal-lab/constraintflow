import copy

class SymExp:
    count = 0
    def __init__(self, mat = [], const = 0.0):
        if SymExp.count < len(mat) :
            SymExp.count = len(mat)
        self.mat = mat 
        self.const = const 

    def copy(self):
        res = SymExp(const=copy.deepcopy(self.const))
        res.mat = copy.deepcopy(self.mat)
        return res 

    def get_const(self):
        return self.const
    
    def new_symbol(self):
        SymExp.count += 1
        return self

    def populate(self, const=0, n=-1, coeff=0):
        if n==-1:
            n = SymExp.count-1
        if len(self.mat) < SymExp.count:
            tmp = [0]*(SymExp.count - len(self.mat))
            self.mat += tmp 
        self.const = const 
        self.mat[n] = coeff
        return self

    def populate_bot(self):
        self.const = 0 
        self.mat = [float('inf')]*SymExp.count 
        return self
    
    def add(self, p):
        if isinstance(p, SymExp):
            self.const = self.const + p.const
            if len(self.mat) < len(p.mat):
                tmp = [0]*(len(p.mat) - len(self.mat))
                self.mat += tmp 
            for i in range(len(p.mat)):
                self.mat[i] = self.mat[i] + p.mat[i]
        else:
            self.const += p
        return self

    def minus(self, p):
        if isinstance(p, SymExp):
            self.const = self.const - p.const 
            if len(self.mat) < len(p.mat):
                tmp = [0]*(len(p.mat) - len(self.mat))
                self.mat += tmp 
            for i in range(len(p.mat)):
                self.mat[i] = self.mat[i] - p.mat[i]
        else:
            self.const = self.const - p
        return self

    def mult(self, c):
        self.const = self.const*c 
        for i in range(len(self.mat)):
            self.mat[i] = self.mat[i]*c
        return self
    
    def map(self, abs_elem, neighbours, f):
        res = SymExp(const = self.const)
        for i in range(len(self.mat)):
            if self.mat[i] != 0:
                tmp = f(i, self.mat[i], abs_elem, neighbours)
                res.add(tmp)
        return res 