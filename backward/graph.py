from z3 import *

x = Real('x')
y = Real('y')
b = True
plus = (x + y).decl()
conjunction = (And(b, b)).decl()
lt = (x < y).decl()
le = (x <= y).decl()
gt = (x > y).decl()
ge = (x >= y).decl()
eq = (x == y).decl()
comparison = [lt, le, gt, ge, eq]

class Vertex():
    def __init__(self, counter, v):
        self.v = v 
        self.counter = counter 
        self.eq = []
        self.le = []
        self.lt = []
        self.ge = []
        self.gt = []

        

class Opt_graph():
    def __init__(self):
        self.form_num = dict()
        self.num_form = dict()
        self.num_vertex = dict()
        self.counter = 0

        self.up = set()
        self.down = set()

    def __init__(self, lhs):
        self.form_num = dict()
        self.num_form = dict()
        self.num_vertex = dict()
        self.counter = 0

        self.up = set()
        self.down = set()

        self.arrange(lhs)

    def get_clauses(self, x):
        if isinstance(x, list):
            l = []
            children = x 
            for child in children:
                l = l + self.get_clauses(child)
            return l
        top_level = x.decl()
        if top_level != conjunction:
            return [x]
        else:
            l = []
            children = x.children()
            for child in children:
                l = l + self.get_clauses(child)
            return l

    def arrange(self, lhs):
        clauses = self.get_clauses(lhs)
        for clause in clauses:
            if clause.decl() in comparison:
                lhs, rhs = clause.children()
                if lhs not in self.form_num.keys():
                    self.form_num[lhs] = self.counter
                    self.num_form[self.counter] = lhs
                    v = Vertex(self.counter, lhs)
                    self.num_vertex[self.counter] = v
                    self.counter += 1
                if rhs not in self.form_num.keys():
                    self.form_num[rhs] = self.counter
                    self.num_form[self.counter] = rhs
                    v = Vertex(self.counter, rhs)
                    self.num_vertex[self.counter] = v
                    self.counter += 1
                
                num_lhs = self.form_num[lhs]
                num_rhs = self.form_num[rhs]

                v_lhs = self.num_vertex[num_lhs]
                v_rhs = self.num_vertex[num_rhs]

                if clause.decl()==eq:
                    v_lhs.eq.append(num_rhs)
                    v_rhs.eq.append(num_lhs)

                if clause.decl()==lt:
                    v_lhs.gt.append(num_rhs)
                    v_rhs.lt.append(num_lhs)
                
                if clause.decl()==gt:
                    v_lhs.lt.append(num_rhs)
                    v_rhs.gt.append(num_lhs)

                if clause.decl()==le:
                    v_lhs.ge.append(num_rhs)
                    v_rhs.le.append(num_lhs)
                
                if clause.decl()==ge:
                    v_lhs.le.append(num_rhs)
                    v_rhs.ge.append(num_lhs)

    def travel_up(self, num):
        v = self.num_vertex[num]
        for i in v.eq + v.ge + v.gt:
            if i not in self.up:
                self.up.add(i)
                self.travel_up(i)

    def travel_down(self, num):
        v = self.num_vertex[num]
        for i in v.eq + v.le + v.lt:
            if i not in self.down:
                self.down.add(i)
                self.travel_down(i)
        
    def get_queries_right(self, top_level, m , rhs):
        return [top_level(self.num_form[i], rhs) for i in m]
    
    def get_queries_left(self, top_level, m , lhs):
        return [top_level(lhs, self.num_form[i]) for i in m]
    
    def get_sufficient_queries(self, query):
        top_level = query.decl()
        lhs, rhs = query.children()
        if lhs in self.form_num.keys():
            num_lhs = self.form_num[lhs]
            if (top_level == lt) or (top_level == le):
                self.up.clear()
                self.travel_up(num_lhs)
                return self.get_queries_right(top_level, self.up, rhs)
            if (top_level == gt) or (top_level == ge):
                self.down.clear()
                self.travel_down(num_lhs)
                return self.get_queries_right(top_level, self.down, rhs)
        
        elif rhs in self.form_num.keys():
            num_rhs = self.form_num[rhs]
            if (top_level == lt) or (top_level == le):
                self.down.clear()
                self.travel_down(num_rhs)
                return self.get_queries_left(top_level, self.down, lhs)
            if (top_level == gt) or (top_level == ge):
                self.up.clear()
                self.travel_up(num_rhs)
                return self.get_queries_left(top_level, self.up, lhs)
        return []