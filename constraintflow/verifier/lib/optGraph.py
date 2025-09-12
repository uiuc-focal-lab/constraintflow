from z3 import *

from constraintflow.verifier.lib.utils import *

class Node():
    def __init__(self, counter, v):
        self.v = v 
        self.counter = counter 
        self.eqq = []
        self.le = []
        self.lt = []
        self.ge = []
        self.gt = []

        

class OptGraph():
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
                    v = Node(self.counter, lhs)
                    self.num_vertex[self.counter] = v
                    self.counter += 1
                if rhs not in self.form_num.keys():
                    self.form_num[rhs] = self.counter
                    self.num_form[self.counter] = rhs
                    v = Node(self.counter, rhs)
                    self.num_vertex[self.counter] = v
                    self.counter += 1
                
                num_lhs = self.form_num[lhs]
                num_rhs = self.form_num[rhs]

                v_lhs = self.num_vertex[num_lhs]
                v_rhs = self.num_vertex[num_rhs]

                if clause.decl()==eqq:
                    v_lhs.eqq.append(num_rhs)
                    v_rhs.eqq.append(num_lhs)

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
        for i in v.eqq + v.ge + v.gt:
            if i not in self.up:
                self.up.add(i)
                self.travel_up(i)

    def travel_down(self, num):
        v = self.num_vertex[num]
        for i in v.eqq + v.le + v.lt:
            if i not in self.down:
                self.down.add(i)
                self.travel_down(i)
        
    def get_queries_right(self, top_level, m , rhs):
        return [top_level(self.num_form[i], rhs) for i in m]
    
    def get_queries_left(self, top_level, m , lhs):
        return [top_level(lhs, self.num_form[i]) for i in m]
    


    def get_sufficient_queries(self, query):
        top_level = query.decl()
        if(len(query.children()) == 2):
            lhs, rhs = query.children()
            # if lhs in self.form_num.keys() and rhs in self.form_num.keys():
            if str(lhs.decl()) == '+':
                lhs_summands = get_summands(lhs)
                if len(lhs_summands)>0:
                    new_summands = []
                    num_changed = 0
                    for i in range(len(lhs_summands)):
                        new_simple_expr, changed = self.get_easy_expr_if(lhs_summands[i], top_level)
                        num_changed += changed
                        new_summands.append(new_simple_expr)
                    if num_changed >= 2:
                        new_lhs = new_summands[0]
                        for i in range(1, len(new_summands)):
                            new_lhs += new_summands[i]
                        return [top_level(new_lhs, rhs)]
                        # if (top_level == lt) or (top_level == le):
                        # elif (top_level == gt) or (top_level == ge):
                        #     return [top_level(new_lhs, rhs)]
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
    def get_easy_expr_if(self, expr, top_level):
        if str(expr.decl())=='If':
            lhs, rhs = expr.children()[1], expr.children()[2]
            if str(lhs.decl())=='*' and str(rhs.decl())=='*':
                common = []
                if simplify(lhs.children()[0] == rhs.children()[0]):
                    common = lhs.children()[0]
                    lhs_substrate = lhs.children()[1]
                    rhs_substrate = rhs.children()[1]
                elif simplify(lhs.children()[1] == rhs.children()[0]):
                    common = lhs.children()[1]
                    lhs_substrate = lhs.children()[0]
                    rhs_substrate = rhs.children()[1]
                elif simplify(lhs.children()[1] == rhs.children()[1]):
                    common = lhs.children()[1]
                    lhs_substrate = lhs.children()[0]
                    rhs_substrate = rhs.children()[0]
                elif simplify(lhs.children()[0] == rhs.children()[1]):
                    common = lhs.children()[0]
                    lhs_substrate = lhs.children()[1]
                    rhs_substrate = rhs.children()[0]
                else:
                    return expr, 0
                if lhs_substrate in self.form_num.keys() and rhs_substrate in self.form_num.keys():
                    num_lhs_substrate = self.form_num[lhs_substrate]
                    num_rhs_substrate = self.form_num[rhs_substrate]
                    self.up.clear()
                    self.down.clear()
                    if simplify(expr.children()[0] == (common >= 0)):
                        if str(top_level) == '<=':
                            self.travel_up(num_lhs_substrate)
                            self.travel_down(num_rhs_substrate)
                        elif str(top_level) == '>=':
                            self.travel_up(num_rhs_substrate)
                            self.travel_down(num_lhs_substrate)
                    elif simplify(expr.children()[0] == (common <= 0)):
                        if str(top_level) == '>=':
                            self.travel_up(num_lhs_substrate)
                            self.travel_down(num_rhs_substrate)
                        elif str(top_level) == '<=':
                            self.travel_up(num_rhs_substrate)
                            self.travel_down(num_lhs_substrate)
                    l1 = list(self.up)
                    l2 = list(self.down)
                    for i in range(len(l1)):
                        for j in range(len(l2)):
                            if simplify(self.num_form[l1[i]] == self.num_form[l2[j]]):
                                return common * self.num_form[l1[i]], 1
        return expr, 0


    
