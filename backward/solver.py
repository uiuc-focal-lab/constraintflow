from z3 import *
from graph import Opt_graph

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
if_ = If(x>0, x, y).decl()
comparison = [lt, le, gt, ge, eq]

class Opt_solver:
    def __init__(self):
        pass

    def check(self, lhs, rhs):
        s = Solver()
        s.add(Not(Implies(lhs, rhs))) 
        return (s.check()==unsat)
    
    def common_vars(self, a, b):
        return len(a.intersection(b))>0

    def vars(self, x):
        if x.children() == []:
            if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool) or isinstance(x, z3.z3.RatNumRef)  or isinstance(x, z3.z3.IntNumRef) or isinstance(x, z3.z3.BoolRef):
                return set()
            return {x}
        s = set()
        for i in x.children():
            s = s.union(self.vars(i))
        return s

    def get_summands(self, x):
        top_level = x.decl()
        if top_level != plus:
            return [x]
        else:
            left = x.children()[0]
            right = x.children()[1]
            return self.get_summands(left) + self.get_summands(right)


    def get_clauses(self, x):
        top_level = x.decl()
        if top_level != conjunction:
            return [x]
        else:
            l = []
            children = x.children()
            for child in children:
                l = l + self.get_clauses(child)
            return l

    def priority(self, q):
        lhs, rhs = q.children()
        lhs = self.vars(lhs)
        rhs = self.vars(rhs)
        return len(lhs.intersection(rhs))

    def get_sufficient_formulae(self, lhs, rhs):        
        g = Opt_graph(lhs)
        l = g.get_sufficient_queries(rhs)
        l.sort(reverse=True, key=self.priority)
        return l

    def get_sub_lemmas(self, rhs):
        top_level = rhs.decl()
        if top_level not in comparison:
            return None 
        left, right = rhs.children()
        if left.decl() == if_:
            m1 = self.get_sub_lemmas(top_level(left.children()[1], right))
            m2 = self.get_sub_lemmas(top_level(left.children()[2], right))
            if not m1:
                m1 = [(left.children()[1], right)]
            if not m2:
                m2 = [(left.children()[2], right)]
            m = m1 + m2
            # print('start printing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # for i in m:
            #     print(type(i))
            #     print(i)
            #     print()
            # jsdhgj
            return m 
        left = self.get_summands(left)
        right = self.get_summands(right)
        if len(left)!=len(right):
            return None 
        m = []
        for i in range(len(left)):
            flag = False
            v1 = self.vars(left[i])
            for j in range(len(right)):
                v2 = self.vars(right[j])
                if len(v1.intersection(v2)) > 0:
                    m.append((left[i], right[j]))
                    del right[j]
                    flag = True 
                    break 
            if not flag:
                return None 
        return m 
    
    def solve_sub_lemma(self, lhs, m, top_level):
        # print(len(m))
        # if len(m)==5:
        #     print(m[-1])
        #     jhsdg
        # i=1
        for r in m:
            res = self.check(lhs, top_level(*r))
            # print(i)
            # i+=1
            if not res:
                return False  
        return True 
    
    def opt_solve(self, lhs, rhs):
        m = self.get_sub_lemmas(rhs)
        if m:
            if self.solve_sub_lemma(lhs, m, rhs.decl()):
                return True 
        return self.check(lhs, rhs) 

    def check_if(self, lhs, rhs):
        top_level1 = rhs.decl()
        # print(top_level)
        if top_level1 in comparison:
            rhs_l, rhs_r = rhs.children()
            top_level2 = rhs_l.decl()
            if top_level2 == if_:
                lhs1 = And(lhs, rhs_l.children()[0])
                rhs1 = top_level1(rhs_l.children()[1], rhs_r)
                # m1 = self.check_if(lhs1, rhs1)

                # lhs2 = And(lhs, Not(rhs.children()[0]))
                # rhs2 = rhs.children()[2]
                lhs2 = And(lhs, Not(rhs_l.children()[0]))
                rhs2 = top_level1(rhs_l.children()[2], rhs_r)
                # m2 = self.check_if(lhs2, rhs2)
                return [(lhs1, rhs1), (lhs2, rhs2)]
                return m1 + m2 
        return [(lhs, rhs)]
    
    def solve_temp(self, lhs, rhs):
        # print()
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(lhs)
        # print()
        # print(rhs)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print()
        m = self.get_sufficient_formulae(lhs, rhs)
        if m:
            for r in m:
                if self.opt_solve(lhs, r):
                    return True 
        return self.opt_solve(lhs, rhs)
    
    def solve(self, lhs, rhs):
        # print()
        # print(lhs)
        # print()
        # print(rhs)
        # print()
        m_if = self.check_if(lhs, rhs)
        print(len(m_if))
        # sjdh
        # print(m_if)
        for i in m_if:
            ret = self.solve_temp(*i)
            if not ret:
                return False 
        return True