from z3 import *
from graph import Opt_graph
import time

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
        self.solved = []
        self.counter = -1

    def clear(self):
        self.solved = []
        self.counter = -1


    def reveal(self):
        for l, r in self.solved:
            print(l)
            print(r)
            print()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def check(self, lhs, rhs):
        # for i in range(len(self.solved)):
        #     clauses_l, r = self.solved[self.counter]
        #     self.counter = (self.counter+1)%len(self.solved)
        #     clauses_lhs = set(self.get_clauses(lhs))
        #     if rhs==r:
        #         if len(clauses_l - clauses_lhs)==0:
        #             # print(i)
        #             return True             
        s = Solver()
        s.add(Not(Implies(lhs, rhs))) 
        # print('reached here')
        # print(time.time())
        ret = s.check()
        # print(time.time())
        # if ret==sat:
        #     print(lhs)
        #     print()
        #     print(rhs)
        #     mjh
        # if ret==unsat:
        #     self.solved.append((set(self.get_clauses(lhs)), rhs))
        return (ret==unsat)
    
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
        flag = 0
        if lhs.decl()==plus and rhs.decl()==plus:
            if len(self.get_summands(lhs)) == len(self.get_summands(rhs)):
                flag = 1
                # return 100
        lhs = self.vars(lhs)
        rhs = self.vars(rhs)
        return (flag, len(lhs.intersection(rhs)))

    def get_sufficient_formulae(self, lhs, rhs):        
        g = Opt_graph(lhs)
        l = g.get_sufficient_queries(rhs)
        # print('!!!!!!!!!!!!!!!!!!!!!!')
        # print(l)
        l.sort(reverse=True, key=self.priority)
        # print(l)
        # for i in l:

        return l

    def get_sub_lemmas(self, rhs, default_order = False):
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
            return m 
        left_ = self.get_summands(left)
        right_ = self.get_summands(right)
        m = []
        if default_order:
            for i in range(len(left_)):
                m.append((left_[i], right_[i]))
            return m
        if len(left_)!=len(right_):
            return None 
        for i in range(len(left_)):
            flag = False
            v1 = self.vars(left_[i])
            for j in range(len(right_)):
                v2 = self.vars(right_[j])
                if len(v1.intersection(v2)) > 0:
                    m.append((left_[i], right_[j]))
                    del right_[j]
                    flag = True 
                    break 
            if not flag:
                m = []
                break
        if flag:
            return m 
        else:
            left_ = self.get_summands(left)
            right_ = self.get_summands(right)
            for i in range(len(left_)):
                m.append((left_[i], right_[i]))
            return m
    
    def solve_sub_lemma(self, lhs, m, top_level):
        for r in m:
            res = self.check(lhs, top_level(*r))
            if not res:
                return False 
            # print('done') 
        return True 
    
    def opt_solve(self, lhs, rhs):
        m = self.get_sub_lemmas(rhs)
        if m:
            if self.solve_sub_lemma(lhs, m, rhs.decl()):
                return True 
            m = self.get_sub_lemmas(rhs, default_order=True)
            if self.solve_sub_lemma(lhs, m, rhs.decl()):
                return True 
        # print(m)
        # print(lhs)
        # print()
        # print(rhs)
        # dsjg
        return self.check(lhs, rhs) 

    def check_if(self, lhs, rhs):
        top_level1 = rhs.decl()
        if top_level1 in comparison:
            rhs_l, rhs_r = rhs.children()
            top_level2 = rhs_l.decl()
            if top_level2 == if_:
                lhs1 = And(lhs, rhs_l.children()[0])
                rhs1 = top_level1(rhs_l.children()[1], rhs_r)

                lhs2 = And(lhs, Not(rhs_l.children()[0]))
                rhs2 = top_level1(rhs_l.children()[2], rhs_r)
                return [(lhs1, rhs1), (lhs2, rhs2)]
        return [(lhs, rhs)]
    
    def fast_solve(self, lhs, rhs):
        top_level = rhs.decl()
        rhs_left, rhs_right = rhs.children()
        lhs_ = self.get_clauses(lhs)
        if top_level in [le]:
            if le(rhs_left, rhs_right) in lhs_:
                return True 
            if eq(rhs_left, rhs_right) in lhs_:
                return True 
        if top_level in [ge]:
            if ge(rhs_left, rhs_right) in lhs_:
                return True 
            if eq(rhs_left, rhs_right) in lhs_:
                return True 
        # print(lhs)
        # print()
        # print(rhs)
        # dsj
        return False
    
    def solve_temp(self, lhs, rhs):
        m = self.get_sufficient_formulae(lhs, rhs)
        # m.append(rhs)
        # print(lhs)
        # print()
        # for r in m:
        #     print(r)
        #     print()
        # jghsd
        if m:
            for r in m:
                if self.fast_solve(lhs, r):
                    return True 
            for r in m:
                if self.opt_solve(lhs, r):
                    return True 
        # return False 
        # print(lhs)
        # print()
        # print(rhs)
        # dsh
        return self.opt_solve(lhs, rhs)
    
    def check_quantifier(self, lhs, rhs):
        return isinstance(rhs, z3.z3.QuantifierRef)
    
    # def handle_exists(self, lhs, rhs):
        

    def solve(self, lhs, rhs):
        print(lhs)
        print()
        print(rhs)
        print()
        if self.check_quantifier(lhs, rhs):
            return self.check(lhs, rhs)
        m_if = self.check_if(lhs, rhs)
        for i in m_if:
            ret = self.solve_temp(*i)
            if not ret:
                return False 
        return True