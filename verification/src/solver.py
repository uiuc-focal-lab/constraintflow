from z3 import *
from graph import Opt_graph
import time
import threading

x = Real('x')
y = Real('y')
b = True
plus = (x + y).decl()
conjunction = (And(b, b)).decl()
lt = (x < y).decl()
le = (x <= y).decl()
gt = (x > y).decl()
ge = (x >= y).decl()
eqq = (x == y).decl()
if_ = If(x>0, x, y).decl()
comparison = [lt, le, gt, ge, eqq]

class Opt_solver:
    def __init__(self):
        self.solved = []
        self.counter = -1
        self.result_event = None
        self.final_answer = None

    def clear(self):
        self.solved = []
        self.counter = -1


    def check(self, lhs, rhs):
        # print('CHECKING NOW')
        # print(lhs)
        # print()
        # print(rhs)
        s = Solver()
        s.add(Not(Implies(lhs, rhs))) 
        ret = s.check()
        # if(ret == sat):
        #     print(s.model())
        # print('CHECKED')
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
        lhs = self.vars(lhs)
        rhs = self.vars(rhs)
        return (flag, len(lhs.intersection(rhs)))

    def get_sufficient_formulae(self, lhs, rhs):        
        g = Opt_graph(lhs)
        l = g.get_sufficient_queries(rhs)
        l.sort(reverse=True, key=self.priority)
        # print(len(l))
        # if len(l)==5:
        #     for i in range(len(l)):
        #         print(l[i])
        return l

    def get_sub_lemmas(self, rhs, default_order = False):
        top_level = rhs.decl()
        if top_level not in comparison:
            return None 
        left, right = rhs.children()
        if left.decl() == if_:
            # return [(left, right)]
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
            # if (len(left_)-1) % (len(right_)-1):
            #     m.append((left_[-1], right_[-1]))
            #     r = (len(left_)-1) / (len(right_)-1)
            #     for i in range(len(right_)-1):
            #         l = left_[i*r]
            #         for j in range(1, r):
            #             l += left_[i*r+j]
            #         m.append((l, right_[i]))
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
        if(len(m) == 1):
            res = self.check(lhs, top_level(*m[0]))
            if not res:
                return False 
            return True
        i = 0
        for r in m:
            #print(i)
            i= i+1
            # print(r)
            l_, r_ = r
            l_top_level = l_.decl()
            r_top_level = r_.decl()
            if l_top_level == if_ and r_top_level == if_:
                # print('here')
                if l_.children()[0] == r_.children()[0]:
                    # print('here')
                    # t1 = self.check(And(lhs, l_.children()[0]), top_level(l_.children()[1], r_.children()[1]))
                    ll = lhs 
                    rr = top_level(l_.children()[1], r_.children()[1])
                    # print(ll)
                    # print(rr)
                    t1 = self.check(ll, rr)
                    # t1 = self.check(lhs, top_level(l_.children()[1], r_.children()[1]))
                    # print('here')
                    # t2 = self.check(And(lhs, Not(l_.children()[0])), top_level(l_.children()[2], r_.children()[2]))
                    t2 = self.check(lhs, top_level(l_.children()[2], r_.children()[2]))
                    # print('here')
                    if  (t1 and t2):
                        continue 
            res = self.check(lhs, top_level(*r))
            if not res:
                return False 
        return True 
    
    def opt_solve(self, lhs, rhs):
        
        m = self.get_sub_lemmas(rhs)
        if m:
            if self.solve_sub_lemma(lhs, m, rhs.decl()):
                return True 
            m = self.get_sub_lemmas(rhs, default_order=True)
            if self.solve_sub_lemma(lhs, m, rhs.decl()):
                return True 
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
            if eqq(rhs_left, rhs_right) in lhs_:
                return True 
        if top_level in [ge]:
            if ge(rhs_left, rhs_right) in lhs_:
                return True 
            if eqq(rhs_left, rhs_right) in lhs_:
                return True 
        return False
    
    def solve_temp(self, lhs, rhs):
        m = self.get_sufficient_formulae(lhs, rhs)
        if m:
            # i=0
            # for r in m:
            #     if self.fast_solve(lhs, r):
            #         return True 
            # i=0
            for r in m:
                # print(i)
                # i = i+1
                if self.opt_solve(lhs, r):
                    return True 
        return self.opt_solve(lhs, rhs)
    
    def check_quantifier(self, lhs, rhs):
        return isinstance(rhs, z3.z3.QuantifierRef)
    
    # def solve_without_opt(self, lhs, rhs):
    #     ret = self.check(lhs, rhs)
    #     # if ret:
    #     self.final_answer = ret 
    #     self.result_event.set()

    #     return  
    
    # def solve_with_opt(self, lhs, rhs):
    #     if self.check_quantifier(lhs, rhs):
    #         return self.check(lhs, rhs)
    #     m_if = self.check_if(lhs, rhs)
    #     for i in m_if:
    #         ret = self.solve_temp(*i)
    #         if not ret:
    #             # print(lhs)
    #             # print()
    #             # print(rhs)
    #             self.final_answer = ret 
    #             self.result_event.set()
    #             return  
    #     self.final_answer = ret 
    #     self.result_event.set()
    #     return 

    # def solve(self, lhs, rhs):
    #     self.final_answer = None
    #     self.result_event = threading.Event()
    #     thread1 = threading.Thread(target=self.solve_without_opt, args=(lhs, rhs))
    #     thread2 = threading.Thread(target=self.solve_with_opt, args=(lhs, rhs))

    #     thread1.start()
    #     thread2.start()

    #     # Wait for one of the threads to finish
    #     self.result_event.wait()
    #     # print('hdsgfjhgsd')
    #     # thread1.join()
    #     # thread2.join()
    #     return self.final_answer
    def solve(self, lhs, rhs):
        # print(lhs)
        # print()
        # print(rhs)
        # print()
        # return self.check(lhs, rhs)
        
        if self.check_quantifier(lhs, rhs):
            return self.check(lhs, rhs)

        m_if = self.check_if(lhs, rhs)

        # if len(m_if)<=1:
        #     ret = self.solve_temp(*m_if[0])
        #     if not ret:
        #         # print(lhs)
        #         # print()
        #         # print(rhs)
        #         return False
        #     return True
        # for i in m_if:
        #     ret = self.solve(*i)
        #     if not ret:
        #         # print(lhs)
        #         # print()
        #         # print(rhs)
        #         return False
        # return True
        for i in m_if:
            ret = self.solve_temp(*i)
            if not ret:
                # print(lhs)
                # print()
                # print(rhs)
                return False 
        return True