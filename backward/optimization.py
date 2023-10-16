from z3 import *

x = Real('x')
y = Real('y')
b = True
plus = (x + y).decl()
conjunction = (And(b, b)).decl()
lt = (x<y).decl()
le = (x<=y).decl()
gt = (x>y).decl()
ge = (x>=y).decl()
eq = (x==y).decl()
comparison = [lt, le, gt, ge, eq]

def common_vars(a,b):
    return len(a.intersection(b))>0

def vars(x):
    if x.children() == []:
        if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool) or isinstance(x, z3.z3.RatNumRef)  or isinstance(x, z3.z3.IntNumRef) or isinstance(x, z3.z3.BoolRef):
            return set()
        return {x}
    s = set()
    for i in x.children():
        s = s.union(vars(i))
    return s

def get_summands(x):
    top_level = x.decl()
    if top_level != plus:
        return [x]
    else:
        left = x.children()[0]
        right = x.children()[1]
        return get_summands(left) + get_summands(right)


def get_clauses(x):
    top_level = x.decl()
    if top_level != conjunction:
        return [x]
    else:
        l = []
        children = x.children()
        for child in children:
            l = l + get_clauses(child)
            # l.append(get_clauses(child))
        # left = x.children()[0]
        # right = x.children()[1]
        # return get_clauses(left) + get_clauses(right)
        return l

def opt_insert(lhs, rhs):
    top_level = rhs.decl()
    if top_level not in comparison:
        return None 
    rhs_left = rhs.children()[0]
    rhs_right = rhs.children()[1]
    lhs = get_clauses(lhs)
    m = []
    for l in lhs:
        if l.decl()==top_level:
            l_left, l_right = l.children()
            if l_right == rhs_right:
                m.append(top_level(rhs_left, l_left))
            if l_left == rhs_left:
                m.append(top_level(l_right, rhs_right))
    if len(m)==0:
        return None 
    return m


def optimization(z3constraint):
    top_level = z3constraint.decl()
    if top_level not in comparison:
        return None 
    left = get_summands(z3constraint.children()[0])
    right = get_summands(z3constraint.children()[1])
    if len(left)!=len(right):
        return None 
    m = []
    print(left)
    print(right)
    for i in range(len(left)):
        flag = False
        v1 = vars(left[i])
        for j in range(len(right)):
            v2 = vars(right[j])
            if len(v1.intersection(v2)) > 0:
                m.append((left[i], right[j]))
                del right[j]
                flag = True 
                break 
        if not flag:
            return None 
    print(m)
    return m 