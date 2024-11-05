from z3 import *

one = Real('one')
minus_one = Real('minus_one')

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

def z3_vars(x):
	# if not isinstance(x, z3.z3.ArithRef):
	# 	return set()
	if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool):
		return set()
	if x.children() == []:
		if isinstance(x, float) or isinstance(x, int) or isinstance(x, bool) or isinstance(x, z3.z3.RatNumRef)  or isinstance(x, z3.z3.IntNumRef) or isinstance(x, z3.z3.BoolRef):
			return set()
		return {x}
	s = set()
	for i in x.children():
		s = s.union(z3_vars(i))
	return s

def z3min(a, b):
	return If(a<b, a, b)

def z3max(a, b):
	return If(a>b, a, b)

def z3max_list(l):
	if len(l)==0:
		return None 
	if len(l)==1:
		return l[0]
	l_temp = z3max_list(l[1:])
	return If(l[0]>l_temp, l[0], l_temp)

def z3min_list(l):
	if len(l)==0:
		return None 
	if len(l)==1:
		return l[0]
	l_temp = z3min_list(l[1:])
	return If(l[0]<l_temp, l[0], l_temp)

def get_all_combs(E, es, i):
	if i>=len(E):
		return es 
	eps = E[i]
	e_vars = z3_vars(es[0])
	if not eps in e_vars:
		return get_all_combs(E, es, i+1)
	es_new = []
	for e in es:
		temp_1 = substitute(e, (eps, one))
		temp_2 = substitute(e, (eps, minus_one))
		es_new.append(temp_1)
		es_new.append(temp_2)
	return get_all_combs(E, es_new, i+1)

def get_z3max_eps(E, e):
	es = get_all_combs(E, [e], 0)
	return z3max_list(es)

def get_z3min_eps(E, e):
	es = get_all_combs(E, [e], 0)
	return z3min_list(es)


def get_summands(expr):
    if str(expr.decl()) == '+':
        return get_summands(expr.children()[0]) + get_summands(expr.children()[1])
    else:
        return [expr]



class Number:

	def __init__(self):
		self.i = 0

	def nextn(self):
		self.i = self.i + 1
		return self.i

