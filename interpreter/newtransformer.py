from common.abs_elem import Abs_elem
from common.polyexp import PolyExp
from common.symexp import SymExp
from common.transformer import Transformer
import torch
import functools


def deepz_lower(n, c, abs_elem, neighbours):
	return -1 if c >= 0 else 1

def deepz_upper(n, c, abs_elem, neighbours):
	return 1 if c >= 0 else 1

def priority(n, abs_elem, neighbours):
	return n[0]

class Cflowzono(Transformer):
	def relu(self, abs_elem, neighbours, prev, curr):
		if abs_elem.get_elem('l', prev) >= 0:
			l_new = abs_elem.get_elem('l', prev)
			u_new = abs_elem.get_elem('u', prev)
			Z_new = abs_elem.get_elem('Z', prev)
		else:
			if abs_elem.get_elem('u', prev) <= 0:
				l_new = 0.0
				u_new = 0.0
				Z_new = 0.0
			else:
				l_new = 0.0
				u_new = abs_elem.get_elem('u', prev)
				Z_new = ((SymExp().new_symbol().populate(coeff=1)).mult(abs_elem.get_elem('u', prev) / 2.0)).add(abs_elem.get_elem('u', prev) / 2.0)

		if isinstance(l_new,PolyExp) or isinstance(l_new,SymExp):
			l_new = l_new.get_const()
		if isinstance(u_new,PolyExp) or isinstance(u_new,SymExp):
			u_new = u_new.get_const()

		if isinstance(Z_new,float) or isinstance(Z_new,int):
			Z_new = SymExp().populate(Z_new)
		return l_new,u_new,Z_new

	def fc(self, abs_elem, neighbours, prev, curr, w, b):
		temp = PolyExp(abs_elem.shapes)
		temp.populate(b, prev, w)
		l_new = (functools.reduce(lambda a_dot, b_dot: a_dot.add(b_dot),[i_dot[0].mult(i_dot[1]) for i_dot in zip([abs_elem.get_elem('Z', i_element) for i_element in prev],w)])).add(b).map(abs_elem, neighbours, deepz_lower)
		u_new = (functools.reduce(lambda a_dot, b_dot: a_dot.add(b_dot),[i_dot[0].mult(i_dot[1]) for i_dot in zip([abs_elem.get_elem('Z', i_element) for i_element in prev],w)])).add(b).map(abs_elem, neighbours, deepz_upper)
		Z_new = (functools.reduce(lambda a_dot, b_dot: a_dot.add(b_dot),[i_dot[0].mult(i_dot[1]) for i_dot in zip([abs_elem.get_elem('Z', i_element) for i_element in prev],w)])).add(b)

		if isinstance(l_new,PolyExp) or isinstance(l_new,SymExp):
			l_new = l_new.get_const()
		if isinstance(u_new,PolyExp) or isinstance(u_new,SymExp):
			u_new = u_new.get_const()

		if isinstance(Z_new,float) or isinstance(Z_new,int):
			Z_new = SymExp().populate(Z_new)
		return l_new,u_new,Z_new
