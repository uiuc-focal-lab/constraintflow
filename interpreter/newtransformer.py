from lib.abs_elem import Abs_elem
# from lib.polyexp import PolyExp
from lib.symexp import SymExp
from lib.transformer import Transformer
import torch
import functools


def simplify_lower(n, coeff, abs_elem, neighbours):
	return (coeff) * (abs_elem.get_elem('l', n)) if coeff >= 0 else (coeff) * (abs_elem.get_elem('u', n))

def simplify_upper(n, coeff, abs_elem, neighbours):
	return (coeff) * (abs_elem.get_elem('u', n)) if coeff >= 0 else (coeff) * (abs_elem.get_elem('l', n))

def priority(n, abs_elem, neighbours):
	return n[0]

class Cflowibp(Transformer):
	def fc(self, abs_elem, neighbours, prev, curr, w, b):
		temp = PolyExp(abs_elem.shapes)
		temp.populate(b, prev, w)
		l_new = temp.copy().map(abs_elem, neighbours, simplify_lower)
		u_new = temp.copy().map(abs_elem, neighbours, simplify_upper)

		if isinstance(l_new,PolyExp) or isinstance(l_new,SymExp):
			l_new = l_new.get_const()
		if isinstance(u_new,PolyExp) or isinstance(u_new,SymExp):
			u_new = u_new.get_const()

		return l_new,u_new

	def relu(self, abs_elem, neighbours, prev, curr):
		if abs_elem.get_elem('l', prev) >= 0:
			l_new = abs_elem.get_elem('l', prev)
			u_new = abs_elem.get_elem('u', prev)
		else:
			if abs_elem.get_elem('u', prev) <= 0:
				l_new = 0.0
				u_new = 0.0
			else:
				l_new = 0.0
				u_new = abs_elem.get_elem('u', prev)

		if isinstance(l_new,PolyExp) or isinstance(l_new,SymExp):
			l_new = l_new.get_const()
		if isinstance(u_new,PolyExp) or isinstance(u_new,SymExp):
			u_new = u_new.get_const()

		return l_new,u_new
