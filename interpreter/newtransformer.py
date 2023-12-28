from common.abs_elem import Abs_elem
from common.polyexp import PolyExp
from common.symexp import SymExp
from common.transformer import Transformer
import torch


def simplify_lower(n, coeff, abs_elem, neighbors):
	return coeff * abs_elem.get_elem('l', n) if coeff >= 0 else coeff * abs_elem.get_elem('u', n)

def simplify_upper(n, coeff, abs_elem, neighbors):
	return coeff * abs_elem.get_elem('u', n) if coeff >= 0 else coeff * abs_elem.get_elem('l', n)

def replace_lower(n, coeff, abs_elem, neighbors):
	return (abs_elem.get_elem('L', n)).mult(coeff) if coeff >= 0 else (abs_elem.get_elem('U', n)).mult(coeff)

def replace_upper(n, coeff, abs_elem, neighbors):
	return (abs_elem.get_elem('U', n)).mult(coeff) if coeff >= 0 else (abs_elem.get_elem('L', n)).mult(coeff)

def priority(n, abs_elem, neighbors):
	return n[0]

def priority2(n, abs_elem, neighbors):
	return -n[0]

def backsubs_lower(e, n, abs_elem, neighbors):
	return e.traverse(abs_elem, neighbors, True, priority2, replace_lower).map(abs_elem, simplify_lower)

def backsubs_upper(e, n, abs_elem, neighbors):
	return e.traverse(abs_elem, neighbors, True, priority2, replace_upper).map(abs_elem, simplify_upper)

def f(n1, n2, abs_elem, neighbors):
	return abs_elem.get_elem('l', n1) >= abs_elem.get_elem('u', n2)

class Cflowdeeppoly(Transformer):
	def relu(self, abs_elem, neighbors, prev, curr):
		if abs_elem.get_elem('l', prev) >= 0:
			l_new = abs_elem.get_elem('l', prev)
			u_new = abs_elem.get_elem('u', prev)
			L_new = prev
			U_new = prev
		else:
			if abs_elem.get_elem('u', prev) <= 0:
				l_new = 0
				u_new = 0
				L_new = 0
				U_new = 0
			else:
				l_new = 0
				u_new = abs_elem.get_elem('u', prev)
				L_new = 0
				U_new = ((prev).mult(abs_elem.get_elem('u', prev) / abs_elem.get_elem('u', prev) - abs_elem.get_elem('l', prev))).minus(abs_elem.get_elem('u', prev) * abs_elem.get_elem('l', prev) / abs_elem.get_elem('u', prev) - abs_elem.get_elem('l', prev))

		if isinstance(l_new,PolyExp) or isinstance(l_new,SymExp):
			l_new = l_new.get_const()
		if isinstance(u_new,PolyExp) or isinstance(u_new,SymExp):
			u_new = u_new.get_const()
		return l_new,u_new,L_new,U_new
