import torch
import sys
from common.polyexp import PolyExpNew, Nlist
from common.abs_elem import Abs_elem
from utils import *

class ibp:
	# def affine(self, abs_elem, neighbours, prev: Nlist, curr: Nlist, poly_size, curr_size, prev_size, debug_flag):
	# 	l_new = plus(torch.where(ge(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, torch.tensor(0).unsqueeze(2)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('l', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('u', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1))).sum(dim=0), PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const)		
	# 	u_new = plus(torch.where(ge(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, torch.tensor(0).unsqueeze(2)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('u', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('l', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1))).sum(dim=0), PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const)		
	# 	return l_new, u_new
	def Relu(self, abs_elem, neighbours, prev, curr, poly_size, curr_size, prev_size, debug_flag=False):
		if ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(1)):
			l_new = abs_elem.get_elem_new('l', prev)			
			u_new = abs_elem.get_elem_new('u', prev)			
			return l_new, u_new
		else:
			if le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(1)):
				l_new = 0.0				
				u_new = 0.0				
				return l_new, u_new
			else:
				l_new = 0.0				
				u_new = abs_elem.get_elem_new('u', prev)				
				return l_new, u_new


	def Affine(self, abs_elem, neighbours, prev: Nlist, curr: Nlist, poly_size, curr_size, prev_size, debug_flag):
		cond = ge(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).get_mat(abs_elem), plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*curr_size, 1*poly_size))
		lhs_l = PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).get_mat(abs_elem), plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).get_mat(abs_elem)
		lhs_r = abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1)
		lhs = mult(lhs_l, lhs_r)
		rhs = mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).get_mat(abs_elem), plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).get_mat(abs_elem), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1))
		temp = torch.where(cond, lhs, rhs)
		print(temp.shape)
		p1 = temp.sum(dim=1).unsqueeze(1)
		
		
		
		
		p2 = PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).get_mat(abs_elem), plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const.unsqueeze(1)
		l_new = plus(p1, p2)		
		u_new = plus(torch.where(ge(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, torch.tensor(0).unsqueeze(2)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('u', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1)), mult(PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('l', Nlist(poly_size, 0, poly_size-1, None)).repeat( 1*curr_size, 1))).sum(dim=0), PolyExpNew(poly_size, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const)		
		return l_new, u_new