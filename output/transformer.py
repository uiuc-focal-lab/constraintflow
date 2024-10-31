import torch
import sys
from lib.polyexp import PolyExpNew
from utils import *
class ibp:
	def Affine(self, abs_elem, neighbours, prev, curr, debug_flag=False):
		l_new = plus(torch.where(ge(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, torch.tensor(0).unsqueeze(2)), mult(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('l', Nlist(N, 0, N-1, None)).repeat( 1, 1)), mult(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('u', Nlist(N, 0, N-1, None)).repeat( 1, 1))).sum(dim=0), PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const)		
		u_new = plus(torch.where(ge(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, torch.tensor(0).unsqueeze(2)), mult(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('u', Nlist(N, 0, N-1, None)).repeat( 1, 1)), mult(PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).mat, abs_elem.get_elem_new('l', Nlist(N, 0, N-1, None)).repeat( 1, 1))).sum(dim=0), PolyExpNew(N, prev.dot(curr.get_metadata('weight')).mat, plus(curr.get_metadata('bias'), prev.dot(curr.get_metadata('weight')).const)).const)		
		return l_new, u_new
	def Relu(self, abs_elem, neighbours, prev, curr, debug_flag=False):
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
