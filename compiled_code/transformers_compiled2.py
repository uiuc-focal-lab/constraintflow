import torch
from common.polyexp import PolyExp, Nlist
from utils import *
class zono:
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		rewrite_0 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_1 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		l_new = plus(mult(rewrite_1, abs_elem.get_elem_new('l', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_1), plus(mult(rewrite_0, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_0), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))))
		rewrite_2 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_3 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		u_new = plus(mult(rewrite_3, abs_elem.get_elem_new('u', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_3), plus(mult(rewrite_2, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_2), abs_elem.get_elem_new('u', prev)))))
