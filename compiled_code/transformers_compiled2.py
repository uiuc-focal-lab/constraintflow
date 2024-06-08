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
		var_0 = SymExp(curr_size, SymExp.count, torch.zeros(curr_size, SymExp.count), torch.zeros(curr_size), SymExp.count, SymExp.count).add_eps(curr_size)
		var_1 = SymExp(curr_size, SymExp.count, mult(divide(abs_elem.get_elem_new('u', prev), torch.tensor(2.0).unsqueeze(0).repeat(curr_size)).unsqueeze(1).repeat(1, sym_size), var_0.get_mat(SymExp.count)), mult(divide(abs_elem.get_elem_new('u', prev), torch.tensor(2.0).unsqueeze(0).repeat(curr_size)), var_0.get_const())0, SymExp.count)
		rewrite_4 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_5 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		Z_new = plus(mult(rewrite_5, abs_elem.get_elem_new('Z', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_5), plus(mult(rewrite_4, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_4), SymExp(curr_size, SymExp.count, var_1.get_mat(SymExp.count), plus(divide(abs_elem.get_elem_new('u', prev), torch.tensor(2.0).unsqueeze(0).repeat(curr_size)), var_1.get_const())0, SymExp.count)))))
		return l_new, u_new, Z_new
	
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
