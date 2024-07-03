import torch
from common.polyexp import PolyExp, Nlist, SymExp
from utils import *
class zono:
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_7 = abs_elem.get_elem_new('u', prev)
		cse_var_9 = torch.tensor(0).unsqueeze(0).repeat(curr_size)
		cse_var_6 = le(cse_var_7, cse_var_9)
		cse_var_10 = abs_elem.get_elem_new('l', prev)
		cse_var_8 = ge(cse_var_10, cse_var_9)
		cse_var_18 = convert_to_float(cse_var_6)
		cse_var_17 = convert_to_float(cse_var_8)
		cse_var_19 = torch.tensor(1.0).unsqueeze(0).repeat(curr_size)
		cse_var_23 = minus(cse_var_19, cse_var_18)
		cse_var_24 = mult(cse_var_18, cse_var_9)
		cse_var_25 = minus(cse_var_19, cse_var_17)
		l_new = plus(mult(cse_var_17, cse_var_10), mult(cse_var_25, plus(cse_var_24, mult(cse_var_23, cse_var_9))))
		u_new = plus(mult(cse_var_17, cse_var_7), mult(cse_var_25, plus(cse_var_24, mult(cse_var_23, cse_var_7))))
		var_0 = SymExp(curr_size, SymExp.count, torch.zeros(curr_size, SymExp.count), torch.zeros(curr_size), SymExp.count, SymExp.count).add_eps(curr_size)
		cse_var_5 = divide(cse_var_7, torch.tensor(2.0).unsqueeze(0).repeat(curr_size))
		var_1 = SymExp(curr_size, SymExp.count, mult(cse_var_5.unsqueeze(1).repeat(1, SymExp.count), var_0.get_mat(SymExp.count)), mult(cse_var_5, var_0.get_const()), 0, SymExp.count)
		cse_var_3 = SymExp(curr_size, SymExp.count, var_1.get_mat(SymExp.count), plus(cse_var_5, var_1.get_const()), 0, SymExp.count)
		cse_var_4 = SymExp(curr_size, SymExp.count, torch.zeros(curr_size, SymExp.count), 0.0, 0, SymExp.count)
		rewrite_4 = cse_var_18.unsqueeze(1).repeat(1, SymExp.count)
		cse_var_1 = SymExp(curr_size, SymExp.count, plus(mult(rewrite_4, cse_var_4.get_mat(SymExp.count)), mult(cse_var_23.unsqueeze(1).repeat(1, SymExp.count), cse_var_3.get_mat(SymExp.count))), plus(mult(cse_var_18, cse_var_4.get_const()), mult(cse_var_23, cse_var_3.get_const())), 0, SymExp.count)
		cse_var_2 = abs_elem.get_elem_new('Z', prev)
		rewrite_6 = cse_var_17.unsqueeze(1).repeat(1, SymExp.count)
		Z_new = SymExp(curr_size, SymExp.count, plus(mult(rewrite_6, cse_var_2.get_mat(SymExp.count)), mult(cse_var_25.unsqueeze(1).repeat(1, SymExp.count), cse_var_1.get_mat(SymExp.count))), plus(mult(cse_var_17, cse_var_2.get_const()), mult(cse_var_25, cse_var_1.get_const())), 0, SymExp.count)
		return l_new, u_new, Z_new
	
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_14 = abs_elem.get_elem_new('Z', prev)
		cse_var_15 = curr.get_metadata('weight')
		cse_var_13 = SymExp(curr_size, SymExp.count, inner_prod(cse_var_15, cse_var_14.get_mat(SymExp.count)), inner_prod(cse_var_15, cse_var_14.get_const()), 0, SymExp.count)
		cse_var_12 = curr.get_metadata('bias')
		cse_var_16 = SymExp(curr_size, SymExp.count, cse_var_13.get_mat(SymExp.count), plus(cse_var_13.get_const(), cse_var_12), 0, SymExp.count)
		cse_var_22 = cse_var_16.get_mat(SymExp.count)
		cse_var_11 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, SymExp.count)
		cse_var_27 = convert_to_float(gt(cse_var_22, cse_var_11))
		rewrite_8 = cse_var_27
		cse_var_20 = cse_var_16.get_const()
		cse_var_21 = torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, SymExp.count)
		cse_var_26 = neg(cse_var_22)
		l_new = plus(neg(plus(mult(rewrite_8, cse_var_22), mult(minus(cse_var_21, rewrite_8), cse_var_26))).sum(dim=1), cse_var_20)
		rewrite_9 = cse_var_27
		u_new = plus(plus(mult(rewrite_9, cse_var_22).sum(dim=1), mult(minus(cse_var_21, rewrite_9), cse_var_26).sum(dim=1)), cse_var_20)
		return l_new, u_new, cse_var_16
	
