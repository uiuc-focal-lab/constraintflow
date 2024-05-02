import torch
from common.polyexp import PolyExp, Nlist
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_15 = prev.dot(curr.get_metadata('weight'))
		var_0 = cse_var_15
		cse_var_14 = curr.get_metadata('bias')
		cse_var_18 = plus(cse_var_14, var_0.get_const())
		var_1 = PolyExp(curr_size, poly_size, var_0.get_mat(poly_size), cse_var_18)
		trav_size_0_0 = get_shape_0(cse_var_18)
		phi_trav_exp1_1_1 = var_1
		phi_trav_size_1_1 = trav_size_0_0
		phi_trav_exp1_2_3 = var_1
		phi_trav_size_2_3 = trav_size_0_0
		while(True):
			trav_size_1_2 = get_shape_0(phi_trav_exp1_1_1.get_const())
			vertices_stop1 = False
			vertices1_1_1 = ne(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(trav_size_1_2, poly_size))
			vertices_stop_default1 = get_default_stop([trav_size_1_2, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).repeat(trav_size_1_2, poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			phi_trav_size_2_3 = trav_size_1_2
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				phi_trav_size_2_3 = trav_size_1_2
				break
			cse_var_12 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_2 = abs_elem.get_elem_new('L', cse_var_12)
			var_3 = abs_elem.get_elem_new('U', cse_var_12)
			cse_var_10 = phi_trav_exp1_1_1.get_mat(poly_size)
			cse_var_11 = ge(cse_var_10, torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(trav_size_1_2, poly_size))
			rewrite_1 = convert_to_float(cse_var_11.unsqueeze(2).repeat(1, 1, poly_size))
			rewrite_2 = convert_to_float(cse_var_11)
			cse_var_8 = mult(cse_var_10, plus(mult(rewrite_2, var_2.get_const().repeat(trav_size_1_2, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(trav_size_1_2, poly_size), rewrite_2), var_3.get_const().repeat(trav_size_1_2, 1)))).sum(dim=1)
			cse_var_9 = mult(cse_var_10.unsqueeze(2).repeat(1, 1, poly_size), plus(mult(rewrite_1, var_2.get_mat(poly_size).repeat(trav_size_1_2, 1, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(trav_size_1_2, poly_size, poly_size), rewrite_1), var_3.get_mat(poly_size).repeat(trav_size_1_2, 1, 1)))).sum(dim=1)
			trav_exp1_4_2 = PolyExp(trav_size_1_2, poly_size, cse_var_9, plus(phi_trav_exp1_1_1.get_const(), cse_var_8))
			phi_trav_exp1_1_1 = trav_exp1_4_2
			phi_trav_size_1_1 = trav_size_1_2
		cse_var_17 = phi_trav_exp1_2_3.get_mat(poly_size)
		rewrite_0 = convert_to_float(ge(cse_var_17, torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(phi_trav_size_2_3, poly_size)))
		cse_var_16 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		l_new = plus(mult(cse_var_17, plus(mult(rewrite_0, abs_elem.get_elem_new('l', cse_var_16).repeat(phi_trav_size_2_3, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(phi_trav_size_2_3, poly_size), rewrite_0), abs_elem.get_elem_new('u', cse_var_16).repeat(phi_trav_size_2_3, 1)))).sum(dim=1), phi_trav_exp1_2_3.get_const())
		var_5 = cse_var_15
		cse_var_13 = plus(cse_var_14, var_5.get_const())
		var_6 = PolyExp(curr_size, poly_size, var_5.get_mat(poly_size), cse_var_13)
		trav_size_2_4 = get_shape_0(cse_var_13)
		phi_trav_exp2_5_1 = var_6
		phi_trav_size_5_5 = trav_size_2_4
		phi_trav_exp2_6_3 = var_6
		phi_trav_size_6_7 = trav_size_2_4
		while(True):
			trav_size_5_6 = get_shape_0(phi_trav_exp2_5_1.get_const())
			vertices_stop2 = False
			vertices2_5_1 = ne(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(trav_size_5_6, poly_size))
			vertices_stop_default2 = get_default_stop([trav_size_5_6, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).repeat(trav_size_5_6, poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			phi_trav_size_6_7 = trav_size_5_6
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				phi_trav_size_6_7 = trav_size_5_6
				break
			cse_var_5 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_7 = abs_elem.get_elem_new('U', cse_var_5)
			var_8 = abs_elem.get_elem_new('L', cse_var_5)
			cse_var_3 = phi_trav_exp2_5_1.get_mat(poly_size)
			cse_var_4 = ge(cse_var_3, torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(trav_size_5_6, poly_size))
			rewrite_4 = convert_to_float(cse_var_4.unsqueeze(2).repeat(1, 1, poly_size))
			rewrite_5 = convert_to_float(cse_var_4)
			cse_var_1 = mult(cse_var_3, plus(mult(rewrite_5, var_7.get_const().repeat(trav_size_5_6, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(trav_size_5_6, poly_size), rewrite_5), var_8.get_const().repeat(trav_size_5_6, 1)))).sum(dim=1)
			cse_var_2 = mult(cse_var_3.unsqueeze(2).repeat(1, 1, poly_size), plus(mult(rewrite_4, var_7.get_mat(poly_size).repeat(trav_size_5_6, 1, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(trav_size_5_6, poly_size, poly_size), rewrite_4), var_8.get_mat(poly_size).repeat(trav_size_5_6, 1, 1)))).sum(dim=1)
			trav_exp2_8_2 = PolyExp(trav_size_5_6, poly_size, cse_var_2, plus(phi_trav_exp2_5_1.get_const(), cse_var_1))
			phi_trav_exp2_5_1 = trav_exp2_8_2
			phi_trav_size_5_5 = trav_size_5_6
		cse_var_7 = phi_trav_exp2_6_3.get_mat(poly_size)
		rewrite_3 = convert_to_float(ge(cse_var_7, torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(phi_trav_size_6_7, poly_size)))
		cse_var_6 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		u_new = plus(mult(cse_var_7, plus(mult(rewrite_3, abs_elem.get_elem_new('u', cse_var_6).repeat(phi_trav_size_6_7, 1)), mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(phi_trav_size_6_7, poly_size), rewrite_3), abs_elem.get_elem_new('l', cse_var_6).repeat(phi_trav_size_6_7, 1)))).sum(dim=1), phi_trav_exp2_6_3.get_const())
		var_10 = cse_var_15
		L_new = PolyExp(curr_size, poly_size, var_10.get_mat(poly_size), plus(cse_var_14, var_10.get_const()))
		var_11 = cse_var_15
		U_new = PolyExp(curr_size, poly_size, var_11.get_mat(poly_size), plus(cse_var_14, var_11.get_const()))
		return l_new, u_new, L_new, U_new
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_32 = abs_elem.get_elem_new('u', prev)
		cse_var_37 = torch.tensor(0).unsqueeze(0).repeat(curr_size)
		cse_var_27 = le(cse_var_32, cse_var_37)
		cse_var_34 = convert_to_float(cse_var_27)
		rewrite_6 = cse_var_34
		cse_var_36 = abs_elem.get_elem_new('l', prev)
		cse_var_31 = ge(cse_var_36, cse_var_37)
		cse_var_33 = convert_to_float(cse_var_31)
		rewrite_7 = cse_var_33
		cse_var_35 = torch.tensor(1.0).unsqueeze(0).repeat(curr_size)
		l_new = plus(mult(rewrite_7, cse_var_36), mult(minus(cse_var_35, rewrite_7), mult(cse_var_37, plus(rewrite_6, minus(cse_var_35, rewrite_6)))))
		rewrite_8 = cse_var_34
		rewrite_9 = cse_var_33
		u_new = plus(mult(rewrite_9, cse_var_32), mult(minus(cse_var_35, rewrite_9), plus(mult(rewrite_8, cse_var_37), mult(minus(cse_var_35, rewrite_8), cse_var_32))))
		rewrite_10 = cse_var_34
		cse_var_26 = convert_to_float(cse_var_31.unsqueeze(1).repeat(1, poly_size))
		rewrite_11 = cse_var_26
		rewrite_12 = cse_var_33
		cse_var_30 = prev.convert_to_poly()
		cse_var_22 = cse_var_30.get_const()
		cse_var_24 = torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		cse_var_25 = cse_var_30.get_mat(poly_size)
		cse_var_29 = PolyExp(curr_size, poly_size, None, mult(cse_var_37, plus(rewrite_10, minus(cse_var_35, rewrite_10))))
		L_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_11, cse_var_25), mult(minus(cse_var_24, rewrite_11), cse_var_29.get_mat(poly_size))), plus(mult(rewrite_12, cse_var_22), mult(minus(cse_var_35, rewrite_12), cse_var_29.get_const())))
		var_12 = cse_var_30
		cse_var_20 = minus(cse_var_32, cse_var_36)
		cse_var_28 = divide(cse_var_32, cse_var_20)
		cse_var_19 = mult(cse_var_28, var_12.get_const())
		cse_var_23 = mult(cse_var_28.unsqueeze(1).repeat(1, poly_size), var_12.get_mat(poly_size))
		rewrite_13 = convert_to_float(cse_var_27.unsqueeze(1).repeat(1, poly_size))
		rewrite_14 = cse_var_26
		rewrite_15 = cse_var_34
		rewrite_16 = cse_var_33
		cse_var_21 = PolyExp(1, poly_size, None, 0.0)
		U_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_14, cse_var_25), mult(minus(cse_var_24, rewrite_14), plus(mult(rewrite_13, torch.tensor(cse_var_21.get_mat(poly_size)).unsqueeze(0).repeat(curr_size).unsqueeze(1).repeat(1, poly_size)), mult(minus(cse_var_24, rewrite_13), cse_var_23)))), plus(mult(rewrite_16, cse_var_22), mult(minus(cse_var_35, rewrite_16), plus(mult(rewrite_15, torch.tensor(cse_var_21.get_const()).unsqueeze(0).repeat(curr_size)), mult(minus(cse_var_35, rewrite_15), plus(divide(mult(mult(cse_var_32, torch.tensor(-1).unsqueeze(0).repeat(curr_size)), cse_var_36), cse_var_20), cse_var_19))))))
		return l_new, u_new, L_new, U_new
	
