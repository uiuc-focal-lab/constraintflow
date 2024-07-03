import torch
from common.polyexp import PolyExp, Nlist, SymExp
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_21 = prev.dot(curr.get_metadata('weight'))
		cse_var_20 = curr.get_metadata('bias')
		var_1 = PolyExp(curr_size, poly_size, cse_var_21.get_mat(poly_size), plus(cse_var_21.get_const(), cse_var_20))
		trav_size_0_0 = get_shape_1(var_1.get_mat(poly_size))
		cse_var_10 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		cse_var_18 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		phi_trav_exp1_1_1 = var_1
		phi_trav_size_1_1 = trav_size_0_0
		phi_vertices1_1_0 = 0
		phi_trav_exp1_2_3 = var_1
		phi_trav_size_2_3 = trav_size_0_0
		phi_vertices1_2_3 = 0
		while(True):
			cse_var_24 = phi_trav_exp1_1_1.get_mat(poly_size)
			trav_size_1_2 = get_shape_1(cse_var_24)
			vertices_stop1 = False
			vertices1_1_1 = ne(cse_var_24, cse_var_10)
			vertices_stop_default1 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			phi_trav_size_2_3 = trav_size_1_2
			phi_vertices1_2_3 = vertices1_1_2
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				phi_trav_size_2_3 = trav_size_1_2
				phi_vertices1_2_3 = vertices1_1_2
				break
			cse_var_19 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			# vertices_priority1 = neg(cse_var_19.get_metadata('layer'))
			var_2 = abs_elem.get_elem_new('L', cse_var_19)
			var_3 = abs_elem.get_elem_new('U', cse_var_19)
			cse_var_17 = phi_trav_exp1_1_1.get_mat(poly_size)
			cse_var_16 = cse_var_17.unsqueeze(2).repeat(1, 1, poly_size)
			cse_var_13 = PolyExp(poly_size, poly_size, mult(cse_var_16, var_3.get_mat(poly_size).repeat(curr_size, 1, 1)), mult(cse_var_17, var_3.get_const().repeat(curr_size, 1)))
			cse_var_14 = PolyExp(poly_size, poly_size, mult(cse_var_16, var_2.get_mat(poly_size).repeat(curr_size, 1, 1)), mult(cse_var_17, var_2.get_const().repeat(curr_size, 1)))
			cse_var_15 = ge(cse_var_17, cse_var_18)
			cse_var_12 = PolyExp(poly_size, poly_size, torch.where(cse_var_15.unsqueeze(2).repeat(1, 1, poly_size), cse_var_14.get_mat(poly_size), cse_var_13.get_mat(poly_size)), torch.where(cse_var_15, cse_var_14.get_const(), cse_var_13.get_const()))
			var_4 = PolyExp(curr_size, poly_size, cse_var_12.get_mat(poly_size).sum(dim=1), cse_var_12.get_const().sum(dim=1))
			trav_exp1_4_2 = PolyExp(curr_size, poly_size, var_4.get_mat(poly_size), plus(var_4.get_const(), phi_trav_exp1_1_1.get_const()))
			phi_trav_exp1_1_1 = trav_exp1_4_2
			phi_trav_size_1_1 = trav_size_1_2
			phi_vertices1_1_0 = vertices1_1_2
		cse_var_22 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_23 = phi_trav_exp1_2_3.get_mat(poly_size)
		l_new = plus(torch.where(ge(cse_var_23, cse_var_18), mult(cse_var_23, abs_elem.get_elem_new('l', cse_var_22).repeat(curr_size, 1)), mult(cse_var_23, abs_elem.get_elem_new('u', cse_var_22).repeat(curr_size, 1))).sum(dim=1), phi_trav_exp1_2_3.get_const())
		var_6 = PolyExp(curr_size, poly_size, cse_var_21.get_mat(poly_size), plus(cse_var_21.get_const(), cse_var_20))
		trav_size_2_4 = get_shape_1(var_6.get_mat(poly_size))
		phi_trav_exp2_5_1 = var_6
		phi_trav_size_5_5 = trav_size_2_4
		phi_vertices2_5_0 = 0
		phi_trav_exp2_6_3 = var_6
		phi_trav_size_6_7 = trav_size_2_4
		phi_vertices2_6_3 = 0
		while(True):
			cse_var_11 = phi_trav_exp2_5_1.get_mat(poly_size)
			trav_size_5_6 = get_shape_1(cse_var_11)
			vertices_stop2 = False
			vertices2_5_1 = ne(cse_var_11, cse_var_10)
			vertices_stop_default2 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			phi_trav_size_6_7 = trav_size_5_6
			phi_vertices2_6_3 = vertices2_5_2
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				phi_trav_size_6_7 = trav_size_5_6
				phi_vertices2_6_3 = vertices2_5_2
				break
			cse_var_7 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			# vertices_priority2 = neg(cse_var_7.get_metadata('layer'))
			var_7 = abs_elem.get_elem_new('U', cse_var_7)
			var_8 = abs_elem.get_elem_new('L', cse_var_7)
			cse_var_6 = phi_trav_exp2_5_1.get_mat(poly_size)
			cse_var_5 = cse_var_6.unsqueeze(2).repeat(1, 1, poly_size)
			cse_var_2 = PolyExp(poly_size, poly_size, mult(cse_var_5, var_8.get_mat(poly_size).repeat(curr_size, 1, 1)), mult(cse_var_6, var_8.get_const().repeat(curr_size, 1)))
			cse_var_3 = PolyExp(poly_size, poly_size, mult(cse_var_5, var_7.get_mat(poly_size).repeat(curr_size, 1, 1)), mult(cse_var_6, var_7.get_const().repeat(curr_size, 1)))
			cse_var_4 = ge(cse_var_6, cse_var_18)
			cse_var_1 = PolyExp(poly_size, poly_size, torch.where(cse_var_4.unsqueeze(2).repeat(1, 1, poly_size), cse_var_3.get_mat(poly_size), cse_var_2.get_mat(poly_size)), torch.where(cse_var_4, cse_var_3.get_const(), cse_var_2.get_const()))
			var_9 = PolyExp(curr_size, poly_size, cse_var_1.get_mat(poly_size).sum(dim=1), cse_var_1.get_const().sum(dim=1))
			trav_exp2_8_2 = PolyExp(curr_size, poly_size, var_9.get_mat(poly_size), plus(var_9.get_const(), phi_trav_exp2_5_1.get_const()))
			phi_trav_exp2_5_1 = trav_exp2_8_2
			phi_trav_size_5_5 = trav_size_5_6
			phi_vertices2_5_0 = vertices2_5_2
		cse_var_8 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_9 = phi_trav_exp2_6_3.get_mat(poly_size)
		u_new = plus(torch.where(ge(cse_var_9, cse_var_18), mult(cse_var_9, abs_elem.get_elem_new('u', cse_var_8).repeat(curr_size, 1)), mult(cse_var_9, abs_elem.get_elem_new('l', cse_var_8).repeat(curr_size, 1))).sum(dim=1), phi_trav_exp2_6_3.get_const())
		L_new = PolyExp(curr_size, poly_size, cse_var_21.get_mat(poly_size), plus(cse_var_21.get_const(), cse_var_20))
		U_new = PolyExp(curr_size, poly_size, cse_var_21.get_mat(poly_size), plus(cse_var_21.get_const(), cse_var_20))
		return l_new, u_new, L_new, U_new
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_45 = abs_elem.get_elem_new('l', prev)
		cse_var_44 = abs_elem.get_elem_new('u', prev)
		cse_var_43 = torch.tensor(0).unsqueeze(0).repeat(curr_size)
		cse_var_41 = le(cse_var_44, cse_var_43)
		cse_var_42 = ge(cse_var_45, cse_var_43)
		l_new = torch.where(cse_var_42, cse_var_45, torch.where(cse_var_41, cse_var_43, torch.where(lt(torch.where(gt(cse_var_45, cse_var_43), cse_var_45, neg(cse_var_45)), torch.where(gt(cse_var_44, cse_var_43), cse_var_44, neg(cse_var_44))), cse_var_45, cse_var_43)))
		u_new = torch.where(cse_var_42, cse_var_44, torch.where(cse_var_41, cse_var_43, cse_var_44))
		cse_var_36 = PolyExp(1, poly_size, None, 0)
		cse_var_39 = cse_var_36.get_mat(poly_size)
		cse_var_28 = torch.tensor(cse_var_39).unsqueeze(0).repeat(curr_size).unsqueeze(1).repeat(1, poly_size)
		cse_var_29 = cse_var_41.unsqueeze(1).repeat(1, poly_size)
		cse_var_30 = cse_var_42.unsqueeze(1).repeat(1, poly_size)
		cse_var_37 = prev.convert_to_poly()
		cse_var_33 = cse_var_37.get_const()
		cse_var_35 = torch.tensor(cse_var_36.get_const()).unsqueeze(0).repeat(curr_size)
		cse_var_38 = lt(torch.where(gt(cse_var_45, cse_var_43), cse_var_45, neg(cse_var_45)), torch.where(gt(cse_var_44, cse_var_43), cse_var_44, neg(cse_var_44)))
		cse_var_40 = cse_var_37.get_mat(poly_size)
		cse_var_34 = PolyExp(curr_size, poly_size, torch.where(cse_var_38.unsqueeze(1).repeat(1, poly_size), cse_var_40, torch.tensor(cse_var_39).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)), torch.where(cse_var_38, cse_var_33, cse_var_35))
		cse_var_32 = PolyExp(curr_size, poly_size, torch.where(cse_var_29, cse_var_28, cse_var_34.get_mat(poly_size)), torch.where(cse_var_41, cse_var_35, cse_var_34.get_const()))
		L_new = PolyExp(curr_size, poly_size, torch.where(cse_var_30, cse_var_40, cse_var_32.get_mat(poly_size)), torch.where(cse_var_42, cse_var_33, cse_var_32.get_const()))
		cse_var_27 = minus(cse_var_44, cse_var_45)
		cse_var_31 = divide(cse_var_44, cse_var_27)
		var_17 = PolyExp(curr_size, poly_size, mult(cse_var_31.unsqueeze(1).repeat(1, poly_size), cse_var_37.get_mat(poly_size)), mult(cse_var_31, cse_var_37.get_const()))
		cse_var_26 = PolyExp(curr_size, poly_size, var_17.get_mat(poly_size), plus(var_17.get_const(), divide(mult(mult(cse_var_44, torch.tensor(-1).unsqueeze(0).repeat(curr_size)), cse_var_45), cse_var_27)))
		cse_var_25 = PolyExp(curr_size, poly_size, torch.where(cse_var_29, cse_var_28, cse_var_26.get_mat(poly_size)), torch.where(cse_var_41, cse_var_35, cse_var_26.get_const()))
		U_new = PolyExp(curr_size, poly_size, torch.where(cse_var_30, cse_var_40, cse_var_25.get_mat(poly_size)), torch.where(cse_var_42, cse_var_33, cse_var_25.get_const()))
		return l_new, u_new, L_new, U_new
	




























