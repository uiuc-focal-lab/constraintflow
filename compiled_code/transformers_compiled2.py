
import torch
from common.polyexp import PolyExp, Nlist
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, debug_flag):
		var_0 = prev.dot(curr.get_metadata('weight'))
		var_1 = PolyExp(curr_size, poly_size, var_0.get_mat(poly_size), plus(curr.get_metadata('bias'), var_0.get_const()))
		trav_size_0_0 = get_shape_1(var_0.get_mat(poly_size))
		phi_trav_exp1_1_1 = var_1
		phi_trav_exp1_2_3 = var_1
		while(True):
			print(phi_trav_exp1_1_1.mat)
			print(phi_trav_exp1_1_1.mat.sum())
			trav_size_1_2 = get_shape_1(phi_trav_exp1_1_1.get_mat(poly_size))
			vertices_stop1 = False
			vertices1_1_1 = ne(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))
			vertices_stop_default1 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				break
			var_2 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_3 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			rewrite_2 = convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
			trav_exp1_4_2 = PolyExp(curr_size, poly_size, plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp1_1_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_2.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), phi_trav_exp1_1_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_3.get_mat(poly_size).squeeze(0))), plus(phi_trav_exp1_1_1.get_const(), plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp1_1_1.get_mat(poly_size)), var_2.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_2), phi_trav_exp1_1_1.get_mat(poly_size)), var_3.get_const().squeeze(0)))))
			phi_trav_exp1_1_1 = trav_exp1_4_2
		rewrite_0 = convert_to_float(ge(phi_trav_exp1_2_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
		l_new = plus(plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp1_2_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp1_2_3.get_mat(poly_size)), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_0), phi_trav_exp1_2_3.get_mat(poly_size)), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).squeeze(0))), phi_trav_exp1_2_3.get_const())
		var_5 = prev.dot(curr.get_metadata('weight'))
		var_6 = PolyExp(curr_size, poly_size, var_5.get_mat(poly_size), plus(curr.get_metadata('bias'), var_5.get_const()))
		trav_size_2_4 = get_shape_1(var_5.get_mat(poly_size))
		phi_trav_exp2_5_1 = var_6
		phi_trav_exp2_6_3 = var_6
		while(True):
			trav_size_5_6 = get_shape_1(phi_trav_exp2_5_1.get_mat(poly_size))
			vertices_stop2 = False
			vertices2_5_1 = ne(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))
			vertices_stop_default2 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				break
			var_7 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_8 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			rewrite_5 = convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
			trav_exp2_8_2 = PolyExp(curr_size, poly_size, plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp2_5_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_7.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), phi_trav_exp2_5_1.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_8.get_mat(poly_size).squeeze(0))), plus(phi_trav_exp2_5_1.get_const(), plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp2_5_1.get_mat(poly_size)), var_7.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_5), phi_trav_exp2_5_1.get_mat(poly_size)), var_8.get_const().squeeze(0)))))
			phi_trav_exp2_5_1 = trav_exp2_8_2
		rewrite_3 = convert_to_float(ge(phi_trav_exp2_6_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
		u_new = plus(plus(inner_prod(mult(convert_to_float(ge(phi_trav_exp2_6_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), phi_trav_exp2_6_3.get_mat(poly_size)), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_3), phi_trav_exp2_6_3.get_mat(poly_size)), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).squeeze(0))), phi_trav_exp2_6_3.get_const())
		var_10 = prev.dot(curr.get_metadata('weight'))
		L_new = PolyExp(curr_size, poly_size, var_10.get_mat(poly_size), plus(curr.get_metadata('bias'), var_10.get_const()))
		var_11 = prev.dot(curr.get_metadata('weight'))
		U_new = PolyExp(curr_size, poly_size, var_11.get_mat(poly_size), plus(curr.get_metadata('bias'), var_11.get_const()))
		return l_new, u_new, L_new, U_new
	
	def Relu2(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		rewrite_6 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_7 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		l_new = plus(mult(rewrite_7, abs_elem.get_elem_new('l', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_7), plus(mult(rewrite_6, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_6), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))))
		rewrite_8 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_9 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		u_new = plus(mult(rewrite_9, abs_elem.get_elem_new('u', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_9), plus(mult(rewrite_8, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_8), abs_elem.get_elem_new('u', prev)))))
		rewrite_10 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_11 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))).unsqueeze(1).repeat(1, poly_size)
		rewrite_12 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		L_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_11, prev.convert_to_poly().get_mat(poly_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))).unsqueeze(1).repeat(1, poly_size), PolyExp(curr_size, poly_size, None, plus(mult(rewrite_10, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_10), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))).get_mat(poly_size))), plus(mult(rewrite_12, prev.convert_to_poly().get_const()), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_12), PolyExp(curr_size, poly_size, None, plus(mult(rewrite_10, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_10), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))).get_const())))
		var_12 = prev.convert_to_poly()
		rewrite_14 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))).unsqueeze(1).repeat(1, poly_size)
		rewrite_15 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_16 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		U_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_14, prev.convert_to_poly().get_mat(poly_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))).unsqueeze(1).repeat(1, poly_size), plus(mult(convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_mat(poly_size)).unsqueeze(0).repeat(curr_size)).unsqueeze(1).repeat(1, poly_size), mult(mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))), divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)))).unsqueeze(1).repeat(1, poly_size), var_12.get_mat(poly_size))))), plus(mult(rewrite_16, prev.convert_to_poly().get_const()), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_16), plus(mult(rewrite_15, torch.tensor(PolyExp(1, poly_size, None, 0.0).get_const()).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_15), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)), torch.tensor(-1).unsqueeze(0).repeat(curr_size)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_12.get_const())))))))
		return l_new, u_new, L_new, U_new
	

	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, debug_flag):
		rewrite_6 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_7 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		l_new = plus(mult(rewrite_7, abs_elem.get_elem_new('l', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_7), plus(mult(rewrite_6, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_6), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))))
		rewrite_8 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_9 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		u_new = plus(mult(rewrite_9, abs_elem.get_elem_new('u', prev)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_9), plus(mult(rewrite_8, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_8), abs_elem.get_elem_new('u', prev)))))
		var_12 = abs_elem.get_elem_new('L', prev)
		var_13 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		var_14 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		rewrite_11 = convert_to_float(ge(var_12.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
		rewrite_12 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_13 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))).unsqueeze(1).repeat(1, poly_size)
		rewrite_14 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		L_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_13, plus(inner_prod(mult(convert_to_float(ge(var_12.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_12.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_13.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(var_12.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), var_12.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_14.get_mat(poly_size).squeeze(0)))), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))).unsqueeze(1).repeat(1, poly_size), PolyExp(curr_size, poly_size, None, plus(mult(rewrite_12, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_12), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))).get_mat(poly_size))), plus(mult(rewrite_14, plus(var_12.get_const(), plus(inner_prod(mult(convert_to_float(ge(var_12.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_12.get_mat(poly_size)), var_13.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_11), var_12.get_mat(poly_size)), var_14.get_const().squeeze(0))))), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_14), PolyExp(curr_size, poly_size, None, plus(mult(rewrite_12, torch.tensor(0.0).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_12), torch.tensor(0.0).unsqueeze(0).repeat(curr_size)))).get_const())))
		var_16 = abs_elem.get_elem_new('U', prev)
		var_17 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		var_18 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		rewrite_16 = convert_to_float(ge(var_16.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
		var_20 = abs_elem.get_elem_new('U', prev)
		var_21 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		var_22 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
		rewrite_18 = convert_to_float(ge(var_20.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))
		rewrite_20 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))).unsqueeze(1).repeat(1, poly_size)
		rewrite_21 = convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		rewrite_22 = convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))
		U_new = PolyExp(curr_size, poly_size, plus(mult(rewrite_20, plus(inner_prod(mult(convert_to_float(ge(var_16.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_16.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_17.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(var_16.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), var_16.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_18.get_mat(poly_size).squeeze(0)))), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))).unsqueeze(1).repeat(1, poly_size), plus(mult(convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size))), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_mat(poly_size)).unsqueeze(0).repeat(curr_size)).unsqueeze(1).repeat(1, poly_size), mult(mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), convert_to_float(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat(curr_size)))), divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)))).unsqueeze(1).repeat(1, poly_size), plus(inner_prod(mult(convert_to_float(ge(var_20.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_20.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_21.get_mat(poly_size).squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), convert_to_float(ge(var_20.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)))), var_20.get_mat(poly_size)).unsqueeze(2).squeeze(2), var_22.get_mat(poly_size).squeeze(0))))))), plus(mult(rewrite_22, plus(var_16.get_const(), plus(inner_prod(mult(convert_to_float(ge(var_16.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_16.get_mat(poly_size)), var_17.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_16), var_16.get_mat(poly_size)), var_18.get_const().squeeze(0))))), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_22), plus(mult(rewrite_21, torch.tensor(PolyExp(1, poly_size, None, 0.0).get_const()).unsqueeze(0).repeat(curr_size)), mult(minus(torch.tensor(1.0).unsqueeze(0).repeat(curr_size), rewrite_21), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)), torch.tensor(-1).unsqueeze(0).repeat(curr_size)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), plus(var_20.get_const(), plus(inner_prod(mult(convert_to_float(ge(var_20.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))), var_20.get_mat(poly_size)), var_21.get_const().squeeze(0)), inner_prod(mult(minus(torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), rewrite_18), var_20.get_mat(poly_size)), var_22.get_const().squeeze(0)))))))))))
		
		if debug_flag:
			print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
			print(L_new.mat[0,:])	
			print(U_new.mat[0,:])	
		return l_new, u_new, L_new, U_new