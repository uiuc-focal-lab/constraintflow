import torch
from common.polyexp import PolyExp, Nlist, SymExp
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_21 = prev.dot(curr.get_metadata('weight'))
		cse_var_20 = curr.get_metadata('bias')
		cse_var_82 = cse_var_21.get_mat(poly_size)
		cse_var_41 = PolyExp(curr_size, poly_size, cse_var_82, plus(cse_var_21.get_const(), cse_var_20))
		cse_var_79 = get_shape_1(cse_var_82)
		cse_var_10 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		cse_var_18 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		cse_var_75 = torch.tensor(1.0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size)
		phi_trav_exp1_1_1 = cse_var_41
		phi_trav_exp1_2_3 = cse_var_41
		while(True):
			cse_var_24 = phi_trav_exp1_1_1.get_mat(poly_size)
			trav_size_1_2 = get_shape_1(cse_var_24)
			vertices_stop1 = False
			vertices1_1_1 = ne(cse_var_24, cse_var_10)
			vertices_stop_default1 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				break
			cse_var_19 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_2 = abs_elem.get_elem_new('L', cse_var_19)
			var_3 = abs_elem.get_elem_new('U', cse_var_19)
			cse_var_77 = phi_trav_exp1_1_1.get_mat(poly_size)
			cse_var_16 = cse_var_77.unsqueeze(2).repeat(1, 1, poly_size)
			cse_var_64 = var_3.get_const()
			cse_var_66 = var_3.get_mat(poly_size)
			cse_var_71 = cse_var_64.repeat(curr_size, 1)
			cse_var_74 = cse_var_66.repeat(curr_size, 1, 1)
			cse_var_65 = var_2.get_const()
			cse_var_68 = var_2.get_mat(poly_size)
			cse_var_72 = cse_var_65.repeat(curr_size, 1)
			cse_var_76 = cse_var_68.repeat(curr_size, 1, 1)
			cse_var_15 = ge(cse_var_77, cse_var_18)
			cse_var_78 = convert_to_float(cse_var_15)
			cse_var_99 = mult(minus(cse_var_75, cse_var_78), cse_var_77)
			cse_var_67 = cse_var_99.unsqueeze(2)
			cse_var_73 = mult(cse_var_78, cse_var_77)
			cse_var_69 = cse_var_73.unsqueeze(2)
			cse_var_62 = plus(inner_prod(cse_var_73, cse_var_65.squeeze(0)), inner_prod(cse_var_99, cse_var_64.squeeze(0)))
			cse_var_63 = plus(inner_prod(cse_var_69.squeeze(2), cse_var_68.squeeze(0)), inner_prod(cse_var_67.squeeze(2), cse_var_66.squeeze(0)))
			trav_exp1_4_2 = PolyExp(curr_size, poly_size, cse_var_63, plus(cse_var_62, phi_trav_exp1_1_1.get_const()))
			phi_trav_exp1_1_1 = trav_exp1_4_2
		cse_var_22 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_80 = phi_trav_exp1_2_3.get_mat(poly_size)
		cse_var_81 = convert_to_float(ge(cse_var_80, cse_var_18))
		l_new = plus(plus(inner_prod(mult(cse_var_81, cse_var_80), abs_elem.get_elem_new('l', cse_var_22).squeeze(0)), inner_prod(mult(minus(cse_var_75, cse_var_81), cse_var_80), abs_elem.get_elem_new('u', cse_var_22).squeeze(0))), phi_trav_exp1_2_3.get_const())
		phi_trav_exp2_5_1 = cse_var_41
		phi_trav_exp2_6_3 = cse_var_41
		while(True):
			cse_var_11 = phi_trav_exp2_5_1.get_mat(poly_size)
			trav_size_5_6 = get_shape_1(cse_var_11)
			vertices_stop2 = False
			vertices2_5_1 = ne(cse_var_11, cse_var_10)
			vertices_stop_default2 = get_default_stop([curr_size, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				break
			cse_var_7 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_7 = abs_elem.get_elem_new('U', cse_var_7)
			var_8 = abs_elem.get_elem_new('L', cse_var_7)
			cse_var_58 = phi_trav_exp2_5_1.get_mat(poly_size)
			cse_var_5 = cse_var_58.unsqueeze(2).repeat(1, 1, poly_size)
			cse_var_46 = var_8.get_const()
			cse_var_48 = var_8.get_mat(poly_size)
			cse_var_53 = cse_var_46.repeat(curr_size, 1)
			cse_var_56 = cse_var_48.repeat(curr_size, 1, 1)
			cse_var_47 = var_7.get_const()
			cse_var_50 = var_7.get_mat(poly_size)
			cse_var_54 = cse_var_47.repeat(curr_size, 1)
			cse_var_57 = cse_var_50.repeat(curr_size, 1, 1)
			cse_var_4 = ge(cse_var_58, cse_var_18)
			cse_var_59 = convert_to_float(cse_var_4)
			cse_var_98 = mult(minus(cse_var_75, cse_var_59), cse_var_58)
			cse_var_49 = cse_var_98.unsqueeze(2)
			cse_var_55 = mult(cse_var_59, cse_var_58)
			cse_var_51 = cse_var_55.unsqueeze(2)
			cse_var_44 = plus(inner_prod(cse_var_55, cse_var_47.squeeze(0)), inner_prod(cse_var_98, cse_var_46.squeeze(0)))
			cse_var_45 = plus(inner_prod(cse_var_51.squeeze(2), cse_var_50.squeeze(0)), inner_prod(cse_var_49.squeeze(2), cse_var_48.squeeze(0)))
			trav_exp2_8_2 = PolyExp(curr_size, poly_size, cse_var_45, plus(cse_var_44, phi_trav_exp2_5_1.get_const()))
			phi_trav_exp2_5_1 = trav_exp2_8_2
		cse_var_8 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_60 = phi_trav_exp2_6_3.get_mat(poly_size)
		cse_var_61 = convert_to_float(ge(cse_var_60, cse_var_18))
		u_new = plus(plus(inner_prod(mult(cse_var_61, cse_var_60), abs_elem.get_elem_new('u', cse_var_8).squeeze(0)), inner_prod(mult(minus(cse_var_75, cse_var_61), cse_var_60), abs_elem.get_elem_new('l', cse_var_8).squeeze(0))), phi_trav_exp2_6_3.get_const())
		return l_new, u_new, cse_var_41, cse_var_41
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_92 = abs_elem.get_elem_new('u', prev)
		cse_var_39 = torch.tensor(0).unsqueeze(0).repeat(curr_size)
		cse_var_36 = le(cse_var_92, cse_var_39)
		cse_var_96 = convert_to_float(cse_var_36)
		cse_var_97 = torch.tensor(1.0).unsqueeze(0).repeat(curr_size)
		cse_var_102 = minus(cse_var_97, cse_var_96)
		cse_var_103 = mult(cse_var_96, cse_var_39)
		cse_var_35 = plus(cse_var_103, mult(cse_var_102, cse_var_39))
		cse_var_91 = abs_elem.get_elem_new('l', prev)
		cse_var_38 = ge(cse_var_91, cse_var_39)
		cse_var_95 = convert_to_float(cse_var_38)
		cse_var_104 = minus(cse_var_97, cse_var_95)
		l_new = plus(mult(cse_var_95, cse_var_91), mult(cse_var_104, cse_var_35))
		u_new = plus(mult(cse_var_95, cse_var_92), mult(cse_var_104, plus(cse_var_103, mult(cse_var_102, cse_var_92))))
		cse_var_34 = prev.convert_to_poly()
		cse_var_42 = cse_var_34.get_const()
		cse_var_89 = cse_var_34.get_mat(poly_size)
		cse_var_33 = PolyExp(curr_size, poly_size, None, cse_var_35)
		cse_var_86 = cse_var_95.unsqueeze(1).repeat(1, poly_size)
		cse_var_85 = cse_var_104.unsqueeze(1).repeat(1, poly_size)
		cse_var_100 = mult(cse_var_95, cse_var_42)
		cse_var_101 = mult(cse_var_86, cse_var_89)
		L_new = PolyExp(curr_size, poly_size, plus(cse_var_101, mult(cse_var_85, cse_var_33.get_mat(poly_size))), plus(cse_var_100, mult(cse_var_104, cse_var_33.get_const())))
		cse_var_29 = minus(cse_var_92, cse_var_91)
		cse_var_90 = divide(cse_var_92, cse_var_29)
		cse_var_93 = mult(cse_var_90, cse_var_42)
		cse_var_94 = mult(cse_var_90.unsqueeze(1).repeat(1, poly_size), cse_var_89)
		cse_var_88 = plus(cse_var_93, divide(mult(mult(cse_var_92, cse_var_91), torch.tensor(-1).unsqueeze(0).repeat(curr_size)), cse_var_29))
		cse_var_28 = PolyExp(1, poly_size, None, 0.0)
		cse_var_83 = torch.tensor(cse_var_28.get_const()).unsqueeze(0).repeat(curr_size)
		cse_var_84 = plus(mult(cse_var_96, torch.tensor(cse_var_28.get_mat(poly_size)).unsqueeze(0).repeat(curr_size)).unsqueeze(1).repeat(1, poly_size), mult(mult(cse_var_102, cse_var_90).unsqueeze(1).repeat(1, poly_size), cse_var_89))
		U_new = PolyExp(curr_size, poly_size, plus(cse_var_101, mult(cse_var_85, cse_var_84)), plus(cse_var_100, mult(cse_var_104, plus(mult(cse_var_96, cse_var_83), mult(cse_var_102, cse_var_88)))))
		return l_new, u_new, L_new, U_new
	
