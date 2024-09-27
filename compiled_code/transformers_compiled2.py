import torch
import copy
from common.polyexp import PolyExpSparse, SymExp
from common.sparse_tensor import SparseTensorBlock
from common.nlist import Llist
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):
		cse_var_24 = prev.dot(curr.get_metadata('weight'), abs_elem.get_poly_size())
		cse_var_23 = curr.get_metadata('bias')
		cse_var_65 = cse_var_24.get_mat(abs_elem)
		cse_var_45 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_65) , copy.deepcopy(plus(cse_var_24.get_const(), cse_var_23)))
		cse_var_12 = torch.tensor(0.0).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, curr_size, poly_size)
		cse_var_21 = torch.tensor(0.0).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, curr_size, poly_size)
		phi_trav_exp1_1_1 = cse_var_45
		phi_trav_exp1_2_3 = cse_var_45
		while(True):
			cse_var_28 = phi_trav_exp1_1_1.get_mat(abs_elem)
			vertices_stop1 = False
			vertices1_1_1 = ne(cse_var_28, cse_var_12)
			vertices_stop_default1 = get_default_stop([1, curr_size, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop1).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, curr_size, poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				break
			cse_var_22 = Llist(abs_elem.network, [1]*(phi_trav_exp1_1_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_2 = abs_elem.get_elem_new('L', cse_var_22)
			var_3 = abs_elem.get_elem_new('U', cse_var_22)
			cse_var_20 = phi_trav_exp1_1_1.get_mat(abs_elem)
			cse_var_18 = repeat(cse_var_20, torch.tensor([batch_size, 1, 1]))
			cse_var_19 = repeat(cse_var_20.unsqueeze(3), torch.tensor([batch_size, 1, 1, poly_size]))
			cse_var_60 = mult(cse_var_18, repeat(var_3.get_const(), torch.tensor([1, curr_size, 1])))
			cse_var_62 = mult(cse_var_19, repeat(var_3.get_mat(abs_elem), torch.tensor([1, curr_size, 1, 1])))
			cse_var_61 = mult(cse_var_18, repeat(var_2.get_const(), torch.tensor([1, curr_size, 1])))
			cse_var_63 = mult(cse_var_19, repeat(var_2.get_mat(abs_elem), torch.tensor([1, curr_size, 1, 1])))
			cse_var_17 = ge(cse_var_20, cse_var_21)
			cse_var_58 = where(repeat(cse_var_17, torch.tensor([batch_size, 1, 1])), cse_var_61, cse_var_60)
			cse_var_59 = where(repeat(cse_var_17.unsqueeze(3), torch.tensor([batch_size, 1, 1, poly_size])), cse_var_63, cse_var_62)
			cse_var_56 = cse_var_58.sum(dim=2)
			cse_var_57 = cse_var_59.sum(dim=2)
			trav_exp1_4_2 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_57) , copy.deepcopy(plus(cse_var_56, repeat(phi_trav_exp1_1_1.get_const(), torch.tensor([batch_size, 1])))))
			phi_trav_exp1_1_1 = trav_exp1_4_2
		cse_var_25 = Llist(abs_elem.network, [1]*(phi_trav_exp1_2_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_27 = phi_trav_exp1_2_3.get_mat(abs_elem)
		cse_var_26 = repeat(cse_var_27, torch.tensor([batch_size, 1, 1]))
		l_new = plus(where(repeat(ge(cse_var_27, cse_var_21), torch.tensor([batch_size, 1, 1])), mult(cse_var_26, repeat(abs_elem.get_elem_new('l', cse_var_25), torch.tensor([1, curr_size, 1]))), mult(cse_var_26, repeat(abs_elem.get_elem_new('u', cse_var_25), torch.tensor([1, curr_size, 1])))).sum(dim=2), repeat(phi_trav_exp1_2_3.get_const(), torch.tensor([batch_size, 1])))
		phi_trav_exp2_5_1 = cse_var_45
		phi_trav_exp2_6_3 = cse_var_45
		while(True):
			cse_var_13 = phi_trav_exp2_5_1.get_mat(abs_elem)
			vertices_stop2 = False
			vertices2_5_1 = ne(cse_var_13, cse_var_12)
			vertices_stop_default2 = get_default_stop([1, curr_size, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop2).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, curr_size, poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				break
			cse_var_8 = Llist(abs_elem.network, [1]*(phi_trav_exp2_5_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_7 = abs_elem.get_elem_new('U', cse_var_8)
			var_8 = abs_elem.get_elem_new('L', cse_var_8)
			cse_var_7 = phi_trav_exp2_5_1.get_mat(abs_elem)
			cse_var_5 = repeat(cse_var_7, torch.tensor([batch_size, 1, 1]))
			cse_var_6 = repeat(cse_var_7.unsqueeze(3), torch.tensor([batch_size, 1, 1, poly_size]))
			cse_var_52 = mult(cse_var_5, repeat(var_8.get_const(), torch.tensor([1, curr_size, 1])))
			cse_var_54 = mult(cse_var_6, repeat(var_8.get_mat(abs_elem), torch.tensor([1, curr_size, 1, 1])))
			cse_var_53 = mult(cse_var_5, repeat(var_7.get_const(), torch.tensor([1, curr_size, 1])))
			cse_var_55 = mult(cse_var_6, repeat(var_7.get_mat(abs_elem), torch.tensor([1, curr_size, 1, 1])))
			cse_var_4 = ge(cse_var_7, cse_var_21)
			cse_var_50 = where(repeat(cse_var_4, torch.tensor([batch_size, 1, 1])), cse_var_53, cse_var_52)
			cse_var_51 = where(repeat(cse_var_4.unsqueeze(3), torch.tensor([batch_size, 1, 1, poly_size])), cse_var_55, cse_var_54)
			cse_var_48 = cse_var_50.sum(dim=2)
			cse_var_49 = cse_var_51.sum(dim=2)
			trav_exp2_8_2 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_49) , copy.deepcopy(plus(cse_var_48, repeat(phi_trav_exp2_5_1.get_const(), torch.tensor([batch_size, 1])))))
			phi_trav_exp2_5_1 = trav_exp2_8_2
		cse_var_9 = Llist(abs_elem.network, [1]*(phi_trav_exp2_6_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_11 = phi_trav_exp2_6_3.get_mat(abs_elem)
		cse_var_10 = repeat(cse_var_11, torch.tensor([batch_size, 1, 1]))
		u_new = plus(where(repeat(ge(cse_var_11, cse_var_21), torch.tensor([batch_size, 1, 1])), mult(cse_var_10, repeat(abs_elem.get_elem_new('u', cse_var_9), torch.tensor([1, curr_size, 1]))), mult(cse_var_10, repeat(abs_elem.get_elem_new('l', cse_var_9), torch.tensor([1, curr_size, 1])))).sum(dim=2), repeat(phi_trav_exp2_6_3.get_const(), torch.tensor([batch_size, 1])))
		return l_new, u_new, cse_var_45, cse_var_45
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):
		cse_var_41 = abs_elem.get_elem_new('u', prev)
		cse_var_43 = torch.tensor(0.0).unsqueeze(0).unsqueeze(1).repeat(batch_size, curr_size)
		cse_var_40 = le(cse_var_41, cse_var_43)
		cse_var_39 = where(cse_var_40, cse_var_43, cse_var_43)
		cse_var_44 = abs_elem.get_elem_new('l', prev)
		cse_var_42 = ge(cse_var_44, cse_var_43)
		l_new = where(cse_var_42, cse_var_44, cse_var_39)
		u_new = where(cse_var_42, cse_var_41, where(cse_var_40, cse_var_43, cse_var_41))
		cse_var_38 = prev.convert_to_poly(abs_elem)
		cse_var_46 = repeat(cse_var_38.get_const(), torch.tensor([batch_size, 1]))
		cse_var_47 = repeat(cse_var_38.get_mat(abs_elem), torch.tensor([batch_size, 1, 1]))
		cse_var_35 = repeat(cse_var_42.unsqueeze(2), torch.tensor([1, 1, poly_size]))
		cse_var_37 = PolyExpSparse(abs_elem.network, 0.0, cse_var_39)
		L_new = PolyExpSparse(abs_elem.network, copy.deepcopy(where(cse_var_35, cse_var_47, cse_var_37.get_mat(abs_elem))) , copy.deepcopy(where(cse_var_42, cse_var_46, cse_var_37.get_const())))
		cse_var_33 = minus(cse_var_41, cse_var_44)
		cse_var_36 = divide(cse_var_41, cse_var_33)
		cse_var_69 = mult(cse_var_36, cse_var_46)
		cse_var_70 = mult(repeat(cse_var_36.unsqueeze(2), torch.tensor([1, 1, poly_size])), cse_var_47)
		temp2 = mult(mult(cse_var_41, torch.tensor(minus(0.0, 1.0)).unsqueeze(0).unsqueeze(1).repeat(batch_size, curr_size)), cse_var_44)
		temp1 = divide(temp2, cse_var_33)
		cse_var_68 = plus(cse_var_69, temp1)
		cse_var_32 = PolyExpSparse(abs_elem.network, 0.0, 0.0)
		cse_var_66 = where(cse_var_40, torch.tensor(cse_var_32.get_const()).unsqueeze(0).unsqueeze(1).repeat(batch_size, curr_size), cse_var_68)
		cse_var_67 = where(repeat(cse_var_40.unsqueeze(2), torch.tensor([1, 1, poly_size])), repeat(torch.tensor(cse_var_32.get_mat(abs_elem)).unsqueeze(0).unsqueeze(1).repeat(batch_size, curr_size).unsqueeze(2), torch.tensor([1, 1, poly_size])), cse_var_70)
		U_new = PolyExpSparse(abs_elem.network, copy.deepcopy(where(cse_var_35, cse_var_47, cse_var_67)) , copy.deepcopy(where(cse_var_42, cse_var_46, cse_var_66)))
		return l_new, u_new, L_new, U_new
	
