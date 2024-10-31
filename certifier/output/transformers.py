import torch
import copy
from certifier.lib.polyexp import PolyExpSparse, SymExp
from certifier.lib.sparse_tensor import SparseTensorBlock
from certifier.lib.nlist import Llist
from certifier.lib.tensor_ops import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):
		cse_var_24 = prev.dot(curr.get_metadata('weight'), abs_elem.get_poly_size())
		cse_var_23 = curr.get_metadata('bias')
		cse_var_87 = plus(cse_var_24.get_const(), cse_var_23)
		cse_var_88 = cse_var_24.get_mat(abs_elem)
		cse_var_84 = repeat(cse_var_87, torch.tensor([batch_size, 1]))
		trav_exp1_0_0 = PolyExpSparse(abs_elem.network, repeat(cse_var_88, torch.tensor([batch_size, 1, 1])) , cse_var_84)
		cse_var_12 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size]))
		cse_var_20 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size]))
		cse_var_80 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size]))
		phi_trav_exp1_1_1 = trav_exp1_0_0
		phi_trav_exp1_2_3 = trav_exp1_0_0
		while(True):
			cse_var_27 = phi_trav_exp1_1_1.get_mat(abs_elem)
			vertices_stop1 = False
			vertices1_1_1 = ne(cse_var_27, cse_var_12)
			vertices_stop_default1 = get_default_stop([batch_size, curr_size, poly_size])
			vertices_stop_temp1 = disj(SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=vertices_stop1, type= type(vertices_stop1)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size])), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				break
			cse_var_21 = Llist(abs_elem.network, [1]*(phi_trav_exp1_1_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_2 = abs_elem.get_elem_new('L', cse_var_21)
			var_3 = abs_elem.get_elem_new('U', cse_var_21)
			cse_var_82 = phi_trav_exp1_1_1.get_mat(abs_elem)
			cse_var_69 = var_3.get_const()
			cse_var_71 = var_3.get_mat(abs_elem)
			cse_var_70 = var_2.get_const()
			cse_var_73 = var_2.get_mat(abs_elem)
			cse_var_17 = ge(cse_var_82, cse_var_20)
			cse_var_83 = convert_to_float(cse_var_17)
			cse_var_72 = mult(minus(cse_var_80, cse_var_83), cse_var_82).unsqueeze(3)
			cse_var_78 = mult(cse_var_83, cse_var_82)
			cse_var_74 = cse_var_78.unsqueeze(3)
			cse_var_67 = plus(inner_prod(cse_var_78, cse_var_70.squeeze(1)), inner_prod(mult(minus(cse_var_80, cse_var_83), cse_var_82), cse_var_69.squeeze(1)))
			cse_var_68 = plus(inner_prod(cse_var_74.squeeze(3), cse_var_73.squeeze(1)), inner_prod(cse_var_72.squeeze(3), cse_var_71.squeeze(1)))
			trav_exp1_4_2 = PolyExpSparse(abs_elem.network, cse_var_68 , plus(cse_var_67, phi_trav_exp1_1_1.get_const()))
			phi_trav_exp1_1_1 = trav_exp1_4_2
		cse_var_25 = Llist(abs_elem.network, [1]*(phi_trav_exp1_2_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_85 = phi_trav_exp1_2_3.get_mat(abs_elem)
		cse_var_86 = convert_to_float(ge(cse_var_85, cse_var_20))
		l_new = plus(plus(inner_prod(mult(cse_var_86, cse_var_85), abs_elem.get_elem_new('l', cse_var_25).squeeze(1)), inner_prod(mult(minus(cse_var_80, cse_var_86), cse_var_85), abs_elem.get_elem_new('u', cse_var_25).squeeze(1))), phi_trav_exp1_2_3.get_const())
		trav_exp2_2_0 = PolyExpSparse(abs_elem.network, repeat(cse_var_88, torch.tensor([batch_size, 1, 1])) , cse_var_84)
		phi_trav_exp2_5_1 = trav_exp2_2_0
		phi_trav_exp2_6_3 = trav_exp2_2_0
		while(True):
			cse_var_13 = phi_trav_exp2_5_1.get_mat(abs_elem)
			vertices_stop2 = False
			vertices2_5_1 = ne(cse_var_13, cse_var_12)
			vertices_stop_default2 = get_default_stop([batch_size, curr_size, poly_size])
			vertices_stop_temp2 = disj(SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=vertices_stop2, type= type(vertices_stop2)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([batch_size, curr_size, poly_size])), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				break
			cse_var_7 = Llist(abs_elem.network, [1]*(phi_trav_exp2_5_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_7 = abs_elem.get_elem_new('U', cse_var_7)
			var_8 = abs_elem.get_elem_new('L', cse_var_7)
			cse_var_62 = phi_trav_exp2_5_1.get_mat(abs_elem)
			cse_var_50 = var_8.get_const()
			cse_var_52 = var_8.get_mat(abs_elem)
			cse_var_51 = var_7.get_const()
			cse_var_54 = var_7.get_mat(abs_elem)
			cse_var_4 = ge(cse_var_62, cse_var_20)
			cse_var_63 = convert_to_float(cse_var_4)
			cse_var_53 = mult(minus(cse_var_80, cse_var_63), cse_var_62).unsqueeze(3)
			cse_var_59 = mult(cse_var_63, cse_var_62)
			cse_var_55 = cse_var_59.unsqueeze(3)
			cse_var_48 = plus(inner_prod(cse_var_59, cse_var_51.squeeze(1)), inner_prod(mult(minus(cse_var_80, cse_var_63), cse_var_62), cse_var_50.squeeze(1)))
			cse_var_49 = plus(inner_prod(cse_var_55.squeeze(3), cse_var_54.squeeze(1)), inner_prod(cse_var_53.squeeze(3), cse_var_52.squeeze(1)))
			trav_exp2_8_2 = PolyExpSparse(abs_elem.network, cse_var_49 , plus(cse_var_48, phi_trav_exp2_5_1.get_const()))
			phi_trav_exp2_5_1 = trav_exp2_8_2
		cse_var_10 = Llist(abs_elem.network, [1]*(phi_trav_exp2_6_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_65 = phi_trav_exp2_6_3.get_mat(abs_elem)
		cse_var_66 = convert_to_float(ge(cse_var_65, cse_var_20))
		u_new = plus(plus(inner_prod(mult(cse_var_66, cse_var_65), abs_elem.get_elem_new('u', cse_var_10).squeeze(1)), inner_prod(mult(minus(cse_var_80, cse_var_66), cse_var_65), abs_elem.get_elem_new('l', cse_var_10).squeeze(1))), phi_trav_exp2_6_3.get_const())
		cse_var_64 = PolyExpSparse(abs_elem.network, repeat(cse_var_88, torch.tensor([batch_size, 1, 1])) , cse_var_84)
		return l_new, u_new, cse_var_64, cse_var_64
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):
		cse_var_98 = abs_elem.get_elem_new('u', prev)
		cse_var_43 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_40 = le(cse_var_98, cse_var_43)
		cse_var_102 = convert_to_float(cse_var_40)
		cse_var_103 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_39 = plus(mult(cse_var_102, cse_var_43), mult(minus(cse_var_103, cse_var_102), cse_var_43))
		cse_var_97 = abs_elem.get_elem_new('l', prev)
		cse_var_42 = ge(cse_var_97, cse_var_43)
		cse_var_101 = convert_to_float(cse_var_42)
		l_new = plus(mult(cse_var_101, cse_var_97), mult(minus(cse_var_103, cse_var_101), cse_var_39))
		u_new = plus(mult(cse_var_101, cse_var_98), mult(minus(cse_var_103, cse_var_101), plus(mult(cse_var_102, cse_var_43), mult(minus(cse_var_103, cse_var_102), cse_var_98))))
		cse_var_38 = prev.convert_to_poly(abs_elem)
		cse_var_46 = repeat(cse_var_38.get_const(), torch.tensor([batch_size, 1]))
		cse_var_95 = repeat(cse_var_38.get_mat(abs_elem), torch.tensor([batch_size, 1, 1]))
		cse_var_37 = PolyExpSparse(abs_elem.network, 0.0, cse_var_39)
		cse_var_92 = repeat(cse_var_101.unsqueeze(2), torch.tensor([1, 1, poly_size]))
		cse_var_91 = repeat(minus(cse_var_103, cse_var_101).unsqueeze(2), torch.tensor([1, 1, poly_size]))
		L_new = PolyExpSparse(abs_elem.network, plus(mult(cse_var_92, cse_var_95), mult(cse_var_91, cse_var_37.get_mat(abs_elem))) , plus(mult(cse_var_101, cse_var_46), mult(minus(cse_var_103, cse_var_101), cse_var_37.get_const())))
		cse_var_33 = minus(cse_var_98, cse_var_97)
		cse_var_96 = divide(cse_var_98, cse_var_33)
		cse_var_99 = mult(cse_var_96, cse_var_46)
		cse_var_94 = plus(cse_var_99, divide(mult(mult(cse_var_98, cse_var_97), SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=minus(0.0, 1.0), type= type(minus(0.0, 1.0))).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))), cse_var_33))
		cse_var_32 = PolyExpSparse(abs_elem.network, 0.0, 0.0)
		cse_var_89 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=cse_var_32.get_const(), type= type(cse_var_32.get_const())).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_90 = plus(repeat(mult(cse_var_102, SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=cse_var_32.get_mat(abs_elem), type= type(cse_var_32.get_mat(abs_elem))).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))).unsqueeze(2), torch.tensor([1, 1, poly_size])), mult(repeat(mult(minus(cse_var_103, cse_var_102), cse_var_96).unsqueeze(2), torch.tensor([1, 1, poly_size])), cse_var_95))
		U_new = PolyExpSparse(abs_elem.network, plus(mult(cse_var_92, cse_var_95), mult(cse_var_91, cse_var_90)) , plus(mult(cse_var_101, cse_var_46), mult(minus(cse_var_103, cse_var_101), plus(mult(cse_var_102, cse_var_89), mult(minus(cse_var_103, cse_var_102), cse_var_94)))))
		return l_new, u_new, L_new, U_new
	
