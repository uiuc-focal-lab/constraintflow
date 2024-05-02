import torch
from common.polyexp import PolyExp, Nlist
from utils import *
import time
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		ttime = time.time()
		var_0 = prev.dot(curr.get_metadata('weight'))
		var_1 = PolyExp(curr_size, poly_size, var_0.get_mat(poly_size), plus(curr.get_metadata('bias'), var_0.get_const()))
		cse_var_17 = curr.get_metadata('bias')
		cse_var_18 = prev.dot(curr.get_metadata('weight'))
		trav_exp1_0_0 = var_1
		phi_trav_exp1_1_1 = trav_exp1_0_0
		phi_trav_size_1_0 = 0
		phi_vertices1_1_0 = 0
		phi_vertices_priority_1_0 = 0
		phi_vertices_stop_1_0 = 0
		phi_trav_exp1_2_3 = trav_exp1_0_0
		phi_trav_size_2_2 = 0
		phi_vertices1_2_3 = 0
		phi_vertices_priority_2_2 = 0
		phi_vertices_stop_2_2 = 0
		print('Time 1', time.time() - ttime)
		ttime = time.time() - ttime
		while(True):
			trav_size_1_1 = get_shape_0(phi_trav_exp1_1_1.get_const())
			vertices_stop_1_1 = False
			vertices1_1_1 = ne(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size))
			vertices_stop_default1 = get_default_stop([trav_size_1_1, poly_size])
			vertices_stop_temp1 = disj(torch.tensor(vertices_stop_1_1).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			phi_trav_size_2_2 = trav_size_1_1
			phi_vertices1_2_3 = vertices1_1_2
			phi_vertices_priority_2_2 = phi_vertices_priority_1_0
			phi_vertices_stop_2_2 = vertices_stop_1_1
			print('Time 2', time.time() - ttime)
			ttime = time.time() - ttime
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				phi_trav_size_2_2 = trav_size_1_1
				phi_vertices1_2_3 = vertices1_1_2
				phi_vertices_priority_2_2 = phi_vertices_priority_1_0
				phi_vertices_stop_2_2 = vertices_stop_1_1
				break
			vertices_priority_4_1 = None
			var_2 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_3 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_4 = PolyExp(trav_size_1_1, poly_size, PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_const())).get_mat(poly_size).sum(dim=1), PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_const())).get_const().sum(dim=1))
			cse_var_10 = PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1))).get_const()))
			cse_var_11 = PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_3.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_3.get_const().repeat( 1*trav_size_1_1, 1)))
			cse_var_12 = PolyExp(poly_size, poly_size, mult(phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_2.get_mat(poly_size).repeat( 1*trav_size_1_1, 1, 1)), mult(phi_trav_exp1_1_1.get_mat(poly_size), var_2.get_const().repeat( 1*trav_size_1_1, 1)))
			cse_var_13 = ge(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_1_1, 1*poly_size))
			cse_var_14 = phi_trav_exp1_1_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size)
			cse_var_15 = phi_trav_exp1_1_1.get_mat(poly_size)
			cse_var_16 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			trav_exp1_4_2 = PolyExp(trav_size_1_1, poly_size, var_4.get_mat(poly_size), plus(phi_trav_exp1_1_1.get_const(), var_4.get_const()))
			phi_trav_exp1_1_1 = trav_exp1_4_2
			phi_trav_size_1_0 = trav_size_1_1
			phi_vertices1_1_0 = vertices1_1_2
			phi_vertices_priority_1_0 = vertices_priority_4_1
			phi_vertices_stop_1_0 = vertices_stop_1_1
		print('Time 3', time.time() - ttime)
		ttime = time.time() - ttime
		l_new = plus(torch.where(ge(phi_trav_exp1_2_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*phi_trav_size_2_2, 1*poly_size)), mult(phi_trav_exp1_2_3.get_mat(poly_size), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*phi_trav_size_2_2, 1)), mult(phi_trav_exp1_2_3.get_mat(poly_size), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*phi_trav_size_2_2, 1))).sum(dim=1), phi_trav_exp1_2_3.get_const())
		var_5 = prev.dot(curr.get_metadata('weight'))
		var_6 = PolyExp(curr_size, poly_size, var_5.get_mat(poly_size), plus(curr.get_metadata('bias'), var_5.get_const()))
		cse_var_19 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_20 = phi_trav_exp1_2_3.get_mat(poly_size)
		trav_exp2_2_0 = var_6
		phi_trav_exp2_5_1 = trav_exp2_2_0
		phi_trav_size_5_3 = phi_trav_size_2_2
		phi_vertices2_5_0 = 0
		phi_vertices_priority_5_3 = phi_vertices_priority_2_2
		phi_vertices_stop_5_3 = phi_vertices_stop_2_2
		phi_trav_exp2_6_3 = trav_exp2_2_0
		phi_trav_size_6_5 = phi_trav_size_2_2
		phi_vertices2_6_3 = 0
		phi_vertices_priority_6_5 = phi_vertices_priority_2_2
		phi_vertices_stop_6_5 = phi_vertices_stop_2_2
		print('Time 4', time.time() - ttime)
		ttime = time.time() - ttime
		while(True):
			trav_size_5_4 = get_shape_0(phi_trav_exp2_5_1.get_const())
			vertices_stop_5_4 = False
			vertices2_5_1 = ne(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size))
			vertices_stop_default2 = get_default_stop([trav_size_5_4, poly_size])
			vertices_stop_temp2 = disj(torch.tensor(vertices_stop_5_4).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			phi_trav_size_6_5 = trav_size_5_4
			phi_vertices2_6_3 = vertices2_5_2
			phi_vertices_priority_6_5 = phi_vertices_priority_5_3
			phi_vertices_stop_6_5 = vertices_stop_5_4
			print('Time 5', time.time() - ttime)
			ttime = time.time() - ttime
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				phi_trav_size_6_5 = trav_size_5_4
				phi_vertices2_6_3 = vertices2_5_2
				phi_vertices_priority_6_5 = phi_vertices_priority_5_3
				phi_vertices_stop_6_5 = vertices_stop_5_4
				break
			vertices_priority_8_4 = None
			var_7 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_8 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_9 = PolyExp(trav_size_5_4, poly_size, PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_const())).get_mat(poly_size).sum(dim=1), PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_const())).get_const().sum(dim=1))
			cse_var_1 = PolyExp(poly_size, poly_size, torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_mat(poly_size)), torch.where(ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size)), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1))).get_const(), PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1))).get_const()))
			cse_var_2 = PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_8.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_8.get_const().repeat( 1*trav_size_5_4, 1)))
			cse_var_3 = PolyExp(poly_size, poly_size, mult(phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(poly_size).repeat( 1*trav_size_5_4, 1, 1)), mult(phi_trav_exp2_5_1.get_mat(poly_size), var_7.get_const().repeat( 1*trav_size_5_4, 1)))
			cse_var_4 = ge(phi_trav_exp2_5_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size_5_4, 1*poly_size))
			cse_var_5 = phi_trav_exp2_5_1.get_mat(poly_size).unsqueeze(2).repeat( 1, 1, 1*poly_size)
			cse_var_6 = phi_trav_exp2_5_1.get_mat(poly_size)
			cse_var_7 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			trav_exp2_8_2 = PolyExp(trav_size_5_4, poly_size, var_9.get_mat(poly_size), plus(phi_trav_exp2_5_1.get_const(), var_9.get_const()))
			phi_trav_exp2_5_1 = trav_exp2_8_2
			phi_trav_size_5_3 = trav_size_5_4
			phi_vertices2_5_0 = vertices2_5_2
			phi_vertices_priority_5_3 = vertices_priority_8_4
			phi_vertices_stop_5_3 = vertices_stop_5_4
		print('Time 6', time.time() - ttime)
		ttime = time.time() - ttime
		u_new = plus(torch.where(ge(phi_trav_exp2_6_3.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*phi_trav_size_6_5, 1*poly_size)), mult(phi_trav_exp2_6_3.get_mat(poly_size), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*phi_trav_size_6_5, 1)), mult(phi_trav_exp2_6_3.get_mat(poly_size), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*phi_trav_size_6_5, 1))).sum(dim=1), phi_trav_exp2_6_3.get_const())
		var_10 = prev.dot(curr.get_metadata('weight'))
		L_new = PolyExp(curr_size, poly_size, var_10.get_mat(poly_size), plus(curr.get_metadata('bias'), var_10.get_const()))
		var_11 = prev.dot(curr.get_metadata('weight'))
		U_new = PolyExp(curr_size, poly_size, var_11.get_mat(poly_size), plus(curr.get_metadata('bias'), var_11.get_const()))
		cse_var_8 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_9 = phi_trav_exp2_6_3.get_mat(poly_size)
		print('Time 7', time.time() - ttime)
		ttime = time.time() - ttime
		return l_new, u_new, L_new, U_new
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		l_new = torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size)))
		u_new = torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('u', prev), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), abs_elem.get_elem_new('u', prev)))
		L_new = PolyExp(curr_size, poly_size, torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), prev.convert_to_poly().get_mat(poly_size), PolyExp(curr_size, poly_size, None, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size))).get_mat(poly_size)), torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), prev.convert_to_poly().get_const(), PolyExp(curr_size, poly_size, None, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size))).get_const()))
		var_12 = prev.convert_to_poly()
		var_13 = PolyExp(curr_size, poly_size, mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))).unsqueeze(1).repeat( 1, 1*poly_size), var_12.get_mat(poly_size)), mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_12.get_const()))
		U_new = PolyExp(curr_size, poly_size, torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), prev.convert_to_poly().get_mat(poly_size), PolyExp(curr_size, poly_size, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_mat(poly_size)).unsqueeze(0).repeat( 1*curr_size).unsqueeze(1).repeat( 1, 1*poly_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_mat(poly_size)), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_const()).unsqueeze(0).repeat( 1*curr_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_const())).get_mat(poly_size)), torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), prev.convert_to_poly().get_const(), PolyExp(curr_size, poly_size, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_mat(poly_size)).unsqueeze(0).repeat( 1*curr_size).unsqueeze(1).repeat( 1, 1*poly_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_mat(poly_size)), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_const()).unsqueeze(0).repeat( 1*curr_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_const())).get_const()))
		cse_var_21 = PolyExp(curr_size, poly_size, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_mat(poly_size)).unsqueeze(0).repeat( 1*curr_size).unsqueeze(1).repeat( 1, 1*poly_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_mat(poly_size)), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(PolyExp(1, poly_size, None, 0.0).get_const()).unsqueeze(0).repeat( 1*curr_size), PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const())).get_const()))
		cse_var_22 = prev.convert_to_poly().get_const()
		cse_var_23 = PolyExp(curr_size, poly_size, var_13.get_mat(poly_size), plus(divide(mult(mult(abs_elem.get_elem_new('u', prev), torch.tensor(-1).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_13.get_const()))
		cse_var_24 = PolyExp(1, poly_size, None, 0.0)
		cse_var_25 = minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))
		cse_var_26 = prev.convert_to_poly().get_mat(poly_size)
		cse_var_27 = ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size)
		cse_var_28 = divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)))
		cse_var_29 = PolyExp(curr_size, poly_size, None, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size)))
		cse_var_30 = prev.convert_to_poly()
		cse_var_31 = torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size))
		cse_var_32 = le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size))
		cse_var_33 = abs_elem.get_elem_new('u', prev)
		cse_var_34 = ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size))
		cse_var_35 = torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)
		cse_var_36 = abs_elem.get_elem_new('l', prev)
		return l_new, u_new, L_new, U_new
	
