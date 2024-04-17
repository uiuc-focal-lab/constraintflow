import torch
from common.polyexp import PolyExpNew, Nlist
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_8 = prev.dot(curr.get_metadata('weight'))
		cse_var_7 = curr.get_metadata('bias')
		cse_var_26 = PolyExpNew(poly_size, cse_var_8.get_mat(abs_elem), plus(cse_var_7, cse_var_8.get_const()))
		var_1 = cse_var_26
		trav_exp1 = var_1
		while(True):
			trav_size = trav_exp1.const.shape[0]
			var_2 = trav_exp1
			vertices_stop = False
			vertices = trav_exp1.mat != 0
			vertices_stop = convert_to_tensor(vertices_stop, poly_size) != True
			vertices_stop_default = torch.zeros(vertices_stop.size())
			vertices_stop_default[0:input_size] = 1
			vertices_stop_default = vertices_stop_default.bool()
			vertices_stop = vertices_stop | vertices_stop_default
			vertices = vertices & (~ vertices_stop)
			if not(vertices.any()):
				break
			var_3 = trav_exp1
			# vertices_priority = minus(cse_var_12, abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)).get_metadata('layer'))
			var_4 = trav_exp1
			cse_var_15 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_5 = abs_elem.get_elem_new('L', cse_var_15)
			var_6 = abs_elem.get_elem_new('U', cse_var_15)
			cse_var_14 = var_4.get_mat(abs_elem)
			cse_var_13 = cse_var_14.unsqueeze(2).repeat( 1, 1, 1*poly_size)
			var_7 = PolyExpNew(poly_size, mult(cse_var_13, var_5.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(cse_var_14, var_5.get_const().repeat( 1*trav_size, 1)))
			var_8 = PolyExpNew(poly_size, mult(cse_var_13, var_6.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(cse_var_14, var_6.get_const().repeat( 1*trav_size, 1)))
			cse_var_11 = ge(cse_var_14, cse_var_12)
			trav_exp1 = plus(PolyExpNew(poly_size, torch.where(cse_var_11.unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(abs_elem), var_8.get_mat(abs_elem)), torch.where(cse_var_11, var_7.get_const(), var_8.get_const())).sum(dim=1), var_4.get_const())
		cse_var_9 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_10 = trav_exp1.get_mat(abs_elem)
		cse_var_12 = torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size, 1*poly_size)
		l_new = plus(torch.where(ge(cse_var_10, cse_var_12), mult(cse_var_10, abs_elem.get_elem_new('l', cse_var_9).repeat( 1*curr_size, 1)), mult(cse_var_10, abs_elem.get_elem_new('u', cse_var_9).repeat( 1*curr_size, 1))).sum(dim=1), trav_exp1.get_const())
		var_10 = cse_var_26
		trav_exp2 = var_10
		while(True):
			trav_size = trav_exp2.const.shape[0]
			var_11 = trav_exp2
			vertices_stop = False
			vertices = trav_exp2.mat != 0
			vertices_stop = convert_to_tensor(vertices_stop, poly_size) != True
			vertices_stop_default = torch.zeros(vertices_stop.size())
			vertices_stop_default[0:input_size] = 1
			vertices_stop_default = vertices_stop_default.bool()
			vertices_stop = vertices_stop | vertices_stop_default
			vertices = vertices & (~ vertices_stop)
			if not(vertices.any()):
				break
			var_12 = trav_exp2
			# vertices_priority = minus(cse_var_12, abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)).get_metadata('layer'))
			var_13 = trav_exp2
			cse_var_6 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
			var_14 = abs_elem.get_elem_new('U', cse_var_6)
			var_15 = abs_elem.get_elem_new('L', cse_var_6)
			cse_var_5 = var_13.get_mat(abs_elem)
			cse_var_4 = cse_var_5.unsqueeze(2).repeat( 1, 1, 1*poly_size)
			var_16 = PolyExpNew(poly_size, mult(cse_var_4, var_14.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(cse_var_5, var_14.get_const().repeat( 1*trav_size, 1)))
			var_17 = PolyExpNew(poly_size, mult(cse_var_4, var_15.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(cse_var_5, var_15.get_const().repeat( 1*trav_size, 1)))
			cse_var_3 = ge(cse_var_5, cse_var_12)
			trav_exp2 = plus(PolyExpNew(poly_size, torch.where(cse_var_3.unsqueeze(2).repeat( 1, 1, 1*poly_size), var_16.get_mat(abs_elem), var_17.get_mat(abs_elem)), torch.where(cse_var_3, var_16.get_const(), var_17.get_const())).sum(dim=1), var_13.get_const())
		cse_var_1 = abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))
		cse_var_2 = trav_exp2.get_mat(abs_elem)
		u_new = plus(torch.where(ge(cse_var_2, cse_var_12), mult(cse_var_2, abs_elem.get_elem_new('u', cse_var_1).repeat( 1*curr_size, 1)), mult(cse_var_2, abs_elem.get_elem_new('l', cse_var_1).repeat( 1*curr_size, 1))).sum(dim=1), trav_exp2.get_const())
		L_new = cse_var_26
		U_new = cse_var_26
		return l_new, u_new, L_new, U_new
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		cse_var_22 = abs_elem.get_elem_new('u', prev)
		cse_var_24 = torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)
		cse_var_21 = le(cse_var_22, cse_var_24)
		cse_var_20 = torch.where(cse_var_21, cse_var_24, cse_var_24)
		cse_var_25 = abs_elem.get_elem_new('l', prev)
		cse_var_23 = ge(cse_var_25, cse_var_24)
		l_new = torch.where(cse_var_23, cse_var_25, cse_var_20)
		u_new = torch.where(cse_var_23, cse_var_22, torch.where(cse_var_21, cse_var_24, cse_var_22))
		cse_var_19 = prev.convert_to_poly()
		var_21 = PolyExpNew(poly_size, None, cse_var_20)
		cse_var_16 = cse_var_23.unsqueeze(1).repeat( 1, 1*poly_size)
		cse_var_27 = cse_var_19.get_const()
		cse_var_28 = cse_var_19.get_mat(abs_elem)
		L_new = PolyExpNew(poly_size, torch.where(cse_var_16, cse_var_28, var_21.get_mat(abs_elem)), torch.where(cse_var_23, cse_var_27, var_21.get_const()))
		cse_var_17 = minus(cse_var_22, cse_var_25)
		cse_var_18 = divide(cse_var_22, cse_var_17)
		var_23 = PolyExpNew(poly_size, mult(cse_var_18.unsqueeze(1).repeat( 1, 1*poly_size), cse_var_28), mult(cse_var_18, cse_var_27))
		var_24 = PolyExpNew(poly_size, None, 0.0)
		var_25 = PolyExpNew(poly_size, var_23.get_mat(abs_elem), minus(divide(mult(cse_var_22, cse_var_25), cse_var_17), var_23.get_const()))
		var_27 = PolyExpNew(poly_size, torch.where(cse_var_21.unsqueeze(1).repeat( 1, 1*poly_size), torch.tensor(var_24.get_mat(abs_elem)).unsqueeze(0).unsqueeze(1).repeat( 1*curr_size, 1*poly_size), var_25.get_mat(abs_elem)), torch.where(cse_var_21, torch.tensor(var_24.get_const()).unsqueeze(0).repeat( 1*curr_size), var_25.get_const()))
		U_new = PolyExpNew(poly_size, torch.where(cse_var_16, cse_var_28, var_27.get_mat(abs_elem)), torch.where(cse_var_23, cse_var_27, var_27.get_const()))
		return l_new, u_new, L_new, U_new
	
