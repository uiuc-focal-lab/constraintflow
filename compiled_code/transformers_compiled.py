import torch
from common.polyexp import PolyExpNew, Nlist
from utils import *
class deeppoly:
	def Affine(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		var_0 = prev.dot(curr.get_metadata('weight'))
		var_1 = PolyExpNew(poly_size, var_0.get_mat(abs_elem), plus(curr.get_metadata('bias'), var_0.get_const()))
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
			vertices_priority = minus(torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size*1, 1*poly_size), abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)).get_metadata('layer'))
			var_4 = trav_exp1
			var_5 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_6 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_7 = PolyExpNew(poly_size, mult(var_4.get_mat(abs_elem).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_5.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(var_4.get_mat(abs_elem), var_5.get_const().repeat( 1*trav_size, 1)))
			var_8 = PolyExpNew(poly_size, mult(var_4.get_mat(abs_elem).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_6.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(var_4.get_mat(abs_elem), var_6.get_const().repeat( 1*trav_size, 1)))
			trav_exp1 = plus(PolyExpNew(poly_size, torch.where(ge(var_4.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_7.get_mat(abs_elem), var_8.get_mat(abs_elem)), torch.where(ge(var_4.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size, 1*poly_size)), var_7.get_const(), var_8.get_const())).sum(dim=1), var_4.get_const())
		l_new = plus(torch.where(ge(trav_exp1.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*curr_size, 1*poly_size)), mult(trav_exp1.get_mat(abs_elem), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1)), mult(trav_exp1.get_mat(abs_elem), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1))).sum(dim=1), trav_exp1.get_const())
		var_9 = prev.dot(curr.get_metadata('weight'))
		var_10 = PolyExpNew(poly_size, var_9.get_mat(abs_elem), plus(curr.get_metadata('bias'), var_9.get_const()))
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
			vertices_priority = minus(torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size*1, 1*poly_size), abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)).get_metadata('layer'))
			var_13 = trav_exp2
			var_14 = abs_elem.get_elem_new('U', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_15 = abs_elem.get_elem_new('L', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None)))
			var_16 = PolyExpNew(poly_size, mult(var_13.get_mat(abs_elem).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_14.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(var_13.get_mat(abs_elem), var_14.get_const().repeat( 1*trav_size, 1)))
			var_17 = PolyExpNew(poly_size, mult(var_13.get_mat(abs_elem).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_15.get_mat(abs_elem).repeat( 1*trav_size, 1, 1)), mult(var_13.get_mat(abs_elem), var_15.get_const().repeat( 1*trav_size, 1)))
			trav_exp2 = plus(PolyExpNew(poly_size, torch.where(ge(var_13.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size, 1*poly_size)).unsqueeze(2).repeat( 1, 1, 1*poly_size), var_16.get_mat(abs_elem), var_17.get_mat(abs_elem)), torch.where(ge(var_13.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*trav_size, 1*poly_size)), var_16.get_const(), var_17.get_const())).sum(dim=1), var_13.get_const())
		u_new = plus(torch.where(ge(trav_exp2.get_mat(abs_elem), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat( 1*curr_size, 1*poly_size)), mult(trav_exp2.get_mat(abs_elem), abs_elem.get_elem_new('u', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1)), mult(trav_exp2.get_mat(abs_elem), abs_elem.get_elem_new('l', abs_elem.get_live_nlist(Nlist(poly_size, 0, poly_size-1, None))).repeat( 1*curr_size, 1))).sum(dim=1), trav_exp2.get_const())
		var_18 = prev.dot(curr.get_metadata('weight'))
		L_new = PolyExpNew(poly_size, var_18.get_mat(abs_elem), plus(curr.get_metadata('bias'), var_18.get_const()))
		var_19 = prev.dot(curr.get_metadata('weight'))
		U_new = PolyExpNew(poly_size, var_19.get_mat(abs_elem), plus(curr.get_metadata('bias'), var_19.get_const()))
		return l_new, u_new, L_new, U_new
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size):
		l_new = torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('l', prev), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size)))
		u_new = torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), abs_elem.get_elem_new('u', prev), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), abs_elem.get_elem_new('u', prev)))
		var_20 = prev.convert_to_poly()
		var_21 = PolyExpNew(poly_size, None, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size), torch.tensor(0.0).unsqueeze(0).repeat( 1*curr_size)))
		L_new = PolyExpNew(poly_size, torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), var_20.get_mat(abs_elem), var_21.get_mat(abs_elem)), torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), var_20.get_const(), var_21.get_const()))
		var_22 = prev.convert_to_poly()
		var_23 = PolyExpNew(poly_size, mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))).unsqueeze(1).repeat( 1, 1*poly_size), var_22.get_mat(abs_elem)), mult(divide(abs_elem.get_elem_new('u', prev), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_22.get_const()))
		var_24 = PolyExpNew(poly_size, None, 0.0)
		var_25 = PolyExpNew(poly_size, var_23.get_mat(abs_elem), minus(divide(mult(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev)), minus(abs_elem.get_elem_new('u', prev), abs_elem.get_elem_new('l', prev))), var_23.get_const()))
		var_26 = prev.convert_to_poly()
		var_27 = PolyExpNew(poly_size, torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), torch.tensor(var_24.get_mat(abs_elem)).unsqueeze(0).unsqueeze(1).repeat( 1*curr_size, 1*poly_size), var_25.get_mat(abs_elem)), torch.where(le(abs_elem.get_elem_new('u', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), torch.tensor(var_24.get_const()).unsqueeze(0).repeat( 1*curr_size), var_25.get_const()))
		U_new = PolyExpNew(poly_size, torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)).unsqueeze(1).repeat( 1, 1*poly_size), var_26.get_mat(abs_elem), var_27.get_mat(abs_elem)), torch.where(ge(abs_elem.get_elem_new('l', prev), torch.tensor(0).unsqueeze(0).repeat( 1*curr_size)), var_26.get_const(), var_27.get_const()))
		return l_new, u_new, L_new, U_new
	
