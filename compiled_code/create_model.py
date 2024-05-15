curr_size = 50
poly_size = 934
prev_size = 50
var_1 = PolyExp(curr_size, poly_size, self.abs_elem.d['L'].mat[934:984, :poly_size], self.abs_elem.d['L'].const[934:984])
trav_size_0_0 = get_shape_1(var_0.get_mat(poly_size))
phi_trav_exp1_1_1 = var_1
phi_trav_exp1_2_3 = var_1
if debug_flag:
    print('@@@@@@@@@@@')
    print(var_1.mat[0,:][-prev_size:])
while(True):
    if debug_flag:
        print('@@@@@@@@@@@')
        print(phi_trav_exp1_1_1.mat[0, :][-2*prev_size:-prev_size])
        # print(phi_trav_exp1_1_1.mat[0, :].sum())
    trav_size_1_2 = get_shape_1(phi_trav_exp1_1_1.get_mat(poly_size))
    vertices_stop1 = False
    vertices1_1_1 = ne(phi_trav_exp1_1_1.get_mat(poly_size), torch.tensor(0).unsqueeze(0).unsqueeze(1).repeat(curr_size, poly_size))
    vertices_stop_default1 = get_default_stop2([curr_size, poly_size])
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
    vertices_stop_default2 = get_default_stop2([curr_size, poly_size])
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
print(L_new.mat[:, 784:834])