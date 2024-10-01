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
		cse_var_90 = cse_var_24.get_mat(abs_elem)
		cse_var_45 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_90) , copy.deepcopy(plus(cse_var_24.get_const(), cse_var_23)))
		cse_var_12 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([1, curr_size, poly_size]))
		cse_var_21 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([1, curr_size, poly_size]))
		cse_var_83 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([1, curr_size, poly_size]))
		phi_trav_exp1_1_1 = cse_var_45
		phi_trav_exp1_2_3 = cse_var_45
		while(True):
			cse_var_28 = phi_trav_exp1_1_1.get_mat(abs_elem)
			vertices_stop1 = False
			vertices1_1_1 = ne(cse_var_28, cse_var_12)
			vertices_stop_default1 = get_default_stop([1, curr_size, poly_size])
			vertices_stop_temp1 = disj(SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=vertices_stop1, type= type(vertices_stop1)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([1, curr_size, poly_size])), vertices_stop_default1)
			vertices1_1_2 = conj(vertices1_1_1, boolNeg(vertices_stop_temp1))
			phi_trav_exp1_2_3 = phi_trav_exp1_1_1
			if(boolNeg(any(vertices1_1_2))):
				phi_trav_exp1_2_3 = phi_trav_exp1_1_1
				break
			cse_var_22 = Llist(abs_elem.network, [1]*(phi_trav_exp1_1_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_2 = abs_elem.get_elem_new('L', cse_var_22)
			var_3 = abs_elem.get_elem_new('U', cse_var_22)
			cse_var_85 = phi_trav_exp1_1_1.get_mat(abs_elem)
			cse_var_72 = var_3.get_mat(abs_elem)
			cse_var_78 = repeat(var_3.get_const(), torch.tensor([1, curr_size, 1]))
			cse_var_74 = var_2.get_mat(abs_elem)
			cse_var_80 = repeat(var_2.get_const(), torch.tensor([1, curr_size, 1]))
			cse_var_17 = ge(cse_var_85, cse_var_21)
			cse_var_86 = convert_to_float(cse_var_17)
			cse_var_79 = mult(minus(cse_var_83, cse_var_86), cse_var_85)
			cse_var_70 = mult(repeat(cse_var_79, torch.tensor([batch_size, 1, 1])), cse_var_78)
			cse_var_81 = mult(cse_var_86, cse_var_85)
			cse_var_71 = mult(repeat(cse_var_81, torch.tensor([batch_size, 1, 1])), cse_var_80)
			cse_var_73 = cse_var_79.unsqueeze(3)
			cse_var_75 = cse_var_81.unsqueeze(3)
			cse_var_68 = plus(cse_var_71.sum(dim=2), cse_var_70.sum(dim=2))
			cse_var_69 = plus(inner_prod(cse_var_75.squeeze(3), cse_var_74.squeeze(1)), inner_prod(cse_var_73.squeeze(3), cse_var_72.squeeze(1)))
			trav_exp1_4_2 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_69) , copy.deepcopy(plus(cse_var_68, repeat(phi_trav_exp1_1_1.get_const(), torch.tensor([batch_size, 1])))))
			phi_trav_exp1_1_1 = trav_exp1_4_2
		cse_var_25 = Llist(abs_elem.network, [1]*(phi_trav_exp1_2_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_88 = phi_trav_exp1_2_3.get_mat(abs_elem)
		cse_var_89 = convert_to_float(ge(cse_var_88, cse_var_21))
		l_new = plus(plus(mult(repeat(mult(cse_var_89, cse_var_88), torch.tensor([batch_size, 1, 1])), repeat(abs_elem.get_elem_new('l', cse_var_25), torch.tensor([1, curr_size, 1]))).sum(dim=2), mult(repeat(mult(minus(cse_var_83, cse_var_89), cse_var_88), torch.tensor([batch_size, 1, 1])), repeat(abs_elem.get_elem_new('u', cse_var_25), torch.tensor([1, curr_size, 1]))).sum(dim=2)), repeat(phi_trav_exp1_2_3.get_const(), torch.tensor([batch_size, 1])))
		phi_trav_exp2_5_1 = cse_var_45
		phi_trav_exp2_6_3 = cse_var_45
		while(True):
			cse_var_13 = phi_trav_exp2_5_1.get_mat(abs_elem)
			vertices_stop2 = False
			vertices2_5_1 = ne(cse_var_13, cse_var_12)
			vertices_stop_default2 = get_default_stop([1, curr_size, poly_size])
			vertices_stop_temp2 = disj(SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=vertices_stop2, type= type(vertices_stop2)).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(torch.tensor([1, curr_size, poly_size])), vertices_stop_default2)
			vertices2_5_2 = conj(vertices2_5_1, boolNeg(vertices_stop_temp2))
			phi_trav_exp2_6_3 = phi_trav_exp2_5_1
			if(boolNeg(any(vertices2_5_2))):
				phi_trav_exp2_6_3 = phi_trav_exp2_5_1
				break
			cse_var_8 = Llist(abs_elem.network, [1]*(phi_trav_exp2_5_1.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
			var_7 = abs_elem.get_elem_new('U', cse_var_8)
			var_8 = abs_elem.get_elem_new('L', cse_var_8)
			cse_var_64 = phi_trav_exp2_5_1.get_mat(abs_elem)
			cse_var_52 = var_8.get_mat(abs_elem)
			cse_var_58 = repeat(var_8.get_const(), torch.tensor([1, curr_size, 1]))
			cse_var_54 = var_7.get_mat(abs_elem)
			cse_var_60 = repeat(var_7.get_const(), torch.tensor([1, curr_size, 1]))
			cse_var_4 = ge(cse_var_64, cse_var_21)
			cse_var_65 = convert_to_float(cse_var_4)
			cse_var_59 = mult(minus(cse_var_83, cse_var_65), cse_var_64)
			cse_var_50 = mult(repeat(cse_var_59, torch.tensor([batch_size, 1, 1])), cse_var_58)
			cse_var_61 = mult(cse_var_65, cse_var_64)
			cse_var_51 = mult(repeat(cse_var_61, torch.tensor([batch_size, 1, 1])), cse_var_60)
			cse_var_53 = cse_var_59.unsqueeze(3)
			cse_var_55 = cse_var_61.unsqueeze(3)
			cse_var_48 = plus(cse_var_51.sum(dim=2), cse_var_50.sum(dim=2))
			cse_var_49 = plus(inner_prod(cse_var_55.squeeze(3), cse_var_54.squeeze(1)), inner_prod(cse_var_53.squeeze(3), cse_var_52.squeeze(1)))
			trav_exp2_8_2 = PolyExpSparse(abs_elem.network, copy.deepcopy(cse_var_49) , copy.deepcopy(plus(cse_var_48, repeat(phi_trav_exp2_5_1.get_const(), torch.tensor([batch_size, 1])))))
			phi_trav_exp2_5_1 = trav_exp2_8_2
		cse_var_9 = Llist(abs_elem.network, [1]*(phi_trav_exp2_6_3.mat.dims-1), None, None,torch.nonzero(abs_elem.d['llist']).flatten().tolist())
		cse_var_66 = phi_trav_exp2_6_3.get_mat(abs_elem)
		cse_var_67 = convert_to_float(ge(cse_var_66, cse_var_21))
		u_new = plus(plus(mult(repeat(mult(cse_var_67, cse_var_66), torch.tensor([batch_size, 1, 1])), repeat(abs_elem.get_elem_new('u', cse_var_9), torch.tensor([1, curr_size, 1]))).sum(dim=2), mult(repeat(mult(minus(cse_var_83, cse_var_67), cse_var_66), torch.tensor([batch_size, 1, 1])), repeat(abs_elem.get_elem_new('l', cse_var_9), torch.tensor([1, curr_size, 1]))).sum(dim=2)), repeat(phi_trav_exp2_6_3.get_const(), torch.tensor([batch_size, 1])))
		return l_new, u_new, cse_var_45, cse_var_45
	
	def Relu(self, abs_elem, prev, curr, poly_size, curr_size, prev_size, input_size, batch_size):
		cse_var_100 = abs_elem.get_elem_new('u', prev)
		cse_var_43 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=0.0, type= type(0.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_40 = le(cse_var_100, cse_var_43)
		cse_var_104 = convert_to_float(cse_var_40)
		cse_var_105 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=1.0, type= type(1.0)).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_39 = plus(mult(cse_var_104, cse_var_43), mult(minus(cse_var_105, cse_var_104), cse_var_43))
		cse_var_99 = abs_elem.get_elem_new('l', prev)
		cse_var_42 = ge(cse_var_99, cse_var_43)
		cse_var_103 = convert_to_float(cse_var_42)
		l_new = plus(mult(cse_var_103, cse_var_99), mult(minus(cse_var_105, cse_var_103), cse_var_39))
		u_new = plus(mult(cse_var_103, cse_var_100), mult(minus(cse_var_105, cse_var_103), plus(mult(cse_var_104, cse_var_43), mult(minus(cse_var_105, cse_var_104), cse_var_100))))
		cse_var_38 = prev.convert_to_poly(abs_elem)
		cse_var_46 = repeat(cse_var_38.get_const(), torch.tensor([batch_size, 1]))
		cse_var_97 = repeat(cse_var_38.get_mat(abs_elem), torch.tensor([batch_size, 1, 1]))
		cse_var_37 = PolyExpSparse(abs_elem.network, 0.0, cse_var_39)
		cse_var_94 = repeat(cse_var_103.unsqueeze(2), torch.tensor([1, 1, poly_size]))
		cse_var_93 = repeat(minus(cse_var_105, cse_var_103).unsqueeze(2), torch.tensor([1, 1, poly_size]))
		L_new = PolyExpSparse(abs_elem.network, copy.deepcopy(plus(mult(cse_var_94, cse_var_97), mult(cse_var_93, cse_var_37.get_mat(abs_elem)))) , copy.deepcopy(plus(mult(cse_var_103, cse_var_46), mult(minus(cse_var_105, cse_var_103), cse_var_37.get_const()))))
		cse_var_33 = minus(cse_var_100, cse_var_99)
		cse_var_98 = divide(cse_var_100, cse_var_33)
		cse_var_101 = mult(cse_var_98, cse_var_46)
		cse_var_96 = plus(cse_var_101, divide(mult(mult(cse_var_100, cse_var_99), SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=minus(0.0, 1.0), type= type(minus(0.0, 1.0))).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))), cse_var_33))
		cse_var_32 = PolyExpSparse(abs_elem.network, 0.0, 0.0)
		cse_var_91 = SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=cse_var_32.get_const(), type= type(cse_var_32.get_const())).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))
		cse_var_92 = plus(repeat(mult(cse_var_104, SparseTensorBlock([], [], 0, torch.tensor([]), dense_const=cse_var_32.get_mat(abs_elem), type= type(cse_var_32.get_mat(abs_elem))).unsqueeze(0).unsqueeze(1).repeat(torch.tensor([batch_size, curr_size]))).unsqueeze(2), torch.tensor([1, 1, poly_size])), mult(repeat(mult(minus(cse_var_105, cse_var_104), cse_var_98).unsqueeze(2), torch.tensor([1, 1, poly_size])), cse_var_97))
		U_new = PolyExpSparse(abs_elem.network, copy.deepcopy(plus(mult(cse_var_94, cse_var_97), mult(cse_var_93, cse_var_92))) , copy.deepcopy(plus(mult(cse_var_103, cse_var_46), mult(minus(cse_var_105, cse_var_103), plus(mult(cse_var_104, cse_var_91), mult(minus(cse_var_105, cse_var_104), cse_var_96))))))
		return l_new, u_new, L_new, U_new
	
def matmul_new(self, sp_tensor):
        if isinstance(sp_tensor, torch.Tensor):
            sp_tensor = convert_dense_to_sparse(sp_tensor)
        

        assert(self.type == float or self.type == int)
        assert(sp_tensor.type == float or sp_tensor.type == int)
        assert(self.dense_const == 0.0)
        assert(sp_tensor.dense_const == 0.0)
        
        if sp_tensor.dims <= 2:
            assert(sp_tensor.total_size[-1].item() == self.total_size[-1].item())
        else:
            assert(sp_tensor.total_size[-2].item() == self.total_size[-1].item())
            assert(sp_tensor.total_size[:-2] == self.total_size[-sp_tensor.dims:-2])
        
        
        start_indices = []
        end_indices = []
        blocks = []
        res_total_size = None
        
        if self.dims == sp_tensor.dims:
            res_total_size = torch.concat([self.total_size[:-1], sp_tensor.total_size[-1:]])
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 2:
                        start_index = torch.concat([self.start_indices[i][:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([self.end_indices[i][:-1], sp_tensor.end_indices[j][-1:]])
                    else:
                        if not self.block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-2], sp_tensor.end_indices[j][:-2]]):
                            continue
                        if not self.block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-2:-1], sp_tensor.end_indices[j][-2:-1]]):
                            continue
                        block = self.blocks[i].type(torch.float) @ sp_tensor.blocks[j].type(torch.float)
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-2], self.start_indices[i][:-2]), self.start_indices[i][-2:-1], sp_tensor.start_indices[j][-1:]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-2], self.end_indices[i][:-2]), self.end_indices[i][-2:-1], sp_tensor.end_indices[j][-1:]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        elif self.dims > sp_tensor.dims:
            res_total_size = self.total_size[:-1]
            for i in range(self.num_blocks):
                for j in range(sp_tensor.num_blocks):
                    if sp_tensor.dims == 1:
                        start_index = self.start_indices[i][:-1]
                        end_index = self.end_indices[i][:-1]
                    else:
                        if not self.block_overlap([self.start_indices[i][:-2], self.end_indices[i][:-2]], [sp_tensor.start_indices[j][:-1], sp_tensor.end_indices[j][:-1]]):
                            continue
                        if not self.block_overlap([self.start_indices[i][-1:], self.end_indices[i][-1:]], [sp_tensor.start_indices[j][-1:], sp_tensor.end_indices[j][-1:]]):
                            continue
                        block = (self.blocks[i].type(torch.float) @ sp_tensor.blocks[j].type(torch.float).unsqueeze(-1)).squeeze(-1)
                        blocks.append(block)
                        start_index = torch.concat([torch.min(sp_tensor.start_indices[j][:-1], self.start_indices[i][:-2]), self.start_indices[i][-2:-1]])
                        end_index = torch.concat([torch.max(sp_tensor.end_indices[j][:-1], self.end_indices[i][:-2]), self.end_indices[i][-2:-1]])
                    start_indices.append(start_index)
                    end_indices.append(end_index)
        else:
            assert(False)

        overlap_classes = dict()
        for i in range(len(start_indices)):
            if i not in overlap_classes:
                flag = False
                for key in overlap_classes.keys():
                    if i in overlap_classes[key]:
                        flag = True 
                        break
                if not flag:
                    overlap_classes[i] = []
            for j in range(i+1, len(start_indices)):
                if self.block_overlap([start_indices[i], end_indices[i]], [start_indices[j], end_indices[j]]):
                    flag = False
                    for key in overlap_classes.keys():
                        if i==key:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                        elif i in overlap_classes[key]:
                            flag = True
                            if j not in overlap_classes[key]:
                                overlap_classes[key].append(j)
                    if not flag:
                        overlap_classes[i] = [j]
        res_start_indices = []
        res_end_indices = []
        res_blocks = []
        for i, key in enumerate(overlap_classes.keys()):
            start_index = start_indices[key]
            end_index = end_indices[key]
            for j in overlap_classes[key]:
                start_index, end_index = self.union_block([start_index, end_index], [start_indices[j], end_indices[j]])
            block = SparseTensorBlock([start_index], [blocks[key]], len(start_index), res_total_size)
            for j in overlap_classes[key]:
                new_block = SparseTensorBlock([start_index], [blocks[j]], len(start_index), res_total_size)
                block = block.binary(new_block, '+')
            res_start_indices.append(start_index)
            res_end_indices.append(end_index)
            res_blocks.append(block.blocks[0])
        return SparseTensorBlock(res_start_indices, res_blocks, len(res_total_size), res_total_size, res_end_indices) 