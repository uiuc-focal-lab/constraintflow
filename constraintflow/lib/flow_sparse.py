from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.polyexp import *
from constraintflow.lib.symexp import *
from constraintflow.lib.llist import *
from constraintflow.lib.network import Network, LayerType, JIT_OP_FOR_LAYER, JIT_OPS
from constraintflow.lib.globals import *
from constraintflow.lib import entry_capture
from constraintflow.lib import globals as G
from constraintflow.gbcsr.sparse_tensor import _tdtype

import torch
import torch.nn.functional as F
import time

_affine_skip_total = 0  # DEBUG: running count of Affine_skip dispatches across the whole run


def get_dense_inlined(t):
    # Inlined get_dense; the simulacrum needs the real meta-tensor path.
    if dummy_mode:
        return t.get_dense()
    res = torch.ones(list(t.total_size), dtype=_tdtype(t.type)) * t.dense_const
    for i in range(t.num_blocks):
        s = [slice(int(t.start_indices[i][j]), int(t.end_indices[i][j])) for j in range(t.start_indices[i].shape[0])]
        b = t.blocks[i]
        if b.block_type == 'D':
            res[tuple(s)] = b.block
        elif b.block_type == 'C':
            res[tuple(s)] = torch.ones(*b.total_shape.tolist()) * b.block
        elif b.block_type == 'R':
            res[tuple(s)] = b.block.expand(*b.total_shape)
        elif b.block_type == 'Diag':
            if b.diag_index == len(b.total_shape):
                res[tuple(s)] = torch.diag_embed(b.block)
            else:
                shape = list(b.block.shape)
                d = b.diag_index - 1
                perm = list(range(len(shape)))
                perm.pop(d)
                perm.append(d)
                new_perm = list(range(len(shape)-1))
                new_perm.insert(d, len(new_perm))
                new_perm.insert(d+1, len(new_perm))
                res[tuple(s)] = torch.diag_embed(b.block.permute(perm)).permute(new_perm)
        elif b.block_type == 'K':
            new_px = (b.ix + 2*b.px - b.kx) % b.sx
            new_py = (b.iy + 2*b.py - b.ky) % b.sy
            curr_size = b.num_kernels*b.ox*b.oy
            eye = torch.eye(curr_size).unsqueeze(0).reshape(curr_size, b.num_kernels, b.ox, b.oy)
            res[tuple(s)] = F.conv_transpose2d(eye, b.block.float(), stride=(b.sx, b.sy), padding=(b.px, b.py), output_padding=(new_px, new_py)).reshape(1, curr_size, -1)
        elif b.block_type == 'P':
            batch_size = b.total_shape[0]
            output_channel, output_x, output_y = b.num_kernels, b.ox, b.oy
            input_channel, kernel_x, kernel_y = b.num_channels, b.kx, b.ky
            input_x, input_y = b.ix, b.iy
            padding = (b.py, b.py, b.px, b.px)
            stride = b.sx
            pieces = b.block.view(-1, output_channel, output_x, output_y, input_channel, kernel_x, kernel_y)
            if pieces.shape[0] < batch_size:
                pieces = pieces.expand(batch_size, *pieces.shape[1:])
            A_matrix = torch.zeros(batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
            orig_stride = A_matrix.stride()
            matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1])
            first_indices = torch.arange(output_x * output_y, device=pieces.device)
            second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
            third_indices = torch.fmod(first_indices, output_y)
            matrix_strided[:,:,second_indices,third_indices,second_indices,third_indices,:,:,:] = pieces.reshape(*pieces.shape[:2], -1, *pieces.shape[4:])
            A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
            A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]
            A_matrix = A_matrix.reshape(A_matrix.shape[0], A_matrix.shape[1], -1)
            if len(A_matrix.shape)!=len(b.total_shape):
                if (torch.tensor(A_matrix.shape) == b.total_shape[:-1]).all():
                    A_matrix = A_matrix.unsqueeze(-1).expand(*b.total_shape)
                else:
                    diffdim = -1
                    for i in range(len(A_matrix.shape)):
                        if(diffdim == -1 and b.total_shape[i] != A_matrix.shape[i]):
                                diffdim = i
                        if diffdim != -1 and b.total_shape[i+1] != A_matrix.shape[i]:
                            raise NotImplementedError(f'PatchesBlock get_dense: {A_matrix.shape} != {b.total_shape[:-1]}')
                    A_matrix = A_matrix.unsqueeze(diffdim).expand(*b.total_shape)
            res[tuple(s)] = A_matrix
        else:
            raise NotImplementedError(f'get_dense: unknown block_type {b.block_type}')
    return res


class Flow:
    def __init__(self, abs_elem: Abs_elem_sparse, transformer, model: Network, print_intermediate_results=False, no_sparsity=False):
        self.abs_elem = abs_elem 
        # type of self.transformer: generated abstract transformer.
        self.transformer = transformer 
        self.model = model
        self.input_size = model.input_size
        self.batch_size = abs_elem.batch_size
        self.print_intermediate_results = print_intermediate_results
        self.no_sparsity = no_sparsity
        self.logfile = None

    def _ensure_logfile(self):
        if self.logfile is None:
            self.logfile = open("flow_log_1.txt", "w")

    def flow(self):
        begin_time = time.time()
        if dummy_mode:
            G.capture_layers = tuple((int(l.identifier), int(l.start), int(l.end)) for l in self.model)
            entry_capture.save_entry_capture(self.abs_elem)
        prev_size = self.model.input_size
        size = self.model.input_size

        json_obj = {op.lower(): [] for op in JIT_OPS}
        json_obj.setdefault('affine_skip', [])
        # Per-layer scalars the generated flow() bakes in as literals (jit builds).
        flow_obj = []

        affine_total = 0  # DEBUG
        affine_skip_count = 0  # DEBUG

        for tmp, layer in enumerate(self.model):
            t_time = time.time()
            poly_size = self.model[self.abs_elem.live_layers[-1]].end
            curr_size = self.model[tmp].end-size

            if layer.type == LayerType.ReLU:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Relu(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)

            elif layer.type == LayerType.Sigmoid:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Sigmoid(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
            #FEAT: Tanh layers dispatch like Sigmoid.
            elif layer.type == LayerType.Tanh:
                prev = Llist(self.model, [1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Tanh(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
            elif layer.type == LayerType.Linear or layer.type == LayerType.Conv2D:
                prev = Llist(self.model, [1, 1], None, None, layer.parents)
                curr = Llist(self.model, [1], None, None, [tmp])
                # Affine_skip is unconditionally injected by single_bound.py's
                # inject_affine_skip, so every compiled transformer has it; the
                # hasattr guard just protects against a stale output/ directory
                # compiled before this optimization existed.
                affine_op = 'Affine'
                if (not layer.feeds_nonlin) and hasattr(self.transformer, 'Affine_skip'):
                    affine_op = 'Affine_skip'
                affine_total += 1  # DEBUG
                if affine_op == 'Affine_skip':
                    affine_skip_count += 1  # DEBUG
                abs_shape = getattr(self.transformer, affine_op)(self.abs_elem, prev, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)

            elif layer.type == LayerType.Input:
                continue
            elif layer.type == LayerType.Add:
                assert len(layer.parents) == 2, 'LayerType.Add always has exactly 2 parents (parse.py)'
                prev1 = Llist(self.model, [1], None, None, [layer.parents[0]])
                prev2 = Llist(self.model, [1], None, None, [layer.parents[1]])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Add(self.abs_elem, prev1, prev2, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
            #FEAT: product of two network values, dispatched like Add to the program's Neuron_mult rule.
            elif layer.type == LayerType.Mul:
                assert len(layer.parents) == 2, 'LayerType.Mul always has exactly 2 parents (parse.py)'
                prev1 = Llist(self.model, [1], None, None, [layer.parents[0]])
                prev2 = Llist(self.model, [1], None, None, [layer.parents[1]])
                curr = Llist(self.model, [1], None, None, [tmp])
                if not hasattr(self.transformer, 'Mult'):
                    raise NotImplementedError('the network multiplies two values, but the program has no Neuron_mult rule')
                abs_shape = self.transformer.Mult(self.abs_elem, prev1, prev2, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
            elif layer.type == LayerType.Concat:
                assert len(layer.parents) == 2, 'LayerType.Concat always has exactly 2 parents (parse.py)'
                prev1 = Llist(self.model, [1], None, None, [layer.parents[0]])
                prev2 = Llist(self.model, [1], None, None, [layer.parents[1]])
                curr = Llist(self.model, [1], None, None, [tmp])
                abs_shape = self.transformer.Concat(self.abs_elem, prev1, prev2, curr, poly_size, curr_size, prev_size, self.input_size, self.batch_size, layer_index = tmp)
            else:
                raise NotImplementedError(f'Flow.flow(): unsupported layer type {layer.type}')
            op_key = affine_op.lower() if layer.type in (LayerType.Linear, LayerType.Conv2D) else JIT_OP_FOR_LAYER[layer.type].lower()
            json_obj[op_key].append(tmp)
            flow_obj.append({'layer': tmp, 'op_key': op_key, 'poly_size': int(poly_size),
                             'curr_size': int(curr_size), 'prev_size': int(prev_size),
                             'input_size': int(self.input_size)})
            size += curr_size
            prev_size = self.model[tmp].size
            self.abs_elem.update(curr, abs_shape, layer_index=tmp)
            # print(f"abs_elem: {tmp}")
            # print(f"LList")
            # print(self.abs_elem.d['llist'])
            # print("\n")
            # print(f"l")
            # print(self.abs_elem.d['l'])
            # print("\n")
            # print(f"u")
            # print(self.abs_elem.d['u'])
            # print("\n")
            # print(f"L")
            # print(self.abs_elem.d['L'])
            # print("\n")
            # print(f"U")
            # print(self.abs_elem.d['U'])


            if self.print_intermediate_results:
                self._ensure_logfile()
                self.logfile.write(f"Layer {tmp+1}: {layer.type}\n")
                print(tmp+1, layer.type, layer.shape)
                print(time.time()-t_time)
                print('---------------------------')
                lb = get_dense_inlined(abs_shape[0])
                ub = get_dense_inlined(abs_shape[1])
                self.logfile.write(f'l: {lb}\n')
                self.logfile.write(f'u: {ub}\n')
                print(f'l: {lb}')
                print(f'u: {ub}')
                # if len(abs_shape) > 3:
                #     L = (abs_shape[2].mat)
                #     U = (abs_shape[3].mat)
                #     print(f'L: {L}')
                #     print(f'U: {U}')
                # elif len(abs_shape) > 2 and hasattr(abs_shape[2], 'mat'):
                #     print(f'Z: {abs_shape[2].mat}')
        lb = get_dense_inlined(abs_shape[0])
        ub = get_dense_inlined(abs_shape[1])

        if affine_skip_count:  # DEBUG
            global _affine_skip_total
            _affine_skip_total += affine_skip_count
            print(f"[single_bound] skipped {affine_skip_count}/{affine_total} affine layers this flow() call "
                  f"(total so far: {_affine_skip_total})")

        if dummy_mode:
            save_capture("jit_layers/layers.json", json_obj)
            if fused_flow:
                save_capture("jit_flow/flow.json", flow_obj)


        return lb, ub
