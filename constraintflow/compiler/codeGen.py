import builtins
import io
import json
import operator
import os
import re

from . import irVisitor
import copy
from .ir import *
from constraintflow.lib.globals import dummy_mode, reuse_mode, load_capture, capture_exists, inductor_mode, fused_flow, sroa, compile_flow
from constraintflow.compiler.optimizations import flow_split
from constraintflow.compiler.optimizations import flow_shapes
from constraintflow.compiler.optimizations import subexp_inlining
from constraintflow.compiler.optimizations import early_reductions as early_reductions_pass
from constraintflow.lib.globals import early_reductions, flow_segment_mb, memory_budget_mb, MEMORY_BUDGET_FRACTION, device_mode, cudagraphs, product_grids, training
from constraintflow.compiler.optimizations import conv_partials, pad_inputs, region_reuse, mem_plan
from constraintflow.compiler.optimizations import product_grids as product_grids_pass
#FEAT: per-change toggles for the DeepPoly-efficient passes
from constraintflow.gbcsr.sparse_block import RUNTIME_HELPERS, TRAINING_HELPERS
from constraintflow.lib import globals as globals_mod

_dense_weight_cache = {}


#FIX FOR: see visitIrFConvTranspose2d; the weights come from parsing the network once on the CPU.
def _conv_weight_taps(network_path):
    """flow() parameter names (net_<i>_weight) of the network's 4-D weights -> the taps a channel-selection
    weight uses, or None for a dense weight."""
    if network_path is None:
        return {}
    if network_path not in _dense_weight_cache:
        import onnx
        import torch
        from constraintflow.lib.parse import parse_onnx_layers
        from constraintflow.gbcsr.sparse_block import _channel_selection
        layers = parse_onnx_layers(onnx.load(network_path), torch.zeros(1, 1), torch.zeros(1), no_sparsity=True)
        taps = {}
        for i, layer in enumerate(layers):
            if (isinstance(getattr(layer, 'weight', None), torch.Tensor) and layer.weight.dim() == 4
                    and layer.weight.is_floating_point()):
                sel = _channel_selection(layer.weight.cpu())
                taps[f'net_{i}_weight'] = None if sel is None else sel[2]
        _dense_weight_cache[network_path] = taps
    return _dense_weight_cache[network_path]


def _dense_conv_weights(network_path):
    """flow() parameter names (net_<i>_weight) of the network's 4-D weights that select no channels."""
    return {name for name, taps in _conv_weight_taps(network_path).items() if taps is None}


_runtime_checked = set()


#FIX FOR: product grids wrap weights as torch.cat([net_1_weight], dim=1), which the name match missed.
def _is_dense_weight(ir, dense):
    """A dense network weight, or a concatenation with one (it keeps that weight's dense columns)."""
    if isinstance(ir, IrVar):
        return ir.name in dense
    return isinstance(ir, IrTorchCat) and any(isinstance(c, IrVar) and c.name in dense for c in ir.children)


#FEAT: parameters of the generated build() and run().
RUN_PARAMS = ("network_file, batch_size, eps, dataset_X, dataset_y, dataset, train, print_intermediate_results, "
              "no_sparsity, initializers=None, input_box=None")

def fused_build():
    """True only on the reuse pass of a jit build, where flow() is emitted."""
    return reuse_mode.get_flag() and fused_flow.get_flag()


def sroa_build():
    """True only on the reuse pass of a jit build (SROA on)."""
    return fused_build() and sroa.get_flag()


def compile_decorator(fullgraph):
    options = {}
    if cudagraphs.get_flag():
        options['triton.cudagraphs'] = True  # what mode="reduce-overhead" sets
    # Inductor would move conv operands to channels-last, where cuDNN picks much
    # slower kernels for these batch-heavy, few-channel convolutions.
    options['layout_optimization'] = False
    passes = []  # graph-pass objects, emitted as code (the generated module imports them), not repr'd literals
    #FEAT: training builds CSE the Dynamo graph before AOTAutograd (see lib/pre_grad_cse.py).
    if training.get_flag():
        #FIX FOR: autotuning a pointwise kernel clones the buffers it writes in place; on the first backward of
        # large CROWN programs that transient ran out of memory (N32 batch 16). Heuristic configs ran as fast.
        options['triton.autotune_pointwise'] = False
        passes.append("'pre_grad_custom_pass': PRE_GRAD_CSE")
    # merge matmuls sharing a left operand (optimizations/shared_lhs_mm.py)
    passes.append("'post_grad_custom_post_pass': SharedLhsMM()")
    opts = ''
    if options or passes:
        opts = ', options={' + ', '.join([repr(k) + ': ' + repr(v) for k, v in options.items()] + passes) + '}'
    return '@torch.compile(fullgraph=' + str(fullgraph) + ', backend="inductor"' + opts + ')'


# Matches a torch.tensor(...) call whose contents are a pure numeric literal
# (no identifiers), e.g. torch.tensor([1, 6272], dtype=torch.int64) or
# torch.tensor([1.0, 1.0]). Anything referencing a runtime variable -- a layer
# size, a poly_size -- fails this match and is left inline, since only a pure
# literal is safe to dedupe into a module-level constant shared across sites.
_LITERAL_TENSOR_RE = re.compile(r'^torch\.tensor\(\[[0-9,\.\-\s\[\]]*\](?:, dtype=torch\.\w+)?\)$')

# One prefix per payload kind: tier class is prefix+'Sparse', leaf is prefix+kind.
SPARSE_TIER = {'FloatTensorSparse': 'FloatTensor', 'BoolTensorSparse': 'BoolTensor',
               'BoolScalarSparse': 'BoolScalar', 'ScalarSparse': 'FloatScalar'}

# The arm of the real SparseBlock.__init__ that each tier fixes statically.
SPARSE_PAYLOAD_INIT = {
    'FloatTensorSparse': 'block.to(device_mode.get_device()).type(torch.float)',
    'BoolTensorSparse': 'block.to(device_mode.get_device())',
    'BoolScalarSparse': 'block',
    'ScalarSparse': 'float(block)',
}

SPARSE_TENSOR_PAYLOAD = {'FloatTensorSparse': True, 'BoolTensorSparse': True,
                         'BoolScalarSparse': False, 'ScalarSparse': False}

BLOCK_FIELDS = {
    'DenseBlock': ('D', ('batch_size',)),
    'ConstBlock': ('C', ()),
    'RepeatBlock': ('R', ('repeat_dims', 'only_one_repeat')),
    'DiagonalBlock': ('Diag', ('diag_index', 'batch_size')),
    'KernelBlock': ('K', ('ix', 'iy', 'ox', 'oy', 'sx', 'sy', 'px', 'py',
                          'kx', 'ky', 'num_channels', 'num_kernels')),
    'PatchesBlock': ('P', ('ix', 'iy', 'ox', 'oy', 'sx', 'sy', 'px', 'py',
                           'kx', 'ky', 'num_channels', 'num_kernels')),
}

_BATCH_SIZE_RE = re.compile(r'\bbatch_size\b')


class CodeGen(irVisitor.IRVisitor):
    def __init__(self,folder):
        self.folder = folder 
        if self.folder.endswith('/'):
            self.folder = self.folder[:-1]
        self.main_file = self.folder + '/main.py'
        self.transformers_file = self.folder + '/transformers.py'
        self.shape = None
        open(self.main_file, "w").close()
        open(self.transformers_file, "w").close()

        self.file = open(self.main_file, "a")
        self.indent = 0

        self.current_layer_index = None

        self.write("import sys")
        self.write("import os")
        self.write("from constraintflow.lib.spec import *")
        self.write("from constraintflow.lib.flow_sparse import Flow" +
                   (", get_dense_inlined" if fused_flow.get_flag() else ""))
        self.write("from constraintflow.lib.abs_elem import Abs_elem_sparse")
        self.write("from constraintflow.lib.symexp import *")
        self.write("from constraintflow.lib.globals import save_capture, inductor_mode")
        self.write_sibling_import('transformers')
        #FEAT: --compile-flow wraps the plain build's Flow.flow in torch.compile with the jit build's
        # Inductor options; transformers.py is unchanged.
        if compile_flow.get_flag() and not reuse_mode.get_flag():
            self.write("import torch")
            self.write("from constraintflow.compiler.optimizations.shared_lhs_mm import SharedLhsMM")
            self.write("Flow.flow = torch.compile(Flow.flow, " + compile_decorator(False)[len('@torch.compile('):])
        self.write("\n")
        # self.write("torch.cuda.reset_peak_memory_stats()")
        #FEAT: input_box=(x_L, x_U) lets a caller (certified training) supply the input box directly.
        #FEAT: build() makes the network and abstract element and run() calls it, so the training
        # fast path (constraintflow/lib/training.py) can build once and call flow() directly.
        self.write("def build(" + RUN_PARAMS + "):")
        
        self.indent += 1
        self.visited = set()

        self.counter = 0

        self.used_block_classes = set()

        # Reuse mode inlines its own class; normal mode imports the real one.
        self.polyexp_cls = 'JitPolyExpSparse' if reuse_mode.get_flag() else 'PolyExpSparse'

        # literal torch.tensor(...) text -> hoisted module-level name (_K<n>).
        # Populated by _const, flushed to jit_constants.py by finish().
        self.const_pool = {}

        # Numbers the temporaries _hoist_if_complex introduces.
        self._rebind_tmp_counter = 0

        # Module-level @torch.compiler.disable helpers, one per in-place write
        # into a torch.as_strided(...) view (PatchesBlock/KernelBlock's Conv2D
        # densification). Correct in eager mode, but Inductor silently drops
        # some of these writes when compiled inline, undercounting DeepZ's
        # generator terms and producing unsound bounds. Must be module-level
        # (not a nested closure) because Dynamo refuses to trace defining and
        # decorating a function inline, so the view/value are passed in as
        # plain arguments instead of closed over.
        self.view_write_helpers = []
        # Set while generating a method that used a view-write helper above.
        # torch.compiler.disable forces a graph break, which fullgraph=True
        # treats as a hard error, so that method must compile with
        # fullgraph=False instead.
        self._method_has_view_write = False
        # Sticky across all methods: decides flow()'s fullgraph in a jit build.
        self._any_view_write = False
        # (op, layer_index) in emission order, for the generated flow().
        self.layer_calls = []

    # Llist params the fused flow passes as None; a body that reads one would be miscompiled.
    _LLIST_PARAMS = ('prev', 'curr', 'prev1', 'prev2')

    def _check_no_llist_params(self, opStmtIr, layer_index, body_text):
        """Reject a specialized body that still reads a Llist parameter."""
        body = body_text.split('\n', 1)[1] if '\n' in body_text else ''
        for name in self._LLIST_PARAMS:
            if name in opStmtIr.params and re.search(r'\b' + name + r'\b', body):
                raise RuntimeError(
                    f"fused flow: {opStmtIr.op}_{layer_index} still reads the Llist parameter "
                    f"{name!r}, which flow() does not reconstruct")

    def emit_flow(self, node):
        """Emit the layer-unrolled flow() that replaces the interpretive Flow.flow."""
        certifier = next(n.transformer for n in node.irNodes if isinstance(n, IrFlow))
        layers = {row['layer']: row for row in load_capture('jit_flow/flow.json')}
        calls = sorted(self.layer_calls, key=lambda c: c[0])
        if sorted(layers) != [c[0] for c in calls]:
            raise RuntimeError(
                f"fused flow: flow.json covers layers {sorted(layers)} but the specialized "
                f"methods cover {[c[0] for c in calls]}")
        fields = list(self.shape.keys())
        state = ', '.join('d_' + key for key in fields)
        # Each method returns its abstract shape followed by the new threaded state.
        shape_out = ', '.join('s_' + key for key in fields)

        self.indent = 0
        self.write('')
        self.write('_T = ' + certifier + '()')
        if inductor_mode.get_flag():
            self.write(compile_decorator(not self._any_view_write))
        self.write('def flow(abs_elem, batch_size):')
        self.indent += 1
        self.write(state + ' = ' + ', '.join("abs_elem.d['" + key + "']" for key in fields))
        for layer_index, op, params in calls:
            row = layers[layer_index]
            args = []
            for name in params:
                if name in self._LLIST_PARAMS:
                    args.append('None')
                elif name in ('poly_size', 'curr_size', 'prev_size', 'input_size'):
                    args.append(str(row[name]))
                else:
                    args.append(name)
            self.write(shape_out + ', ' + state + ' = _T.' + op + '_' + str(layer_index)
                       + '(' + ', '.join(args) + ', layer_index = ' + str(layer_index) + ')')
        self.write('return ' + shape_out)
        self.indent -= 1
        self.write('')

    def _render_stmts(self, stmts, trailer=None):
        """Render statements into their own buffer so the caller can wrap them."""
        real_file = self.file
        self.file = io.StringIO()
        self.indent = 1
        for stmt in stmts:
            self.visit(stmt)
        if trailer is not None:
            self.write(trailer)
        body_text = self.file.getvalue()
        self.file = real_file
        return body_text

    def _emit_explode_inputs(self, params):
        self.indent = 0
        self.write('')
        # No leading underscore: `from transformers import *` drops those.
        self.write('def explode_inputs(abs_elem, batch_size):')
        self.indent += 1
        self.write('return (' + ', '.join(path for _, path in params) + ',)')
        self.indent -= 1
        self.write('')

    def emit_sroa_flow_segmented(self, node, segments):
        """Emit the scalarized flow as chained per-segment functions.

        Under fullgraph a decorator on the driver inlines the segments back into
        one graph. Compile each segment instead to preserve the boundaries.
        """
        self._method_has_view_write = False
        bodies = []
        for seg in segments:
            trailer = None if seg is segments[-1] else (
                'return (' + ''.join(n + ', ' for n in seg.live_out) + ')')
            text = self._render_stmts(seg.stmts, trailer)
            names = list(seg.live_in)
            if _BATCH_SIZE_RE.search(text) and 'batch_size' not in names:
                names.append('batch_size')
            bodies.append((seg, names, text))

        params = list(node.flow_params)
        if any('batch_size' in names for _, names, _ in bodies):
            params.append(('batch_size', 'batch_size'))
        self._emit_explode_inputs(params)

        decorator = compile_decorator(not self._method_has_view_write)
        sized = getattr(self, '_flow_sizes', {})
        shapes = dict(getattr(sized, 'inputs', {}))
        shapes.update({v['name']: v for v in sized.values() if 'shape' in v})
        reused, distinct = {}, 0
        for seg, names, text in bodies:
            signature = region_reuse.key(text, names, shapes)
            self.indent = 0
            if signature is not None and signature in reused:
                self.write(seg.name + ' = ' + reused[signature])
                continue
            if signature is not None:
                reused[signature] = seg.name
            distinct += 1
            if inductor_mode.get_flag():
                self.write(decorator)
            self.write('def ' + seg.name + '(' + ', '.join(names) + '):')
            self.file.write(text)
            self.write('')

        self.indent = 0
        print('[region-reuse] {} unique / {} emitted regions'.format(distinct, len(bodies)))
        self._region_count = distinct
        self.write('def flow(' + ', '.join(name for name, _ in params) + '):')
        self.indent += 1
        local_names = {name for name, _ in params}
        local_names.update(n for seg, _, _ in bodies for n in seg.live_out)
        last_call = {name: i for i, (_, names, _) in enumerate(bodies)
                     for name in names if name in local_names}
        for seg, names, _ in bodies:
            call = seg.name + '(' + ', '.join(names) + ')'
            if seg is segments[-1]:
                self.write('return ' + call)
            else:
                if seg.live_out:
                    self.write(', '.join(seg.live_out) + ', = ' + call)
                else:
                    self.write(call)
                dead = sorted(n for n in names if last_call.get(n) == seg.index)
                if dead:
                    self.write('del ' + ', '.join(dead))
        self.indent -= 1
        self.write('')

    #FEAT: record_temporaries for lazy temporaries
    def _probe_sizes(self, node, *, record_temporaries=False):
        """Fake-execute the block for real byte sizes; None when unavailable."""
        meta = getattr(node, 'flow_param_meta', None)
        if not meta:
            return None
        # split deep expressions into temporaries
        if getattr(node, 'flow_functional', False):
            from constraintflow.compiler.optimizations import expression_limits
            split = expression_limits.run(node.flow_block, layers=getattr(node, 'flow_layers', None))
            if split:
                print('[expression-limits] {} temporaries expose deep expressions'.format(split))
        stmts = node.flow_block.children
        helpers = len(self.view_write_helpers)
        had_write = self._method_has_view_write
        sources = [self._render_stmts([s]).strip() for s in stmts]
        prelude = self.view_write_helpers[helpers:]
        sized = flow_shapes.probe(stmts, sources, meta,
                                  getattr(node, 'flow_batch_size', 1), self.const_pool,
                                  #FEAT: lazy temporaries
                                  prelude, record_temporaries=record_temporaries)
        del self.view_write_helpers[helpers:]   # emission re-renders them
        self._method_has_view_write = had_write
        if sized:
            print('[flow-sizes] ' + flow_shapes.summary(stmts, sized))
        return sized

    def _memory_budget(self):
        """Budget in bytes for mem_plan, or 0 when disabled."""
        mb = memory_budget_mb.get_value()
        if mb >= 0:
            return int(mb * 1024 ** 2)
        import torch
        if device_mode.get_device() != 'cuda' or not torch.cuda.is_available():
            return 0
        return int(torch.cuda.get_device_properties(0).total_memory * MEMORY_BUDGET_FRACTION)

    def emit_sroa_flow(self, node):
        """Emit explode_inputs() and the scalarized flow()."""
        from constraintflow.lib.globals import center_radius
        # rerun center/radius fusion on the final flow
        if center_radius.get_flag() and getattr(node, 'flow_functional', False):
            from constraintflow.compiler.optimizations import box_concretize
            count = box_concretize.run(node.flow_block)
            print('[box-reductions] {} sign-split reductions fused'.format(count))
        sized = self._probe_sizes(node)
        # factor row-scaled patches; conv_chunks and conv_sum clean up after it
        # and only run when it factored something
        if sized and getattr(node, 'flow_functional', False):
            from constraintflow.compiler.optimizations import patch_factor, conv_sum, conv_chunks
            count = patch_factor.run(node.flow_block, self.visit, sized,
                                     getattr(node, 'flow_batch_size', 1))
            print('[patch-factor] {} patch compositions factored'.format(count))
            if count:
                chunks = conv_chunks.run(node.flow_block, self.visit, sized,
                                         getattr(node, 'flow_batch_size', 1))
                print('[conv-chunks] {} output-channel chunks recombined'.format(chunks))
                merged = conv_sum.run(node.flow_block)
                print('[conv-sum] {} shared-weight convolutions combined'.format(merged))
                sized = self._probe_sizes(node)
        # rewrite the einsum + F.fold compositions patch_factor left behind into a sum of padded slices
        # (a gather instead of col2im); CPU builds only, until it is measured on the GPU
        if sized and getattr(node, 'flow_functional', False) and device_mode.get_device() == 'cpu':
            from constraintflow.compiler.optimizations import fold_padsum
            count = fold_padsum.run(node.flow_block)
            print('[fold-padsum] {} einsum + fold compositions rewritten'.format(count))
            if count:
                sized = self._probe_sizes(node)
        if product_grids.get_flag() and sized and getattr(node, 'flow_functional', False):
            count = product_grids_pass.run(node.flow_block, self.visit, sized,
                                         getattr(node, 'flow_batch_size', 1))
            print('[product-grids] {} convolutions and mat-vecs evaluated as grids'.format(count))
            if count:
                sized = self._probe_sizes(node)
        if early_reductions.get_flag():
            count = early_reductions_pass.run(node.flow_block, self.visit)
            print('[early-reductions] {} reductions scheduled after their inputs'.format(count))
            sized = self._probe_sizes(node)
        # Expose partial sums explicitly after reduction scheduling, independently
        # of whether the emitted flow will be segmented. Requires functional SSA.
        if getattr(node, 'flow_functional', False):
            count = conv_partials.run(node.flow_block)
            print('[conv-partials] {} partial results exposed'.format(count))
            sized = self._probe_sizes(node)
        barriers = set()
        if inductor_mode.get_flag() and getattr(node, 'flow_functional', False):
            pads = set(pad_inputs.run(node.flow_block))
            barriers.update(pads)
            print('[pad-inputs] {} pad inputs materialized'.format(len(pads)))
            sized = self._probe_sizes(node)
        # inline storage-sharing views before memory planning
        if sized and getattr(node, 'flow_functional', False):
            from constraintflow.compiler.optimizations import view_aliases
            count = view_aliases.run(node.flow_block, sized)
            print('[view-aliases] {} storage-sharing aliases inlined'.format(count))
            if count:
                sized = self._probe_sizes(node)
        # lower pointwise-produced matvecs to row reductions
        if sized and getattr(node, 'flow_functional', False):
            from constraintflow.compiler.optimizations import fused_matvec
            count = fused_matvec.run(node.flow_block, self.visit, sized,
                                     getattr(node, 'flow_batch_size', 1))
            print('[fused-matvec] {} pointwise matvecs lowered to row reductions'.format(count))
            if count:
                sized = self._probe_sizes(node)
        #FEAT: lazy temporaries
        # Transformations only need named shapes. Planning must also account
        # for unnamed operands materialized inside each expression.
        if sized and not getattr(sized, 'temporaries_profiled', True):
            sized = self._probe_sizes(node, record_temporaries=True)
        budget = self._memory_budget()
        if budget and sized:
            log = mem_plan.run(node.flow_block, sized, budget)
            if log is not None:
                print('[mem-plan] ' + log)
                #FEAT: keep the temporaries profile after planning
                sized = self._probe_sizes(node, record_temporaries=True)
        self._flow_sizes = sized or {}
        self._flow_metrics = {
            'statements': len(node.flow_block.children),
            'estimated_peak_tensor_bytes': flow_shapes.peak_live_bytes(node.flow_block.children, sized) if sized else None,
            'workspace_included': False,
        }
        # Row selection consumes final SSA, after memory planning and before
        # segmentation can hide dependencies behind emitted region calls.
        from constraintflow.compiler.optimizations import conditional_rows
        if conditional_rows.emit(self, node):
            return
        if flow_segment_mb.get_value() > 0:
            segments = flow_split.split(
                node.flow_block, sized,
                max_region_bytes=flow_segment_mb.get_value() * 1024 ** 2,
                barriers=barriers)
            if segments is not None:
                print('[flow-split] ' + str(len(segments)) + ' segments: '
                      + flow_split.describe(segments)
                      + (' (compiled per segment)' if inductor_mode.get_flag() else ' (eager)'))
                return self.emit_sroa_flow_segmented(node, segments)
            print('[flow-split] not split, emitting one flow()')
        real_file = self.file
        self.file = io.StringIO()
        self.indent = 1
        self._method_has_view_write = False
        self.visit(node.flow_block)
        body_text = self.file.getvalue()
        self.file = real_file
        params = list(node.flow_params)
        if _BATCH_SIZE_RE.search(body_text):
            # shapes that stayed symbolic render it as raw text in a traced
            # total_size, so the body is the only place it can be spotted
            params.append(('batch_size', 'batch_size'))
        self.indent = 0
        self.write('')
        # No leading underscore: `from transformers import *` drops those.
        self.write('def explode_inputs(abs_elem, batch_size):')
        self.indent += 1
        self.write('return (' + ', '.join(path for _, path in params) + ',)')
        self.indent -= 1
        self.write('')
        if inductor_mode.get_flag():
            self.write(compile_decorator(not self._method_has_view_write))
        self.write('def flow(' + ', '.join(name for name, _ in params) + '):')
        self.file.write(body_text)
        self.write('')

    def _const(self, text):
        """Intern a literal torch.tensor(...) expression into the shared constant
        pool (reuse mode only) and return the reference to use in its place.
        Non-literal text -- and anything outside reuse mode -- passes through
        unchanged, so this is safe to call on every torch.tensor(...) site."""
        if not reuse_mode.get_flag() or not _LITERAL_TENSOR_RE.match(text):
            return text
        name = self.const_pool.get(text)
        if name is None:
            # No leading underscore: `from jit_constants import *` (in
            # transformers.py's header) silently drops underscore-prefixed
            # names under Python's wildcard-import rules.
            name = 'JITCONST' + str(len(self.const_pool))
            self.const_pool[text] = name
        return name

    def finish(self):
        """Flush the hoisted constant pool to <folder>/jit_constants.py. Called
        once after visit(ir) completes, since the pool is only complete then."""
        if hasattr(self, '_flow_metrics'):
            metrics = dict(self._flow_metrics, unique_regions=getattr(self, '_region_count', 1))
            if not self.file.closed:
                self.file.flush()
            metrics['generated_source_bytes'] = sum(os.path.getsize(p) for p in (self.main_file, self.transformers_file))
            with open(os.path.join(self.folder, 'compile_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        if hasattr(self, '_stable_dense_source'):
            # tell() records byte offsets, including any non-ASCII prelude text.
            with open(self.transformers_file, 'rb') as f:
                source = f.read()
            start, end, dense = self._stable_dense_source
            with open(os.path.join(self.folder, 'transformers_dense.py'), 'wb') as f:
                f.write(source[:start] + dense.encode('utf-8') + source[end:])
        if not reuse_mode.get_flag():
            return
        if training.get_flag():
            open(os.path.join(self.folder, '__init__.py'), 'w').close()
        path = self.folder + '/jit_constants.py'
        with open(path, 'w') as f:
            f.write('import torch\n\n')
            for text, name in self.const_pool.items():
                f.write(name + ' = ' + text + '\n')
        if sroa_build():
            self._check_flow_is_closed()

    def _check_flow_is_closed(self):
        """Every name flow() reads must be a parameter, a local, or a module global.
        A leftover ambient name compiles fine and only fails inside dynamo."""
        import ast
        source = open(self.transformers_file).read()
        tree = ast.parse(source)
        globals_, flow = set(dir(builtins)), None
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                globals_.update((a.asname or a.name).split('.')[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                globals_.add(node.name)
                if node.name == 'flow':
                    flow = node
            elif isinstance(node, ast.Assign):
                globals_.update(t.id for t in ast.walk(node)
                                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store))
        if flow is None:
            return
        bound = {a.arg for a in flow.args.args}
        loads = set()
        for node in ast.walk(flow):
            if isinstance(node, ast.Name):
                (bound if isinstance(node.ctx, ast.Store) else loads).add(node.id)
        free = sorted(loads - bound - globals_)
        if free:
            raise RuntimeError('sroa: flow() reads unbound ' + ', '.join(free)
                               + '; it must be a flow parameter')

    def _hoist_if_complex(self, expr_text):
        """visitIrExpandSymExp/visitIrSetBlockTotalShapeLastDim need to reference
        their base object twice (once to mutate, once to read the mutated
        total_size/total_shape back) and then a third time via their return
        value. When symexpCount/subexp_inlining leaves that base inlined as a
        full expression -- typically a SymExpSparse(...)/JitSparseTensor(...)
        construction, not a bare name -- emitting it verbatim at each of those
        sites reconstructs the whole object 2-3 times per rebind. Bind it to a
        fresh local first so it is only built once; a bare name/attribute
        chain (no call in it) is already cheap to repeat and is returned as-is."""
        if '(' not in expr_text:
            return expr_text
        tmp = 'rebind_tmp_' + str(self._rebind_tmp_counter)
        self._rebind_tmp_counter += 1
        self.write(tmp + ' = ' + expr_text)
        return tmp


    def _ttb_comment(self, node):
        if dummy_mode.get_flag() or reuse_mode.get_flag():
            return ' #' + str(node.ttb_counter)
        return ''

    # This function is used to get the branch information for a tuple (block_id, layer_index) from the jit file.
    def get_profiled_branch(self, block_id):
        if not reuse_mode.get_flag():
            return None
        if self.current_layer_index is None:
            return None
        rel_path = f"jit_branch/branch_{self.current_layer_index}_{block_id}.json"
        if not capture_exists(rel_path):
            raise Exception(f"Profiled branch data not found for block_id {block_id} at {rel_path}")
        json_obj = load_capture(rel_path)
        taken = json_obj["taken"]
        if taken in ["then", "else"]:
            return taken
        return None


    def write(self, str, flag=True):
        self.file.write('\t'*self.indent + str)
        if flag:
            self.file.write('\n')

    def write_sibling_import(self, module):
        #FEAT: training builds import siblings relatively, so several programs can be loaded as packages.
        if training.get_flag():
            self.write('try:\n    from .' + module + ' import *\nexcept ImportError:\n    from ' + module + ' import *')
        else:
            self.write('from ' + module + ' import *')

    def write_expr(self, str, flag=True):
        self.file.write(str)
        if flag:
            self.file.write('\n')


    def open(self, file):
        self.file.close()
        self.file = open(file, "a")

    def visitIrProgram(self, node):
        self.shape = node.shape
        temp_shape = copy.deepcopy(self.shape)
        temp_shape['llist'] = 'bool'
        temp_dict = "{"
        key = 'llist'
        i = 0

        temp_dict += "\'" + key + "\' : " + key + ', '
        for i, key in enumerate(node.shape.keys()):
            temp_dict += "\'" + key + "\' : " + key 
            if i < len(node.shape.keys())-1:
                temp_dict += ", "
        temp_dict += '}'


        self.write("network, l, u, L, U, Z, llist = get_network_and_input_spec(network_file, batch_size, dataset_X, dataset_y, dataset, eps=eps, train=train, no_sparsity=no_sparsity, initializers=initializers, input_box=input_box, members=" + repr(tuple(node.shape.keys()))
                   + (", fold_spec=True" if training.get_flag() else "") + ")")
        self.write("abs_elem = Abs_elem_sparse(" + temp_dict + ", " + str(temp_shape) + ", network, batch_size=batch_size, no_sparsity=no_sparsity)")
        #FEAT: end build(); run() continues with the flow lines visitIrFlow appends.
        self.write("return network, abs_elem")
        self.indent -= 1
        self.write("def run(" + RUN_PARAMS + "):")
        self.indent += 1
        self.write("network, abs_elem = build(network_file, batch_size, eps, dataset_X, dataset_y, dataset, train, print_intermediate_results, no_sparsity, initializers=initializers, input_box=input_box)")
        



        # GENERATE TRANSFORMERS
        self.open(self.transformers_file)
        self.indent = 0
        self.write('import gc')
        self.write('import json')
        self.write('import os')
        self.write('import torch')
        # self.write("torch.set_float32_matmul_precision('high')")
        self.write('import operator')
        self.write('from constraintflow.lib.globals import device_mode')
        if not reuse_mode.get_flag():
            self.write('from constraintflow.lib.polyexp import PolyExpSparse')
        if reuse_mode.get_flag():
            self.write('from constraintflow.lib.symexp import SymExpSparse')
            #FEAT: training helpers are imported only by training builds, so verification code is unchanged.
            self.write('from constraintflow.gbcsr.sparse_block import ' + ', '.join(
                h for h in RUNTIME_HELPERS if h not in TRAINING_HELPERS or training.get_flag()))
            #FEAT: pre-autograd CSE pass named in training builds' compile decorator.
            if training.get_flag():
                self.write('from constraintflow.lib.pre_grad_cse import PRE_GRAD_CSE')
            self.write('from constraintflow.compiler.optimizations.shared_lhs_mm import SharedLhsMM')
            # self.write('from constraintflow.lib.symexp import get_new_eps')
            # self.write('from constraintflow.gbcsr.op_helper import binary_to_identity_unary')

        else:
            self.write('from constraintflow.lib.symexp import *')
        if reuse_mode.get_flag():
            self.write('import operator')
            self.write('import torch.nn.functional as F')
            self.write_sibling_import('jit_constants')
            # self.write('from constraintflow.gbcsr.tensor_ops import *')
        else:
            self.write('from constraintflow.gbcsr.sparse_tensor import SparseTensor')
        self.write('from constraintflow.lib.llist import Llist')
        if not reuse_mode.get_flag():
            self.write('from constraintflow.gbcsr.tensor_ops import *')
        for i, transformer_name in enumerate(node.tstore.keys()):
            if sroa_build():
                break
            self.write('class ' + transformer_name + ':')
            self.indent += 1

            transformerIr = node.tstore[transformer_name]

            for j, opStmtIr in enumerate(transformerIr):
                param_list = ', '.join(opStmtIr.params)
                # In jit builds the specialized methods take the threaded state too.
                layerwise_list = ', '.join(opStmtIr.layerwise_params or opStmtIr.params)
                if opStmtIr.layerwise_cfgs is None:
                    self.write('def ' + opStmtIr.op + '(self, ' + param_list + ', layer_index = None):')
                    self.indent += 1
                    # self.write('torch.cuda.memory._record_memory_history(')
                    # self.indent += 1
                    # self.write('max_entries=1000000')
                    # self.indent -= 1
                    # self.write(')')

                    cfg = opStmtIr.cfg
                    self.visit(cfg.ir[cfg.entry_node])
                    self.indent -= 1
                    self.write('', True)
                else:
                    self.write('def ' + opStmtIr.op + '(self, ' + param_list + ', layer_index = None):')
                    self.indent += 1
                    # self.write('torch.cuda.memory._record_memory_history(')
                    # self.indent += 1
                    # self.write('max_entries=1000000')
                    # self.indent -= 1
                    # self.write(')')
                    if opStmtIr.layerwise_params is None:
                        for layer_index in opStmtIr.layerwise_cfgs.keys():
                            self.write('if layer_index == ' + str(layer_index) + ':')
                            self.indent += 1
                            self.write('return self.' + opStmtIr.op + '_' + str(layer_index) + '(' + param_list + ', layer_index = layer_index)')
                            self.indent -= 1
                        self.write("raise RuntimeError(f'no specialized kernel for " + opStmtIr.op + " at layer {layer_index}')")
                    else:
                        # Fused builds drive the specialized methods from flow(), not from here.
                        self.write("raise RuntimeError('" + opStmtIr.op + ": this is a jit build; call flow() instead')")
                    self.indent -= 1
                    self.write('')

                    for layer_index in opStmtIr.layerwise_cfgs.keys():
                        self.layer_calls.append((layer_index, opStmtIr.op, opStmtIr.layerwise_params))
                        # Buffer the body first so we know, after visiting it,
                        # whether it used a view-write helper before deciding
                        # fullgraph=True vs False for this method's decorator.
                        self._method_has_view_write = False
                        real_file = self.file
                        self.file = io.StringIO()
                        self.write('def ' + opStmtIr.op + '_' + str(layer_index) + '(self, ' + layerwise_list + ', layer_index = None):')
                        self.indent += 1
                        self.write('while_iteration = -1')
                        cfg = opStmtIr.layerwise_cfgs[layer_index]
                        self.current_layer_index = layer_index
                        self.visit(cfg.ir[cfg.entry_node])
                        self.indent -= 1
                        self.write('', True)
                        body_text = self.file.getvalue()
                        self.file = real_file
                        self._any_view_write |= self._method_has_view_write
                        # Fused builds carry one decorator on flow() instead of one per method.
                        if inductor_mode.get_flag() and not fused_build():
                            fullgraph = not self._method_has_view_write
                            self.write(compile_decorator(fullgraph))
                        if fused_build():
                            self._check_no_llist_params(opStmtIr, layer_index, body_text)
                        self.file.write(body_text)
            self.indent -=1

        if sroa_build():
            self.emit_sroa_flow(node)
        elif fused_build():
            self.emit_flow(node)

        if reuse_mode.get_flag():
            # Defined after use; names resolve when the methods are called.
            self.indent = 0
            for helper_src in self.view_write_helpers:
                self.write(helper_src, False)
                self.write('')
        if reuse_mode.get_flag() and not (sroa_build() and not self.used_block_classes):
            self.indent = 0
            self.write('class JitSparseTensor:')
            self.indent += 1
            self.write('__slots__ = ("start_indices", "blocks", "dims", "total_size", "end_indices", "type", "dense_const", "delete_indices", "num_blocks")')
            self.write('def __init__(self, start_indices, blocks, dims, total_size, end_indices, type, dense_const, delete_indices):')
            self.indent += 1
            for field in ('start_indices', 'blocks', 'dims', 'total_size', 'end_indices', 'type', 'dense_const', 'delete_indices'):
                self.write('self.' + field + ' = ' + field)
            self.write('self.num_blocks = len(start_indices)')
            self.indent -= 1
            # These preserve structure entirely; only payload and dense_const change.
            self.write('def float(self):')
            self.indent += 1
            self.write('return JitSparseTensor(self.start_indices, [b.float() for b in self.blocks], self.dims, self.total_size, self.end_indices, float, float(self.dense_const), self.delete_indices)')
            self.indent -= 1
            self.write('def check_dense(self):')
            self.indent += 1
            self.write('t = 0')
            self.write('for i in range(self.num_blocks):')
            self.indent += 1
            self.write('p = 1')
            self.write('for d in self.blocks[i].total_shape.tolist():')
            self.indent += 1
            self.write('p = p * d')
            self.indent -= 1
            self.write('t = t + p')
            self.indent -= 1
            self.write('q = 1')
            self.write('for d in self.total_size.tolist():')
            self.indent += 1
            self.write('q = q * d')
            self.indent -= 1
            self.write('return not (t < q)')
            self.indent -= 1
            self.write('def increase_size(self, start_index, new_total_size):')
            self.indent += 1
            self.write('res_start_indices = []')
            self.write('res_end_indices = []')
            self.write('res_blocks = []')
            self.write('for i in range(self.num_blocks):')
            self.indent += 1
            self.write('res_start_indices.append(start_index + self.start_indices[i])')
            self.write('res_end_indices.append(start_index + self.end_indices[i])')
            self.write('res_blocks.append(self.blocks[i].copy())')
            self.indent -= 1
            self.write('return JitSparseTensor(res_start_indices, res_blocks, self.dims, new_total_size, res_end_indices, self.type, self.dense_const, self.delete_indices)')
            self.indent -= 2

            self.write('class ' + self.polyexp_cls + ':')
            self.indent += 1
            self.write('__slots__ = ("network", "mat", "const")')
            self.write('def __init__(self, network, mat, const):')
            self.indent += 1
            for field in ('network', 'mat', 'const'):
                self.write('self.' + field + ' = ' + field)
            self.indent -= 2

            self.write('class JitSparseBlock:')
            self.indent += 1
            self.write('__slots__ = ("block", "total_shape")')
            self.indent -= 1

            # float() names another leaf, so both float targets must exist for every kind used.
            leaves = set(self.used_block_classes)
            leaves |= {(kind, 'FloatTensorSparse') for kind, t in self.used_block_classes
                       if SPARSE_TENSOR_PAYLOAD[t]}
            leaves |= {(kind, 'ScalarSparse') for kind, t in self.used_block_classes
                       if not SPARSE_TENSOR_PAYLOAD[t]}

            # Each tier owns one payload representation, so leaves carry only geometry.
            for sparse_block_type in sorted({t for _, t in leaves}):
                self.write('class ' + SPARSE_TIER[sparse_block_type] + 'Sparse(JitSparseBlock):')
                self.indent += 1
                self.write('__slots__ = ()')
                self.write('def __init__(self, block, total_shape):')
                self.indent += 1
                self.write('self.block = ' + SPARSE_PAYLOAD_INIT[sparse_block_type])
                self.write('self.total_shape = total_shape')
                self.indent -= 2

            for kind, sparse_block_type in sorted(leaves):
                block_type, fields = BLOCK_FIELDS[kind]
                tier = SPARSE_TIER[sparse_block_type]
                is_tensor = SPARSE_TENSOR_PAYLOAD[sparse_block_type]
                float_tier = SPARSE_TIER['FloatTensorSparse' if is_tensor else 'ScalarSparse']
                rest = ''.join(', self.' + f for f in fields)
                self.write('class ' + tier + kind + '(' + tier + 'Sparse):')
                self.indent += 1
                self.write('__slots__ = (' + ''.join('"' + f + '", ' for f in fields) + ')')
                self.write("block_type = '" + block_type + "'")
                self.write('def __init__(self, block, total_shape' + ''.join(', ' + f for f in fields) + '):')
                self.indent += 1
                self.write(tier + 'Sparse.__init__(self, block, total_shape)')
                for f in fields:
                    self.write('self.' + f + ' = ' + f)
                self.indent -= 1
                self.write('def float(self):')
                self.indent += 1
                payload = 'self.block.float()' if is_tensor else 'float(self.block)'
                self.write('return ' + float_tier + kind + '(' + payload + ', self.total_shape' + rest + ')')
                self.indent -= 1
                self.write('def copy(self):')
                self.indent += 1
                self.write('return ' + tier + kind + '(' + ('self.block.clone()' if is_tensor else 'self.block') + ', self.total_shape' + rest + ')')
                self.indent -= 2

        self.open(self.main_file)
        for i in range(len(node.irNodes)):
            self.visit(node.irNodes[i])

    def visitIrBlock(self, node):
        if not node in self.visited:
            self.visited.add(node)
            ir_list = node.children
            for counter, i in enumerate(ir_list):
                self.visit(i)
            if node.inner_jump != None:
                if len(node.inner_jump)==3:
                    cond = self.visit(node.inner_jump[0])
                    block_id = node.block_id
                    profiled_branch = self.get_profiled_branch(block_id)
                    if profiled_branch == 'then':
                        self.visit(node.inner_jump[1])
                    elif profiled_branch == 'else':
                        self.visit(node.inner_jump[2])
                    else:
                        self.write('if(' + str(cond) + '):')
                        self.indent += 1
                        if block_id is not None:
                            self.write('if dummy_mode:')
                            self.indent += 1
                            self.write('save_capture("jit_branch/branch_" + str(layer_index) + "_' + str(block_id) + '.json", {"taken": "then"})')
                            self.indent -= 1
                        self.visit(node.inner_jump[1])
                        self.indent -= 1
                        self.write('else:')
                        self.indent += 1
                        if block_id is not None:
                            self.write('if dummy_mode:')
                            self.indent += 1
                            self.write('save_capture("jit_branch/branch_" + str(layer_index) + "_' + str(block_id) + '.json", {"taken": "else"})')
                            self.indent -= 1
                        self.visit(node.inner_jump[2])
                        self.indent -= 1
                elif not isinstance(node.inner_jump[1], IrWhileBlock):
                    cond = self.visit(node.inner_jump[0])
                    self.write('if(' + str(cond) + '):')
                    self.indent += 1
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                else:
                    cond = self.visit(node.inner_jump[0])
                    self.write('while_iteration = -1')
                    self.write('while(' + str(cond) + '):')
                    self.indent += 1
                    self.write('while_iteration += 1')
                    # self.write('print(\'while_iteration\', while_iteration)')
                    self.visit(node.inner_jump[1])
                    self.indent -= 1
                    self.write('if dummy_mode:')
                    self.indent += 1
                    self.write('json_obj = {"num_iterations": while_iteration}')
                    self.write('save_capture("jit_while/while_iterations_layer_" + str(layer_index) + "_while_" + str(' + str(node.inner_jump[1].while_number) + ') + ".json", json_obj)')
                    self.indent -= 1
            if node.jump != None:
                self.visit(node.jump[1])

    def visitIrAssignment(self, node):
        var = str(self.visit(node.children[0]))
        expr = str(self.visit(node.children[1]))
        self.write(var + ' = ' + expr )
        # node.counter = self.counter
        # self.counter += 1

    # For the del statements
    # Currently not used. 
    def visitIrDel(self, node):
        # self.write('del ' + ', '.join(node.var_names))
        # self.write('gc.collect()')
        # self.write('torch.cuda.empty_cache()')
        pass

    def visitIrBreak(self, node):
        self.write('break')

    def visitIrTransRetBasic(self, node):
        # self.write('try:')
        # self.indent += 1
        # self.write('torch.cuda.memory._dump_snapshot(f"memory_usage_{layer_index}.pickle")')
        # self.indent -= 1
        # self.write('except:')
        # self.indent += 1
        # self.write('raise Exception("CUDA memory snapshot failed. This can happen if the file prefix is too long or if there are issues with the CUDA setup. Please check your CUDA configuration and ensure that the file prefix is valid.")')
        # self.indent -= 1
        
        exprs = []
        for i in range(len(node.children)):
            expr = self.visit(node.children[i])
            exprs.append(expr)
        ret_expr = 'return '
        for i in range(len(node.children)-1):
            ret_expr += exprs[i]
            ret_expr += ', '
        ret_expr += exprs[-1] + ', '
        self.write(ret_expr)

    def visitIrTransRetIf(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1
        else:
            self.write('pass')
        self.write('return')

    def visitIrIte(self, node):
        cond = self.visit(node.children[0])
        self.write('if(' + cond + '):')
        self.indent += 1
        if len(node.children[1])>0:
            for i in node.children[1]:
                self.visit(i)
        else:
            self.write('pass')
        self.indent -= 1
        if len(node.children[2])>0:
            self.write('else:')
            self.indent += 1
            for i in node.children[1]:
                self.visit(i)
            self.indent -= 1

    # def visitIrWhile(self, node):
    #     cond = self.visit(node.children[0])
    #     self.write('while(' + str(cond) + '):')
    #     self.indent += 1
    #     for ir in node.children[1:]:
    #         self.visit(ir)
    #     self.indent -= 1
    

    def visitIrStr(self, node):
        return node
    
    def visitIrConst(self, node):
        return str(node.const)

    def visitList(self, node):
        res = '['
        for i, child in enumerate(node):
            if i > 0:
                res += ', '
            res += self.visit(child)
        res += ']'
        return res

    def renderTraceIndex(self, index):
        parts = []
        for item in index:
            if isinstance(item, list):
                if len(item) == 2:
                    parts.append(self.visit(item[0]) + ':' + self.visit(item[1]))
                else:
                    parts.append(self.visit(item))
            elif isinstance(item, str):
                parts.append(item)
            elif isinstance(item, int) and item == 0:
                parts.append(':')
            else:
                parts.append(self.visit(item))
        return ', '.join(parts)

    def visitIrSparseTensor(self, node):
        start_indices = '[' + ', '.join(
            x if isinstance(x, str) else self._const('torch.tensor(' + str(x.tolist()) + ', dtype=torch.int64)')
            for x in node.start_indices) + ']'
        end_indices = '[' + ', '.join(
            x if isinstance(x, str) else self._const('torch.tensor(' + str(x.tolist()) + ', dtype=torch.int64)')
            for x in node.end_indices) + ']'
        total_size = node.total_size if isinstance(node.total_size, str) else self._const('torch.tensor(' + str(node.total_size.tolist()) + ', dtype=torch.int64)')
        # An empty block list is a plain [], not an IR node.
        blocks = '[]' if isinstance(node.children[0], list) else self.visit(node.children[0])
        # repr of these three does not parse back.
        dense_const = repr(node.dense_const)
        if dense_const in ('inf', '-inf', 'nan'):
            dense_const = "float('" + dense_const + "')"

        return ('JitSparseTensor(' + start_indices + ', ' + blocks + ', ' + str(node.dims) + ', '
                + total_size + ', ' + end_indices + ', ' + node.type.__name__ + ', '
                + dense_const + ', ' + str(node.delete_indices) + ')')

    def visitIrConstBlock(self, node):
        self.used_block_classes.add(('ConstBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'ConstBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ')')

    def visitIrPatchesBlock(self, node):
        self.used_block_classes.add(('PatchesBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'PatchesBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ', '
                + str(node.ix) + ', ' + str(node.iy) + ', '
                + str(node.ox) + ', ' + str(node.oy) + ', '
                + str(node.sx) + ', ' + str(node.sy) + ', '
                + str(node.px) + ', ' + str(node.py) + ', '
                + str(node.kx) + ', ' + str(node.ky) + ', '
                + str(node.num_channels) + ', ' + str(node.num_kernels) + ')')

    def visitIrRepeatBlock(self, node):
        self.used_block_classes.add(('RepeatBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'RepeatBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ', '
                + self._const(node.repeat_dims) + ', ' + str(node.only_one_repeat) + ')')

    def visitIrDiagonalBlock(self, node):
        self.used_block_classes.add(('DiagonalBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'DiagonalBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ', '
                + str(node.diag_index) + ', ' + str(node.batch_size) + ')')

    def visitIrTorchDiagonal(self, node):
        input_expr = self.visit(node.children[0])
        return (
            'torch.diagonal(' + input_expr
            + ', dim1=' + str(node.dim1)
            + ', dim2=' + str(node.dim2) + ')'
        )

    def visitIrTorchPermute(self, node):
        input_expr = self.visit(node.children[0])
        perm_args = ', '.join(str(i) for i in node.permutation)
        return input_expr + '.permute(' + perm_args + ')'

    def visitIrTorchTranspose(self, node):
        return self.visit(node.children[0]) + '.transpose(' + str(node.dim0) + ', ' + str(node.dim1) + ')'

    def visitIrTorchMatmul(self, node):
        fn = 'bcast_matmul(' if reuse_mode.get_flag() else 'torch.matmul('
        #FEAT: training builds use the variant whose backward avoids per-sample weight gradients.
        if reuse_mode.get_flag() and training.get_flag():
            fn = 'training_bcast_matmul('
        return fn + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'

    def visitIrTorchUnsqueeze(self, node):
        return self.visit(node.children[0]) + '.unsqueeze(' + str(node.index) + ')'

    def visitIrTorchSqueeze(self, node):
        return self.visit(node.children[0]) + '.squeeze(' + str(node.index) + ')'

    def visitIrTorchReshape(self, node):
        return self.visit(node.children[0]) + '.reshape(' + self.visit(node.shape) + ')'

    def visitIrTorchView(self, node):
        return self.visit(node.children[0]) + '.view(' + self.visit(node.shape) + ')'

    def visitIrTorchRepeat(self, node):
        #FIX FOR: in training, the kernel tiled over patch positions (W.view(1,K,-1).unsqueeze(-1).repeat(1,1,1,P), read
        # permuted) was saved [K,J,P] and read transposed (uncoalesced) by the backward; an expand keeps it in [K,P,J] order.
        u = node.children[0]
        if (training.get_flag() and isinstance(u, IrTorchUnsqueeze) and u.index in (-1, 3)
                and isinstance(u.children[0], (IrTorchView, IrTorchReshape)) and isinstance(u.children[0].shape, (list, tuple))
                and len(u.children[0].shape) == 3 and isinstance(node.repeats, (list, tuple)) and len(node.repeats) == 4 and list(node.repeats[:3]) == [1, 1, 1]):
            return self.visit(u) + '.expand([-1, -1, -1, ' + str(node.repeats[3]) + '])'
        return self.visit(node.children[0]) + '.repeat(' + self.visit(node.repeats) + ')'

    def visitIrTorchExpand(self, node):
        return self.visit(node.children[0]) + '.expand(' + self.visit(node.shape) + ')'

    def visitIrTorchPad(self, node):
        return 'F.pad(' + self.visit(node.children[0]) + ', ' + str(list(node.pad)) + ')'

    def visitIrTorchSum(self, node):
        return self.visit(node.children[0]) + '.sum(dim=' + str(node.dim) + ')'

    def visitIrTorchZeros(self, node):
        kwargs = []
        if node.device is not None:
            kwargs.append('device=' + self.visit(node.device))
        if node.dtype is not None:
            kwargs.append('dtype=' + self.visit(node.dtype))
        args = [self.visit(node.size)] + kwargs
        return 'torch.zeros(' + ', '.join(args) + ')'

    def visitIrTorchEye(self, node):
        kwargs = []
        if node.device is not None:
            kwargs.append('device=' + self.visit(node.device))
        if node.dtype is not None:
            kwargs.append('dtype=' + self.visit(node.dtype))
        args = [self.visit(node.size)] + kwargs
        return 'torch.eye(' + ', '.join(args) + ')'

    def visitIrTorchFloat(self, node):
        return self.visit(node.children[0]) + '.float()'

    def visitIrTorchDiagEmbed(self, node):
        return 'torch.diag_embed(' + self.visit(node.children[0]) + ')'

    def visitIrTorchStride(self, node):
        return self.visit(node.children[0]) + '.stride()'

    def visitIrTorchAsStrided(self, node):
        return (
            'torch.as_strided(' + self.visit(node.children[0]) + ', '
            + self.visit(node.size) + ', ' + self.visit(node.stride) + ')'
        )

    def visitIrPatchesToDense(self, node):
        return ('patches_to_dense(' + self.visit(node.children[0]) + ', '
                + ', '.join(str(v) for v in node.geometry) + ')')

    def visitIrRowGather(self, node):
        index = self.visit(node.children[1])
        if node.divisor != 1:
            index += ' // ' + str(node.divisor)
        if node.period is not None:
            index = '(' + index + ') % ' + str(node.period)
        return '(' + self.visit(node.children[0]) + ').index_select(1, ' + index + ')'

    def visitIrSelectedPatchesToDense(self, node):
        return ('selected_patches_to_dense(' + self.visit(node.children[0]) + ', '
                + self.visit(node.children[1]) + ', '
                + ', '.join(str(x) for x in node.geometry) + ')')

    def visitIrTensorScatter(self, node):
        # out-of-place: no aliasing for dynamo to reason about, so no graph break
        return (self.visit(node.children[0]) + '.scatter(' + str(node.dim) + ', '
                + node.index + ', ' + self.visit(node.children[1]) + ')')

    def visitIrTorchSlice(self, node):
        return self.visit(node.children[0]) + '[' + self.renderTraceIndex(node.index) + ']'

    def visitIrFConv2d(self, node):
        return (
            'F.conv2d(' + self.visit(node.children[0]) + ', '
            + self.visit(node.children[1])
            + (', ' + self.visit(node.children[2]) if len(node.children) > 2 else '')
            + ', stride='
            + self.visit(node.stride) + ', padding=' + self.visit(node.padding) + ')'
        )

    def visitIrFConvTranspose2d(self, node):
        kwargs = []
        if node.stride is not None:
            kwargs.append('stride=' + self.visit(node.stride))
        if node.padding is not None:
            kwargs.append('padding=' + self.visit(node.padding))
        if node.output_padding is not None:
            kwargs.append('output_padding=' + self.visit(node.output_padding))
        weight = self.visit(node.children[1])
        #FIX FOR: conv_transpose2d checked every weight for channel selection on the GPU at run time (a D2H
        # per conv weight per batch); a network weight known to be dense at compile time skips the check.
        if globals_mod.network_path is not None:
            taps = _conv_weight_taps(globals_mod.network_path)
            if _is_dense_weight(node.children[1], _dense_conv_weights(globals_mod.network_path)):
                kwargs.append('may_select=False')
            #FIX FOR: selection weights (MaxPool expansion) were checked on the host at run time (3 D2H per weight
            # per batch); their taps are known at compile time.
            elif isinstance(node.children[1], IrVar) and taps.get(node.children[1].name) is not None:
                kwargs.append('selection_taps=' + repr([i for tap in taps[node.children[1].name] for i in tap]))
            elif weight not in _runtime_checked:
                _runtime_checked.add(weight)
                print(f'[convT] weight checked for channel selection at run time: {weight[:120]}')
        return (
            'conv_transpose2d(' + self.visit(node.children[0]) + ', ' + weight
            + (', ' + ', '.join(kwargs) if kwargs else '') + ')'
        )

    def visitIrFUnfold(self, node):
        return (
            'F.unfold(' + self.visit(node.children[0]) + ', kernel_size='
            + self.visit(node.kernel_size) + ', padding=' + self.visit(node.padding)
            + ', stride=' + self.visit(node.stride) + ')'
        )

    def visitIrFFold(self, node):
        padding = ', padding=' + self.visit(node.padding) if node.padding is not None else ''
        return (
            'F.fold(' + self.visit(node.children[0]) + ', output_size='
            + self.visit(node.output_size) + ', kernel_size=' + self.visit(node.kernel_size)
            + ', stride=' + self.visit(node.stride) + padding + ')'
        )

    def visitIrTorchEinsum(self, node):
        operands = ', '.join(self.visit(c) for c in node.children)
        #FEAT: training builds split the backward reduction of kernel-patch compositions (see training_einsum).
        if reuse_mode.get_flag() and training.get_flag():
            return 'training_einsum(' + repr(node.equation) + ', ' + operands + ')'
        return 'torch.einsum(' + repr(node.equation) + ', ' + operands + ')'

    def visitIrTorchCat(self, node):
        return ('torch.cat([' + ', '.join(self.visit(c) for c in node.children)
                + '], dim=' + str(node.dim) + ')')

    def visitIrAssignToView(self, node):
        view_expr = self.visit(node.children[0])
        index_expr = self.renderTraceIndex(node.index)
        value_expr = self.visit(node.children[1])
        if not inductor_mode.get_flag():
            self.write(view_expr + '[' + index_expr + '] = ' + value_expr)
            return
        # index_expr only references torch/module-level globals, never a local
        # of the calling method, so it's safe to inline in the module-level helper.
        self._method_has_view_write = True
        helper_name = '_view_write_' + str(len(self.view_write_helpers))
        self.view_write_helpers.append(
            '@torch.compiler.disable\n'
            'def ' + helper_name + '(view, value):\n'
            '\tview[' + index_expr + '] = value\n'
            '\treturn view\n'
        )
        self.write(helper_name + '(' + view_expr + ', ' + value_expr + ')')
    
    def visitIrEmptyList(self, node):
        return '[]'

    def visitType(self, node):
        return node.__name__
    
    def visitFloat(self, node):
        if node == float('inf'):
            return "float('inf')"
        if node == float('-inf'):
            return "float('-inf')"
        return str(node)
    
    def visitInt(self, node):
        return str(node)
    
    def visitIrAppendList(self, node):
        return self.visit(node.children[0]) + '+ [' + self.visit(node.children[1]) + ']'
    

    def visitIrListExtract(self, node):
        return '(' + self.visit(node.children[0]) + ')[' + self.visit(node.children[1]) + ']'
    
    def visitIrBlockExtract(self, node):
        return self.visit(node.children[0]) + '.blocks[' + self.visit(node.children[1]) + ']'

    def visitIrBlockCopy(self, node):
        return self.visit(node.children[0]) + '.copy()'
    
    def visitIrGetSparseTensorBlocks(self, node):
        the_sparse_tensor = self.visit(node.children[0])
        return the_sparse_tensor + '.blocks'

    def visitIrGetAbsElemSparseDKey(self, node):
        # the_abs_elem_sparse = self.visit(node.children[0])
        key = node.key
        # return the_abs_elem_sparse + f'.d[{key}]'
        return f'abs_elem.d[\'{key}\']'
    
    def visitIrGetPolyExpSparseConst(self, node):
        the_pes = self.visit(node.children[0])
        return the_pes + '.const'

    def visitIrGetPolyExpSparseMat(self, node):
        the_pes = self.visit(node.children[0])
        return the_pes + '.mat'

    def visitIrGetSymExpSparseConst(self, node):
        the_ses = self.visit(node.children[0])
        return the_ses + '.const'

    def visitIrGetSymExpSparseMat(self, node):
        the_ses = self.visit(node.children[0])
        return the_ses + '.mat'

    def visitIrGetKthLayerNetworkParam(self, node):
        return f'abs_elem.network[{node.layer_index}].{node.param}'

    def visitIrDenseBlock(self, node):
        self.used_block_classes.add(('DenseBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'DenseBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ', '
                + str(node.batch_size) + ')')

    def visitIrKernelBlock(self, node):
        self.used_block_classes.add(('KernelBlock', node.sparse_block_type))
        return (SPARSE_TIER[node.sparse_block_type] + 'KernelBlock('
                + self.visit(node.children[0]) + ', ' + self._const(node.total_shape) + ', '
                + str(node.ix) + ', ' + str(node.iy) + ', '
                + str(node.ox) + ', ' + str(node.oy) + ', '
                + str(node.sx) + ', ' + str(node.sy) + ', '
                + str(node.px) + ', ' + str(node.py) + ', '
                + str(node.kx) + ', ' + str(node.ky) + ', '
                + str(node.num_channels) + ', ' + str(node.num_kernels) + ')')

    def get_operator_func(self, name: str):
        if not isinstance(name, str):
            name = name.__name__
        OP_MAP = {
            "add": 'operator.add',
            "sub": 'operator.sub',
            "eq": 'operator.eq',
            "ne": 'operator.ne',
            "ge": 'operator.ge',
            "gt": 'operator.gt',
            "le": 'operator.le',
            "lt": 'operator.lt',
            "or_": 'operator.or_',
            "and_": 'operator.and_',
            "mul": 'operator.mul',
            "truediv": 'operator.truediv',
            "floordiv": 'operator.floordiv',
            "mod": 'operator.mod',
            "pow": 'operator.pow',
        }

        try:
            return OP_MAP[name]
        except KeyError:
            raise ValueError(f"Unsupported operator: {name}, {type(name)}")


    def visitIrBlockBinaryOp(self, node):
        return self.visit(node.children[0]) + '.binary(' + self.visit(node.children[1]) + ', ' + self.get_operator_func(node.op) + ')'

    def visitIrBlockWhereBlock(self, node):
        return 'sp_where_block(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ', ' + self.visit(node.children[2]) + ')'

    def visitIrBlockUnaryOp(self, node):
        op = node.op
        if op == '-':
            op_str = 'operator.neg'
        elif op == 'not':
            op_str = 'operator.not_'
        elif op in ('sigma', 'sigma_tangent', 'tanh_tangent'):
            op_str = repr(op)
        else:
            raise Exception('OP NOT IDENTIFIED', op)
        return self.visit(node.children[0]) + '.unary(' + op_str + ')'
    
    def visitIrGetSubBlockCustomRange(self, node):
        return self.visit(node.children[0]) + '.get_sub_block_custom_range(' + self.visit(node.start_index) + ', ' + self.visit(node.end_index) + ', ' + self.visit(node.block_id) + ', ' + str(node.tensor) + ')'

    def visitTorchTensor(self, node):
        return self._const('torch.tensor(' + str(node.tolist()) + ')')

    def visitIrVar(self, node):
        if node.name == 'sym_size':
            return 'SymExpSparse.count'
        return node.name
    
    def visitIrEpsilon(self, node):
        # num = self.visit(node.num)
        shape = '['
        for i in range(len(node.irMetadata)):
            for j in range(len(node.irMetadata[i].shape)):
                shape += self.visit(node.irMetadata[i].shape[j]) + ","
        shape += ']'
        shape = self._const('torch.tensor(' + shape + ')')
        if node.inside_while:
            return 'get_new_eps(abs_elem.network, ' + shape + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = True, while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
        return 'get_new_eps(abs_elem.network, ' + shape + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = False, while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)

    def visitIrNewEps(self, node):
        [matIr, constIr] = node.children
        return 'SymExpSparse(abs_elem.network, ' + self.visit(matIr) + ', ' + self.visit(constIr) + ')'
    
    def visitIrPhi(self, node):
        s = 'phi(['
        for i in range(len(node.children)):
            s += self.visit(node.children[i])
            if i != len(node.children)-1:
                s += ', '
        s += '])'
        return s
    
    def visitIrConvertBoolToFloat(self, node):
        if reuse_mode.get_flag():
            return '('+self.visit(node.children[0])+').float()'
        return 'convert_to_float(' + self.visit(node.children[0]) + ')'

    def visitIrRepeat(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        repeat_dims = self._const('torch.tensor([' + repeat_dims + '])')
        if node.inside_while:
            ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
        else:
            ret = 'repeat(' + self.visit(node.children[0]) + ', ' + repeat_dims + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)
        return ret

    def visitIrAddDimension(self, node):
        [inputIr] = node.children
        size = 0
        for i in range(len(node.irMetadata)-1):
            for j in range(len(node.irMetadata[i].broadcast)):
                size += 1
        size += len(inputIr.irMetadata[-1].shape)
        indices = []
        for i in range(len(node.irMetadata[-1].shape) - len(inputIr.irMetadata[-1].shape)):
            indices.append(str(size))
            size += 1
        if len(indices) == 0:
            return self.visit(inputIr)
        if node.inside_while:
            kw_suffix = (
                'layer_index = layer_index, '
                'counter = ' + str(node.ttb_counter) + ', inside_while = True, '
                'while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
            )
        else:
            kw_suffix = (
                'layer_index = layer_index, '
                'counter = ' + str(node.ttb_counter) + ', inside_while = False, '
                'while_number = ' + str(node.while_number) + ')'
            )
        ret = '(' + self.visit(inputIr) + ')'
        for dim in indices:
            ret = ret + '.unsqueeze(' + dim + ', ' + kw_suffix
        return ret
    
    def visitIrRemoveDimension(self, node):
        [inputIr] = node.children
        if node.inside_while:
            ret = '(' + self.visit(inputIr) + ').squeeze(' + str(node.numDim) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
        else:
            ret = '(' + self.visit(inputIr) + ').squeeze(' + str(node.numDim) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')'
        return ret

    def visitIrAddDimensionConst(self, node):
        assert(isinstance(node, IrAddDimensionConst))
        inputIr = node.children[0]
        size = len(node.children)-1
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        if inputIr.irMetadata[-1].isConst:
            if node.inside_while:
                trace = ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = True, while_number = ' + str(node.while_number) + ', while_iteration=while_iteration'
            else:
                trace = ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = False, while_number = ' + str(node.while_number)
            return 'add_dimension_const(' + str(self.visit(inputIr)) + ', ' + self._const('torch.tensor([' + repeat_dims + '])') + trace + ')'
        ret = str(self.visit(inputIr))
        for i in range(size):
            ret += '.unsqueeze(' + str(i) + ')'
        ret += '.repeat(' + self._const('torch.tensor([' + repeat_dims + '])') + ')'
        return ret
    
    def visitIrBinaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == 'max':
            op_name = 'cf_max'
            flag = True
        elif node.op == 'min':
            op_name = 'cf_min'
            flag = True
        elif node.op == '+':
            op_name = 'operator.add'
        elif node.op == '-':
            op_name = 'operator.sub'
        elif node.op == '<=':
            op_name = 'operator.le'
        elif node.op == '<':
            op_name = 'operator.lt'
        elif node.op == '>=':
            op_name = 'operator.ge'
        elif node.op == '>':
            op_name = 'operator.gt'
        elif node.op == '==':
            op_name = 'operator.eq'
        elif node.op == '!=':
            op_name = 'operator.ne'
        elif node.op == 'and':
            op_name = 'operator.and_'
        elif node.op == 'or':
            op_name = 'operator.or_'
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        if flag:
            if node.inside_while:
                return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=True, while_number=' + str(node.while_number) + ', while_iteration=while_iteration)'
            return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=False, while_number=' + str(node.while_number) + ')'
        else:
            if node.inside_while:
                return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
            

            return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)
    
    def visitIrUnaryOp(self, node):
        op_name = None 
        flag = False
        if node.op == '-':
            op_name = 'operator.neg'
        elif node.op == 'not':
            op_name = 'operator.not_'
        elif node.op in ('sigma', 'sigma_tangent', 'tanh_tangent'):
            op_name = repr(node.op)
        elif node.op == 'any':
            op_name = 'any'
            flag = True
        elif node.op == 'all':
            op_name = 'all'
            flag = True
        elif node.op == 'get_dims':
            op_name = 'get_dims'
            flag = True
        elif node.op == 'get_shape_1':
            op_name = 'get_shape_1'
            flag = True
        elif node.op == 'get_shape_0':
            op_name = 'get_shape_0'
            flag = True
        else:
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [inputIr] = node.children
        if flag and node.op in ('get_shape_1', 'get_shape_0'):
            return op_name + '(' + self.visit(inputIr) + ')'
        elif flag:
            if node.inside_while:
                return f'{op_name}({self.visit(inputIr)}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
            return f'{op_name}({self.visit(inputIr)}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
        else:
            if node.inside_while:
                return 'unary(' + self.visit(inputIr) + ', ' + op_name + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=True, while_number=' + str(node.while_number) + ', while_iteration=while_iteration)'
            return 'unary(' + self.visit(inputIr) + ', ' + op_name + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=False, while_number=' + str(node.while_number) + ')'
        
    def visitIrSimpleUnary(self, node):
        op = node.op
        if isinstance(op, (IrVar, IrLambda)):
            return '(' + self.visit(op) + ')(' + self.visit(node.children[0]) + ')'
        if op == '-':
            op_str = 'operator.neg'
        elif op == 'not':
            op_str = 'operator.not_'
        elif op == 'sigma':
            return 'torch.sigmoid(' + self.visit(node.children[0]) + ')'
        elif op in ('sigma_tangent', 'tanh_tangent'):
            return op + '(' + self.visit(node.children[0]) + ')'
        else:
            raise Exception('OP NOT IDENTIFIED', op)
        return op_str + '(' + self.visit(node.children[0]) + ')'

    def visitIrLambda(self, node):
        if node.op in ('add', 'mul', 'and_', 'or_'):
            return 'lambda x: x'
        if node.op == 'sub':
            return 'lambda x: operator.neg(x)'
        raise Exception('OP NOT IDENTIFIED', node.op)

    def visitIrSimpleBinary(self, node):
        [lhsIr, rhsIr] = node.children
        lhs = self.visit(lhsIr)
        rhs = self.visit(rhsIr)
        #FIX FOR: NaN gradients from divisions in torch.where arms that are not selected (sparse_block.safe_div).
        if training.get_flag() and self.get_operator_func(node.op) == 'operator.truediv':
            return 'safe_div(' + lhs + ', ' + rhs + ')'
        return self.get_operator_func(node.op) + '(' + lhs + ', ' + rhs + ')'

    def visitIrTorchWhere(self, node):
        return (
            'torch.where(' +
            self.visit(node.children[0]) + ', ' +
            self.visit(node.children[1]) + ', ' +
            self.visit(node.children[2]) +
            ')'
        )
    
    def visitIrTensorOnes(self, node):
        # Was defaulting to CPU unconditionally (no device field was ever
        # recorded), silently mismatching every GPU tensor built elsewhere.
        if isinstance(node.total_size, str):
            return 'torch.ones(*' + node.total_size + ', device=device_mode.get_device())'
        return 'torch.ones(*' + str(node.total_size.tolist()) + ', device=device_mode.get_device())'

    def visitTensorRepeat(self, node):
        return 'torch.repeat(' + self.visit(node.children[0]) + ', *' + node.repeat_dims + ')'

    def visitIrTensorRepeat(self, node):
        return self.visitTensorRepeat(node)
    
    def visitTensorClamp(self, node):
        if node.min_true:
            return 'torch.clamp(' + self.visit(node.children[0]) + ', min=' + str(node.const) + ')'
        else:
            return 'torch.clamp(' + self.visit(node.children[0]) + ', max=' + str(node.const) + ')'

    def visitIrTensorClamp(self, node):
        return self.visitTensorClamp(node)

    def visitIrBlockClamp(self, node):
        return self.visit(node.children[0]) + '.clamp(' + str(node.const) + ', min_true=' + str(node.min_true) + ')'

    def visitIrBlockSqueeze(self, node):
        return self.visit(node.children[0]) + '.squeeze(' + str(node.index) + ')'



    def visitIrGetDefaultStop(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        if node.inside_while:
            return f'get_default_stop([{repeat_dims}], abs_elem, batch_size, curr_size, poly_size, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'get_default_stop([{repeat_dims}], abs_elem, batch_size, curr_size, poly_size, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
    
    def visitIrGetPriorityLList(self, node):
        if node.inside_while:
            return f'get_max_priority({self.visit(node.children[0])}, {self.visit(node.children[1])}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'get_max_priority({self.visit(node.children[0])}, {self.visit(node.children[1])}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'
    
    # def visitIrGetPolyexpStop(self, node):
    #     return 'filter_trav_exp_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    # def visitIrGetPolyexpNotStop(self, node):
    #     return 'filter_trav_exp_not_stop(' + self.visit(node.children[0]) + ', ' + self.visit(node.children[1]) + ')'
    
    def visitIrGetPolyexpStop(self, node):
        trav = self.visit(node.children[0])
        stop = self.visit(node.children[1])
        if node.inside_while:
            return f'filter_trav_exp_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'filter_trav_exp_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'

    def visitIrGetPolyexpNotStop(self, node):
        trav = self.visit(node.children[0])
        stop = self.visit(node.children[1])
        if node.inside_while:
            return f'filter_trav_exp_not_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=True, while_number={node.while_number}, while_iteration=while_iteration)'
        return f'filter_trav_exp_not_stop({trav}, {stop}, layer_index=layer_index, counter={node.ttb_counter}, inside_while=False, while_number={node.while_number})'

    def visitIrPolyExpMat(self, node):
        return self.visit(node.children[0]) + '.mat'

    def visitIrPolyExpNotStopFloat(self, node):
        if reuse_mode.get_flag():
            return '('+self.visit(node.children[0]) + '.unary(operator.not_)).float()'
        return 'convert_to_float(' + self.visit(node.children[0]) + '.unary(operator.not_))'

    def visitIrBlockPolyexpStop(self, node):
        recv = self.visit(node.children[0])
        return (self.polyexp_cls + '(' + recv + '.network, ' + self.visit(node.children[1])
                + ', ' + recv + '.const)')

    def visitIrBlockPolyexpNotStop(self, node):
        recv = self.visit(node.children[0])
        return (self.polyexp_cls + '(' + recv + '.network, ' + self.visit(node.children[1])
                + ', ' + self.visit(node.children[2]) + ')')

    def visitIrBlockAny(self, node):
        return self.visit(node.children[0]) + '.any()'

    def visitIrBlockAll(self, node):
        return self.visit(node.children[0]) + '.all()'

    def visitIrBlockGetDims(self, node):
        return self.visit(node.children[0]) + '.dims'
    
    def visitIrConvertToTensor(self, node):
        repeat_dims = ''
        for i in range(1, len(node.children)):
            repeat_dims += self.visit(node.children[i])
            if i<len(node.children)-1:
                repeat_dims += ', '
        return 'convert_to_tensor(' + self.visit(node.children[0]) + ', [' + repeat_dims +  '])'

    def visitIrMult(self, node):
        op_name = None 
        if node.op == '*':
            op_name = 'operator.mul'
        elif node.op == '/':
            op_name = 'operator.truediv'
        else:
            op_name = node.op
            raise Exception('OP NOT IDENTIFIED', node.op)
        
        [lhsIr, rhsIr] = node.children
        # node.ttb_counter = self.counter
        # self.counter += 1
        # return 'binary' + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ')'
        if node.inside_while:
            return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
        return 'binary(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + op_name + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)
    
    def visitIrInnerProduct(self, node):
        op_name = 'inner_prod'
        
        [lhsIr, rhsIr] = node.children
        if node.inside_while:
            return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
        return op_name + '(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)
    
    def visitIrBlockInnerProduct(self, node):
        if node.type == 'equal_dims':
            return self.visit(node.children[0]) + '.matmul_equal_dims(' + self.visit(node.children[1])+ ')'
        elif node.type == 'unequal_dims':
            return self.visit(node.children[0]) + '.matmul_unequal_dims(' + self.visit(node.children[1])+ ')'
        else:
            raise Exception('OP NOT IDENTIFIED', node.type)

    def visitIrDot(self, node):
        [lhsIr, rhsIr] = node.children
        dot_args = (
            ', layer_index = layer_index, '
            + 'counter = ' + str(node.ttb_counter)
            + ', inside_while = ' + ('True' if node.inside_while else 'False')
            + ', while_number = ' + str(node.while_number)
        )
        if node.inside_while:
            dot_args += ', while_iteration = while_iteration'
        if lhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(lhsIr) + '.dot(' + self.visit(rhsIr) + ', abs_elem.get_poly_size()' + dot_args + ", mats_input = 'rhs')"
        elif rhsIr.irMetadata[-1].type == 'Neuron':
            return self.visit(rhsIr) + '.dot(' + self.visit(lhsIr) + ', abs_elem.get_poly_size()' + dot_args + ", mats_input = 'lhs')"
        elif lhsIr.irMetadata[-1].type == 'Float':
            if node.inside_while:
                return 'inner_prod(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
            return 'inner_prod(' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)
        else:
            raise Exception('NOT IMPLEMENTED')

    def visitIrTernary(self, node):
        [condIr, lhsIr, rhsIr] = node.children
        return 'where(' + self.visit(condIr) + ', ' + self.visit(lhsIr) + ', ' + self.visit(rhsIr) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = ' + ('True' if node.inside_while else 'False') + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)

    def visitIrClamp(self, node):
        [inputIr, const] = node.children
        min_true = node.min_true 
        if node.inside_while:
            return 'clamp(' + self.visit(inputIr) + ', ' + str(const) + ', ' + str(min_true) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = True' + ', while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)' + self._ttb_comment(node)
        return 'clamp(' + self.visit(inputIr) + ', ' + str(const) + ', ' + str(min_true) + ', ' + 'layer_index = layer_index, ' + 'counter = ' + str(node.ttb_counter) + ', inside_while = False' + ', while_number = ' + str(node.while_number) + ')' + self._ttb_comment(node)

    def visitIrCombineToPoly(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'poly_size'
        return self.polyexp_cls + '(abs_elem.network, ' + self.visit(coeffIr) + ' , ' + self.visit(constIr) + ')'

    def visitIrCombineToSym(self, node):
        [coeffIr, constIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + self.visit(coeffIr) + ', ' + self.visit(constIr) + ')'


    def visitIrExtractPolyCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return self.visit(inputIr) + '.get_mat(abs_elem' \
            + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
    
    def visitIrExtractSymCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count' \
            + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'

    def visitIrExtractPolyConst(self, node, ):
        [inputIr] = node.children
        if reuse_mode:
            return self.visit(inputIr) + '.const'
        return self.visit(inputIr) + '.get_const()'
    
    def visitIrExtractSymConst(self, node):
        [inputIr] = node.children
        if reuse_mode:
            return self.visit(inputIr) + '.const'
        return self.visit(inputIr) + '.get_const()'

    def visitIrConvertNeuronToPoly(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return self.visit(inputIr) + '.convert_to_poly(abs_elem' \
            + ', layer_index=layer_index, counter=' + str(node.ttb_counter) \
            + ', inside_while=' + str(node.inside_while) \
            + ', while_number=' + str(node.while_number) \
            + ', while_iteration=' + while_iteration + ')'
    
    def visitIrConcatStitch(self, node):
        [prev1Ir, prev2Ir] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return (f"concat_stitch_2d(abs_elem, '{node.key}', '{node.source}', "
                f"{self.visit(prev1Ir)}, {self.visit(prev2Ir)}, layer_index=layer_index, "
                f"counter={node.ttb_counter}, inside_while={node.inside_while}, "
                f"while_number={node.while_number}, while_iteration={while_iteration})")

    def visitIrConcatStitchMat(self, node):
        [prev1Ir, prev2Ir] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        return (f"concat_stitch_mat(abs_elem, '{node.key}', "
                f"{self.visit(prev1Ir)}, {self.visit(prev2Ir)}, layer_index=layer_index, "
                f"counter={node.ttb_counter}, inside_while={node.inside_while}, "
                f"while_number={node.while_number}, while_iteration={while_iteration})")

    def visitIrConvertConstToPoly(self, node):
        [inputIr, rows] = node.children
        cols = 'poly_size'
        return self.polyexp_cls + '(abs_elem.network, 0.0, ' + self.visit(inputIr) + ')'
        
    def visitIrConvertConstToSym(self, node):
        [inputIr, rows] = node.children
        cols = 'SymExpSparse.count'
        rows = self.visit(rows)
        return 'SymExpSparse(abs_elem.network,' + 'None, ' + self.visit(inputIr) + ')'
    
    def visitIrExpandSymExp(self, node):
        [inputIr] = node.children
        if not reuse_mode.get_flag():
            return self.visit(inputIr) + '.expand_symexp_mat(SymExpSparse.count)'
        var_name = self._hoist_if_complex(self.visit(inputIr))
        # Rebind rather than mutate in place: total_size may alias a hoisted
        # constant shared by other blocks/tensors, so writing through it would
        # corrupt every other user of that constant.
        self.write(var_name + '.total_size = torch.cat([' + var_name + '.total_size[:-1], torch.tensor([SymExpSparse.count], dtype=torch.int64)])')
        return var_name

    def visitIrAccess(self, node):
        [lhsIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        if not node.isMetadata:
            return 'abs_elem.get_elem(\'' + node.elem + '\', ' + self.visit(lhsIr) \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        else:
            return self.visit(lhsIr) + '.get_metadata(\'' + node.elem + '\', batch_size' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        
    def visitIrReduce(self, node):
        [inputIr] = node.children
        size = node.reduce_dim
        if node.inside_while:
            return '(' + self.visit(inputIr) + ').sum(' + str(size) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = True, while_number = ' + str(node.while_number) + ', while_iteration=while_iteration)'
        return '(' + self.visit(inputIr) + ').sum(' + str(size) + ', layer_index = layer_index, counter = ' + str(node.ttb_counter) + ', inside_while = False, while_number = ' + str(node.while_number) + ')'

    def visitIrMapCoeff(self, node):
        [inputIr] = node.children
        while_iteration = 'while_iteration' if node.inside_while else 'None'
        if inputIr.irMetadata[-1].type == 'PolyExp':    
            return self.visit(inputIr) + '.get_mat(abs_elem' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'
        return self.visit(inputIr) + '.get_mat(SymExpSparse.count' \
                + ', layer_index=layer_index, counter=' + str(node.ttb_counter) + ', inside_while=' + str(node.inside_while) + ', while_number=' + str(node.while_number) + ', while_iteration=' + while_iteration + ')'

    def visitIrMapNeuron(self, node):
        if node.dims:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '), None, None,' + "abs_elem.live_layers)"
        else:
            return 'Llist(abs_elem.network, [1]*(' + self.visit(node.children[0]) + '.mat.dims-1), None, None,' + "abs_elem.live_layers)"

    def visitIrSymbolic(self, node):
        return node.name
    
    def _emit_while_contents(self, while_block):
        """Emit header stmts, break check, and body stmts of a WhileBlock."""
        for stmt in while_block.children:
            self.visit(stmt)
        if while_block.inner_jump is not None:
            cond = self.visit(while_block.inner_jump[0])
            self.write('if(' + str(cond) + '):')
            self.indent += 1
            for stmt in while_block.inner_jump[1].children:
                self.visit(stmt)
            self.indent -= 1
        if while_block.jump is not None:
            body_block = while_block.jump[1]
            for stmt in body_block.children:
                self.visit(stmt)

    def visitIrParallelBlock(self, node):
        self.write('import threading')
        for var in node.escape1 + node.escape2:
            self.write(var + ' = None')

        # Thread 1 (L-bound loop)
        self.write('def _par_thread_1():')
        self.indent += 1
        for var in node.escape1:
            self.write('nonlocal ' + var)
        for stmt in node.zone1_init:
            self.visit(stmt)
        self.write('while(True):')
        self.indent += 1
        self._emit_while_contents(node.while1)
        self.indent -= 1
        for stmt in node.zone1_post:
            self.visit(stmt)
        self.indent -= 1

        # Thread 2 (U-bound loop)
        self.write('def _par_thread_2():')
        self.indent += 1
        for var in node.escape2:
            self.write('nonlocal ' + var)
        for stmt in node.zone2_init:
            self.visit(stmt)
        self.write('while(True):')
        self.indent += 1
        self._emit_while_contents(node.while2)
        self.indent -= 1
        for stmt in node.zone2_post:
            self.visit(stmt)
        self.indent -= 1

        self.write('_par_n_orig = torch.get_num_threads()')
        self.write('_par_n_cores = torch.multiprocessing.cpu_count()')
        self.write('torch.set_num_threads(max(1, min(_par_n_orig, _par_n_cores // 2)))')
        self.write('_t1 = threading.Thread(target=_par_thread_1)')
        self.write('_t2 = threading.Thread(target=_par_thread_2)')
        self.write('_t1.start(); _t2.start()')
        self.write('_t1.join(); _t2.join()')
        self.write('torch.set_num_threads(_par_n_orig)')

    def visitIrFlow(self, node):
        self.indent += 1
        if sroa_build():
            self.write('res = flow(*explode_inputs(abs_elem, batch_size))')
            self.write('res = apply_spec_offset(network, res)')
            self.write('if not inductor_mode.get_flag():')
            self.write('    print("Peak memory usage:", torch.cuda.max_memory_allocated() / 1024**2, "MB")')
            self.write('return res')
            self.indent -= 1
            return
        if fused_build():
            # The unrolled flow() returns the final abstract shape; densify it here.
            self.write('abs_shape = flow(abs_elem, batch_size)')
            self.write('res = get_dense_inlined(abs_shape[0]), get_dense_inlined(abs_shape[1])')
            self.write('res = apply_spec_offset(network, res)')
            self.write('if not inductor_mode.get_flag():')
            self.write('    print("Peak memory usage:", torch.cuda.max_memory_allocated() / 1024**2, "MB")')
            self.write('return res')
            self.indent -= 1
            return
        self.write('flow = Flow(abs_elem, ' + str(node.transformer) + '(), network, print_intermediate_results, no_sparsity)')
        self.write('res = flow.flow()')
        self.write('res = apply_spec_offset(network, res)')
        self.write('if not inductor_mode.get_flag():')
        self.write('    print("Peak memory usage:", torch.cuda.max_memory_allocated() / 1024**2, "MB")')
        self.write('return res')
        self.indent -= 1

    def visitIrObjectLookup(self, node):
        if node.object_name == "block":
            return self.visit(node.children[0]) + ".block"
        raise Exception("NOT IMPLEMENTED")

    def visitIrBlockCreateSimilar(self, node):
        return (
            self.visit(node.children[0]) +
            ".create_similar(" +
            self.visit(node.children[1]) +
            ")"
        )

    def visitIrSetBlockTotalShapeLastDim(self, node):
        block_var = self._hoist_if_complex(self.visit(node.children[0]))
        value = self.visit(node.children[1])
        # Rebind rather than mutate in place -- see visitIrExpandSymExp.
        self.write(block_var + ".total_shape = torch.cat([" + block_var + ".total_shape[:-1], torch.tensor([" + value + "], dtype=torch.int64)])")

    def visitIrAssignToBlock(self, node):
        block_var = self.visit(node.children[0])
        value = self.visit(node.children[1])
        self.write(block_var + ".block = " + value)

    def visitIrBlockGetSubBlockCustomRange(self, node):
        return (
            self.visit(node.children[0]) +
            ".get_sub_block_custom_range(" +
            self.visit(node.start_index) + ", " +
            self.visit(node.end_index) + ", " +
            self.visit(node.block_start_index) +
            ")"
        )
