import os
import json

# Common parent folder under which every jit_* capture directory is created
jit_root = "jit_captures"

def set_jit_root(path):
    global jit_root
    jit_root = path

# For fuse-affine.
network_path = None
# Compile-local stable ReLU configuration; None disables the final IR pass.
stable_relu_config = None
# Traversal-stop row scheduling: (chunk sizes, refine selected rows for the whole batch).
traversal_stop_config = (128, True)

def set_network_path(path):
    global network_path
    network_path = path

def jit_path(*parts):
    """Resolve a jit capture path under jit_root.

    Accepts either a bare directory name ("jit_binary") or an already-joined
    relative capture path ("jit_binary/binary_0_1.json"); both are placed under
    the common parent folder.
    """
    return os.path.join(jit_root, *parts)


# jit-memory
CAPTURE_ABI = 3
_jit_store = {}
# Host-only network coordinates for capture-time traversal validation.
capture_layers = ()

def jit_store_clear():
    _jit_store.clear()

def save_capture(rel_path, obj):
    """Persist a jit capture under its relative path (dir/file.json)."""
    if in_memory_captures:
        _jit_store[rel_path] = json.dumps({"capture_abi": CAPTURE_ABI, "data": obj})
        return
    path = jit_path(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({"capture_abi": CAPTURE_ABI, "data": obj}, f)

def load_capture(rel_path):
    """Load a jit capture previously written with save_capture."""
    if in_memory_captures:
        record = json.loads(_jit_store[rel_path])
    else:
        with open(jit_path(rel_path), 'r') as f:
            record = json.load(f)
    if not isinstance(record, dict) or record.get('capture_abi') != CAPTURE_ABI:
        raise ValueError(f"Incompatible JIT capture {rel_path}; rebuild captures with this compiler.")
    return record['data']


def capture_exists(rel_path):
    if in_memory_captures:
        return rel_path in _jit_store
    return os.path.exists(jit_path(rel_path))


class Flag:
    def __init__(self, initial=False):
        self.flag = initial

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def get_flag(self):
        return self.flag
    
    def __bool__(self):
        return self.flag
    

    def __str__(self):
        return f'flag: {self.flag}'

debug_flag = Flag()
debug_flag2 = Flag()
debug_flag3 = Flag()
debug_flag4 = Flag()

dummy_mode = Flag()
reuse_mode = Flag()
dense_default_mode = Flag()
inductor_mode = Flag()
#FEAT: `compile --compile-flow`: a plain (non-jit) build runs Flow.flow under torch.compile.
compile_flow = Flag()

# Preserves the captures in memory instead of writing json files.
in_memory_captures = Flag()


fuse_affine_subst = Flag()

# Set by builtin_ops.inject_builtin_ops when the compiled program defines its own
# Neuron_add rule. Such a rule (e.g. deeppoly's prev_0 + prev_1 for both L and U)
# makes an Add layer behave like an Affine one for --fuse-affine-subst: L == U, and
# its parents' concrete l/u are never read. The builtin Add reads both, so it doesn't.
program_add_rule = Flag()

# Turn patches box into dense 
compact_patches = Flag()

# jit builds: emit a layer-unrolled flow() into transformers.py
fused_flow = Flag()

# jit builds: splice every layer into one flow() and scalar-replace the
# Jit* aggregates. Requires reuse_mode + fused_flow.
sroa = Flag(True)


class Count:
    def __init__(self, initial=1):
        self.value = initial

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def __str__(self):
        return f'count: {self.value}'


class Mode:
    def __init__(self, initial):
        self.initial = initial
        self.value = initial

    def set_value(self, value):
        self.value = value

    def reset(self):
        self.value = self.initial

    def get_value(self):
        return self.value

    def __str__(self):
        return f'mode: {self.value}'


# Early reduction scheduling and bounded Inductor compilation regions.
early_reductions = Flag()
flow_segment_mb = Mode(0.0)

# MB the reuse build may keep live at once. Over it, mem_plan reorders and
# reorders and recomputes the flow.
# 0 disables, -1 = auto (MEMORY_BUDGET_FRACTION of the CUDA device's memory at
# compile time; the real peak runs a few GB above the estimate).
memory_budget_mb = Mode(-1.0)
MEMORY_BUDGET_FRACTION = 0.7

# Prove identical-weight positive/negative convolution pairs in tensor SSA.
fuse_sign_convs = Flag()

# Lower a sign-selected concretization over Float bounds, (c >= 0) ? c*X : c*Y, to
# c*(X+Y)/2 + |c|*(X-Y)/2 (center/radius) instead of c+*X + c-*Y. The sign-split form
# subtracts two large nearly equal sums when X ~ Y (narrow intervals), and loses the
# interval width to fp32 rounding; auto_LiRPA concretizes with center/radius.
center_radius = Flag(True)

# Compile flow() with mode="reduce-overhead" (CUDA graph replay).
cudagraphs = Flag()

#FEAT: certified-training build: importable as a package, safe divisions, margin spec folded into the last Linear.
training = Flag()

# Leave Inductor's conv layout optimization (channels-last operands) on; off by default.
inductor_layout_opt = Flag()

# UNFAIR LOADING
# Load ONNX initializers with parse.load_onnx_direct instead of onnx.load.
# direct_onnx_load = Flag()

# Evaluate all-pairs convolution and broadcast mat-vec grids (IBP's W+/W- times l/u) as one operation.
product_grids = Flag()


class DeviceMode:
    _DEVICE_MAP = {
        'cpu':    'cpu',
        'gpu':    'cuda',
        'gpumac': 'mps',
    }

    def __init__(self):
        self._device = 'cpu'

    def set_cpu(self):
        self._device = 'cpu'

    def set_gpu(self):
        self._device = 'cuda'

    def set_gpumac(self):
        self._device = 'mps'

    def set_mode(self, mode: str):
        if mode not in self._DEVICE_MAP:
            raise ValueError(f"Unknown device mode '{mode}'. Choose from: {list(self._DEVICE_MAP)}")
        self._device = self._DEVICE_MAP[mode]

    def get_device(self) -> str:
        return self._device

    def is_accelerated(self) -> bool:
        return self._device != 'cpu'

    def sync(self):
        if self._device == 'cuda':
            import torch
            torch.cuda.synchronize()

device_mode = DeviceMode()

class OperationTime():
    def __init__(self):
        self.total_time = 0
        self.data_transfer_time = 0
        self.actual_op_time = 0
        self.num_ops = 0
    def update_total_time(self, time1):
        self.total_time += time1
        self.num_ops += 1
    def update_data_transfer_time(self, time1):
        self.data_transfer_time += time1
    def update_actual_op_time(self, time1):
        self.actual_op_time += time1
        
    def get_total_time(self):
        return self.total_time
    def get_data_transfer_time(self):
        return self.data_transfer_time
    def get_actual_op_time(self):        
        return self.actual_op_time
    def get_num_ops(self):
        return self.num_ops


class Time:
    def __init__(self):
        self.total_time = 0
        self.op_time = 0
        self.num_used = 0

    def __str__(self):
        if self.total_time == 0:
            percentage_op_time = 0
        else:
            percentage_op_time = 100*self.op_time/self.total_time 
        return f'total time: {self.total_time: 8.3f}s, \
operation time: {self.op_time: 8.3f}s, \
%op time: {percentage_op_time: 8.3f}, \
index time: {self.total_time - self.op_time: 8.3f}s, \
num used: {self.num_used}'
    
    def just_update_total_time(self, time1):
        self.total_time += time1
    def update_total_time(self, time1):
        self.total_time += time1
        self.num_used += 1

    def update_num_used(self):
        self.num_used += 1

    def update_op_time(self, time1):
        self.op_time += time1

    def get_total_time(self):
        return self.total_time

    def get_op_time(self):
        return self.op_time
    
binary_time = Time()
unary_time = Time()
matmul_time = Time()
where_time = Time()
repeat_time = Time()
clamp_time = Time()
any_time = Time()
all_time = Time()
unsqueeze_time = Time()
get_elem_time = Time()
get_sparse_range_time = Time()
reduce_size_time = Time()
filter_non_live_time = Time()
union_tensors_time = Time()
patches_to_mat_time = Time()
sub_block_custom_range_time = Time()
squeeze_time = Time()
sum_time = Time()
sanity_time = Time()
sparse_tensor_init_time = Time()
binary_profilier = OperationTime()
unary_profilier = OperationTime()
equal_matmul_profilier = OperationTime()
unequal_matmul_profilier = OperationTime()
clamp_profilier = OperationTime()



binary_tensor_ops_expenses = Time()
binary_tensor_ops_x_sparsity = Time()
binary_tensor_ops_y_sparsity = Time()
binary_tensor_ops_no_sparse = Time()
total_binary_tensor_ops = Time()

binary_sparse_tensor_expenses = Time()
binary_sparse_tensor_dom2 = Time()
binary_sparse_tensor_dom1 = Time()
binary_sparse_tensor_overlap = Time()
binary_sparse_tensor_overlap_expenses = Time()
binary_sparse_tensor_dom1_expenses = Time()
binary_sparse_tensor_dom2_expenses = Time()
total_binary_sparse_tensor = Time()

binary_sparse_tensor_count = Time()
binary_block_level_tensor_count = Time()

binary_block_expenses = Time()

binary_fixed_costs = Time()


matmul_tensor_ops = Time()
matmul_tensor_ops_expenses = Time()
matmul_sparse_tensor_expenses = Time()
matmul_block_all= Time()
matmul_block_op = Time()

# Added by Heng
clamp_total_time = Time()
clamp_op_expense = Time()
clamp_sparse_tensor_expense = Time()
clamp_sparse_block_expense = Time()
clamp_sparse_block_op_time = Time()
clamp_const_block_expense = Time()
clamp_const_block_op_time = Time() # Should be 0, added for uniformity.
clamp_repeat_block_expense = Time()
clamp_repeat_block_op_time = Time()

stop_condition_time = Time()
