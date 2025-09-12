class Flag:
    def __init__(self):
        self.flag = False

    def set_flag(self):
        self.flag = True

    def reset_flag(self):
        self.flag = False

    def get_flag(self):
        return self.flag
    
    def __str__(self):
        return f'flag: {self.flag}'

debug_flag = Flag()
debug_flag2 = Flag()
debug_flag3 = Flag()
debug_flag4 = Flag()

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
    

    def update_total_time(self, time1):
        self.total_time += time1
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
mat_to_patches_time = Time()
patches_to_mat_time = Time()
union_tensors_time = Time()
sub_block_custom_range_time = Time()
fixed_cost1 = Time()
fixed_cost2 = Time()
fixed_cost3 = Time()
relu_time = Time()
affine_time = Time()
l_time = Time()
u_time = Time()
mult_time = Time()
check_time = Time()

squeeze_time = Time()
sanity_time = Time()
sparse_tensor_init_time = Time()