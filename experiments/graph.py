import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from experiments.experiments_correct import run_verifier as run_experiment_correct

sns.set_theme(style="darkgrid")

folder = "dnn_certifiers/"
certifiers = ["deeppoly", "deepz", "ibp", "refinezono"]
gen_time = dict()
verification_times = dict()
parameter_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

for c in certifiers:
    verification_times[c] = []
    for p in parameter_values:
        print(f"Running {c} with parameters {p}")
        ret_dict_correct = run_experiment_correct(folder + c + "_affine", p, p)
        print()
        verification_times[c].append(ret_dict_correct[list(ret_dict_correct.keys())[0]][0])


# Logarithmic scale for better visualization
log_parameter_values = np.log2(parameter_values)
log_verification_times_deep_poly = np.log2(verification_times["deeppoly"])
log_verification_times_deep_z = np.log2(verification_times["deepz"])
log_verification_times_relu_val = np.log2(verification_times["ibp"])
log_verification_times_refine_zono = np.log2(verification_times["refinezono"])

table = []
row1 = []
for b in parameter_values:
    row1.append(b)
table.append(["Certifier"]+row1)
for c in certifiers:
    row = [c]
    for b in range(len(log_parameter_values)):
        row.append(round(verification_times[c][b], 3))
    table.append(row)
print()
print()
print(tabulate(table))
print(f"The verification times for different parameter values as shown in Fig. 17 of the paper.")



plt.plot(log_parameter_values, log_verification_times_deep_poly, 'o-', label='DeepPoly', color='blue')
plt.plot(log_parameter_values, log_verification_times_deep_z, 'o-', label='DeepZ', color='purple')
plt.plot(log_parameter_values, log_verification_times_relu_val, 'o-', label='IBP', color='green')
plt.plot(log_parameter_values, log_verification_times_refine_zono, 'o-', label='RefineZono', color='red')

plt.xlabel(r'Parameter value $(n_{\text{prev}})$', fontsize=24)
plt.ylabel(r'Verification Time $(V)$', fontsize=24)

y_axis_labels = [2, 8, 32, 128, 512, 2048, 8192, 32768]
plt.xticks(log_parameter_values, parameter_values, fontsize=18)
plt.yticks(np.log2(y_axis_labels), y_axis_labels, fontsize=18)
plt.legend(fontsize=24)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()