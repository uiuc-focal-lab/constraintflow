import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

sns.set_theme(style="darkgrid")

certifiers = ["deeppoly", "deepz", "refinezono", "ibp"]
verification_times = dict()
parameter_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
verification_times["deeppoly"] = [3.084350347518921,6.513941049575806,13.474895238876343,30.60409903526306,79.80948519706726,247.3636612892151,889.6070690155029,3640.4967002868652,16570.10538005829]
verification_times["deepz"] = [2.1721630096435547,4.4436256885528564,9.743277549743652,21.366082429885864,56.4132776260376,190.47812581062317,854.5479183197021,4902.001406431198,36841.756742239]
verification_times["refinezono"] = [0.8362777233123779,1.6762847900390625,3.7793562412261963,8.060346841812134,21.77883791923523,77.10634350776672,329.9939982891083,1820.7125716209412,12882.870867967606]
verification_times["ibp"] = [1.513129711151123,3.0602200031280518,6.464395523071289,14.361914873123169,36.5639922618866,120.90966653823853,540.8649754524231,3128.07422542572,23287.789702177048]


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