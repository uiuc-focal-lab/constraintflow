from tabulate import tabulate
from experiments.experiments_correct import run_verifier as run_experiment_correct
from experiments.experiments_incorrect import run_verifier as run_experiment_incorrect

folder = "dnn_certifiers/"
certifiers = ["balance_cert", "reuse_cert"]
basicops = ["affine", "maxpool", "relu", "mult"]
parameters = {"affine": (10, 10), "maxpool": (3, 3), "relu": (1, 1), "mult": (1, 1)}
gen_time = dict()
verification_time = dict()
bug_time = dict()


for c in certifiers:
    gen_time[c] = dict()
    verification_time[c] = dict()
    bug_time[c] = dict()
    for b in basicops:
        print(f"Running {c} on sound {b}")
        ret_dict_correct = run_experiment_correct(folder + c + "_" + b, parameters[b][0], parameters[b][1])
        print()
        print(f"Running {c} on unsound {b}")
        ret_dict_incorrect = run_experiment_incorrect(folder + c + "_" + b, parameters[b][0], parameters[b][1])
        print()
        print()
        gen_time[c][b] = ret_dict_correct[list(ret_dict_correct.keys())[0]][1]
        verification_time[c][b] = ret_dict_correct[list(ret_dict_correct.keys())[0]][0]
        bug_time[c][b] = ret_dict_incorrect[list(ret_dict_incorrect.keys())[0]][0]
        
table = []
row1 = []
for b in basicops:
    row1.append("")
    row1.append(b)
    row1.append("")
table.append(["Certifier"]+row1)
heading = ['G', 'V', 'B']*len(basicops)
table.append([" "]+heading)
for c in certifiers:
    row = [c]
    for b in basicops:
        row += [round(gen_time[c][b], 3), round(verification_time[c][b], 3), round(bug_time[c][b], 3)]
    table.append(row)
print()
print()
print(tabulate(table))
print(f"(a) New Transformers introduced in § 6.1")
