from tabulate import tabulate
from experiments.experiments_correct import run_verifier as run_experiment_correct
from experiments.experiments_incorrect import run_verifier as run_experiment_incorrect

folder = "dnn_certifiers/"
certifiers = ["deeppoly", "vegas", "deepz", "refinezono", "ibp", "hybrid_zono"]
basicops = ["affine", "maxpool", "minpool", "avgpool"]
parameters = {"affine": (10, 10), "maxpool": (3, 3), "minpool": (3, 3), "avgpool": (3, 3)}
gen_time = dict()
verification_time = dict()
bug_time = dict()


def r(x):
    if isinstance(x, float):
        return round(x, 3)
    return x

for c in certifiers:
    gen_time[c] = dict()
    verification_time[c] = dict()
    bug_time[c] = dict()
    for b in basicops:
        if c == "hybrid_zono" and b == "affine":
            gen_time[c][b] = '-'
            verification_time[c][b] = '-'
            bug_time[c][b] = '-'
            continue
        elif c == "vegas" and b != "affine":
            gen_time[c][b] = '-'
            verification_time[c][b] = '-'
            bug_time[c][b] = '-'
            continue
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
        row += [r(gen_time[c][b]), r(verification_time[c][b]), r(bug_time[c][b])]
    table.append(row)
print()
print()
print(tabulate(table))
print(f"(b) Composite operations")
