import os

python_script="python -m verification.experiments.experiments_correct"
python_script_unsound="python -m verification.experiments.experiments_incorrect"
folder = "verification/data/test_cases_correct/"
certifiers = ["deeppoly", "vegas", "deepz", "refinezono", "ibp", "hybrid_zono", "balance_cert", "reuse_cert"]
basicops = ["relu", "neuron_max",  "neuron_min", "neuron_add", "neuron_mult", "relu6", "abs", "sigmoid", "tanh", "swish"]

print("Sound DNN certifier experiments:")

for c in certifiers:
	print(c)
	for b in basicops:
		cmdline = python_script + " " + folder + c + "_" + b + " 1 1  " + " | ./verification/experiments/parsecmd.py"
		print(b)
		os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "affine" + " 2048 2048  " + " | ./verification/experiments/parsecmd.py"
	print("affine")
	os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "maxpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("maxpool")
	os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "minpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("minpool")
	os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "avgpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("avgpool")
	os.system(cmdline)

print("Unsound DNN certifier experiments:")

for c in certifiers:
	print(c)
	for b in basicops:
		cmdline = python_script_unsound + " " + folder + c + "_" + b + " 1 1  " + " | ./verification/experiments/parsecmd.py"
		print(b)
		os.system(cmdline)

	cmdline = python_script_unsound + " " + folder + c + "_" + "affine" + " 2048 2048  " + " | ./verification/experiments/parsecmd.py"
	print("affine")
	os.system(cmdline)

	cmdline = python_script_unsound + " " + folder + c + "_" + "maxpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("maxpool")
	os.system(cmdline)

	cmdline = python_script_unsound + " " + folder + c + "_" + "minpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("minpool")
	os.system(cmdline)

	cmdline = python_script_unsound + " " + folder + c + "_" + "avgpool" + " 10 10  " + " | ./verification/experiments/parsecmd.py"
	print("avgpool")
	os.system(cmdline)