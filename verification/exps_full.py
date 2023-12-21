import os

python_script="python experiments.py"
folder = "data/test_cases_incorrect/"
certifiers = ["deeppoly", "ibp", "refinezono", "vegas", "zono"]
basicops = ["neuron_add", "neuron_mult", "neuron_max", "neuron_min", "relu"]

for c in certifiers:
	print(c)
	for b in basicops:
		cmdline = python_script + " " + folder + c + "_" + b + " \"1 1 1\" " + " | ./parsecmd.py"
		print(b)
		os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "affine" + " \"2100 2100 1\" " + " | ./parsecmd.py"
	print("affine")
	os.system(cmdline)

	cmdline = python_script + " " + folder + c + "_" + "maxpool" + " \"10 1 1\" " + " | ./parsecmd.py"
	print("maxpool")
	os.system(cmdline)

	# if(c == "ibp" or c == "deeppoly"):
	# 	cmdline = python_script + " " + folder + c + "_" + "neuron_list_mult" + " \"256 1 1\" " + " | ./parsecmd.py"
	# 	print("neuron_list_mult")
	# 	os.system(cmdline)