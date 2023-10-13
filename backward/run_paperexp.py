import os

params = {
"deeppoly_affine": (100,8),
"deeppoly_relu": (100,10),
"deeppoly_maxpool": (10,10),
"zono_affine": (8,10),
"zono_relu": (100,10),
"zono_maxpool": (3,10),
"refinezono_affine": (100,10),
"refinezono_relu": (100,10),
"refinezono_maxpool": (3,10),
"ibp_affine": (8,10),
"ibp_relu": (100,10),
"ibp_maxpool": (19,10),
"fb_affine": (100,10),
"fb_relu": (100,10),
"fb_maxpool": (100,10),
"polyzono_affine": (8,5),
"polyzono_relu": (100,10),
"polyzono_maxpool": (3,10) }

out_folder =  "tempout/"
in_folder = "test_cases_correct/"

for test in params.keys():
	(nprev,nsym) = params[test]
	cmd = "python experiments.py " + in_folder + test + " \"" + str(nprev) + " " + str(nsym) + " " + "1\"" + " > " + out_folder + test  
	print(cmd)
	df
	os.system(cmd)
	cmd = "python paper_parse.py " + test + " " + out_folder + test 
	os.system(cmd) 
	df