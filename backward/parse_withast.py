import os

asttimes = {
"deeppoly_affine": 0.12422,
"deeppoly_relu": 0.161128,
"deeppoly_maxpool": 0.122776,
"zono_affine": 0.09267,
"zono_relu": 0.11229,
"zono_maxpool": 0.09518,
"refinezono_affine": 0.10554,
"refinezono_relu": 0.14984,
"refinezono_maxpool": 0.1069,
"ibp_affine": 0.08383,
"ibp_relu": 0.09718,
"ibp_maxpool": 0.06715,
"fb_affine": 0.067038,
"fb_relu": 0.11861,
"fb_maxpool": 0.0452,
"polyzono_affine": 0.13021,
"polyzono_relu": 0.17815,
"polyzono_maxpool": 0.1509 }

def stripn(s):
	i = 0
	while(not s[i].isdigit()):
		i = i+1
	return s[0:i]

folder = "times_full_correct/"
names = []
for filename in os.listdir(folder):
	names.append(filename)
names.sort()
for filename in names:
	with open(os.path.join(folder, filename)) as f:
		lines = f.readlines()
		if len(lines) > 0:
			start = float(lines[0].split(" ")[1])
			gen = 0
			solve = 0
			for l in lines:
				if(l.split(" ")[0] == "gen"):
					gen = gen + float(l.split(" ")[1]) - start
					start = float(l.split(" ")[1])
				if(l.split(" ")[0] == "end"):
					solve = solve + float(l.split(" ")[1]) - start
					start = float(l.split(" ")[1])
			#Add AST to filenames
			f = stripn(filename[3:])
			gen = gen + asttimes[f]
			print(filename[3:])
			print(gen)
			print(solve)