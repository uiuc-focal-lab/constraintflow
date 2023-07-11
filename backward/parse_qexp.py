import os

folder = "times_incorrect_extra/"
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
			exp = 0
			for l in lines:
				if(l.split(" ")[0] == "gen"):
					gen = gen + float(l.split(" ")[1]) - start
					start = float(l.split(" ")[1])
				if(l.split(" ")[0] == "end"):
					solve = solve + float(l.split(" ")[1]) - start
					start = float(l.split(" ")[1])
				if(l.split(" ")[0] == "graph"):
					exp = float(l.split(" ")[3]) - start
				if(l.split(" ")[0] == "symbolic"):
					symbolic = float(l.split(" ")[2]) - start - exp 

			print(filename[3:])
			print(gen)
			print(solve)
			print(exp)
			print(symbolic)