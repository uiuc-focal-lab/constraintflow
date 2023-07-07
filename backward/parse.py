import os

folder = "times_full_incorrect/"
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

			print(filename[3:])
			print(gen)
			print(solve)
