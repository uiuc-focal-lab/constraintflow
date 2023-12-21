import sys
with open(sys.argv[2]) as f:
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

		print(sys.argv[1])
		print("generation time:")
		print(gen)
		print("verification time:")
		print(solve)
