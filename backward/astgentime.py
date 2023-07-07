import os
names=["tempout"]

import os
gen = 0
for i in range(0,3):
	for filename in names:
		filename = filename+str(i)
		with open(filename) as f:
			lines = f.readlines()
			if len(lines) > 0:
				start = float(lines[0].split(" ")[1])
				for l in lines:
					if(l.split(" ")[0] == "start"):
						gen = gen + float(l.split(" ")[1]) - start
print(gen/3)

				
