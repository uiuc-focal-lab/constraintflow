import os

names = ["deeppoly_relu", "deeppoly_affine", "deeppoly_maxpool", "zono_relu", "zono_affine", "zono_maxpool", "refinezono_relu", "refinezono_affine", "refinezono_maxpool", "ibp_relu", "ibp_affine", "ibp_maxpool", "polyzono_relu", "polyzono_affine", "polyzono_maxpool", "fb_relu", "fb_affine", "fb_maxpool"]
cgen = {}
cverify = {}
igen = {}
iverify = {}
ccount = {}
icount = {}

cfile = "out_correct"
ifile = "out_incorrect"

def get_number(s, prefix):
	numbers = s.split(" ")
	numbers[0] = numbers[0][len(prefix):]
	numbers2 = [int(x) for x in numbers]
	return numbers2

with open(cfile)as f:
	lines = f.readlines()
	for n in names:
		gensum = 0
		count = 0
		versum = 0
		i = 0
		while i < (len(lines)):
			if lines[i].startswith(n) and lines[i+1] != "0":
				#numbers2 = get_number(lines[i], n)
				#if(numbers2[1] >= numbers2[0]):
				gensum = gensum + float(lines[i+1])
				versum = versum + float(lines[i+2])
				count = count + 1
			i = i + 3
		cgen[n] = gensum / count
		cverify[n] = versum / count
		ccount[n] = count

with open(ifile)as f:
	lines = f.readlines()
	for n in names:
		gensum = 0
		count = 0
		versum = 0
		i = 0
		while i < (len(lines)):
			if lines[i].startswith(n) and lines[i+1] != "0":
				#numbers2 = get_number(lines[i], n)
				#if(numbers2[1] >= numbers2[0]):
				gensum = gensum + float(lines[i+1])
				versum = versum + float(lines[i+2])
				count = count + 1
			i = i + 3
		igen[n] = gensum / count
		iverify[n] = versum / count
		icount[n] = count

print("correct gen")
for k in cgen:
	print(k, cgen[k])

print("correct verify")
for k in cverify:
	print(k, cverify[k])


for k in ccount:
	print(k, ccount[k])

print("incorrect gen")
for k in igen:
	print(k, igen[k])

print("incorrect verify")
for k in iverify:
	print(k, iverify[k])


for k in icount:
	print(k, icount[k])
