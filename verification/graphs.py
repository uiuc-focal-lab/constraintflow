import os
import numpy as np
from matplotlib import pyplot as plt

names = ["deeppoly_relu", "deeppoly_affine", "deeppoly_maxpool", "zono_relu", "zono_affine", "zono_maxpool", "refinezono_relu", "refinezono_affine", "refinezono_maxpool", "ibp_relu", "ibp_affine", "ibp_maxpool", "polyzono_relu", "polyzono_affine", "polyzono_maxpool", "fb_relu", "fb_affine", "fb_maxpool"]
cfile = "correct_full" #correct implementation full results
ifile = "incorrect_full" #incorrect implementation full results
cfulldata_gen = {} #correct implementation data
ifulldata_gen = {} #incorrect implementation data
cfulldata_ver = {} #correct implementation data
ifulldata_ver = {} #incorrect implementation data

def get_number(s, prefix):
	numbers = s.split(" ")
	numbers[0] = numbers[0][len(prefix):]
	numbers2 = [int(x) for x in numbers]
	return numbers2

#Collecting data
with open(cfile)as f:
	lines = f.readlines()
	for n in names:
		ndata_g = np.zeros((10,10)) #For generation time
		ndata_v = np.zeros((10,10)) #For verificationtime
		i = 0
		while i < (len(lines)):
			if lines[i].startswith(n) and lines[i+1] != "0":
				numbers2 = get_number(lines[i], n)
				nprev = numbers2[0]
				nsymb = numbers2[1]
				ndata_g[nprev-1][nsymb-1] = float(lines[i+1])
				ndata_v[nprev-1][nsymb-1] = float(lines[i+2])
			i = i + 3
		cfulldata_gen[n] = ndata_g
		cfulldata_ver[n] = ndata_v

with open(ifile)as f:
	lines = f.readlines()
	for n in names:
		ndata_g = np.zeros((10,10)) #For generation time
		ndata_v = np.zeros((10,10)) #For verificationtime
		i = 0
		while i < (len(lines)):
			if lines[i].startswith(n) and lines[i+1] != "0":
				numbers2 = get_number(lines[i], n)
				nprev = numbers2[0]
				nsymb = numbers2[1]
				ndata_g[nprev-1][nsymb-1] = float(lines[i+1])
				ndata_v[nprev-1][nsymb-1] = float(lines[i+2])
			i = i + 3
		ifulldata_gen[n] = ndata_g
		ifulldata_ver[n] = ndata_v

#print(cfulldata)
#print(ifulldata)

#Generating graphs

endnsym = {"deeppoly_affine": 7,
"deeppoly_relu": 10,
"deeppoly_maxpool": 10,
"zono_affine": 10,
"zono_relu": 10,
"zono_maxpool": 10,
"refinezono_affine": 10,
"refinezono_relu": 10,
"refinezono_maxpool": 10,
"ibp_affine": 10,
"ibp_relu": 10,
"ibp_maxpool": 10,
"fb_affine": 10,
"fb_relu": 10,
"fb_maxpool": 10,
"polyzono_affine": 7,
"polyzono_relu": 10,
"polyzono_maxpool": 7}

endnprev = {"deeppoly_affine": 10,
"deeppoly_relu": 10,
"deeppoly_maxpool": 9,
"zono_affine": 7,
"zono_relu": 10,
"zono_maxpool": 3,
"refinezono_affine": 10,
"refinezono_relu": 10,
"refinezono_maxpool": 3,
"ibp_affine": 7,
"ibp_relu": 10,
"ibp_maxpool": 10,
"fb_affine": 10,
"fb_relu": 10,
"fb_maxpool": 10,
"polyzono_affine": 10,
"polyzono_relu": 10,
"polyzono_maxpool": 3}

#nprev fixed
#names2 = ["deeppoly_relu", "deeppoly_affine", "deeppoly_maxpool", "zono_relu", "zono_affine", "zono_maxpool", "refinezono_relu", "refinezono_affine", "refinezono_maxpool", "ibp_relu", "ibp_affine", "ibp_maxpool", "polyzono_relu", "polyzono_affine", "polyzono_maxpool", "fb_relu", "fb_affine", "fb_maxpool"]
#names2 = ["deeppoly_affine"]
names_nprevfixed = ["deeppoly_affine", "polyzono_affine", "polyzono_maxpool", "refinezono_maxpool", "zono_maxpool"]
names_nsymfixed = ["deeppoly_maxpool", "ibp_affine", "ibp_maxpool", "polyzono_maxpool", "refinezono_maxpool", "zono_affine", "zono_maxpool"]
for n in names_nprevfixed:
	e1 = endnsym[n]
	e2 = endnprev[n]
	
	plt.figure()
	name2 = n + "_ver_nprevfixed"
	avg = cfulldata_ver[n][0][0:e1]
	for i in range(1,e2):
		avg = avg + cfulldata_ver[n][i][0:e1]
	avg = avg / e2
	x = np.arange(e1)
	x = x + 1
	
	print(name2)
	for i in range(len(x)):
		print(avg[i])

	plt.plot(x, avg)
	plt.title(name2)
	plt.xlabel('Nsym')
	plt.ylabel('Time in seconds')
	#leg = plt.legend(loc='upper left')
	plt.savefig("graphs_final/"+name2+".png")

#nsymb fixed
for n in names_nsymfixed:
	e1 = endnprev[n]
	e2 = endnsym[n]
	
	plt.figure()
	name2 = n + "_ver_nsymfixed"

	avg = cfulldata_ver[n][:,0][0:e1]
	for i in range(1,e2):
		avg = avg + cfulldata_ver[n][:,i][0:e1]
	avg = avg / e2
	x = np.arange(e1)
	x = x + 1

	print(name2)
	for i in range(len(x)):
		print(avg[i])

	plt.plot(x, avg)
	plt.title(name2)
	plt.xlabel('Nprev')
	plt.ylabel('Time in seconds')
	#leg = plt.legend(loc='upper left')
	plt.savefig("graphs_final/"+name2+".png")
	