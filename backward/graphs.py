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

endnsymb = {"deeppoly_affine": 7,
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
"polyzono_maxpool": 10}

endnprev = {"deeppoly_affine": 10,
"deeppoly_relu": 10,
"deeppoly_maxpool": 9,
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
"polyzono_affine": 10,
"polyzono_relu": 10,
"polyzono_maxpool": 10}

#nprev fixed
names2 = ["deeppoly_relu", "deeppoly_affine", "deeppoly_maxpool", "zono_relu", "zono_affine", "zono_maxpool", "refinezono_relu", "refinezono_affine", "refinezono_maxpool", "ibp_relu", "ibp_affine", "ibp_maxpool", "polyzono_relu", "polyzono_affine", "polyzono_maxpool", "fb_relu", "fb_affine", "fb_maxpool"]
for n in names2:
	e = endnsymb[n]
	'''
	plt.figure()
	name1 = n + "_gen_nprevfixed"
	e = endnsymb[n]
	nprev_1 = cfulldata_gen[n][0][0:e]
	nprev_3 = cfulldata_gen[n][2][0:e]
	nprev_5 = cfulldata_gen[n][4][0:e]
	nprev_7 = cfulldata_gen[n][6][0:e]
	plt.plot(nprev_1, label='nprev=1')
	plt.plot(nprev_3, label='nprev=3')
	plt.plot(nprev_5, label='nprev=5')
	plt.plot(nprev_7, label='nprev=7')
	plt.title(name1)
	plt.xlabel('Nsymb')
	plt.ylabel('Time in seconds')
	leg = plt.legend(loc='upper left')
	plt.savefig("graphs/"+name1+".png")
	'''
	
	plt.figure()
	name2 = n + "_ver_nprevfixed"
	nprev_1 = cfulldata_ver[n][0][0:e]
	nprev_3 = cfulldata_ver[n][2][0:e]
	nprev_5 = cfulldata_ver[n][4][0:e]
	nprev_7 = cfulldata_ver[n][6][0:e]
	x = np.arange(e)
	x = x + 1
	avg = (nprev_1 + nprev_3 + nprev_5 + nprev_7) / 4
	'''
	plt.plot(x,nprev_1, label='nprev=1')
	plt.plot(x,nprev_3, label='nprev=3')
	plt.plot(x,nprev_5, label='nprev=5')
	plt.plot(x,nprev_7, label='nprev=7')
	'''
	plt.plot(x, avg)
	plt.title(name2)
	plt.xlabel('Nsymb')
	plt.ylabel('Time in seconds')
	#leg = plt.legend(loc='upper left')
	plt.savefig("graphs/"+name2+".png")

#nsymb fixed
for n in names2:
	e = endnprev[n]
	'''
	plt.figure()
	name1 = n + "_gen_nsymbfixed"
	e = endnprev[n]
	nprev_1 = cfulldata_gen[n][:,0][0:e]
	nprev_3 = cfulldata_gen[n][:,2][0:e]
	nprev_5 = cfulldata_gen[n][:,4][0:e]
	nprev_7 = cfulldata_gen[n][:,6][0:e]
	plt.plot(nprev_1, label='nsymb=1')
	plt.plot(nprev_3, label='nsymb=3')
	plt.plot(nprev_5, label='nsymb=5')
	plt.plot(nprev_7, label='nsymb=7')
	plt.title(name1)
	plt.xlabel('Nprev')
	plt.ylabel('Time in seconds')
	leg = plt.legend(loc='upper left')
	plt.savefig("graphs/"+name1+".png")
	'''
	
	plt.figure()
	name2 = n + "_ver_nsymbfixed"
	nprev_1 = cfulldata_ver[n][:,0][0:e]
	nprev_3 = cfulldata_ver[n][:,2][0:e]
	nprev_5 = cfulldata_ver[n][:,4][0:e]
	nprev_7 = cfulldata_ver[n][:,6][0:e]
	x = np.arange(e)
	x = x + 1
	avg = (nprev_1 + nprev_3 + nprev_5 + nprev_7) / 4
	'''
	plt.plot(x,nprev_1, label='nsymb=1')
	plt.plot(x,nprev_3, label='nsymb=3')
	plt.plot(x,nprev_5, label='nsymb=5')
	plt.plot(x,nprev_7, label='nsymb=7')
	'''
	plt.plot(x, avg)
	plt.title(name2)
	plt.xlabel('Nprev')
	plt.ylabel('Time in seconds')
	#leg = plt.legend(loc='upper left')
	plt.savefig("graphs/"+name2+".png")
	