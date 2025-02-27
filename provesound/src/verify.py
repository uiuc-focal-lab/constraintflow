from z3 import *
import time

from constraintflow import astVisitor
from constraintflow import astcf as AST

from provesound.src.value import *
from provesound.src.symbolicDNN import SymbolicDNN, populate_vars

from provesound.lib.utils import *
from provesound.lib.optSolver import OptSolver
from provesound.lib.globals import *

exptemp = None
op_ = None 
case_ = 0

# This class implements the ProveSound algorithm. Given a ConstraintFlow 
# program, it visits all the abstract transformers and checks their their soundness w.r.t
# a given soundness property (specified by the user as a part of the program).
# For each transformer, it initializes a SymbolicDNN object and then expands it using 
# symbolicDNN.py. Then, using the soundness property, it generates the verification query 
# using the symbolic semantics and checks its soundness using a z3 based solver implemented in 
# src/optSolver.py. The verification time is stored in a dictionary and returned to the user. 
class Verify(astVisitor.ASTVisitor):

	def __init__(self):
	# Initializes the Verify class with default parameters and data structures.
	# Attributes:
	# 	shape (dict): A dictionary to store the abstract shape.
	# 	F (dict): A dictionary to store function definitions.
	# 	theta (dict): A dictionary to store transformer definitions.
	# 	Nprev (int): Bound parameter, initialized to 3.
	# 	Nsym (int): Bound parameter, initialized to 3.
	# 	number (Number): An instance of the Number class to generate new variable names.
	# 	M (dict): A dictionary to map the outputs of traverse and solver to values.
	# 	V (dict): A dictionary to store symbolic variables and corresponding vertices.
	# 	C (list): A list to store symbolic constraints.
	# 	E (list): A list to store new symbolic variables.
	# 	old_eps (list): A list to store existing symbolic epsilon values.
	# 	old_neurons (list): A list to store existing symbolic neurons.
	# 	store (dict): A dictionary to store variable values.
	# 	arrayLens (dict): A dictionary to store lengths of arrays as z3 variables.
	# 	solver (OptSolver): An instance of the OptSolver class.
	
		self.shape = {}
		self.F = {}
		self.theta = {}
		self.Nprev = 3
		self.Nsym = 3
		self.number = Number()
		self.M = {}
		self.V = {}
		self.C = []
		self.E = []
		self.old_eps = []
		self.old_neurons = []
		self.store = {}
		self.arrayLens = {}
		self.solver = OptSolver()

	def visitShapeDecl(self, node):
		for (t,e) in node.elements.arglist:
			self.shape[e.name] = t.name

		self.constraint = node.p

	def visitFunc(self, node):
		self.F[node.decl.name.name] = node

	def visitTransformer(self, node):
		self.theta[node.name.name] = node

	def visitFlow(self, node):
		global exptemp
		global op_
		global case_
		ret_dict = dict()
		node = self.theta[node.trans.name]
		for op_i in range(len(node.oplist.olist)):
			ret_dict[node.oplist.olist[op_i].op.op_name] = (0, 0)
			self.E.clear()

			hasE = node.oplist.opsE[op_i]
			op = node.oplist.olist[op_i]
			store = self.store
			arrayLens = self.arrayLens
			prevLength = (Int('prevLength'), "Int")
			if self.Nprev > 1:
				self.C.append(prevLength[0]>0)
				self.C.append(prevLength[0] <= self.Nprev)
			
			op_ = op.op.op_name
			case_ = 0

			# print(f"{op_} transformer")
			update_start_time()
			
			if(op_ == "Relu" or op_ == "Relu6" or op_ == "Abs" or op_=='rev_Relu' or op_=='rev_Relu6' or op_=='rev_Abs' or op_ == 'rev_Maxpool' or op_ == "HardTanh" or op_ == "rev_HardTanh" or op_ == "HardSigmoid" or op_ == "rev_HardSigmoid" or op_ == "HardSwish" or op_ == "rev_HardSwish"):
				nprev= 1
			elif op_ == 'Neuron_mult' or op_ == 'Neuron_add' or op_ == 'Neuron_max' or op_ == 'Neuron_min' or op_ == 'rev_Neuron_mult' or op_ == 'rev_Neuron_add' or op_ == 'rev_Neuron_max' or op_ == 'rev_Neuron_min':
				nprev = 2
			else:
				nprev = self.Nprev
				
			required_neurons = ['curr', 'prev']
			is_list = False 
			if 'Affine' in op_ or 'pool' in op_:
				is_list = True


			if op_ == 'rev_Maxpool':
				required_neurons = ['curr', 'prev', 'curr_list']
			elif (op_ == 'Neuron_mult' or op_ == 'Neuron_add' or op_ == 'Neuron_max' or op_ == 'Neuron_min' or op_=='rev_Neuron_mult' or op_ == 'rev_Neuron_add' or op_ == 'rev_Neuron_max' or op_ == 'rev_Neuron_min' or op_ == 'Neuron_list_mult'):
				required_neurons = ['curr', 'prev_0', 'prev_1']

			s = SymbolicDNN(self.store, self.F, self.constraint, self.shape, nprev, self.Nsym, self.number, self.M, self.V, self.C, self.E, self.old_eps, self.old_neurons, self.solver, self.arrayLens, prevLength)
			s.ss.hasE = hasE
			
			if("curr" in required_neurons):
				curr = Vertex('Curr')
				self.V[curr.name] =  curr 
				populate_vars(s.vars, curr, self.C, self.store, s.ss, self.constraint, self.number)
				store["curr"] = (curr.name, "Neuron")
			if("prev" in required_neurons):
				prev = []
				for i in range(nprev):
					p = Vertex('Prev'+str(i))
					prev.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)
				if(is_list):
					store["prev"] = prev
				else:
					store["prev"] = prev[0] 
				if(len(prev) > 1):
					arrayLens[str(prev)] = prevLength
			
			if("curr_list" in required_neurons):
				curr_list = []
				for i in range(self.Nprev):
					p = Vertex('curr_list'+str(i))
					curr_list.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)
				store["curr_list"] = curr_list
				arrayLens[str(curr_list)] = prevLength

			if(op_ == 'Neuron_list_mult'):
				prev_0 = []
				prev_1 = []
				for i in range(nprev):
					p = Vertex('Prev0_'+str(i))
					prev_0.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)

					p = Vertex('Prev1_'+str(i))
					prev_1.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)
				store["prev_0"] = prev_0 
				arrayLens[str(prev_0)] = prevLength
				store["prev_1"] = prev_1
				arrayLens[str(prev_1)] = prevLength 
			elif("prev_1" in required_neurons):
				prev = []
				for i in range(nprev):
					p = Vertex('Prev'+str(i))
					prev.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)
				store["prev_0"] = prev[0] 
				store["prev_1"] = prev[1] 
			elif("prev_0" in required_neurons):
				prev = []
				for i in range(nprev):
					p = Vertex('Prev'+str(i))
					prev.append((p.name, "Neuron"))
					self.V[p.name] = p
					populate_vars(s.vars, p, self.C, self.store, s.ss, self.constraint, self.number)
				store["prev_0"] = prev[0] 
			

			curr_prime = Vertex('curr_prime' + str(op_i))
			s.V[curr_prime.name] = curr_prime

			#Define relationship between curr and prev
			if(op_ == "Affine"):
				if(not "weight" in curr.symmap.keys()):
					curr.symmap["weight"] = [(Real('weight_curr' + str(op_i) + "_" + str(self.number.nextn())), "Float") for i in range(nprev)]
					arrayLens[str(curr.symmap["weight"])] = prevLength
				if(not "bias" in curr.symmap.keys()):
					curr.symmap["bias"] = ((Real('bias_curr' + str(self.number.nextn())), "Float"))
				exptemp = curr.symmap["bias"]
				for i in range(len(prev)):
					exptemp = ADD(exptemp, IF(LEQ(i+1,prevLength),(MULT(prev[i], curr.symmap["weight"][i])),(0, "Int")))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			elif(op_ == "Neuron_list_mult"):
				exptemp = (0, 'Float')
				for i in range(nprev):
					exptemp = ADD(exptemp, IF(LEQ(i+1,prevLength),MULT(prev_0[i], prev_1[i]),(0, "Int")))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			elif(op_ == "rev_Affine"):
				if(not "equations" in curr.symmap.keys()):
					curr.symmap["equations"] = [(Real('equations_' + str(op_i) + "_" + str(self.number.nextn())), "PolyExp") for i in range(nprev)]
					arrayLens[str(curr.symmap["equations"])] = prevLength
				exptemp = (True, "Bool")
				j = 0
				for t in curr.symmap["equations"]:
					exptemp = AND(exptemp, IF(LEQ(j+1,prevLength),EQQ(curr.name, t),(True, "Bool")))
					j = j + 1

				s.currop = s.ss.convertToZ3(exptemp)
			elif(op_ == "Relu"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(GEQ(exptemp, (0, 'Float')), exptemp, (0, 'Float'))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Relu6"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(GEQ(exptemp, (0, 'Float')), exptemp, (0, 'Float'))
				exptemp = IF(LEQ(exptemp, (6, 'Float')), exptemp, (6, 'Float'))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Abs"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(GEQ(exptemp, (0, 'Float')), exptemp, NEG(exptemp))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "HardTanh"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				exptemp = IF(LEQ(exptemp, (-1, 'Float')), -1, IF(GEQ(exptemp, (1, 'Float')), 1, exptemp))

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "HardSigmoid"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				exptemp = If(If((exptemp + 1)/2 > 1, 1, (exptemp + 1)/2) < 0, 0, If((exptemp + 1)/2 > 1, 1, (exptemp + 1)/2))
				
				s.currop = (curr.name == exptemp)

			elif(op_ == "HardSwish"):
				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				exptemp = If(exptemp <= -3, 0, If(exptemp >= 3, exptemp, exptemp * (exptemp+3)/6))
				
				s.currop = (curr.name == exptemp)

			elif(op_ == "Neuron_mult"):
				exptemp = MULT(prev[0], prev[1])

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Neuron_add"):
				exptemp = ADD(prev[0], prev[1])

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)
			
			elif(op_ == "Neuron_max"):
				exptemp = IF(GEQ(prev[0], prev[1]), prev[0], prev[1])

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Neuron_min"):
				exptemp = IF(LEQ(prev[0], prev[1]), prev[0], prev[1])

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "rev_Neuron_mult"): #prev_0 is the output neuron and prev_1 is the other input neuron
				exptemp = EQQ(prev[0], MULT(curr.name, prev[1]))
				s.currop = s.ss.convertToZ3(exptemp)

			elif(op_ == "rev_Neuron_add"):
				exptemp = EQQ(prev[0], ADD(curr.name, prev[1]))
				s.currop = s.ss.convertToZ3(exptemp)

			elif(op_ == "rev_Neuron_max"):
				exptemp = EQQ(prev[0],IF(GEQ(curr.name, prev[1]), curr.name, prev[1]))
				s.currop = s.ss.convertToZ3(exptemp)

			elif(op_ == "rev_Neuron_min"):
				exptemp = EQQ(prev[0],IF(LEQ(curr.name, prev[1]), curr.name, prev[1]))
				s.currop = s.ss.convertToZ3(exptemp)

			elif(op_ == "rev_Relu"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (0, 'Float')), curr.name, (0, 'Float'))
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "rev_Relu6"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (0, 'Float')), curr.name, (0, 'Float'))
				prevexp = IF(LEQ(curr.name, (6, 'Float')), prevexp, (6, 'Float'))
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "rev_Abs"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (0, 'Float')), curr.name, NEG(curr.name))
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "rev_HardTanh"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (-1, 'Float')), curr.name, (-1, 'Float'))
				prevexp = IF(LEQ(curr.name, (1, 'Float')), prevexp, (1, 'Float'))
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "rev_HardSigmoid"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(curr.name, (-1, 'Float')), DIV(ADD(curr.name, (1, "Float")), (2, "Float")), (0, 'Float'))
				prevexp = IF(LEQ(curr.name, (1, 'Float')), prevexp, (1, 'Float'))
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "rev_HardSwish"):

				exptemp = (0, "Float") 
				for i in range(len(prev)):
					exptemp = ADD(exptemp, prev[i])
				
				temp_curr_name = ADD(curr.name, (3, "Float"))
				
				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = IF(GEQ(temp_curr_name, (0, 'Float')), temp_curr_name, (0, 'Float'))
				prevexp = IF(LEQ(temp_curr_name, (6, 'Float')), prevexp, (6, 'Float'))
				prevexp = DIV(prevexp, (6, "Float"))
				prevexp = MULT(prevexp, curr.name)
				s.currop = ( s.ss.convertToZ3(prevexp) == exptemp)

			elif(op_ == "Maxpool"):
				exptemp = prev[0]
				for i in range(1, nprev):
					cond = LEQ(i+1,prevLength)
					for j in range(i):
						cond = AND(cond, GEQ(prev[i], prev[j]))
					for j in range(i+1,nprev):
						cond = AND(cond, OR(GEQ(prev[i], prev[j]),GT(j+1,prevLength)))

					exptemp = IF(cond, prev[i], exptemp)

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Minpool"):
				exptemp = prev[0]
				for i in range(1, nprev):
					cond = LEQ(i+1,prevLength)
					for j in range(i):
						cond = AND(cond, LEQ(prev[i], prev[j]))
					for j in range(i+1,nprev):
						cond = AND(cond, OR(LEQ(prev[i], prev[j]),GT(j+1,prevLength)))

					exptemp = IF(cond, prev[i], exptemp)

				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "Avgpool"):
				exptemp = prev[0]
				summation = prev[0]
				for i in range(1, nprev):
					summation = ADD(summation, prev[i])
					exptemp = IF(EQQ(i+1,prevLength), DIV(summation, prevLength), exptemp)
				
				exptemp = s.ss.convertToZ3(exptemp)
				s.currop = (curr.name == exptemp)

			elif(op_ == "rev_Maxpool"):
				total_list = curr_list + [(curr.name, "Neuron")]
				exptemp = total_list[0]
				for i in range(1, self.Nprev+1):
					cond = (True, 'Bool')
					for j in range(self.Nprev+1):
						if i!=j:
							cond = AND(cond, GEQ(total_list[i], total_list[j]))
					exptemp = IF(cond, total_list[i], exptemp)
				prevexp = (0, "Float") 
				for i in range(len(prev)):
					prevexp = ADD(prevexp, prev[i])

				exptemp = s.ss.convertToZ3(exptemp)
				prevexp = s.ss.convertToZ3(prevexp)
				s.currop = (prevexp == exptemp)
			
			for m in curr.symmap.keys():
				curr_prime.symmap[m] = curr.symmap[m]
			
			computation = (curr_prime.name == curr.name)
			if(len(s.ss.arrayLens) != 0):
				computation = [s.currop, computation, prevLength[0] > 0, prevLength[0] <= nprev] + s.ss.tempC
			else:
				computation = [s.currop, computation] + s.ss.tempC
			s.ss.tempC = []
			s.flag = True
			
			try:
				s.visit(op.ret)
			except:
				print(f"Induction failed for {op_}")
				ret_dict[node.oplist.olist[op_i].op.op_name] = (get_verification_time(), get_generation_time())
				reset_time()
				continue
			s.visit(op.ret)

			update_generation_time()
			vallist = None
			
			if(isinstance(op.ret, AST.TransRetIfNode)):
				vallist = self.visitTransRetIf(op.ret, s)
			else:
				vallist = self.visitTransRetBasic(op.ret, s)
			leftC = computation + s.ss.C 
			
			try:    
				self.applyTrans(leftC, vallist, s, curr_prime, computation)
				print(f"Proved {op_}")
			except:
				print(f"Transformer unsound for {op_}")

			ret_dict[node.oplist.olist[op_i].op.op_name] = (get_verification_time(), get_generation_time())
			reset_time()
		return ret_dict

	def applyTrans(self, leftC, vallist, s, curr_prime, computation):
		global case_
		if(isinstance(vallist, list)):
			for (elem, val) in zip(self.shape.keys(), vallist):
				curr_prime.symmap[elem] = val

			conslist = [self.constraint]
			if isinstance(self.constraint, AST.ExprListNode):
				conslist = self.constraint.exprlist
			set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)

			case_ += 1
			for cons_id, one_cons in enumerate(conslist):
				c = populate_vars(s.vars, curr_prime, self.C, self.store, s.ss, one_cons, self.number, False)
				z3constraint = s.ss.convertToZ3(c)
				if(isinstance(z3constraint, bool)):
					z3constraint = BoolSort().cast(z3constraint)
				if isinstance(exptemp, z3.z3.ArithRef) and not('rev' in op_):
					z3constraint = substitute(z3constraint, (curr_prime.name, exptemp))
				eps_constraints = []
				for eps in self.E:
					eps_constraints.append(eps >= -1)
					eps_constraints.append(eps <= 1)
				
				newLeftC = leftC + s.ss.tempC + eps_constraints + s.ss.C


				
				gen_time = time.time()
				update_generation_time()
				lhs = And(newLeftC)
				rhs = z3constraint
				w = self.solver.solve(lhs, rhs)
				end_time = time.time()
				update_verification_time()
				if(not w):
					raise Exception(f"Constraint Unsound. Proved in {end_time - gen_time : .5f}s")
				s.ss.tempC = []

		else:
			condz3 = s.ss.convertToZ3(vallist.cond)
			preC = leftC + s.ss.tempC
			s.ss.tempC = []
			self.applyTrans(preC + [condz3], vallist.left, s, curr_prime, computation)
			self.applyTrans(preC + [Not(condz3)], vallist.right, s, curr_prime, computation)

	
	def visitTransRetBasic(self, node, s):
		return s.ss.visit(node.exprlist)

	def visitTransRetIf(self, node, s):
		cond = s.ss.visit(node.cond)
		left = None
		right = None
		if(isinstance(node.tret, AST.TransRetIfNode)):
			left = self.visitTransRetIf(node.tret, s)
		else:
			left = self.visitTransRetBasic(node.tret, s)

		if(isinstance(node.fret, AST.TransRetIfNode)):
			right = self.visitTransRetIf(node.fret, s)
		else:
			right = self.visitTransRetBasic(node.fret, s)

		return IF(cond, left, right)

	def visitSeq(self, node: AST.SeqNode):
		r1 = self.visit(node.stmt1)
		r2 = self.visit(node.stmt2)
		return r2

	def visitProg(self, node):
		self.visit(node.shape)
		return self.visit(node.stmt)

