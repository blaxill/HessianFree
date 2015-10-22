import numpy
import numpy.random
import time
import numpy.linalg
from batch import *
from network import *
from opencl_learner import *
import sets
import math
import sys

verbose = False

class HessianFree:
	def __init__(self, network, data,
                batch_size, gradient_compute_batches, loss_compute_batches,
                use_double=False, damping=1.0):

            platform = cl.get_platforms()[0]
            self.devices = platform.get_devices()
            if verbose:
                print "devices: ", self.devices
            self.contexts = [None] * len(self.devices)
            self.gn = [None] * len(self.devices)

            self.gradient_compute_batches = gradient_compute_batches
            self.loss_compute_batches = loss_compute_batches
            self.data = data
            self.batch_size = batch_size

            self.network = network
            self.damping = damping
            self.x0 = None
            self.b = None

            self.initial_loss = 0

            for i, device in enumerate(self.devices):
                    self.contexts[i] = cl.Context(devices=[device])
                    self.gn[i] = GaussNewton(self.contexts[i],network,batch_size,"opencl/", use_double=use_double)
                    #self.gn[i].adjust_for_batch_size(batch_size)
 
	def minimize(self): #(x0,b,network, multipler, damping):
		if self.x0 is None:
			self.x0 = numpy.zeros_like(self.b)

		def func(v):
			for i, gn in enumerate(self.gn):
				w = v.astype(gn.float_type)#,copy=False)
				gn.update_R_weights(gpu_net_array=w)
				gn.gauss_product()
				
			result = [ gn.read_result_weights( self.network.array_like(dtype=gn.float_type) ) for gn in self.gn ]

			return  numpy.mean(result,axis=0, dtype=numpy.float64) + self.damping*v
		
		batches = self.data.sample_split( self.batch_size * len(self.devices), len(self.devices) )
		for i, gn in enumerate(self.gn):
			gn.load_batch( batches[i] )
		for i, gn in enumerate(self.gn):
			gn.forward_pass() # Load states

		x = self.x0
		r = self.b - func(x)
		z = r#*M_inv
		p = z

		last = numpy.inf
		memory = [] #Results for cg-iteration backtrack
		setters = [ zz for zz in range(250) ]
		setters = sets.Set([ int(math.ceil(1.3**zz)) for zz in setters ])
		phi_mem = []
		backspaces=0

		for i in range(1, 250): # Max iterations
			Ap = func(p)
			pAp =  numpy.dot(p,Ap)
			if pAp == 0.:
				phi = 0.5 * numpy.dot(x, func(x)) - numpy.dot(self.b,x)
				if len(memory) ==0 or memory[-1][0] != phi:
					memory.append( (phi, x) )
				break
			rr = numpy.dot(r,z)
			alpha = rr / pAp

			x = x + alpha * p
			last_rr = rr
			r = r - alpha * Ap
			z = r#*M_inv

			phi = 0.5 * numpy.dot(x, func(x)) - numpy.dot(self.b,x)

			if i in setters:
				memory.append( (phi, x) )

			if phi < 0:
				phi_mem.append(phi)
				number_needed = int(max(10, i*0.1))
				if len(phi_mem) > number_needed and (phi - phi_mem[-1-number_needed])/phi < 0.0005*number_needed:
					if i not in setters:
						memory.append( (phi, x) )
					break
			else:
				phi_mem = []

			last = phi

			if last_rr == 0.0:
				if i not in setters:
					memory.append( (phi, x) )
				break

			beta = numpy.dot(z,r) / last_rr
			p = z + beta*p

			progress = ' [CG iter %03i, phi=%+.6f]' % (i, phi)
			sys.stdout.write('\b'*backspaces + progress)
			sys.stdout.flush()
			backspaces = len(progress)


		self.x0 = x*0.95 #Use current decayed direction for next start (unless lower loss is not found in line search)

		return memory, i

	def compute_gradient(self):
		#Set b = -grad
		#Set inital loss
		if verbose: print "Computing gradient..."
		loss_calc = []
		grad = self.network.array_like(dtype=numpy.float64)

		maxer = self.gradient_compute_batches / (self.batch_size * len(self.devices))
		mul = float(self.batch_size) /  self.gradient_compute_batches

		compute_batches = self.data.sample_split( self.batch_size * len(self.devices) * maxer,
			len(self.devices) * maxer )

		for j in xrange(maxer):
			grad_calc = [self.network.array_like(dtype=gn.float_type) for gn in self.gn]
			e = [None] * len(self.gn)
			for i, gn in enumerate(self.gn):
				gn.gradient(compute_batches[j*len(self.devices) + i])
			for i, gn in enumerate(self.gn):
				loss_calc+=[ gn.read_back_loss() ]
				cl.enqueue_copy(gn.queue, grad_calc[i], gn.result_weights, is_blocking=True)
				#print grad_calc[i]
				#gn.read_weights(grad_calc[i])
				#self.network.from_array(grad_calc[i],dtype=gn.float_type)
				#bi = compute_batches[j*len(self.devices) + i]
				#print numpy.array(numpy.mean(					[ self.network.gradient(x[:2],x[2:]) for x in bi.buffers ],axis=0)).flatten() 

			grad+=numpy.mean(grad_calc,axis=0, dtype=numpy.float64)
			if verbose: print(float((j+1)*100.)/maxer),"%"

	
		self.initial_loss = numpy.mean(loss_calc,axis=0, dtype=numpy.float64)
		#print "Inital loss ", self.initial_loss
		self.b =  (-grad) / maxer

	def compute_loss(self):
		if verbose: print "Computing loss..."
		loss_calc = []

		maxer = self.loss_compute_batches / (self.batch_size * len(self.devices))
		mul = float(self.batch_size) /  self.loss_compute_batches

		compute_batches = self.data.sample_split( self.batch_size * len(self.devices) * maxer,
				len(self.devices) * maxer )

		for j in xrange(maxer):
			e = [None] * len(self.gn)
			for i, gn in enumerate(self.gn):
				gn.load_forward_and_loss(compute_batches[j*len(self.gn) + i])
			for i, gn in enumerate(self.gn):
				loss_calc+=[ gn.read_back_loss() ]
			if verbose: print(float((j+1)*100.)/maxer),"%"

		loss = numpy.mean(loss_calc,axis=0, dtype=numpy.float64) 
		if verbose: print  "Loss ", loss
		if numpy.isnan(loss):
			loss = numpy.inf
		return loss

	def optimize_step(self):
		def upload_network(gpu_net_array):
			#if self.network is None:
			#	self.network = self.network
			#network_array = network.to_array(order='F')
			for i, gn in enumerate(self.gn):
				w = gpu_net_array.astype(gn.float_type)#,copy=False)
				gn.update_weights(gpu_net_array=w)

		network_array = self.network.to_array(order='F')
		upload_network(network_array)

		self.compute_gradient()# This loads gradient and loss over a large batch

		ps,cg_iterations = self.minimize()#(next_direction,-grad,network,multipler,damping)

		#Cg backtrack
		divisor,delta = ps.pop()

		upload_network(network_array+delta)
		cg_loss = self.compute_loss()

		for cg_phi,cg_x in ps:
			if numpy.array_equal(cg_x,delta): continue

			upload_network(network_array+cg_x)
			temp = self.compute_loss()

			if temp < cg_loss:
				delta = cg_x
				divisor = cg_phi
				cg_loss = temp
				#print("CG back track successful")
			else:
				break

		#Line search
		if cg_loss >= self.initial_loss:
			#print "Entering line search!"
			found = False
			for i,e in enumerate( [0.8**x for x in range(1,30)] ):
				#print i, 
				upload_network(network_array+delta*e)
				temp = self.compute_loss()
				if temp < self.initial_loss:
					found = True
					cg_loss = temp
					delta = delta*e
					break
			if not found:
				self.x0 = None
				#print("CG direction providing a loss result!")
				#print " CG loss causing reset.",
			else:
				self.network.from_array(network_array + delta)
				#print
		else:
			self.network.from_array(network_array + delta)
			#print

		rho = cg_loss - self.initial_loss
		if divisor == 0:
			rho=-numpy.inf
		else:
			rho /=divisor
		decr  = 2./3
		incr = 3./2

		if rho < 1/4. or numpy.isnan(rho): # the reductino was bad, rho was small
			d = incr
		elif rho > 3/4.:                # the reduction was good since rho was large
			d = decr
		else: 
			d = 1.

		self.damping = self.damping*d

		return cg_loss, cg_iterations
