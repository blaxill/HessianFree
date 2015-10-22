from network import DeepBeliefNetwork
from hessian_free import HessianFree
from batch import Batch
from numpy import log, array, float32

def testBasic():
	net_shape = (2,8,1)
	network = DeepBeliefNetwork( net_shape )
	xor_data = [ [0.0,0.0,0.0],
				[1.0,0.0,1.0],
				[0.0,1.0,1.0],
				[1.0,1.0,0.0]]
	xor_data = xor_data + xor_data #8
	xor_data = xor_data + xor_data#16
	#xor_data = xor_data + xor_data#32
	#xor_data = xor_data + xor_data#64
	#xor_data = xor_data + xor_data#128
	#xor_data = xor_data + xor_data#256
	#xor_data = xor_data + xor_data#512
	#xor_data = xor_data + xor_data#1024
	xor_batch = Batch( [array(x).astype(float32) for x in xor_data],2,1)
	learner = HessianFree(network, xor_batch, 2, 8, 4, use_double=False)

	def eval(xyx):
		return (log(1.-xyx[0]) + log(xyx[1]) + log(xyx[2]) + log(1.-xyx[3]))*0.25

	error = 99
	iteration = 1
	while error > 0.1:
		print "Iteration %03i" % iteration, 
		error, steps = learner.optimize_step()
		print " Error: %.6f, Damping: %+02.2f" % (error, log(learner.damping))
		iteration+=1

	print array([ network.predictions(x[:2]) for x in xor_data[:4] ]).flatten(), eval(array([ network.predictions(x[:2]) for x in xor_data[:4] ]).flatten())
	network = learner.network

if __name__ == "__main__":
    testBasic()
