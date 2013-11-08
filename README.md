# Deep learning via Hessian-free optimization

GPU Implementation of http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf

Trains a neural net using the 2nd order Hessian Free method. Most common methods 
are first order methods that have rapidly decreasing derivatives on deep neural nets.
Other recent techniques i.e. RBM's work by learning layer by layer and finally 
performing full layer optimization. Hessian Free method can be applied to the full
net from an initialization state.

# Status of project

The OpenCL kernel for the backward pass is currently non-optimal in the way it performs
batching. Also, contrastive divergence pre-learning is currently not operational
(but not required!)