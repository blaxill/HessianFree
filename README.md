# Deep learning via Hessian-free optimization

(Multi) GPU Implementation of http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf

Trains a neural net using the 2nd order Hessian Free method. 

# Status of project

The OpenCL kernel for the backward pass is currently an unnecessary bottle-neck. Also, contrastive divergence, dropout and double support are disabled. The code also needs a clean up.
