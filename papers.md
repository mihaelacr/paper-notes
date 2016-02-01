
# Unitary evolution recurrent neural networks
  * http://arxiv.org/abs/1511.06464
  * To avoid exploding and vanishing gradients, we need to use a matrix for recurrent corrections with eigen values 1 => we parametrize the matrix so that we ensure it is unitary
  * parametrize matrices with real entries is difficult and either results in a high computational complexity or too simple parametrization (use a matrix with non zero entries for the diagonal as a matrix for recurrent connections)
  * instead of working in real space, let's work in complex space => easier to find a definition for the matrix of the recurrent connections that is unitary in the complex space. This is computationally feasible and captures enough complexity for learning
  * memory & cost increase almost linearly with the size of the hidden layer
  * new activation function, relu for complex numbers
  * good results for Rnns example tasks (binary addition, etc), but not for a real life example