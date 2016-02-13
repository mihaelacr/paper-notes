
# Unitary evolution recurrent neural networks
  * http://arxiv.org/abs/1511.06464
  * Motivation: avoid exploding / vanishing gradient in RNNs
  * To avoid exploding and vanishing gradients, we need to use a matrix for recurrent corrections with eigen values 1 => we parametrize the matrix so that we ensure it is unitary
  * parametrize matrices with real entries is difficult and either results in a high computational complexity or too simple parametrization (use a matrix with non zero entries for the diagonal as a matrix for recurrent connections)
  * instead of working in real space, let's work in complex space => easier to find a definition for the matrix of the recurrent connections that is unitary in the complex space. This is computationally feasible and captures enough complexity for learning
  * memory & cost increase almost linearly with the size of the hidden layer
  * new activation function, relu for complex numbers
  * good results for Rnns example tasks (binary addition, etc), but not for a real life example


# Generative adversarial networks
  * http://arxiv.org/abs/1406.2661
  * Motivation: generate good samples
  * Used for: start of the art image generation with Laplacian pyramids of GANs
  * min-max game:
     * Generator (G) tries to generate good samples
     * Discriminator (D) tries to learn which samples come from the true distribution of the input and which samples come from G
     * at the end of training you hope that D cannot distinguish between real samples and samples from G
  * loss: = E_(x∼pdata(x))[log D(x)] + E_(z∼pz(z))[log(1 − D(G(z)))]
  * no need for MCMC and approximations
  * general framework, D&G not bound to be neural nets
  * training tricks:
      * update D k times for each update of G
      * min log(1-D(G(z))) does poorly at the begging of training because D has an advantage (easy to distinguish between samples from G and samples from the real distribution), so instead max log(D(G(z))), which has a stronger gradient
  * Theoretic result: the optimal distribution for D is: D*(G(x)) = p_data(x) / (p_data(x) + p_G(x))
