import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random

def combine_gamma(mu1, sd1, mu2, sd2, ng):
    '''
    Simulate the distribution of the sum of two gamma distributions
    '''
    key = random.PRNGKey(0)
    
    # Calculate parameters for the gamma distributions
    p1 = mu1**2 / sd1**2
    p2 = mu1 / sd1**2
    p3 = mu2**2 / sd2**2
    p4 = mu2 / sd2**2

    # Simulate using gamma distribution
    ng_sim = int(1e5)
    gamma1 = dist.Gamma(p1, p2).sample(key, (ng_sim,))
    gamma2 = dist.Gamma(p3, p4).sample(key, (ng_sim,))
    temp = gamma1 + gamma2

    # Define x1 and x2
    x1 = np.array([0] + list(np.arange(0, ng-1) + 1.5))
    x2 = np.array([1.5] + list(np.arange(0, ng-1) + 2.5))

    # Calculate probabilities
    prob = jnp.array([jnp.mean((temp > x1[i]) & (temp <= x2[i])) for i in range(ng)])
    prob = prob / jnp.sum(prob)

    return prob

def main():
    mu1 = 1
    sd1 = 0.5
    mu2 = 2
    sd2 = 1
    ng = 10
    prob = combine_gamma(mu1, sd1, mu2, sd2, ng)
    print(prob)

if __name__ == '__main__':
    main()
