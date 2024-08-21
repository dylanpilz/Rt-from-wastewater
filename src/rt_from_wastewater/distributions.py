import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random


def shedding_load_dist(mu1, sd1, mu2, sd2, ng):
    """
    Simulate the distribution of the sum of two gamma distributions
    """

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
    x1 = np.array([0] + list(np.arange(0, ng - 1) + 1.5))
    x2 = np.array([1.5] + list(np.arange(0, ng - 1) + 2.5))

    # Calculate probabilities
    prob = jnp.array([jnp.mean((temp > x1[i]) & (temp <= x2[i])) for i in range(ng)])
    prob = prob / jnp.sum(prob)

    return prob


def discrete_serial_interval(k, mu=2.8, sigma=1.5):
    """
    Discretized serial interval distribution
    https://rdrr.io/cran/EpiEstim/src/R/discr_si.R
    """
    #' @param k Positive integer, or vector of positive integers for which the
    #' discrete distribution is desired.
    #' @param mu A positive real giving the mean of the Gamma distribution.
    #' @param sigma A non-negative real giving the standard deviation of the Gamma
    #' distribution.
    #' @return Gives the discrete probability \eqn{w_k} that the serial interval is
    #' equal to \eqn{k}.

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if mu <= 1:
        raise ValueError("mu must be greater than 1")
    if any(k < 0):
        raise ValueError("k must be non-negative")

    a = ((mu - 1) / sigma) ** 2
    b = sigma**2 / (mu - 1)

    cdf_gamma = lambda k, a, b: dist.Gamma(a, b).cdf(k)

    res = (
        k * cdf_gamma(k, a, b)
        + (k - 2) * cdf_gamma(k - 2, a, b)
        - 2 * (k - 1) * cdf_gamma(k - 1, a, b)
    )

    res = res + a * b * (
        2 * cdf_gamma(k - 1, a + 1, b)
        - cdf_gamma(k - 2, a + 1, b)
        - cdf_gamma(k, a + 1, b)
    )

    res = res.apply(lambda x: max(0, x))

    pass


def main():
    mu1 = 1
    sd1 = 0.5
    mu2 = 2
    sd2 = 1
    ng = 10
    prob = shedding_load_dist(mu1, sd1, mu2, sd2, ng)
    print(prob)


if __name__ == "__main__":
    main()
