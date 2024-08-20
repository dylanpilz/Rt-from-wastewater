import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def wastewater_model(N0, N, n_SI, SI, n_v, v, M, viral, viral_obs, EpidemicStart, nsd, sampleday):
    
    # Priors
    tau = numpyro.sample('tau', dist.Exponential(0.03))
    y = numpyro.sample('y', dist.Exponential(1.0 / tau))
    kappa = numpyro.sample('kappa', dist.Normal(0, 1))
    R0 = numpyro.sample('R0', dist.Normal(2, 1))
    sigma = numpyro.sample('sigma', dist.Gamma(0.01, 0.01))
    nu0 = numpyro.sample('nu0', dist.Gamma(0.01, 0.01))
    nu_sigma = numpyro.sample('nu_sigma', dist.Gamma(0.01, 0.01))
    nu = numpyro.sample('nu', dist.Normal(nu0, nu_sigma).expand([N]))

    # Initial Rt
    Rt = numpyro.sample('Rt', dist.Normal(R0, kappa).expand([N]))
    
    # Expected number of infections on day t
    prediction = jnp.zeros(N)
    prediction = prediction.at[:N0].set(y)

    for i in range(N0, N):
        convolution = jnp.sum(prediction[i-j-1] * SI[j-1] for j in range(1, min(n_SI, i-1) + 1))
        prediction = prediction.at[i].set(Rt[i] * convolution)

    # E_viral calculation
    E_viral = jnp.zeros(N)
    E_viral = E_viral.at[0].set(1e-9)

    for i in range(1, N):
        E_viral = E_viral.at[i].set(
            jnp.sum(prediction[i-j-1] * v[j-1] for j in range(1, min(n_v, i-1) + 1))
        )

    # Likelihood for observed viral load
    with numpyro.plate('obs', nsd):
        numpyro.sample('obs', dist.Normal(jnp.log(E_viral[sampleday]), nu[sampleday]), obs=viral[sampleday, :viral_obs[sampleday]])

    # Update Rt for t > 1
    for t in range(1, N):
        Rt = Rt.at[t].set(numpyro.sample(f'Rt_{t}', dist.LogNormal(jnp.log(Rt[t-1]), sigma)))
