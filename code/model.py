import pymc as pm
import numpy as np


def create_model(data: np.ndarray) -> pm.Model:
    """Create CCT model with informed priors. Created with help from AI"""
    n_informants, n_items = data.shape
   
    with pm.Model() as model:
        competence = pm.Uniform(
            "competence",
            lower=0.5,
            upper=1,
            shape=n_informants
        )
       
        consensus = pm.Bernoulli(
            "consensus",
            p=0.5,
            shape=n_items
        )
       
        prob = pm.math.switch(
            consensus,
            competence[:, None],
            1 - competence[:, None]
        )
       
        pm.Bernoulli(
            "observations",
            p=prob,
            observed=data
        )
   
    return model


def sample_model(model: pm.Model) -> pm.backends.base.MultiTrace:
    """Perform MCMC sampling with configured settings"""
    with model:
        return pm.sample(
            draws=2000,
        )