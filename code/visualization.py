import matplotlib.pyplot as plt
import arviz as az
import numpy as np


def plot_competence(trace: az.InferenceData) -> plt.Figure:
    """Plot competence distributions - initial code by me, improved by AI"""
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(trace, var_names=["competence"], ax=ax)
    ax.set_title("Informant Competence Distributions")
    plt.close(fig)
    return fig


def plot_consensus(trace: az.InferenceData) -> plt.Figure:
    """Plot consensus answer probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(trace, var_names=["consensus"], ax=ax)
    ax.set_title("Consensus Answer Probabilities")
    plt.close(fig)
    return fig