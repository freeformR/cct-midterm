import numpy as np
import arviz as az


def get_competence_stats(trace: az.InferenceData) -> dict:
    """Calculate competence statistics - code vastly improved by AI, initial structure by me!"""
    means = trace.posterior["competence"].mean(("chain", "draw")).values
    return {
        "means": means,
        "most_competent": np.argmax(means),
        "least_competent": np.argmin(means)
    }


def compare_consensus_methods(trace: az.InferenceData, data: np.ndarray) -> dict:
    """Compare CCT consensus with simple majority vote. code improved by AI but original implementation by me!"""
    # Get CCT consensus probabilities
    cct_probs = trace.posterior["consensus"].mean(("chain", "draw")).values
   
    # Calculate both consensus methods
    cct_consensus = (cct_probs > 0.5).astype(int)
    majority_vote = (data.mean(axis=0) > 0.5).astype(int)
   
    # Find disagreements using proper element-wise comparison
    disagreements = np.where(cct_consensus != majority_vote)[0]
   
    return {
        "cct_consensus": cct_consensus,
        "majority_vote": majority_vote,
        "disagreements": disagreements
    }
