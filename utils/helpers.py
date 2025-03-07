import numpy as np

def logistic_probability(score_diff, beta=0.1):
    return 1 / (1 + np.exp(-beta * score_diff))