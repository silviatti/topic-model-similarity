import pickle as pkl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import gensim.downloader as api
from scipy.spatial.distance import cosine


def _KL(P, Q):
    """
    Perform Kullback-Leibler divergence

    Parameters
    ----------
    P : distribution P
    Q : distribution Q

    Returns
    -------
    divergence : divergence from Q to P
    """
    # add epsilon to grant absolute continuity
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


def _LOR(P, Q):
    lor = 0
    for v, w in zip(P, Q):
        if v > 0 or w > 0:
            lor = lor + np.abs(np.log(v) - np.log(w))
    return lor/len(P)


def kl_divergence(beta):
    kl_div = 0
    count = 0
    for i, j in combinations(range(len(beta)), 2):
        kl_div += _KL(beta[i], beta[j])
        count += 1
    return kl_div/count


def log_odds_ratio(beta):
    lor = 0
    count = 0
    for i, j in combinations(range(len(beta)), 2):
        lor += _KL(beta[i], beta[j])
        count += 1
    return lor/count


def we_weighted_sum_similarity(beta, id2word, wv):
    wess = 0
    count = 0
    for i, j in combinations(range(len(beta)), 2):
        centroid1 = np.zeros(wv.vector_size)
        weights = 0
        for id_beta, w in enumerate(beta[i]):
            centroid1 = centroid1 + wv[id2word[id_beta]] * w
            weights += w
        centroid1 = centroid1 / weights
        centroid2 = np.zeros(wv.vector_size)
        weights = 0
        for id_beta, w in enumerate(beta[i]):
            centroid2 = centroid2 + wv[id2word[id_beta]] * w
            weights += w
        centroid2 = centroid2 / weights
        wess += cosine(centroid1, centroid2)
    return wess/count
