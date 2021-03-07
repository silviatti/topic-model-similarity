import numpy as np
from utils.rbo import rbo as rbo_utils
from itertools import combinations


def proportion_common_words(topics, topk=10):
    """
    compute proportion of unique words

    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed

    Returns
    -------
    pcw : proportion of common words
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = 1 - (len(unique_words) / (topk * len(topics)))
        return puw


def rbo(topics, weight=0.9, topk=10):
    """
    compute rank-biased overlap

    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity
          will be computed
    weight: p (float), default 1.0: Weight of each
            agreement at depth d:p**(d-1). When set
            to 1.0, there is no weight, the rbo returns
            to average overlap.
    Returns
    -------
    rbo : score of the rank biased overlap over the topics
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        collect = []
        for list1, list2 in combinations(topics, 2):
            word2index = get_word2index(list1, list2)
            indexed_list1 = [word2index[word] for word in list1]
            indexed_list2 = [word2index[word] for word in list2]
            rbo_val = rbo_utils(indexed_list1[:topk], indexed_list2[:topk], p=weight)[2]
            collect.append(rbo_val)
        return np.mean(collect)


def pairwise_jaccard_similarity(topics, topk=10):
    sim = 0
    count = 0
    for list1, list2 in combinations(topics, 2):
        intersection = len(list(set(list1[:topk]).intersection(list2[:topk])))
        union = (len(list1[:topk]) + len(list2[:topk])) - intersection
        count = count + 1
        sim = sim + (float(intersection) / union)
    return sim/count


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index
