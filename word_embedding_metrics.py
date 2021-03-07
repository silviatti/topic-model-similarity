from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine

from utils.word_embeddings_rbo import word_embeddings_rbo as wer
from utils.word_embeddings_rbo_centroid import word_embeddings_rbo as werc


def werbo_m(topics, word_embedding_model, weight=0.9, topk=10):
    """
    computes Word embedding based RBO

    Parameters
    ----------
    topics: a list of lists of words
    word_embedding_model: word embedding space in gensim word2vec format
    weight: p (float), default 1.0: Weight of each agreement at depth d:
    p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
    topk: top k words on which the topic diversity will be computed
    """

    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        collect = []
        for list1, list2 in combinations(topics, 2):
            word2index = get_word2index(list1, list2)
            index2word = {v: k for k, v in word2index.items()}
            indexed_list1 = [word2index[word] for word in list1]
            indexed_list2 = [word2index[word] for word in list2]
            rbo_val = wer(indexed_list1[:topk], indexed_list2[:topk], p=weight,
                          index2word=index2word, word2vec=word_embedding_model,
                          norm=False)[2]
            collect.append(rbo_val)
        return np.mean(collect)


def werbo_c(topics, word_embedding_model, weight=0.9, topk=10):
    """
    computes Word embedding based RBO - centroid

    Parameters
    ----------
    topics: a list of lists of words
    word_embedding_model: word embedding space in gensim word2vec format
    weight: p (float), default 1.0: Weight of each agreement at depth d:
    p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
    topk: top k words on which the topic diversity will be computed
    """

    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        collect = []
        for list1, list2 in combinations(topics, 2):
            word2index = get_word2index(list1, list2)
            index2word = {v: k for k, v in word2index.items()}
            indexed_list1 = [word2index[word] for word in list1]
            indexed_list2 = [word2index[word] for word in list2]
            rbo_val = werc(indexed_list1[:topk], indexed_list2[:topk], p=weight,
                           index2word=index2word, word2vec=word_embedding_model, norm=False)[2]
            collect.append(rbo_val)
        return np.mean(collect)


def we_pairwise_similarity(topics, wv, topk=10):
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        count = 0
        sum_sim = 0
        for list1, list2 in combinations(topics, 2):
            word_counts = 0
            sim = 0
            for word1 in list1[:topk]:
                for word2 in list2[:topk]:
                    if word1 in wv.wv.vocab and word2 in wv.wv.vocab:
                        sim = sim + wv.similarity(word1, word2)
                        word_counts = word_counts + 1
            sim = sim / word_counts
            sum_sim = sum_sim + sim
            count = count + 1
        return sum_sim / count


def we_centroid_similarity(topics, wv, topk=10):
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        sim = 0
        count = 0
        for list1, list2 in combinations(topics, 2):
            centroid1 = np.zeros(wv.vector_size)
            centroid2 = np.zeros(wv.vector_size)
            count1, count2 = 0, 0
            for word1 in list1[:topk]:
                if word1 in wv.wv.vocab:
                    centroid1 = centroid1 + wv[word1]
                    count1 += 1
            for word2 in list2[:topk]:
                if word2 in wv.wv.vocab:
                    centroid2 = centroid2 + wv[word2]
                    count2 += 1
            centroid1 = centroid1 / count1
            centroid2 = centroid2 / count2
            sim = sim + (1 - cosine(centroid1, centroid2))
            count += 1
        return sim / count


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index
