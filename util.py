import numpy as np
import math
import dill


def save(data, path):
    output = open(path, "wb")
    dill.dump(data, output)
    output.close()


def load(path):
    tmp = open(path, 'rb')
    data = dill.load(tmp)
    tmp.close()
    return data


def getHitK(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return hit


def getNdcgK(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    ndcg = np.mean(ndcgs)
    return ndcg