import json
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx import pagerank
from collections import Counter
import torch
import numpy as np

def subgraph_pagerank(train, ent2id):
    ranks = []

    for k, v in train.items():
        edge = []
        for triple in v:
            edge.append([ent2id[triple[0]], ent2id[triple[2]]])
        data = to_networkx(Data(edge_index=torch.LongTensor(edge).t().contiguous()))
        rank = pagerank(data)
        ranks.append(rank)

    head_page_rank = np.zeros((len(ent2id), len(train)))
    for i, rank in enumerate(ranks):
        for k, v in rank.items():
            head_page_rank[k, i] = v

    return head_page_rank

def rel_rank(train, ent2id):
    head = {}
    head_num = {}

    # get head id
    for k, v in train.items():
        head[k] = []
        for triple in v:
            head[k].append(ent2id[triple[0]])

    # count head nums for each relation
    for k, v in head.items():
        count = dict(Counter(v))
        head_num[k] = sorted(count.items(), key=lambda x: x[1], reverse=True)

    head_count = np.zeros((len(ent2id), len(train)))
    for i, (k, v) in enumerate(head_num.items()):
        for count in v:
            head_count[count[0], i] += count[1]
    return head_count > 0

def entity_score(head, rel, alpha):
    return np.power(rel.sum(-1), alpha) * np.power(head.sum(-1), 1 - alpha)

if __name__ == "__main__":
    dataset = 'NELL'
    train = json.load(open(f'{dataset}/train_tasks.json'))
    ent2id = json.load(open(f'{dataset}/ent2ids'))

    head_rank = subgraph_pagerank(train, ent2id)
    head_rel = rel_rank(train, ent2id)
    
    head_rank = head_rel * head_rank
    head_score = entity_score(head_rank, head_rel, 0.5)
    print(np.argwhere(head_rank == head_rank.max()))   # get the highest score entity
    




