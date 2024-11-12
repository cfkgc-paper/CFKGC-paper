import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.tasks = dataset[step + '_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        if parameter['is_shuffle']:
            random.shuffle(self.all_rels)

        self.curr_rel_idx = 0
        self.num_rels = len(self.all_rels)

        # base class settings
        self.bfew = parameter['base_classes_few']
        self.bnq = parameter['base_classes_num_query']
        self.br = parameter['base_classes_relation']

        # novel class settings
        self.few = parameter['few']
        self.nq = parameter['num_query']
        self.bs = parameter['batch_size']

        if step == 'fw_dev':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0
        if step == 'dev':
            self.eval_triples = []
            self.tasks_relations_num = []

    def next_one(self, is_base):
        few = self.bfew if is_base else self.few
        nq = self.bnq if is_base else self.nq

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        curr_cand = self.rel2candidates[curr_rel]

        # while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
        #     curr_rel = self.all_rels[self.curr_rel_idx]
        #     self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
        #     curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks 
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, few + nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[few:]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            support_negative_triples.append([e1, rel, negative])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

        # shift current relation idx to next
        self.curr_rel_idx = self.curr_rel_idx + 1

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    def next_batch(self, is_last, is_base):
        last_rel_idx = self.curr_rel_idx
        rel_num = self.br if is_base else self.bs
        next_batch_all = [self.next_one(is_base) for _ in range(rel_num)]

        self.curr_rel_idx = self.curr_rel_idx if is_last is True else last_rel_idx
        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)

        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel
