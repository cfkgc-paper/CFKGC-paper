from embedding import *
from collections import OrderedDict
import torch
from network.subnet import SubnetLinear, EntityMask


class PERelationMetaLearner(nn.Module):
    def __init__(self, few, base_relation, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5,
                 sparsity=0.5,  novel_relation=3):  # TODO: update relation_num
        super(PERelationMetaLearner, self).__init__()
        self.base_mask = EntityMask(base_relation, few, 2 * embed_size)
        self.novel_mask = EntityMask(novel_relation, few, 2 * embed_size)
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.fc1 = SubnetLinear(2 * embed_size, num_hidden1, sparsity=sparsity, bias=False)
        self.rel_fc1 = nn.Sequential(OrderedDict([
            # ('bn', nn.BatchNorm1d(few)),    # TODO: why batchnorm1d with few
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.fc2 = SubnetLinear(num_hidden1, num_hidden2, sparsity=sparsity, bias=False)
        self.rel_fc2 = nn.Sequential(OrderedDict([
            # ('bn', nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            # ('bn', nn.BatchNorm1d(few)),
        ]))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs, mask, mode, epoch, is_base):
        if mask is None:
            mask = self.none_masks
        if epoch == 0 and not is_base:
            self.novel_mask.init_mask_parameters()

        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        # if mode == "train" and is_base:
        #     x = self.base_mask(x)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode, is_base=is_base)
        x = self.rel_fc1(x)
        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc1.bias'], mode=mode, is_base=is_base)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)

    def get_masks(self):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.long)
                task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.long) \
                    if getattr(module, 'bias') is not None else None
        return task_mask


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class PEMetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(PEMetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = PERelationMetaLearner(parameter['few'], parameter['base_classes_relation'], embed_size=50, num_hidden1=250,
                                                          num_hidden2=100, out_size=50, dropout_p=self.dropout_p,
                                                          sparsity=0.5)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = PERelationMetaLearner(parameter['few'], parameter['base_classes_relation'], embed_size=100, num_hidden1=500,
                                                          num_hidden2=200, out_size=100, dropout_p=self.dropout_p,
                                                          sparsity=0.5)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, mode, epoch, is_base, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        few = support.shape[1]  # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative

        rel = self.relation_learner(support, None, mode, epoch, is_base)  # FC
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few + num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]  # TODO: problem!
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                y = torch.ones(p_score.shape[0], 1).to(device)
                # y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta * grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q

        rel_prototype = rel_q.clone()
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score, rel_prototype
