import pickle
from copy import deepcopy

import torch.nn.functional as F
import numpy as np

from network.param_emb_models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging
from torch.autograd import Variable
from utils import NCELoss


random_cache = [([["concept:insect:grubs", "concept:arthropodandotherarthropod", "concept:insect:pests"],
                  ["concept:insect:grubs", "concept:arthropodandotherarthropod",
                      "concept:insect:pests"],
                  ["concept:insect:grubs", "concept:arthropodandotherarthropod", "concept:insect:pests"]],),
                ([["concept:insect:grubs", "concept:arthropodandotherarthropod", "concept:arthropod:paper_wasps"],
                  ["concept:insect:grubs", "concept:arthropodandotherarthropod",
                      "concept:arthropod:paper_wasps"],
                  ["concept:insect:grubs", "concept:arthropodandotherarthropod", "concept:arthropod:paper_wasps"]],),
                ([["concept:insect:earthworms", "concept:arthropodandotherarthropod", "concept:insect:invertebrates"],
                  ["concept:insect:earthworms", "concept:arthropodandotherarthropod",
                      "concept:insect:invertebrates"],
                  ["concept:insect:earthworms", "concept:arthropodandotherarthropod", "concept:insect:invertebrates"]],
                 ),
                ([["concept:insect:earthworms", "concept:arthropodandotherarthropod", "concept:arthropod:copperheads"],
                  ["concept:insect:earthworms", "concept:arthropodandotherarthropod",
                      "concept:arthropod:copperheads"],
                  ["concept:insect:earthworms", "concept:arthropodandotherarthropod", "concept:arthropod:copperheads"]],)]

cache_task = [([["concept:insect:pests", "concept:animalthatfeedoninsect", "concept:insect:insects"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect",
                    "concept:insect:insects"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect", "concept:insect:insects"]],),
              ([["concept:insect:pests", "concept:animalthatfeedoninsect", "concept:invertebrate:soldier"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect",
                    "concept:invertebrate:soldier"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect", "concept:invertebrate:soldier"]],),
              ([["concept:insect:bugs", "concept:animalthatfeedoninsect", "concept:insect:insects"],
                ["concept:insect:bugs", "concept:animalthatfeedoninsect",
                    "concept:insect:insects"],
                ["concept:insect:bugs", "concept:animalthatfeedoninsect", "concept:insect:insects"]],
               ),
              ([["concept:insect:pests", "concept:animalthatfeedoninsect", "concept:invertebrate:derbid_planthopper"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect",
                    "concept:invertebrate:derbid_planthopper"],
                ["concept:insect:pests", "concept:animalthatfeedoninsect",
                 "concept:invertebrate:derbid_planthopper"]],)]

worse_cache = [([["concept:animal:invertebrates001", "concept:animalthatfeedoninsect", "concept:insect:snails"],
                 ["concept:animal:invertebrates001",
                     "concept:animalthatfeedoninsect", "concept:insect:snails"],
                 ["concept:animal:invertebrates001", "concept:animalthatfeedoninsect", "concept:insect:snails"]],),
               ([["concept:animal:invertebrates001", "concept:animalthatfeedoninsect", "concept:invertebrate:soldier"],
                 ["concept:animal:invertebrates001",
                     "concept:animalthatfeedoninsect", "concept:invertebrate:soldier"],
                 ["concept:animal:invertebrates001", "concept:animalthatfeedoninsect",
                  "concept:invertebrate:soldier"]],),
               ([["concept:animal:creatures", "concept:animalthatfeedoninsect", "concept:insect:garden_pests"],
                 ["concept:animal:creatures", "concept:animalthatfeedoninsect",
                     "concept:insect:garden_pests"],
                 ["concept:animal:creatures", "concept:animalthatfeedoninsect", "concept:insect:garden_pests"]],),
               ([["concept:animal:creatures", "concept:animalthatfeedoninsect", "concept:invertebrate:derbid_planthopper"],
                 ["concept:animal:creatures", "concept:animalthatfeedoninsect",
                     "concept:invertebrate:derbid_planthopper"],
                 ["concept:animal:creatures", "concept:animalthatfeedoninsect", "concept:invertebrate:derbid_planthopper"]],)]


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter

        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.fw_dev_data_loader = data_loaders[3]

        self.bfew = parameter['base_classes_few']
        self.bnq = parameter['base_classes_num_query']
        self.br = parameter['base_classes_relation']
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.num_tasks = parameter['num_tasks']

        # model
        self.metaR = PEMetaR(dataset, parameter)
        self.vice_metaR = PEMetaR(dataset, parameter)
        self.lambda_ = parameter['lambda']
        self.nceloss = NCELoss(parameter['temperature'], self.device)
        self.metaR.to(self.device)
        self.vice_metaR.to(self.device)

        # training
        self.device = parameter['device']
        self.optimizer = torch.optim.Adam(
            self.metaR.parameters(), self.learning_rate)

        self.epoch = parameter['epoch']
        self.base_epoch = parameter['base_epoch']
        self.print_epoch = parameter['print_epoch']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.early_NOVEL_stopping_patience = parameter['early_NOVEL_stopping_patience']
        self.eval_epoch = parameter['eval_epoch']
        self.base_eval_epoch = parameter['base_eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']

        self.learning_rate = parameter['learning_rate']

        # dir
        self.state_dir = os.path.join(
            self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(
            self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(
            filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")
        self.csv_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'])

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(
            self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def write_fw_validating_log(self, data, record, task, epoch):
        if epoch + self.eval_epoch >= self.epoch:
            record[0][task, task] = data['MRR']
            record[1][task, task] = data['Hits@10']
            record[2][task, task] = data['Hits@5']
            record[3][task, task] = data['Hits@1']

    def write_cl_validating_log(self, metrics, record, task):
        for i, data in enumerate(metrics):
            record[0][task, i] = data['MRR']
            record[1][task, i] = data['Hits@10']
            record[2][task, i] = data['Hits@5']
            record[3][task, i] = data['Hits@1']

    def logging_fw_training_data(self, data, epoch, task):
        if epoch == self.eval_epoch:
            logging.info(f"Few_Shot_Learning_task {task}")
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_cl_training_data(self, metrics, task):
        logging.info(f"Eval_Continual_Learning_task {task}")
        for i, data in enumerate(metrics):
            logging.info("Task: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                str(i), data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path):
        setname = 'val set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def get_total_mrr(self, arr):
        idx = [i for i in range(arr.shape[0])]
        idx = [idx, idx]
        fw_metric = arr[idx]
        arr[idx] = 0
        cl_metric = arr.sum(axis=1)
        for i, j in enumerate(cl_metric):
            if i != 0:
                cl_metric[i] = j / i

        return np.array([fw_metric, cl_metric]).T

    def save_metrics(self, Hit10_val_mat, Hit1_val_mat, Hit5_val_mat, MRR_val_mat):
        np.savetxt(os.path.join(self.csv_dir, 'MRR.csv'),
                   MRR_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@10.csv'),
                   Hit10_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@5.csv'),
                   Hit5_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@1.csv'),
                   Hit1_val_mat, delimiter=",")
        mrr = self.get_total_mrr(MRR_val_mat)
        np.savetxt(os.path.join(self.csv_dir, 'metric.csv'),
                   mrr, delimiter=",", fmt='%s')

    def triple_rehearsal(self, task, base):
        base_mask = F.sigmoid(self.metaR.relation_learner.base_mask.w_m)
        mask = base_mask.sum(axis=-1).sum(axis=-
                                          1).max() == base_mask.sum(axis=-1).sum(axis=-1)
        idx = (mask > 0).nonzero(as_tuple=True)[0]
        for i in idx:
            for j, cur in enumerate(task):
                # optimal repaly strategy
                task[j] = task[j] + (base[j][i.item()],)
                # train_task[j] = train_task[j] + worse_cache[j] # worse replay
                # train_task[j] = train_task[j] + cache_task[j] # ours
                # train_task[j] = train_task[j] + random_cache[j] # random replay

    def random_vice_param(self):
        for (name, param), (vice_name, vice_param) in zip(self.metaR.named_parameters(), self.vice_metaR.named_parameters()):
            vice_param.data = param.data.clone().detach() + self.lambda_ * torch.normal(0,
                                                                                        torch.ones_like(
                                                                                            param.data.clone().detach()) * param.data.clone().detach().std())

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, consolidated_masks, epoch=None, is_base=None, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()

            # MODULE 3 START: Multi-view Relation Augmentation
            self.random_vice_param()
            _, _, vice_rel = self.vice_metaR(
                task, 'train', epoch, is_base, iseval, curr_rel)
            vice_rel = Variable(
                vice_rel.clone().detach().data, requires_grad=False)

            p_score, n_score, rel = self.metaR(
                task, 'train', epoch, is_base, iseval, curr_rel)

            y = torch.ones(p_score.shape[0], 1).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y) + 0.1 * self.nceloss(vice_rel, rel) if not is_base else self.metaR.loss_func(
                p_score, n_score, y)
            loss.backward()

            # Continual Subnet no backprop
            if consolidated_masks is not None and consolidated_masks != {}:  # Only do this for tasks 1 and beyond
                for key in consolidated_masks.keys():
                    module_name, module_attr = key.split(
                        '.')  # e.g. fc1.weight
                    # Zero-out gradients
                    if hasattr(getattr(self.metaR.relation_learner, module_name), module_attr):
                        if getattr(getattr(self.metaR.relation_learner, module_name), module_attr) is not None:
                            getattr(getattr(self.metaR.relation_learner, module_name), module_attr).grad[
                                consolidated_masks[key] == 1.0] = 0
            self.optimizer.step()

        elif curr_rel != '':
            # TODO: update iseval and mode
            p_score, n_score, _ = self.metaR(task, 'val', iseval, curr_rel)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score
    
    def get_epoch_score(self, curr_rel, data, eval_task, ranks, t, temp):
        _, p_score, n_score = self.do_one_step(
            eval_task, None, iseval=True, curr_rel=curr_rel)
        x = torch.cat([n_score, p_score], 1).squeeze()
        self.rank_predict(data, x, ranks)

        # print current temp data dynamically
        for k in data.keys():
            temp[k] = data[k] / t
        sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
        sys.stdout.flush()
        
    def novel_continual_eval(self, previous_rel, task):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        data_loader = self.dev_data_loader
        current_eval_num = 0

        for rel in previous_rel:
            data_loader.eval_triples.extend(data_loader.tasks[rel][self.few:])
            current_eval_num += len(data_loader.tasks[rel][self.few:])

        data_loader.num_tris = len(data_loader.eval_triples)
        data_loader.tasks_relations_num.append(current_eval_num)
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        tasks_data = []
        ranks = []

        t = 0
        i = 0
        previous_t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()

            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1
            self.get_epoch_score(curr_rel, data, eval_task, ranks, t, temp)

        tasks_data.append(data)

        # print overall evaluation result and return it
        for data in tasks_data:
            for k in data.keys():
                data[k] = round(data[k] / t, 3)

        print('continual learning', tasks_data)
        if self.parameter['step'] == 'train':
            self.logging_cl_training_data(tasks_data, task)
        return tasks_data

    def fw_eval(self, task, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        data_loader = self.fw_dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1
            self.get_epoch_score(curr_rel, data, eval_task, ranks, t, temp)

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_fw_training_data(data, epoch, task)
        else:
            self.logging_eval_data(data, self.state_dict_file)

        print("few shot {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

    def train(self):
        # initialization
        # TODO: recorder
        Hit10_val_mat, Hit1_val_mat, Hit5_val_mat, MRR_val_mat, val_mat = self.init_val_mat()
        per_task_masks, consolidated_masks = {}, {}

        for task in range(self.num_tasks):
            # early stop setting
            best_loss = 100  
            now_waiting = 0
            best_e = 0

            # training by epoch
            epoch = self.base_epoch if task == 0 else self.epoch
            eval_epoch = self.base_eval_epoch if task == 0 else self.eval_epoch

            for e in range(epoch):
                is_last = False if e != epoch - 1 else True
                is_base = True if task == 0 else False
                patience = self.early_stopping_patience if is_base else self.early_NOVEL_stopping_patience

                # sample one batch from data_loader
                train_task, curr_rel = self.train_data_loader.next_batch(
                    is_last, is_base)

                # MODULE 2 START: Triple Rehearsal
                if not is_base:
                    self.triple_rehearsal(train_task, base_task)
                # MODULE 2 END

                loss, _, _ = self.do_one_step(train_task, consolidated_masks, epoch, is_base, iseval=False,
                                              curr_rel=curr_rel)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_e = e
                    now_waiting = 0
                else:
                    now_waiting += 1

                # print the loss on specific epoch
                if e % self.print_epoch == 0:
                    loss_num = loss.item()
                    print("Epoch: {}\tLoss: {:.4f} {}, sofar best epo is : {}".format(e, loss_num,
                                                                                      self.train_data_loader.curr_rel_idx,
                                                                                      best_e))

                if now_waiting > patience:
                    print(
                        f"stop at {e} for {patience} epoches, loss hasn't been better.")
                    print(f"best loss is {best_loss}, best epoch is {best_e}")
                    valid_data = self.fw_eval(task, epoch=e)  # few shot val
                    self.write_fw_validating_log(valid_data, val_mat, task, e)
                    valid_data = self.novel_continual_eval(curr_rel, task)
                    self.write_cl_validating_log(valid_data, val_mat, task)
                    break

                # save checkpoint on specific epoch
                if e % self.checkpoint_epoch == 0 and e != 0:
                    print('Epoch  {} has finished, saving...'.format(e))
                    self.save_checkpoint(e)

                # do evaluation on specific epoch
                if e % eval_epoch == 0 and e != 0:
                    print('Epoch  {} has finished, validating few shot...'.format(e))
                    valid_data = self.fw_eval(task, epoch=e)  # few shot eval
                    self.write_fw_validating_log(valid_data, val_mat, task, e)

                if e == eval_epoch - 1:
                    print(
                        'Epoch  {} has finished, validating continual learning...'.format(e))
                    valid_data = self.novel_continual_eval(
                        curr_rel, task)  # continual learning eval
                    self.write_cl_validating_log(valid_data, val_mat, task)

            base_task = train_task if is_base else base_task

            # MODULE 2 START: Meta-learner Modulation
            # Consolidate task masks to keep track of parameters to-update or not
            per_task_masks[task] = self.metaR.relation_learner.get_masks()
            if task == 0:
                consolidated_masks = deepcopy(per_task_masks[task])
            else:
                for key in per_task_masks[task].keys():
                    # Operation on sparsity
                    if consolidated_masks[key] is not None and per_task_masks[task][key] is not None:
                        consolidated_masks[key] = 1 - (
                            (1 - consolidated_masks[key]) * (1 - per_task_masks[task][key]))

        self.save_metrics(Hit10_val_mat, Hit1_val_mat,
                          Hit5_val_mat, MRR_val_mat)

        self.save_val_mat(Hit10_val_mat, Hit1_val_mat,
                          Hit5_val_mat, MRR_val_mat)
        print('Training has finished')

    def save_val_mat(self, Hit10_val_mat, Hit1_val_mat, Hit5_val_mat, MRR_val_mat):
        np.savetxt(os.path.join(self.csv_dir, 'MRR.csv'),
                   MRR_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@10.csv'),
                   Hit10_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@5.csv'),
                   Hit5_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@1.csv'),
                   Hit1_val_mat, delimiter=",")

        idx = [i for i in range(MRR_val_mat.shape[0])]
        idx = [idx, idx]
        fw_metric = MRR_val_mat[tuple(idx)]
        MRR_val_mat[tuple(idx)] = 0
        cl_metric = MRR_val_mat.sum(axis=1)
        for i, j in enumerate(cl_metric):
            if i != 0:
                cl_metric[i] = j / i
        metric = np.array([fw_metric, cl_metric]).T
        np.savetxt(os.path.join(self.csv_dir, 'metric.csv'),
                   metric, delimiter=",")

    def init_val_mat(self):
        # record fw and cl vl MRR metrics
        MRR_val_mat = np.zeros((self.num_tasks, self.num_tasks))
        # record fw and cl vl MRR metrics
        Hit1_val_mat = np.zeros((self.num_tasks, self.num_tasks))
        # record fw and cl vl MRR metrics
        Hit5_val_mat = np.zeros((self.num_tasks, self.num_tasks))
        # record fw and cl vl MRR metrics
        Hit10_val_mat = np.zeros((self.num_tasks, self.num_tasks))
        val_mat = [MRR_val_mat, Hit10_val_mat, Hit5_val_mat, Hit1_val_mat]
        return Hit10_val_mat, Hit1_val_mat, Hit5_val_mat, MRR_val_mat, val_mat

    def test_train_relation_support_query(self, is_base, train_task, epoch):
        print(
            f"Assert relation num {len(train_task[0])} few num {len(train_task[0][0])} "
            f"query num {len(train_task[2][0])}")
        if is_base:
            assert len(train_task[0]) == self.br
            assert len(train_task[0][0]) == self.bfew
            assert len(train_task[2][0]) == self.bnq
            assert self.train_data_loader.curr_rel_idx == 0 if epoch != self.base_epoch - 1 else self.br
        else:
            assert len(train_task[0]) == self.batch_size + 1  # replay one node
            assert len(train_task[0][0]) == self.few
            assert len(train_task[2][0]) == self.num_query
            assert self.train_data_loader.curr_rel_idx != 0 if epoch != self.epoch - 1 else 51
            print(f'Test idx {self.train_data_loader.curr_rel_idx}')

