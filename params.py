import torch
import argparse

BEPO = 9999999
EPO = 22222


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-seed", "--seed", default=42, type=int)  
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    
    # dataset setting
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="./NELL", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-if", "--is_shuffle", default=True, type=bool)  # whether shuffle the training relations
    
    # dataloader setting
    args.add_argument("-bfew", "--base_classes_few", default=3, type=int)  # base support num
    args.add_argument("-bnq", "--base_classes_num_query", default=3, type=int)
    args.add_argument("-few", "--few", default=3, type=int) # novel support num
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-br", "--base_classes_relation", default=30, type=int)
    args.add_argument("-bs", "--batch_size", default=3, type=int)  # novel relations num
    args.add_argument("-nt", "--num_tasks", default=8, type=int)
    
    # model setting
    args.add_argument("-t", "--temperature", default=0.5, type=float)
    args.add_argument("-l", "--lambda", default=0.1, type=float)  # [0.1 for NELL, 1 for Wiki]
    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)

    # training setting
    args.add_argument("-gpu", "--device", default=0, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-bepo", "--base_epoch", default=BEPO, type=int)  # [1832 for NELL, 3584 for Wiki]
    args.add_argument("-epo", "--epoch", default=EPO, type=int)  # novel epoch
    args.add_argument("-es_p", "--early_stopping_patience", default=500, type=int) # base patience
    args.add_argument("-es_np", "--early_NOVEL_stopping_patience", default=50, type=int)    # [50 for NELL, 300 for Wiki]
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-beval_epo", "--base_eval_epoch", default=BEPO - 1, type=int)  # [1831 for NELL, 3583 for Wiki]   
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=10000, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=EPO - 1, type=int)

    # other setting    
    args.add_argument("-abla", "--ablation", default=False, type=bool)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    return params


data_dir = {
    'train_tasks': '/continual_train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/con_base100_n100_dev_tasks.json',  # continual testing evaluation
    'few_shot_dev_tasks': '/dev_tasks.json',  

    'rel2candidates': '/rel2candidates.json',
    'e1rel_e2': '/e1rel_e2.json',
    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
}
