import json
import random



def print_session_continual_val_num():
    base_rel_num = 73
    novel_seesion = 7
    novel_rel = 7
    continual = json.load(open('Wiki/con_base100_n100_dev_tasks.json'))
    count = []
    for j, (k, v) in enumerate(continual.items()):
        count.append(len(v) - 3)
    print(sum(count[0:base_rel_num]))
    for i in range(novel_seesion):
        print(sum(count[base_rel_num+novel_rel*i:base_rel_num+novel_rel*(i+1)]))

if __name__ == "__main__":
    few_dev = json.load(open("Wiki/continual_train_tasks.json"))
    print(len(few_dev))

# if __name__ == "__main__":
#     wiki_train = json.load(open('Wiki/train_tasks.json'))
#     rel2candidates = json.load(open("Wiki/rel2candidates.json"))
#     rel = sorted(list(wiki_train.keys()))

#     random.seed(41)
#     val = {}
#     continual_train = {}
#     for k, (rel, triples) in enumerate(wiki_train.items()):
#         if len(rel2candidates[rel]) <= 10 or len(triples) <= 10:
#             continue
#         val[rel] = random.sample(triples, k=int(len(triples)*0.3))
#         continual_train[rel] = [i for i in triples if i not in val[rel]]
    
#     print(len(continual_train))
#     with open("Wiki/continual_dev_tasks.json", 'w') as f:
#         json.dump(val, f)
#     with open("Wiki/continual_train_tasks.json", 'w') as f:
#         json.dump(continual_train, f)
              
# if __name__ == "__main__":
#     train = json.load(open("Wiki/continual_train_tasks.json"))
#     rel2candidates = json.load(open("Wiki/rel2candidates.json"))
#     rel = sorted(list(train.keys()))

#     for j, (k, v) in enumerate(train.items()):
#         if len(rel2candidates[k]) <= 10 or len(train[k]) <= 10:
#             print(k)


# if __name__ == "__main__":
#     con_dev = json.load(open("NELL/continual_dev_tasks.json"))
#     con_dev = dict(sorted(con_dev.items(), key=lambda d: d[0]))
#     con_new = {}
#     for j, (k, v) in enumerate(con_dev.items()):
#         con_new[k] = []
#         random.shuffle(v)
#         if j < 30:
#             for i in range(6):
#                 con_new[k].append(v[i])
#         else:
#             for i in range(len(v)):
#                 con_new[k].append(v[i])

#     json.dump(con_new, open('NELL/con_base100_n100_dev_tasks.json', 'w'))


# if __name__ == "__main__":
#     con_dev = json.load(open("Wiki/continual_dev_tasks.json"))
#     con_dev = dict(sorted(con_dev.items(), key=lambda x: x[0]))
#     con_new = {}
#     for j, (k, v) in enumerate(con_dev.items()):
#         con_new[k] = []
#         random.shuffle(v)
#         if j < 73:
#             for i in range(12):
#                 con_new[k].append(v[i])
#         else:
#             for i in range(len(v)):
#                 con_new[k].append(v[i])
    
#     assert len(con_new) == 122
#     json.dump(con_new, open('Wiki/con_base100_n100_dev_tasks.json', 'w'))


# if __name__ == "__main__":
#     few_dev = json.load(open("Wiki/dev_tasks.json"))
#     con_new = {}
#     count = []
#     half_dev = {}
#     for j, (k, v) in enumerate(few_dev.items()):
#         half_dev[k] = random.sample(v, k=int(len(v)*0.2))
#         print(len(half_dev[k]))
    
    
#     assert len(half_dev) == 16
#     json.dump(half_dev, open('Wiki/0.2_dev.json', 'w'))
    

