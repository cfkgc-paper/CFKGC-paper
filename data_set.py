import argparse
import json

parser = argparse.ArgumentParser(description='generate continual learning dataset')
parser.add_argument('--val_rate', default=0.3, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    dev_tasks = {}
    train_tasks = {}
    dataset = {'train_tasks': json.load(open('NELL/train_tasks.json'))}

    for k, v in dataset['train_tasks'].items():
        dev_tasks[k] = []
        train_tasks[k] = []
        for i in range(len(v)):
            if i < int(args.val_rate * len(v)):
                dev_tasks[k].append(v.pop())
            else:
                train_tasks[k].append(v.pop())

    with open("NELL/continual_train_tasks.json", "w") as outfile:
        json.dump(train_tasks, outfile)
    with open("NELL/continual_dev_tasks.json", "w") as outfile:
        json.dump(dev_tasks, outfile)
