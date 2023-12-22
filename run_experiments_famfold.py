import subprocess
import sys

commands = []
device_id = sys.argv[1] 
mode = sys.argv[2]
partitions = sys.argv[3].split(',')
# partitions is something like 0,1,2,3,4
for idx in partitions:
    # for every partition on the list, generate the train/test command

    # file containing all the embeddings representations of Archive II dataset
    all_embeddings_file = "data/all_repr_archiveii.pt"
    # where to store .txt file (epoch :train_loss train_f1 val_loss val_f1 format)
    # and weights from training 
    results_file = f"./results-emb-famfold-{idx}-pairwise"
    if mode == 'train':
        train_file = f"data/famfold-data/train-partition-{idx}-famfold.csv"
        valid_file = f"data/famfold-data/valid-partition-{idx}-famfold.csv"
        # make a list 
        commands.append(
            f"sincFold -d cuda:{device_id} {mode} {train_file} \
                {all_embeddings_file} --valid-file {valid_file} \
                -o {results_file} -r {idx}famfold-pairwise"
        )
    elif mode == 'test':
        test_file = f"data/famfold-data/test-partition-{idx}-famfold.csv"
        commands.append(
            f"sincFold -d cuda:{device_id} {mode} {test_file} \
                {all_embeddings_file} \
                -w {results_file}/weights.pmt -o {results_file}-test -r {idx}famfoldtest-pairwise"
        )
    else:
        print('you must specify a mode')
for command in commands:
    # print(command)
    subprocess.call(command, shell=True)
