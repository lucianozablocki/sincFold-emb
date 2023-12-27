import subprocess
import sys

commands = []
data_dir = sys.argv[1]
working_dir = sys.argv[2]
device_id = sys.argv[3] 
mode = sys.argv[4]
partitions = sys.argv[5].split(',')
# partitions is something like 0,1,2,3,4
for idx in partitions:
    # for every partition on the list, generate the train/test command

    # file containing all the embeddings representations of Archive II dataset
    all_embeddings_file = "data/all_repr_archiveii.pt"
    # where to store .txt file (epoch :train_loss train_f1 val_loss val_f1 format)
    # and weights from training 
    results_dir = f"{working_dir}-{idx}"
    if mode == 'train':
        train_file = f"{data_dir}/train-partition-{idx}.csv"
        valid_file = f"{data_dir}/famfold-data/valid-partition-{idx}.csv"
        # make a list 
        commands.append(
            f"sincFold -d cuda:{device_id} {mode} {train_file} \
                {all_embeddings_file} --valid-file {valid_file} \
                -o {results_dir} -r {idx}"
        )
    elif mode == 'test':
        test_file = f"{data_dir}/test-partition-{idx}.csv"
        commands.append(
            f"sincFold -d cuda:{device_id} {mode} {test_file} \
                {all_embeddings_file} \
                -w {results_dir}/weights.pmt -o {results_dir}-test -r {idx}test"
        )
    else:
        print('you must specify a mode')
for command in commands:
    # print(command)
    subprocess.call(command, shell=True)
