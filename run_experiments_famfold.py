import subprocess
import sys
import os

commands = []
data_dir = sys.argv[1]
working_dir = sys.argv[2]
device_id = sys.argv[3] 
mode = sys.argv[4]
partitions = sys.argv[5].split(',')
# partitions is something like 0,1,2,3,4 (list of data partitions to use)
config_dir = sys.argv[6]

# Specify the directory path
# directory_path = "config-peek"
configs = []
# List all files in the directory
for entry in os.listdir(config_dir):
    entry_path = os.path.join(config_dir, entry)
    if os.path.isfile(entry_path):
        parent_dir_name = os.path.basename(config_dir)
        file_name = entry
        configs.append(os.path.join(parent_dir_name, file_name))
print(configs)

for idx in partitions:
    # for every partition on the list, generate the train/test command

    # file containing all the embeddings representations of Archive II dataset
    all_embeddings_file = "data/all_repr_archiveii.pt"
    # where to store .txt file (epoch :train_loss train_f1 val_loss val_f1 format)
    # and weights from training 
    results_dir = f"{working_dir}-{idx}"
    if mode == 'train':
        train_file = f"{data_dir}/train-partition-{idx}.csv"
        # using test partition as validation, completely wrong, but we
        # want to know how the test loss progress over time (experimental phase)
        valid_file = f"{data_dir}/test-partition-{idx}.csv"
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
